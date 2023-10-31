import casadi as cd
import numpy as np
import scipy.sparse as sp_sparse

from do_mpc.differentiator._nlpdifferentiator import NLPDifferentiator, DoMPCDifferentiator

class NLPDifferentiator_SecondOrder(NLPDifferentiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_size_metrics(self):
        
        self.n_x = self.nlp["x"].shape[0]
        self.n_g = self.nlp["g"].shape[0]
        self.n_p = self.nlp["p"].shape[0]
        self.n_z = 2 * self.n_x + self.n_g

        self.n_x_unreduced = self.n_x
        self.n_z_unreduced = self.n_z
        self.n_p_unreduced = self.n_p

        if self.status.reduced_nlp:
            self.n_x_unreduced = self.nlp_unreduced["x"].shape[0]
            self.n_p_unreduced = self.nlp_unreduced["p"].shape[0]

    def _prepare_differentiator(self):
        super()._prepare_differentiator()
        self.status.prepare_first_order_differentiator = True

        # 7. Get symbolic expressions for second order sensitivity matrices
        self._prepare_second_order_sensitivity_matrices()

    def _prepare_second_order_sensitivity_matrices(self):
        self._get_C_matrices()
        self.status.sym_KKT_system = True
    
    def _get_C_matrices(self):
        first_order_sensitivities = cd.SX.sym("first_order_sensitivities", (self.n_z, self.n_p))

        p = self.nlp["p"]
        z = self.nlp["z"]
        C = []
        for idx_p in range(self.n_p):
            A_2 = cd.reshape(cd.jacobian(self.A_sym, p[idx_p]), self.A_sym.shape)
            
            sens_row = first_order_sensitivities[:, idx_p]
            A_1 = []
            for idx_z in range(self.n_z):
                # NOTE: This is horribly slow. Double check if the expression can be simplified due to some differentiation rules (reversing chain rule? product rule?)
                A_1_col = cd.reshape(cd.jacobian(self.A_sym, z[idx_z]), self.A_sym.shape) @ sens_row
                A_1.append(A_1_col)
            A_1 = cd.horzcat(*A_1)

            A = A_1 + A_2

            B_3 = cd.reshape(cd.jacobian(self.B_sym, p[idx_p]), self.B_sym.shape)
            
            B_1 = []
            for idx_p2 in range(self.n_p):
                # NOTE: This is horribly slow. Double check if the expression can be simplified due to some differentiation rules (reversing chain rule? product rule?)
                B_1_col = cd.reshape(cd.jacobian(self.A_sym, p[idx_p2]), self.A_sym.shape) @ sens_row
                B_1.append(B_1_col)
            B_1 = cd.horzcat(*B_1)

            B = B_1 + B_3

            C.append(B + A @ first_order_sensitivities)
        C = cd.horzcat(*C)
        self.C_sym = cd.simplify(C)
        self.C_func = cd.Function("C_func", [z, p, first_order_sensitivities], [self.C_sym], ["z_opt", "p_opt", "first_order_sensitivities"], ["C"])
        return
    
    def _get_second_order_sensitivity_matrices(self, z_num: cd.DM, p_num: cd.DM, dz_dp_num: cd.DM):
        A_num = self.A_func(z_num, p_num)       
        C_num = self.C_func(z_num, p_num, dz_dp_num)
        return A_num, C_num

    def _calculate_second_order_sensitivities(self, z_num: cd.DM, p_num: cd.DM, dz_dp_num: cd.DM, where_cons_active: np.ndarray):
        
        # returns
        residuals = None

        A_num, C_num = self._get_second_order_sensitivity_matrices(z_num, p_num, dz_dp_num)
        A_num, C_num = self._reduce_sensitivity_matrices(A_num, C_num, where_cons_active)

        if self.settings.check_rank:
            self._check_rank(A_num)

        try:
            param_sens = self._solve_linear_system(A_num, C_num, lin_solver=self.settings.lin_solver)
        # except np.linalg.LinAlgError:
        except:
            if self.settings.lstsq_fallback:
                print("Solving LSE failed. Falling back to least squares solution.")
                param_sens = self._solve_linear_system(A_num, C_num, lin_solver="lstsq")
            else:
                raise np.linalg.LinAlgError("Solving LSE failed.")
            
        if self.settings.track_residuals:
            residuals = self._track_residuals(A_num, C_num, param_sens)
        return param_sens, residuals
    
    def _remove_unused_sym_vars(self):
        """
        Reduces the NLP by removing symbolic variables for x and p that are not contained in the objective function or the constraints.

        """
        # detect undetermined symbolic variables
        undet_opt_x_idx, det_opt_x_idx = self._detect_undetermined_sym_var("x")
        undet_opt_p_idx, det_opt_p_idx = self._detect_undetermined_sym_var("p")
        
        # copy nlp and nlp_bounds
        nlp_red = self.nlp.copy()
        nlp_bounds_red = self.nlp_bounds.copy()

        # adapt nlp
        nlp_red["x"] = self.nlp["x"][det_opt_x_idx]
        nlp_red["p"] = self.nlp["p"][det_opt_p_idx]

        # adapt nlp_bounds
        nlp_bounds_red["lbx"] = self.nlp_bounds["lbx"][det_opt_x_idx]
        nlp_bounds_red["ubx"] = self.nlp_bounds["ubx"][det_opt_x_idx]

        det_sym_idx_dict = {"opt_x":det_opt_x_idx, "opt_p":det_opt_p_idx}
        undet_sym_idx_dict = {"opt_x":undet_opt_x_idx, "opt_p":undet_opt_p_idx}

        N_vars_to_remove = len(undet_sym_idx_dict["opt_x"])+len(undet_sym_idx_dict["opt_p"])
        if N_vars_to_remove > 0:
            self.nlp_unreduced, self.nlp_bounds_unreduced = self.nlp, self.nlp_bounds
            self.nlp, self.nlp_bounds = nlp_red, nlp_bounds_red
            self.status.reduced_nlp = True
            # self.flags["fully_determined_nlp"] = True
        else:
            self.status.reduced_nlp = False
            # self.flags["fully_determined_nlp"] = True
            print("NLP formulation does not contain unused variables.")
        self.det_sym_idx_dict, self.undet_sym_idx_dict = det_sym_idx_dict, undet_sym_idx_dict


    def differentiate_twice(self, nlp_sol: dict, convert_to_tensor: bool = True):
        """
        Differentiates the derivative of the NLP solution.
        """

        # Calculate the first derivative
        # dx_dp_num, dlam_dp_num, residuals_1, LICQ_status, SC_status, where_cons_active = self.differentiate(nlp_sol, nlp_sol["p"])
        dx_dp_num, dlam_dp_num = self.differentiate(nlp_sol, nlp_sol["p"])

        # get parameters of optimal solution
        p_num = nlp_sol["p"]
        
        # reduce NLP solution if necessary
        if self.status.reduced_nlp:
            nlp_sol, p_num = self._reduce_nlp_solution_to_determined(nlp_sol, p_num)

        

        # extract active primal and dual solution
        z_num, where_cons_active = self._extract_active_primal_dual_solution(nlp_sol)

        # calculate second order parametric sensitivities
        dz_dp_num = cd.vertcat(dx_dp_num, dlam_dp_num)
        dz_dp_num = dz_dp_num[:, self.det_sym_idx_dict["opt_p"]]
        param_second_order_sens, residuals_2 = self._calculate_second_order_sensitivities(z_num, p_num, dz_dp_num, where_cons_active)


        # map sensitivities to original decision variables and lagrange multipliers
        d2x_dp_num_red2, d2lam_dp_num_red2 = self._map_param_sens(param_second_order_sens, where_cons_active)
        if self.status.reduced_nlp:
            d2x_dp_num2, d2lam_dp_num2 = self._map_second_order_param_sens_to_full(d2x_dp_num_red2, d2lam_dp_num_red2)
        else:
            d2x_dp_num2 = d2x_dp_num_red2
            d2lam_dp_num2 = d2lam_dp_num_red2

        if convert_to_tensor:
            d2x_dp_num2 = self._convert_to_tensor(d2x_dp_num2)
            d2lam_dp_num2 = self._convert_to_tensor(d2lam_dp_num2)

        return dx_dp_num, dlam_dp_num, d2x_dp_num2, d2lam_dp_num2

    def _convert_to_tensor(self, array: cd.DM) -> np.ndarray:
        if isinstance(array, cd.DM):
            array = array.full()

        tensor_array = np.empty((array.shape[0], self.n_p_unreduced, self.n_p_unreduced))
        for idx in range(self.n_p_unreduced):
            lower = idx*self.n_p_unreduced
            upper = (idx+1)*self.n_p_unreduced
            tensor_array[:, :, idx] = array[:, lower:upper]

        return tensor_array
    
    def _map_dlamdp(self, param_sens: np.ndarray, where_cons_active: np.ndarray) -> np.ndarray:
        """
        Maps the parametric sensitivities to the original sensitivities of the lagrange multipliers.
        """
        dlam_dp = np.zeros((self.n_g+self.n_x, param_sens.shape[-1]))
        assert len(where_cons_active) == param_sens.shape[0]-self.n_x, "Number of non-zero dual variables does not match number of parametric sensitivities for lagrange multipliers."
        
        if sp_sparse.issparse(param_sens):
            dlam_dp[where_cons_active,:] = param_sens[self.n_x:,:].toarray()
        else:
            dlam_dp[where_cons_active,:] = param_sens[self.n_x:,:]

        return dlam_dp

    def _map_param_sens_to_full(self, dx_dp_num_red: cd.DM, dlam_dp_num_red: cd.DM) -> tuple[cd.DM]:
        """
        Maps the reduced parametric sensitivities to the full decision variables.
        """
        idx_x_determined, idx_p_determined = self.det_sym_idx_dict["opt_x"], self.det_sym_idx_dict["opt_p"]

        dx_dp_num = np.zeros((self.n_x_unreduced,self.n_p_unreduced))
        # dx_dp_num = sp_sparse.csc_matrix(dx_dp_num)
        dx_dp_num[idx_x_determined[:,None],idx_p_determined] = dx_dp_num_red
        
        dlam_dp_num = np.zeros((self.n_g+self.n_x_unreduced,self.n_p_unreduced))
        # dlam_dp_num = sp_sparse.csc_matrix(dlam_dp_num)

        idx_lam_determined = np.hstack([np.arange(0,self.n_g,dtype=np.int64),idx_x_determined+self.n_g])

        dlam_dp_num[idx_lam_determined[:,None],idx_p_determined] = dlam_dp_num_red

        return dx_dp_num, dlam_dp_num
    
    def _map_second_order_param_sens_to_full(self, d2x_dp2_num_red: cd.DM, d2lam_dp2_num_red: cd.DM) -> tuple[cd.DM]:
        """
        Maps the reduced parametric sensitivities to the full decision variables.
        """
        idx_x_determined, idx_p_determined = self.det_sym_idx_dict["opt_x"], self.det_sym_idx_dict["opt_p"]

        idx_p_determined = np.concatenate([k * self.n_p_unreduced + idx_p_determined for k in range(self.n_p_unreduced) if k in idx_p_determined])

        d2x_dp2_num = np.zeros((self.n_x_unreduced, self.n_p_unreduced ** 2))
        # dx_dp_num = sp_sparse.csc_matrix(dx_dp_num)
        d2x_dp2_num[idx_x_determined[:,None], idx_p_determined] = d2x_dp2_num_red
        
        d2lam_dp2_num = np.zeros((self.n_g+self.n_x_unreduced,self.n_p_unreduced ** 2))
        # dlam_dp_num = sp_sparse.csc_matrix(dlam_dp_num)

        idx_lam_determined = np.hstack([np.arange(0,self.n_g,dtype=np.int64),idx_x_determined+self.n_g])

        d2lam_dp2_num[idx_lam_determined[:,None], idx_p_determined] = d2lam_dp2_num_red

        return d2x_dp2_num, d2lam_dp2_num


class DoMPCDifferentiator_RL(DoMPCDifferentiator):

    def __init__(self, optimizer, *args, **kwargs):
        super(DoMPCDifferentiator_RL, self).__init__(optimizer,*args, **kwargs)
        # del self.optimizer
    
    def _get_do_mpc_nlp_sol(self, mpc_object):
        nlp_sol = {}
        nlp_sol["x"] = cd.vertcat(mpc_object.opt_x_num)
        nlp_sol["x_unscaled"] = cd.vertcat(mpc_object.opt_x_num_unscaled)
        nlp_sol["g"] = cd.vertcat(mpc_object.opt_g_num)
        nlp_sol["lam_g"] = cd.vertcat(mpc_object.lam_g_num)
        nlp_sol["lam_x"] = cd.vertcat(mpc_object.lam_x_num)
        nlp_sol["p"] = cd.vertcat(mpc_object.opt_p_num)
        return nlp_sol
    
    def differentiate(self, optimizer):
        nlp_sol = self._get_do_mpc_nlp_sol(optimizer)
        dx_dp_num, dlam_dp_num = super(DoMPCDifferentiator, self).differentiate(nlp_sol, nlp_sol["p"])
        
        # rescale dx_dp_num
        dx_dp_num = cd.times(dx_dp_num,self.x_scaling_factors.tocsc())

        return dx_dp_num, dlam_dp_num

    def _map_param_sens_to_full(self, dx_dp_num_red, dlam_dp_num_red):
        """
        Maps the reduced parametric sensitivities to the full decision variables.
        """
        idx_x_determined, idx_p_determined = self.det_sym_idx_dict["opt_x"], self.det_sym_idx_dict["opt_p"]

        dx_dp_num = np.zeros((self.n_x_unreduced,self.n_p_unreduced))
        # dx_dp_num = sp_sparse.csc_matrix(dx_dp_num)
        dx_dp_num[idx_x_determined[:,None],idx_p_determined] = dx_dp_num_red
        
        dlam_dp_num = np.zeros((self.n_g+self.n_x_unreduced,self.n_p_unreduced))
        # dlam_dp_num = sp_sparse.csc_matrix(dlam_dp_num)

        idx_lam_determined = np.hstack([np.arange(0,self.n_g,dtype=np.int64),idx_x_determined+self.n_g])

        dlam_dp_num[idx_lam_determined[:,None],idx_p_determined] = dlam_dp_num_red

        return dx_dp_num, dlam_dp_num

class DoMPCSecondOrderDifferentiator_RL(NLPDifferentiator_SecondOrder):
        
    def __init__(self, optimizer, *args, **kwargs):
        nlp, nlp_bounds = self._get_do_mpc_nlp(optimizer)        
        self.optimizer = optimizer
        self.x_scaling_factors = self.optimizer.opt_x_scaling.master
        super().__init__(nlp, nlp_bounds, *args, **kwargs)

    def _get_do_mpc_nlp(self, mpc_object):
        """
        This function is used to extract the symbolic expressions and bounds of the underlying NLP of the MPC.
        It is used to initialize the NLPDifferentiator class.
        """

        # 1 get symbolic expressions of NLP
        nlp = {'x': cd.vertcat(mpc_object.opt_x), 'f': mpc_object.nlp_obj, 'g': mpc_object.nlp_cons, 'p': cd.vertcat(mpc_object.opt_p)}

        # 2 extract bounds
        nlp_bounds = {}
        nlp_bounds['lbg'] = mpc_object.nlp_cons_lb
        nlp_bounds['ubg'] = mpc_object.nlp_cons_ub
        nlp_bounds['lbx'] = cd.vertcat(mpc_object._lb_opt_x)
        nlp_bounds['ubx'] = cd.vertcat(mpc_object._ub_opt_x)

        # return nlp, nlp_bounds
        # return {"nlp": nlp.copy(), "nlp_bounds": nlp_bounds.copy()}
        return nlp, nlp_bounds
    
    def _get_do_mpc_nlp_sol(self, mpc_object):
        nlp_sol = {}
        nlp_sol["x"] = cd.vertcat(mpc_object.opt_x_num)
        nlp_sol["x_unscaled"] = cd.vertcat(mpc_object.opt_x_num_unscaled)
        nlp_sol["g"] = cd.vertcat(mpc_object.opt_g_num)
        nlp_sol["lam_g"] = cd.vertcat(mpc_object.lam_g_num)
        nlp_sol["lam_x"] = cd.vertcat(mpc_object.lam_x_num)
        nlp_sol["p"] = cd.vertcat(mpc_object.opt_p_num)
        return nlp_sol

    def differentiate_twice(self, optimizer, convert_to_tensor: bool = True):
        """
        Differentiates the derivative of the NLP solution.
        """

        nlp_sol = self._get_do_mpc_nlp_sol(optimizer)

        dx_dp_num, dlam_dp_num, d2x_dp_num2, d2lam_dp_num2 = super().differentiate_twice(nlp_sol, convert_to_tensor=convert_to_tensor)

        # rescale dx_dp_num
        dx_dp_num = cd.times(dx_dp_num, self.x_scaling_factors.tocsc())

        # rescale d2x_dp_num
        d2x_dp_num2 = d2x_dp_num2 * self.x_scaling_factors.full().reshape(-1, 1, 1)
        return dx_dp_num, dlam_dp_num, d2x_dp_num2, d2lam_dp_num2