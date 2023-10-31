import pickle
import os

import casadi as cd
import casadi.tools as castools
import numpy as np

from do_mpc.controller import MPC
import copy
import time
import warnings


class RL_AC_MPC(MPC):

    def __init__(self, model = None):
        if model is None:
            print("Warning: RL_AC_MPC requires a model in the initialization phase. You need to load a model later via RL_AC_MPC.load(path).")
            return
        
        super().__init__(model)
           
        # Updating the data fields
        self.settings.gamma = 1

        # Default Parameters (param. details in set_param method):

        self.settings.nlpsol_opts.update(
            {
                "ipopt.fixed_variable_treatment": "make_constraint",
            }
        )


        self.flags['prepare_qnlp'] = False
    


    # Setting up the objective
    def set_mterm(self, mterm = None):
        if mterm is None:
            self.mterm = cd.DM(0)
        else:
            self.mterm = mterm
        
        # Check if mterm is valid:
        if not isinstance(self.mterm, (cd.DM, cd.SX, cd.MX)):
            raise Exception('mterm must be of type casadi.DM, casadi.SX or casadi.MX. You have: {}.'.format(type(mterm)))

        assert self.mterm.shape == (1,1), 'mterm must have shape=(1,1). You have {}'.format(mterm.shape)

        _x, _u, _z, _tvp, _p = self.model['x','u','z','tvp','p']

        
        # TODO: This function should be evaluated with scaled variables.
        self.mterm_fun = cd.Function('mterm', [_x, _z, _tvp, _p], [self.mterm], ["_x", "_z", "_tvp", "_p"], ["mterm"])

        # Check if mterm use invalid variables as inputs.
        # For the check we evaluate the function with dummy inputs and expect a DM output.
        err_msg = '{} contains invalid symbolic variables as inputs. Must contain only: {}'
        try:
            self.mterm_fun(_x(0), _z(0), _tvp(0), _p(0))
        except:
            raise Exception(err_msg.format('mterm','_x, _z, _tvp, _p'))

        return

    def set_lterm(self, lterm = None):

        if lterm is None:
            self.lterm = cd.DM(0)
        else:
            self.lterm = lterm
        
        # Check if lterm is valid:
        if not isinstance(self.lterm, (cd.DM, cd.SX, cd.MX)):
            raise Exception('lterm must be of type casadi.DM, casadi.SX or casadi.MX. You have: {}.'.format(type(lterm)))

        assert self.lterm.shape == (1,1), 'lterm must have shape=(1,1). You have {}'.format(lterm.shape)

        _x, _u, _z, _tvp, _p = self.model['x','u','z','tvp','p']
        
        self.lterm_fun = cd.Function('lterm', [_x, _u, _z, _tvp, _p], [self.lterm], ["_x", "_u", "_z", "_tvp", "_p"], ["lterm"])

        # Check if lterm use invalid variables as inputs.
        # For the check we evaluate the function with dummy inputs and expect a DM output.
        err_msg = '{} contains invalid symbolic variables as inputs. Must contain only: {}'
        try:
            self.lterm_fun(_x(0), _u(0), _z(0), _tvp(0), _p(0))
        except:
            raise Exception(err_msg.format('lterm','_x, _u, _z, _tvp, _p'))

        return
    
    def set_lambdaterm(self, lambdaterm = None):

        if lambdaterm is None:
            self.lambdaterm = cd.DM(0)
        else:
            self.lambdaterm = lambdaterm
        
        # Check if lambdaterm is valid:
        if not isinstance(self.lambdaterm, (cd.DM, cd.SX, cd.MX)):
            raise Exception('lambdaterm must be of type casadi.DM, casadi.SX or casadi.MX. You have: {}.'.format(type(lambdaterm)))

        assert self.lambdaterm.shape == (1,1), 'lambdaterm must have shape=(1,1). You have {}'.format(lambdaterm.shape)

        _x, _u, _z, _tvp, _p = self.model['x','u','z','tvp','p']
        
        self.lambdaterm_fun = cd.Function('lambdaterm', [_x, _z, _tvp, _p], [self.lambdaterm], ["_x", "_z", "_tvp", "_p"], ["lambdaterm"])

        # Check if lambdaterm use invalid variables as inputs.
        # For the check we evaluate the function with dummy inputs and expect a DM output.
        err_msg = '{} contains invalid symbolic variables as inputs. Must contain only: {}'
        try:
            self.lambdaterm_fun(_x(0), _z(0), _tvp(0), _p(0))
        except:
            raise Exception(err_msg.format('lambdaterm','_x, _z, _tvp, _p'))

        return
    
    def set_fterm(self, fterm = None):

        if fterm is None:
            self.fterm = cd.DM(0)
        else:
            self.fterm = fterm
        
        # Check if fterm is valid:
        if not isinstance(self.fterm, (cd.DM, cd.SX, cd.MX)):
            raise Exception('fterm must be of type casadi.DM, casadi.SX or casadi.MX. You have: {}.'.format(type(fterm)))

        assert self.fterm.shape == (1,1), 'fterm must have shape=(1,1). You have {}'.format(fterm.shape)

        _x, _u, _z, _tvp, _p = self.model['x','u','z','tvp','p']
        
        self.fterm_fun = cd.Function('fterm', [_x, _u, _z, _tvp, _p], [self.fterm], ["_x", "_u", "_z", "_tvp", "_p"], ["fterm"])

        # Check if fterm use invalid variables as inputs.
        # For the check we evaluate the function with dummy inputs and expect a DM output.
        err_msg = '{} contains invalid symbolic variables as inputs. Must contain only: {}'
        try:
            self.fterm_fun(_x(0), _u(0), _z(0), _tvp(0), _p(0))
        except:
            raise Exception(err_msg.format('fterm','_x, _u, _z, _tvp, _p'))

        return

    def set_objective(self, mterm=None, lterm=None, lambdaterm = None, fterm = None):

        assert self.flags['setup'] == False, 'Cannot call .set_objective after .setup().'

        self.set_mterm(mterm = mterm)
        self.set_lterm(lterm = lterm)
        self.set_fterm(fterm = fterm)
        self.set_lambdaterm(lambdaterm = lambdaterm)

        self.flags['set_objective'] = True
        return


    # Modifying methods for the setup
    def _prepare_nlp(self):
        """
        Internal method. See detailed documentation with optimizer.prepare_nlp
        """
        nl_cons_input = self.model['x', 'u', 'z', 'tvp', 'p']
        self._setup_nl_cons(nl_cons_input)
        self._check_validity()

        # Obtain an integrator (collocation, discrete-time) and the amount of intermediate (collocation) points
        ifcn, n_total_coll_points = self._setup_discretization()
        n_branches, n_scenarios, child_scenario, parent_scenario, branch_offset = self._setup_scenario_tree()

        # How many scenarios arise from the scenario tree (robust multi-stage MPC)
        n_max_scenarios = self.n_combinations ** self.settings.n_robust

        # If open_loop option is active, all scenarios (at a given stage) have the same input.
        if self.settings.open_loop:
            n_u_scenarios = 1
        else:
            # Else: Each scenario has its own input.
            n_u_scenarios = n_max_scenarios

        # How many slack variables (for soft constraints) are introduced over the horizon.
        if self.settings.nl_cons_single_slack:
            n_eps = 1
        else:
            n_eps = self.settings.n_horizon

        # Create struct for optimization variables:
        self._opt_x = opt_x = self.model.sv.sym_struct([
            # One additional point (in the collocation dimension) for the final point.
            cd.tools.entry('_x', repeat=[self.settings.n_horizon+1, n_max_scenarios,
                                1+n_total_coll_points], struct=self.model._x),
            cd.tools.entry('_z', repeat=[self.settings.n_horizon+1, n_max_scenarios,
                                max(n_total_coll_points,1)], struct=self.model._z),
            cd.tools.entry('_u', repeat=[self.settings.n_horizon, n_u_scenarios], struct=self.model._u),
            cd.tools.entry('_eps', repeat=[n_eps, n_max_scenarios], struct=self._eps),
        ])
        self.n_opt_x = self._opt_x.shape[0]
        # NOTE: The entry _x[k,child_scenario[k,s,b],:] starts with the collocation points from s to b at time k
        #       and the last point contains the child node
        # NOTE: Currently there exist dummy collocation points for the initial state (for each branch)

        # Create scaling struct as assign values for _x, _u, _z.
        self.opt_x_scaling = opt_x_scaling = opt_x(1)
        opt_x_scaling['_x'] = self._x_scaling
        opt_x_scaling['_z'] = self._z_scaling
        opt_x_scaling['_u'] = self._u_scaling
        # opt_x are unphysical (scaled) variables. opt_x_unscaled are physical (unscaled) variables.
        self.opt_x_unscaled = opt_x_unscaled = opt_x(opt_x.cat * opt_x_scaling)


        # Create struct for optimization parameters:
        self._opt_p = opt_p = self.model.sv.sym_struct([
            cd.tools.entry('_x0', struct=self.model._x),
            cd.tools.entry('_tvp', repeat=self.settings.n_horizon+1, struct=self.model._tvp),
            cd.tools.entry('_p', repeat=self.n_combinations, struct=self.model._p),
            cd.tools.entry('_u_prev', struct=self.model._u),
        ])
        # self.rterm_factor = self.rterm_factor_func(self._opt_p["_p", 0])
        _w = self.model._w(0)

        self.n_opt_p = opt_p.shape[0]

        # Dummy struct with symbolic variables
        self.aux_struct = self.model.sv.sym_struct([
            cd.tools.entry('_aux', repeat=[self.settings.n_horizon, n_max_scenarios], struct=self.model._aux_expression)
        ])
        # Create mutable symbolic expression from the struct defined above.
        self._opt_aux = opt_aux = self.model.sv.struct(self.aux_struct)

        self.n_opt_aux = opt_aux.shape[0]

        self._lb_opt_x = opt_x(-np.inf)
        self._ub_opt_x = opt_x(np.inf)

        # Initialize objective function and constraints
        obj = cd.DM(0)
        cons = []
        cons_lb = []
        cons_ub = []

        # Initial condition:
        cons.append(opt_x['_x', 0, 0, -1]-opt_p['_x0']/self._x_scaling)

        cons_lb.append(np.zeros((self.model.n_x, 1)))
        cons_ub.append(np.zeros((self.model.n_x, 1)))

        # NOTE: Weigthing factors for the tree assumed equal. They could be set from outside
        # Weighting factor for every scenario
        omega = [1. / n_scenarios[k + 1] for k in range(self.settings.n_horizon)]
        omega_delta_u = [1. / n_scenarios[k + 1] for k in range(self.settings.n_horizon)]
        
        # For all control intervals
        for k in range(self.settings.n_horizon):
            # For all scenarios (grows exponentially with n_robust)
            for s in range(n_scenarios[k]):
                # For all childen nodes of each node at stage k, discretize the model equations

                # Scenario index for u is always 0 if self.open_loop = True
                s_u = 0 if self.settings.open_loop else s
                for b in range(n_branches[k]):
                    # Obtain the index of the parameter values that should be used for this scenario
                    current_scenario = b + branch_offset[k][s]

                    if k == 0:
                        # The rotational cost term to the cost function
                        obj += self.lambdaterm_fun(opt_x_unscaled['_x', 0, 0, -1], opt_x_unscaled['_z', 0, 0, -1], opt_p['_tvp', 0], opt_p['_p', current_scenario])


                    # Compute constraints and predicted next state of the discretization scheme
                    col_xk = cd.vertcat(*opt_x['_x', k+1, child_scenario[k][s][b], :-1])
                    col_zk = cd.vertcat(*opt_x['_z', k, child_scenario[k][s][b]])
                    [g_ksb, xf_ksb] = ifcn(opt_x['_x', k, s, -1], col_xk,
                                           opt_x['_u', k, s_u], col_zk, opt_p['_tvp', k],
                                           opt_p['_p', current_scenario], _w)

                    # Add the collocation equations
                    cons.append(g_ksb)
                    cons_lb.append(np.zeros(g_ksb.shape[0]))
                    cons_ub.append(np.zeros(g_ksb.shape[0]))

                    # Add continuity constraints
                    cons.append(xf_ksb - opt_x['_x', k+1, child_scenario[k][s][b], -1])
                    cons_lb.append(np.zeros((self.model.n_x, 1)))
                    cons_ub.append(np.zeros((self.model.n_x, 1)))

                    k_eps = min(k, n_eps-1)
                    if self.settings.nl_cons_check_colloc_points:
                        # Ensure nonlinear constraints on all collocation points
                        for i in range(n_total_coll_points):
                            nl_cons_k = self._nl_cons_fun(
                                opt_x_unscaled['_x', k, s, i], opt_x_unscaled['_u', k, s_u], opt_x_unscaled['_z', k, s, i],
                                opt_p['_tvp', k], opt_p['_p', current_scenario], opt_x_unscaled['_eps', k_eps, s])
                            cons.append(nl_cons_k)
                            cons_lb.append(self._nl_cons_lb)
                            cons_ub.append(self._nl_cons_ub)
                    else:
                        # Ensure nonlinear constraints only on the beginning of the FE
                        nl_cons_k = self._nl_cons_fun(
                            opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u], opt_x_unscaled['_z', k, s, 0],
                            opt_p['_tvp', k], opt_p['_p', current_scenario], opt_x_unscaled['_eps', k_eps, s])
                        cons.append(nl_cons_k)
                        cons_lb.append(self._nl_cons_lb)
                        cons_ub.append(self._nl_cons_ub)

                    # Add terminal constraints
                    # TODO: Add terminal constraints with an additional nl_cons

                    # Add contribution to the cost
                    obj += self.settings.gamma ** k * omega[k] * self.lterm_fun(opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u],
                                                     opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])
                    # Add slack variables to the cost
                    obj += self.settings.gamma ** k * self.epsterm_fun(opt_x_unscaled['_eps', k_eps, s])

                    # Add fterm
                    obj += self.settings.gamma ** k * omega[k] * self.fterm_fun(opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u],
                                                     opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])

                    # In the last step add the terminal cost too
                    if k == self.settings.n_horizon - 1:
                        obj += self.settings.gamma ** k * omega[k] * self.mterm_fun(opt_x_unscaled['_x', k + 1, s, -1], opt_x_unscaled['_z', k + 1, s, -1], opt_p['_tvp', k+1],
                                                         opt_p['_p', current_scenario])

                    # U regularization:
                    if k == 0:
                        obj += self.settings.gamma ** k * self.rterm_factor.master.T@((opt_x['_u', 0, s_u]-opt_p['_u_prev']/self._u_scaling)**2)
                    else:
                        obj += self.settings.gamma ** k * self.rterm_factor.master.T@((opt_x['_u', k, s_u]-opt_x['_u', k-1, parent_scenario[k][s_u]])**2)

                    # Calculate the auxiliary expressions for the current scenario:
                    opt_aux['_aux', k, s] = self.model._aux_expression_fun(
                        opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u], opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])

                    # For some reason when working with MX, the "unused" aux values in the scenario tree must be set explicitly (they are not ever used...)
                for s_ in range(n_scenarios[k],n_max_scenarios):
                    opt_aux['_aux', k, s_] = self.model._aux_expression_fun(
                        opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u], opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])

        # Set bounds for all optimization variables
        self._update_bounds()


        # Write all created elements to self:
        self._nlp_obj = obj
        self._nlp_cons = cons
        self._nlp_cons_lb = cons_lb
        self._nlp_cons_ub = cons_ub


        # Initialize copies of structures with numerical values (all zero):
        self._opt_x_num = self._opt_x(0)
        self.opt_x_num_unscaled = self._opt_x(0)
        self._opt_p_num = self._opt_p(0)
        self.opt_aux_num = self._opt_aux(0)

        self.flags['prepare_nlp'] = True

    def make_step(self, state, old_action: np.ndarray = None):
        if isinstance(old_action, (np.ndarray)):
            self._u0.master = old_action

        u0 = super().make_step(state)
        return u0

    def get_solution_dict(self):
        r = copy.copy(self.r_prev)
        r["p"] = self.opt_p_num.master.full()
        return r

    def solve(self):
        """Solves the optmization problem.

        The current problem is defined by the parameters in the
        :py:attr:`opt_p_num` CasADi structured Data.

        Typically, :py:attr:`opt_p_num` is prepared for the current iteration in the :py:func:`make_step` method.
        It is, however, valid and possible to directly set paramters in :py:attr:`opt_p_num` before calling :py:func:`solve`.

        The method updates the :py:attr:`opt_p_num` and :py:attr:`opt_x_num` attributes of the class.
        By resetting :py:attr:`opt_x_num` to the current solution, the method implicitly
        enables **warmstarting the optimizer** for the next iteration, since this vector is always used as the initial guess.

        .. warning::

            The method is part of the public API but it is generally not advised to use it.
            Instead we recommend to call :py:func:`make_step` at each iterations, which acts as a wrapper
            for :py:func:`solve`.

        :raises asssertion: Optimizer was not setup yet.

        :return: None
        :rtype: None
        """
        assert self.flags['setup'] == True, 'optimizer was not setup yet. Please call optimizer.setup().'

        solver_call_kwargs = {
            'x0': self.opt_x_num,
            'lbx': self._lb_opt_x,
            'ubx': self._ub_opt_x,
            'lbg': self.nlp_cons_lb,
            'ubg': self.nlp_cons_ub,
            'p': self.opt_p_num,
        }

        # Warmstarting the optimizer after the initial run:
        if self.flags['initial_run']:
            solver_call_kwargs.update({
                'lam_x0': self.lam_x_num,
                'lam_g0': self.lam_g_num,
            })



        r = self.S(**solver_call_kwargs)
        self.r_prev = copy.copy(r)
        # Note: .master accesses the underlying vector of the structure.
        self.opt_x_num.master = r['x']
        self.opt_f_num = r["f"]
        self.opt_x_num_unscaled.master = r['x']*self.opt_x_scaling
        self.opt_g_num = r['g']
        # Values of lagrange multipliers:
        self.lam_g_num = r['lam_g']
        self.lam_x_num = r['lam_x']
        self.solver_stats = self.S.stats()

        # Calculate values of auxiliary expressions (defined in model)
        self.opt_aux_num.master = self.opt_aux_expression_fun(
                self.opt_x_num,
                self.opt_p_num
            )
        
        # For warmstarting purposes: Flag that initial run has been completed.
        self.flags['initial_run'] = True


    def _check_validity(self):
        """Private method to be called in :py:func:`setup`. Checks if the configuration is valid and
        if the optimization problem can be constructed.
        Furthermore, default values are set if they were not configured by the user (if possible).
        Specifically, we set dummy values for the ``tvp_fun`` and ``p_fun`` if they are not present in the model.
        """
        # Objective mus be defined.
        if self.flags['set_objective'] == False:
            raise Exception('Objective is undefined. Please call .set_objective() prior to .setup().')
        # rterm should have been set (throw warning if not)
        if self.flags['set_rterm'] == False:
            cd.warnings.warn('rterm was not set and defaults to zero. Changes in the control inputs are not penalized. Can lead to oscillatory behavior.')
            time.sleep(2)
        # tvp_fun must be set, if tvp are defined in model.
        if self.flags['set_tvp_fun'] == False and self.model._tvp.size > 0:
            raise Exception('You have not supplied a function to obtain the time-varying parameters defined in model. Use .set_tvp_fun() prior to setup.')
        # p_fun must be set, if p are defined in model.
        if self.flags['set_p_fun'] == False and self.model._p.size > 0:
            raise Exception('You have not supplied a function to obtain the parameters defined in model. Use .set_p_fun() (low-level API) or .set_uncertainty_values() (high-level API) prior to setup.')

        if not isinstance(self.rterm_factor.cat, (cd.SX, cd.MX)):
            if np.any(self.rterm_factor.master.full() < 0):
                warnings.warn('You have selected negative values for the rterm penalizing changes in the control input.')
                time.sleep(2)

        # Lower bounds should be lower than upper bounds:
        for lb, ub in zip([self._x_lb, self._u_lb, self._z_lb], [self._x_ub, self._u_ub, self._z_ub]):
            bound_check = lb.cat > ub.cat
            bound_fail = [label_i for i,label_i in enumerate(lb.labels()) if bound_check[i]]
            if np.any(bound_check):
                raise Exception('Your bounds are inconsistent. For {} you have lower bound > upper bound.'.format(bound_fail))

        # Are terminal bounds for the states set? If not use default values (unless MPC is setup to not use terminal bounds)
        if np.all(self._x_terminal_ub.cat == np.inf) and self.settings.use_terminal_bounds:
            self._x_terminal_ub = self._x_ub
        if np.all(self._x_terminal_lb.cat == -np.inf) and self.settings.use_terminal_bounds:
            self._x_terminal_lb = self._x_lb

        # Set dummy functions for tvp and p in case these parameters are unused.
        if 'tvp_fun' not in self.__dict__:
            _tvp = self.get_tvp_template()

            def tvp_fun(t): return _tvp
            self.set_tvp_fun(tvp_fun)

        if 'p_fun' not in self.__dict__:
            _p = self.get_p_template(1)

            def p_fun(t): return _p
            self.set_p_fun(p_fun)


    # Datenverwaltung
    def __deepcopy__(self, memo):

        # Initialize a new instance of the class
        copied_mpc = RL_AC_MPC(self.model)


        # Copy all settings
        copied_mpc.set_param(**self.settings.__dict__)


        # Copy the objective function
        _x = copied_mpc.model._x
        _z = copied_mpc.model._z
        _u = copied_mpc.model._u
        _tvp = copied_mpc.model._tvp
        _p = copied_mpc.model._p
        objective_list = ["lterm", "mterm", "fterm", "lambdaterm"]
        objective = {term: getattr(self, term + "_fun") for term in objective_list}

        for term in objective_list:
            if term in ["lterm", "fterm"]:
                objective[term] = objective[term](_x, _u, _z, _tvp, _p)
            else:
                objective[term] = objective[term](_x, _z, _tvp, _p)
        copied_mpc.set_objective(**objective)

        copied_mpc.set_rterm(**{key: float(self.rterm_factor[key]) for key in self.rterm_factor.keys()[1:]})


        # Copy the (uncertain) parameters
        p_template = copied_mpc.get_p_template(1) # NOTE: This could cause trouble at some point
        p_template.master = self.p_fun(0).master
        copied_mpc.set_p_fun(lambda tnow: p_template)


        # Copy the (uncertain) time-varying parameters
        tvp_template = copied_mpc.get_tvp_template() # NOTE: This could cause trouble at some point
        tvp_template.master = self.tvp_fun(0).master
        copied_mpc.set_tvp_fun(lambda tnow: tvp_template)


        # Copy the bounds/state constraints
        for key in self._x_lb.keys():
            bound = self._x_lb[key]
            if isinstance(bound, cd.DM):
                bound = bound.full().copy()
            copied_mpc.bounds["lower", "_x", key] = bound

        for key in self._x_ub.keys():
            bound = self._x_ub[key]
            if isinstance(bound, cd.DM):
                bound = bound.full().copy()
            copied_mpc.bounds["upper", "_x", key] = bound

        for key in self._z_ub.keys():
            bound = self._z_ub[key]
            if isinstance(bound, cd.DM):
                bound = bound.full().copy()
            copied_mpc.bounds["upper", "_z", key] = bound

        for key in self._z_lb.keys():
            bound = self._z_lb[key]
            if isinstance(bound, cd.DM):
                bound = bound.full().copy()
            copied_mpc.bounds["lower", "_z", key] = bound

        for key in self._u_ub.keys():
            bound = self._u_ub[key]
            if isinstance(bound, cd.DM):
                bound = bound.full().copy()
            copied_mpc.bounds["upper", "_u", key] = bound

        for key in self._u_lb.keys():
            bound = self._u_lb[key]
            if isinstance(bound, cd.DM):
                bound = bound.full().copy()
            copied_mpc.bounds["lower", "_u", key] = bound


        # Copy the nonlinear constraints
        _nl_cons = self._nl_cons_fun(self.model._x, self.model._u, self.model._z, self.model._tvp, self.model._p, self._eps)
        _nl_cons += self._eps
        _nl_cons_fun = cd.Function("nl_cons_fun_copy", [self.model._x, self.model._u, self.model._z, self.model._tvp, self.model._p], [_nl_cons])
        _nl_cons = _nl_cons_fun(_x, _u, _z, _tvp, _p)
        
        nl_cons_list = []
        slack_vars_list = []
        for cons, slack_vars in zip(self.nl_cons_list[1:], self.slack_vars_list[1:]):
            expr_shape = cons["expr"].shape
            expr = _nl_cons[:expr_shape[0], :]
            _nl_cons = _nl_cons[expr_shape[0]:, :]
            cons["expr"] = expr

            slack = {
                "penalty_term_cons": slack_vars["penalty"],
                "maximum_violation": slack_vars["ub"],
            }
            slack["soft_constraint"] = False if slack_vars["penalty"] == 0 else True
            nl_cons_list.append(cons)
            slack_vars_list.append(slack)

        for cons, slack in zip(nl_cons_list, slack_vars_list):
            copied_mpc.set_nl_cons(**cons, **slack)


        # TODO: DO this for the scaling part
        for key in self.model._x.keys():
            copied_mpc.scaling["_x", key] = self.scaling["_x", key].full().copy()
        for key in self.model._z.keys()[1:]:
            copied_mpc.scaling["_z", key] = self.scaling["_z", key].full().copy()
        for key in self.model._u.keys()[1:]:
            copied_mpc.scaling["_u", key] = self.scaling["_u", key].full().copy()

        # TODO: Do this for the uncertain parameter stuff

        # Setup the final model and provide the initial state
        copied_mpc.setup()
        
        x0 = self.x0.master
        if isinstance(x0, cd.DM):
            x0 = x0.full().copy()
        copied_mpc.x0 = self.x0(x0)

        z0 = self.z0.master
        if isinstance(z0, cd.DM):
            z0 = z0.full().copy()
        copied_mpc.z0 = self.z0(z0)

        u0 = self.u0.master
        if isinstance(u0, cd.DM):
            u0 = u0.full().copy()
        copied_mpc.u0 = self.u0(u0)

        copied_mpc.set_initial_guess()

        return copied_mpc

           
    def save(self, path):
        if not path.endswith("\\"):
            path += "\\"

        if not os.path.exists(path):
            os.makedirs(path)

        not_saved_variablesAttributeError = []
        not_saved_variablesTypeError = []
        for key, value in self.__dict__.items():
            try:
                if isinstance(value, cd.tools.structure3.SXStruct):
                    lvalue = {key: value[key] for key in value.keys()}
                    value = lvalue.copy()
                if key == "p_fun":
                    value = value(0)
                with open(path + key + ".pkl", "wb") as f:
                    pickle.dump(value, f)
            except TypeError:
                not_saved_variablesTypeError.append(key)
                os.remove(path + key + ".pkl")
                
                if key == "nl_cons_list":
                    not_saved_variablesTypeError.pop(-1)
                    nl_cons_list = []
                    for item in value:
                        nl_cons_list.append((item["expr_name"], item["ub"], item["expr"].shape))
                    
                    with open(path + key + ".pkl", "wb") as f:
                        pickle.dump(nl_cons_list, f)
            except AttributeError:
                not_saved_variablesAttributeError.append(key)
                os.remove(path + key + ".pkl")

        return
    
    def load(self, path):
        
        # with open(path + "mpc.pkl", "rb") as f:
        #     self = pickle.load(f)

        with open(path + "model.pkl", "rb") as f:
            model = pickle.load(f)
        self.__init__(model)

        files = os.listdir(path)

        # with open(path + "_eps.pkl", "rb") as f:
        #     value = pickle.load(f)
        # value
        for file in files:
            key = file[:-4]
            with open(path + file, "rb") as f:
                value = pickle.load(f)
            setattr(self, key, value)

        self.flags = {key: False for key in self.flags}

        # Load subfunctions of objective function
        _x = self.model._x
        _u = self.model._u
        _z = self.model._z
        _tvp = self.model._tvp
        _p = self.model._p
        _eps = self._eps
        del self._eps

        with open(path + "lterm_fun.pkl", "rb") as f:
            lterm_fun = pickle.load(f)
        lterm = lterm_fun(_x, _u, _z, _tvp, _p)

        with open(path + "mterm_fun.pkl", "rb") as f:
            mterm_fun = pickle.load(f)
        mterm = mterm_fun(_x, _z, _tvp, _p)

        with open(path + "fterm_fun.pkl", "rb") as f:
            fterm_fun = pickle.load(f)
        fterm = fterm_fun(_x, _u, _z, _tvp, _p)

        with open(path + "lambdaterm_fun.pkl", "rb") as f:
            lambdaterm_fun = pickle.load(f)
        lambdaterm = lambdaterm_fun(_x, _z, _tvp, _p)
        self.set_objective(mterm = mterm, lterm = lterm, fterm = fterm, lambdaterm = lambdaterm)
        


        # Set nl_constraints
        _nl_cons = self._nl_cons_fun(_x, _u, _z, _tvp, _p, _eps)
        del self._nl_cons_fun
        _nl_cons += _eps
        _nl_cons_list = []
        _nl_cons_list.extend(copy.deepcopy(self.nl_cons_list))
        self.nl_cons_list.append("dummy")
        self.nl_cons_list.pop(-1)
        _slack_vars_list = []
        _slack_vars_list.extend(copy.deepcopy(self.slack_vars_list))
        self.slack_vars_list.append("dummy")
        self.slack_vars_list.pop(-1)
        self.nl_cons_list = [{'expr_name': 'default', 'expr': cd.DM([]), 'ub': cd.DM([])}]
        self.slack_vars_list = [{"slack_name": "default", "shape": 0, "ub": cd.DM([]), "penalty": 0}]

        # Gather the nl_constraints
        nl_cons_list = []

        for expr_name, ub, shape in _nl_cons_list[1:]:
            expr = []
            for index, item in enumerate(cd.vertsplit(_nl_cons)):
                expr.append(item)
                if index >= shape[0] - 1:
                    break
            expr = cd.vcat(expr)
            cons = {
                "expr_name": expr_name,
                "expr": expr,
                "ub": ub
            }
            nl_cons_list.append(cons)

            _nl_cons = _nl_cons[shape[0]:, :]

        for param in nl_cons_list:
            for item in _slack_vars_list:
                if item["slack_name"] == expr_name:
                    slack_exp = item

            param["soft_constraint"] = False if slack_exp["penalty"] == 0 else True
            param["penalty_term_cons"] = slack_exp["penalty"]
            param["maximum_violation"] = slack_exp["ub"]

            self.set_nl_cons(**param)

        self.flags["set_rterm"] = True

        # TODO: Not sure, but maybe I have to do something for the scaling
        # TODO: Not sure, but maybe I also have to specify something about the uncertain parameters
        return