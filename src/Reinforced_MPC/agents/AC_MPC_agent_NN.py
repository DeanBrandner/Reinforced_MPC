
import random
import pickle

import casadi as cd
import numpy as np

import tensorflow as tf

from dataclasses import dataclass
from collections import deque
from sys import stdout

from .AC_MPC_agent import AC_MPC_agent
from ..tools.DoMPCDifferentiator_RL import DoMPCDifferentiator_RL as Differentiator
from ..tools.DoMPCDifferentiator_RL import DoMPCSecondOrderDifferentiator_RL as SecondOrderDifferentiator


class Critic_Scaler():
    def __init__(self, log_basis: float = 10., log_offset: float = 1e-1):
        self.log_basis = log_basis
        self.log_offset = log_offset
    
    def fit_input(self, data: np.ndarray):
        self.input_shift = data.mean(axis = 0)
        self.input_scale = data.std(axis = 0)
        return
    
    def transform_input(self, data: np.ndarray):
        data = (data - self.input_shift) / self.input_scale
        return data
    
    def fit_transform_input(self, data: np.ndarray):
        self.fit_input(data = data)
        return self.transform_input(data = data)
    
    def reverse_transform_input(self, scaled_data: np.ndarray):
        data = scaled_data * self.input_scale + self.input_shift
        return data
    
    def fit_output(self, data: np.ndarray):
        data = np.log(data + self.log_offset) / np.log(self.log_basis)
        self.output_shift = data.min()
        self.output_scale = data.max() - data.min()
        return
    
    def transform_output(self, data: np.ndarray):
        data = np.log(data + self.log_offset) / np.log(self.log_basis)
        data = (data - self.output_shift) / self.output_scale
        return data
    
    def fit_transform_output(self, data: np.ndarray):
        self.fit_output(data = data)
        return self.transform_output(data = data)
    
    def reverse_transform_output(self, scaled_data: np.ndarray):
        data = scaled_data * self.output_scale + self.output_shift
        data = self.log_basis ** data - self.log_offset
        return data
    
    def reverse_tranform_gradient(self, scaled_output: np.ndarray, scaled_gradient: np.ndarray):
        # Rescale input part
        # Scalar part
        scalar_part = np.log(self.log_basis) * self.output_scale * (self.reverse_transform_output(scaled_output) + self.log_offset)
        
        # Matrix part
        scaled_gradient = tf.expand_dims(scaled_gradient, axis = -1)
        diag_output_scaling = np.tile(np.diag(1/self.input_scale.flatten()), [scaled_output.shape[0], 1, 1])
        matrix_part = diag_output_scaling @ scaled_gradient
        matrix_part = matrix_part[:,:,0]

        # Combine both parts
        gradient = scalar_part * matrix_part
        return gradient

    def reverse_transform_hessian(self, scaled_output: np.ndarray, scaled_gradient: np.ndarray, scaled_hessian: np.ndarray):
        # First part
        scalar_part_1 = np.log(self.log_basis) * self.output_scale
        rescaled_gradient = self.reverse_tranform_gradient(scaled_output = scaled_output, scaled_gradient = scaled_gradient)
        rescaled_gradient = np.expand_dims(rescaled_gradient, axis = -1)
        scaled_gradient = np.expand_dims(scaled_gradient, axis = -1)
        diag_input_scaling = np.tile(np.diag(1/self.input_scale.flatten()), [scaled_output.shape[0], 1, 1])
        matrix_part_1 = diag_input_scaling @ scaled_gradient
        hessian_part_1 = scalar_part_1 * matrix_part_1 @ np.transpose(rescaled_gradient, axes = [0, 2, 1])

        # Second part
        scalar_part_2 = np.log(self.log_basis) * self.output_scale * (self.reverse_transform_output(scaled_output) + self.log_offset)
        scalar_part_2 = tf.expand_dims(scalar_part_2, axis = -1)
        matrix_part_2 = diag_input_scaling @ scaled_hessian @ diag_input_scaling
        hessian_part_2 = scalar_part_2 * matrix_part_2

        # Combine both parts
        hessian = hessian_part_1 + hessian_part_2
        return hessian
    



class AC_MPC_agent_NN(AC_MPC_agent):

    def __init__(self, actor_mpc = None, critic = None, shortterm_memory_size: int = 50, n_critic_horizon: int = 20, critic_settings: dict = {}, **kwargs):

        super().__init__(actor_mpc, **kwargs)

        self.mpc_differentiator = Differentiator(self.replay_mpc)
        self.mpc_differentiator.settings.check_SC = False
        self.mpc_differentiator.settings.check_LICQ = False
        self.mpc_differentiator.settings.lin_solver = "lstsq"

        self.flags["differentiator_initialized"] = True

        # TODO: Currently not sure if this can also be a deque since the policy only changes very slowly.
        self.episode = []

        self.critic = critic
        self.critic_scaler = Critic_Scaler(**critic_settings)

        self.n_samples = 0
        self.shortterm_memory = deque(maxlen = shortterm_memory_size)

        self.n_critic_horizon = n_critic_horizon

        

    def remember_episode(self):
        self.memory.append(self.episode)
        self.shortterm_memory.append(self.episode)
        self.episode = []
        self.n_samples = np.sum([len(episode) for episode in self.memory])

    def remember(self, old_state: np.ndarray, old_action: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done_flag: bool):
        self.episode.append((old_state, old_action, action, reward, next_state, done_flag))

    def _get_minibatch_episodes(self, batch_size: int):
        episode_batch = random.sample(self.memory, batch_size)
        return episode_batch

    def _get_critic_features(self, old_states: np.ndarray, old_actions: np.ndarray, actions: np.ndarray, next_states: np.ndarray, next_actions: np.ndarray, rewards: np.ndarray, first_time: bool = False):

        state_action_features = np.hstack([old_states[:,:, 0], old_actions[:,:, 0], actions[:,:, 0]])

        if first_time:    
            targets = rewards[:,:,0]
            return state_action_features, targets
        next_state_action_features = np.hstack([next_states[:,:, 0], actions[:,:, 0], next_actions[:,:, 0]])
        next_state_action_features = self.critic_scaler.transform_input(next_state_action_features)
        targets = rewards[:,:,0] + self.settings.gamma * self.critic_scaler.reverse_transform_output(self.critic(next_state_action_features))

        return state_action_features, targets
    
    def train_critic(self, old_states: np.ndarray, old_actions: np.ndarray, actions: np.ndarray, next_states: np.ndarray, next_actions: np.ndarray, rewards: np.ndarray, **kwargs):
        first_time = True

        last_loss = 1000.
        for idx in range(self.n_critic_horizon):

            bar_len = 60
            filled_len = int(round(bar_len * (idx + 1) / self.n_critic_horizon))
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            stdout.write(f'Critic progress (n_horizon):\t[' + bar  + f'] {idx + 1}/{self.n_critic_horizon} {(100.0 * (idx + 1)/self.n_critic_horizon):.1f}%\t loss: {last_loss:.3e}\r')

            state_features, targets = self._get_critic_features(old_states, old_actions, actions, next_states, next_actions, rewards, first_time)

            scaled_state_features = self.critic_scaler.fit_transform_input(state_features)
            scaled_target_features = self.critic_scaler.fit_transform_output(targets)
            

            history = self.critic.fit(
                x = scaled_state_features,
                y = scaled_target_features,
                **kwargs)
            
            last_loss = history.history["loss"][-1]
            
            first_time = False
        print()
        return history
    

    def get_closed_loop_cost(self, n_last_samples: (int, str) = "all"):
        if n_last_samples == "all":
            n_last_samples = len(self.memory)

        closed_loop_cost = []
        for episode in self.memory[-n_last_samples:]:
            cost = episode[-1][3] * 1/(1 - self.settings.gamma)
            for step in episode[::-1]:
                cost = step[3] + self.settings.gamma * cost
                # cost += step[3]
            closed_loop_cost.append(cost)
        closed_loop_cost = np.mean(closed_loop_cost)
        return closed_loop_cost
    
    def _get_closed_loop_cost(self):
        
        closed_loop_cost = []
        for episode in self.shortterm_memory:
            # cost = 0
            if self.settings.gamma < 1:
                cost = episode[-1][3] * 1/(1 - self.settings.gamma)
            else:
                cost = 0
                
            for step in episode[::-1]:
                cost = step[3] + self.settings.gamma * cost
                # cost += step[3]
            closed_loop_cost.append(cost)
        closed_loop_cost = np.mean(closed_loop_cost)
        return closed_loop_cost

    def _get_batch(self, n_samples: (int, str)):
        if n_samples == "all":
            n_samples = self.n_samples

        batch = []
        for episode in self.memory:
            batch.extend(episode)
        
        if len(batch) > n_samples:
            n_samples = len(batch)
        return batch

    def _get_minibatches(self, batch: list, batch_size: int = 32):
        random.shuffle(batch)
        minibatches = []
        
        while len(batch) > batch_size:
            minibatches.append(batch[:batch_size])
            batch = batch[batch_size:]
        minibatches.append(batch)

        return minibatches
    
    def _get_minibatch_data(self, minibatch: list):
        
        old_states = []
        old_actions = []
        actions = []
        rewards = []
        next_states = []

        for item in minibatch:
            old_states.append(np.expand_dims(item[0].T, 0))
            old_actions.append(np.expand_dims(item[1].T, 0))
            actions.append(np.expand_dims(item[2].T, 0))
            rewards.append(item[3])
            next_states.append(np.expand_dims(item[4].T, 0))

        old_states = np.concatenate(old_states, axis = 0)
        old_actions = np.concatenate(old_actions, axis = 0)
        actions = np.concatenate(actions, axis = 0)
        next_states = np.concatenate(next_states, axis = 0)
        rewards = np.array(rewards).reshape(-1, 1, 1)

        old_states = np.transpose(old_states, axes = [0, 2, 1])
        old_actions = np.transpose(old_actions, axes = [0, 2, 1])
        actions = np.transpose(actions, axes = [0, 2, 1])
        next_states = np.transpose(next_states, axes = [0, 2, 1])
        rewards = np.transpose(rewards, axes = [0, 2, 1])

        return old_states, old_actions, actions, next_states, rewards
    
    def _get_replay_actions(self, old_states: np.ndarray, old_actions: np.ndarray):

        opt_actions = np.empty(old_actions.shape) # (batch_size, n_actions, 1)
        for idx, (s_old, a_old) in enumerate(zip(old_states, old_actions)):
            bar_len = 60
            filled_len = int(round(bar_len * (idx + 1) / old_states.shape[0]))
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            stdout.write(f'Next optimal actions:\t[' + bar  + f'] {idx + 1}/{old_states.shape[0]} {(100.0 * (idx + 1)/old_states.shape[0]):.1f}%\r')
            
            self.replay_mpc.u0 = a_old.copy()

            action_opt_local = self.replay_mpc.make_step(s_old.copy())
            opt_actions[idx, :, :]  = action_opt_local
        
        self.replay_mpc.reset_history()
        print()
        return opt_actions
    
    def _get_replay_actions_and_sensitivities(self, old_states: np.ndarray, old_actions: np.ndarray):
        n_a = self.actor.model._u.shape[0]

        n_x0 = self.replay_mpc.opt_p(0)["_x0"].shape[0]
        n_tvp = self.replay_mpc.opt_p(0)["_tvp", 0].shape[0]
        n_p = self.replay_mpc.opt_p(0)["_p", 0].shape[0]
        n_u_prev = self.replay_mpc.opt_p(0)["_u_prev"].shape[0]

        opt_actions = np.empty(old_actions.shape) # (batch_size, n_actions, 1)
        sensitivities = np.empty((old_actions.shape[0], n_a, n_p)) # (batch_size, n_actions, n_parameters)
        for idx, (s_old, a_old) in enumerate(zip(old_states, old_actions)):
            bar_len = 60
            filled_len = int(round(bar_len * (idx + 1) / old_states.shape[0]))
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            stdout.write(f'Optimal action and Sensitivity:\t[' + bar  + f'] {idx + 1}/{old_states.shape[0]} {(100.0 * (idx + 1)/old_states.shape[0]):.1f}%\r')
            
            self.replay_mpc.u0 = a_old.copy()

            action_opt_local = self.replay_mpc.make_step(s_old.copy())
            opt_actions[idx, :, :]  = action_opt_local

            dx_dp_num, dlam_dp_num = self.mpc_differentiator.differentiate(self.replay_mpc)

            dx_dp_num = dx_dp_num[:, n_x0 + n_tvp: - n_u_prev].full()
            relevant_sens = []
            for col in dx_dp_num.T:
                u_sens = self.actor.opt_x(col)["_u", 0, 0].full()
                relevant_sens.append(u_sens)
            sensitivities[idx, :, :] = np.concatenate(relevant_sens, axis = -1)
        
        self.replay_mpc.reset_history()
        print()
        return opt_actions, sensitivities
            
    def _get_closed_loop_gradient(self, old_states: np.ndarray, old_actions: np.ndarray, actions: np.ndarray, sensitivities: np.ndarray):     

        old_states_tf = tf.convert_to_tensor(old_states[:,:,0], dtype = tf.float32)
        old_actions_tf = tf.convert_to_tensor(old_actions[:,:,0], dtype = tf.float32)
        actions_tf = tf.convert_to_tensor(actions[:,:,0], dtype = tf.float32)

        critic_input = tf.concat([old_states_tf, old_actions_tf, actions_tf], axis = 1)
        critic_input = self.critic_scaler.transform_input(critic_input)

        with tf.GradientTape(persistent = False) as tape:
            tape.watch(critic_input)
            scaled_Q_values = self.critic(critic_input)

        scaled_grad_Q_input = tape.gradient(scaled_Q_values, critic_input).numpy() # This is the scaled one. It must be rescaled to get the real gradient
        grad_Q_input = self.critic_scaler.reverse_tranform_gradient(scaled_Q_values, scaled_grad_Q_input)
        grad_Q_a = grad_Q_input[:, -actions_tf.shape[-1]:]
        grad_Q_a = np.expand_dims(grad_Q_a, axis = -1)

        grad = np.transpose(sensitivities, axes = [0, 2, 1]) @ grad_Q_a
        grad = np.mean(grad, axis = 0)

        return grad
    
    def _get_update(self, gradient: np.ndarray):
        update = -self.settings.learning_rate * gradient
        return update
    
    def _normalize_update(self, update: np.ndarray, normalize_value: float):
        norm = np.sqrt(update.T @ update)
        if norm > normalize_value:
            update = normalize_value * (update / norm)

        return update

    def _update_parameters(self, update: np.ndarray):

        p_template = self.actor.get_p_template(1)
        p_template.master = self.actor.p_fun(0)["_p", 0] + update # NOTE: Minus!

        self.actor.set_p_fun(lambda t_now: p_template)
        self.actor_behavior.set_p_fun(lambda t_now: p_template)
        self.replay_mpc.set_p_fun(lambda t_now: p_template)

        return
    
    # Training section
    def replay(
        self,
        critic_training_kwargs: dict = {},
        normalize: bool = False,
        normalize_value: float = 1.0,
        ):
       
        batch = self._get_batch(n_samples = "all")
        old_states, old_actions, actions, next_states, rewards = self._get_minibatch_data(minibatch = batch)
        
        next_opt_actions = self._get_replay_actions(old_states = next_states, old_actions = actions)
        self.train_critic(old_states = old_states, old_actions= old_actions, actions = actions, next_states = next_states, next_actions = next_opt_actions, rewards = rewards, **critic_training_kwargs)

        opt_actions, sensitivities = self._get_replay_actions_and_sensitivities(old_states = old_states, old_actions = old_actions)
        grad = self._get_closed_loop_gradient(old_states = old_states, old_actions = old_actions, actions = opt_actions, sensitivities = sensitivities)
        
        update = self._get_update(gradient = grad)
        if normalize:
            update = self._normalize_update(update = update, normalize_value = normalize_value)
        
        self._update_parameters(update = update)
        # self.memory.clear()
        return True

    def save_critic(self, path: str):
        self.critic.save(path + "critic.hdf5")
        return
    
    def save(self, path: str):
        if not path.endswith("\\"):
            path += "\\"
        
        super().save(path)
        self.save_critic(path = path)
        return
    
    @staticmethod
    def load_critic(path: str):
        critic = tf.keras.models.load_model(path)
        return critic
    
    @classmethod
    def load(cls, path: str):
        # NOTE This is only experimental and not tested yet
        if not path.endswith("\\"):
            path += "\\"

        actor = cls.load_actor(path + "actor")
        critic = cls.load_critic(path + "critic.hdf5")

        with open(path + "settings.pkl", "rb") as f:
            settings = pickle.load(f)
        with open(path + "memory.pkl", "rb") as f:
            memory = pickle.load(f)

        agent = cls(actor_mpc = actor, critic = critic)
        agent.settings = settings
        agent.memory = memory

        agent.n_samples = np.sum([len(episode) for episode in agent.memory])
        return agent


class AC_MPC_agent_NN_TR(AC_MPC_agent_NN):

    def __init__(self, actor_mpc = None, critic = None, TR_kwargs: dict = {}, **kwargs):
        super().__init__(actor_mpc = actor_mpc, critic = critic, **kwargs)
        self.old_closed_loop_cost = None

        self.TR_settings = TR_settings()
        for key, value in TR_kwargs.items():
            setattr(self.TR_settings, key, value)

        # Fix all hyperparameters, which are not needed
        self.settings.learning_rate = 1.0
        self.settings.batch_size = None

    def _revert_update(self):
        self._update_parameters(update = -self.old_update)
        return 
    
    def _update_TR_settings(self, rho: float):
        if rho < 0.25:
            self.TR_settings.delta *= 0.25
            if self.TR_settings.delta < self.TR_settings.delta_min:
                self.TR_settings.delta = self.TR_settings.delta_min
        else:
            if rho > 0.75 and np.linalg.norm(self.old_update) <= self.TR_settings.delta:
                self.TR_settings.delta = min(2 * self.TR_settings.delta, self.TR_settings.delta_max)
        return
    
    def _continue_check(self, gradient: np.ndarray):
        continue_bool = True
        if np.linalg.norm(gradient) < self.TR_settings.delta_min:
            continue_bool = False
        return continue_bool
    
    def _get_update(self, gradient: np.ndarray):
        update = -gradient
        update_norm = np.linalg.norm(update)
        if update_norm > self.TR_settings.delta:
            update = self.TR_settings.delta * (update / update_norm)
        return update
    
    def replay(self, critic_training_kwargs: dict = {}):
        continue_bool = True
        closed_loop_cost = self._get_closed_loop_cost()

        if self.old_closed_loop_cost is None:

            minibatch = self._get_batch(n_samples = "all")
            old_states, old_actions, actions, next_states, rewards = self._get_minibatch_data(minibatch = minibatch)
            
            next_opt_actions = self._get_replay_actions(old_states = next_states, old_actions = actions)
            self.train_critic(old_states = old_states, old_actions= old_actions, actions = actions, next_states = next_states, next_actions = next_opt_actions, rewards = rewards, **critic_training_kwargs)

            opt_actions, sensitivities = self._get_replay_actions_and_sensitivities(old_states = old_states, old_actions = old_actions)

            grad = self._get_closed_loop_gradient(old_states = old_states, old_actions = old_actions, actions = opt_actions, sensitivities = sensitivities)

            update = self._get_update(gradient = grad)
            self.old_update = update
            self._update_parameters(update = update)
            # self.memory.clear()

            self.old_closed_loop_cost = closed_loop_cost
            self.old_grad = grad

            continue_bool = self._continue_check(grad)

            return continue_bool
        
        numerator = float(self.old_closed_loop_cost - closed_loop_cost)
        denominator = -float(self.old_grad.T @ self.old_update)
        rho = numerator / denominator
        
        self._update_TR_settings(rho)
        
        #NOTE: This check could be faulty
        if (rho <= self.TR_settings.eta) and (np.linalg.norm(self.old_update) > self.TR_settings.delta_min or not np.isclose(np.linalg.norm(self.old_update), self.TR_settings.delta_min)):
            self._revert_update()
            grad = self.old_grad
        else:
            minibatch = self._get_batch(n_samples = "all")
            old_states, old_actions, actions, next_states, rewards = self._get_minibatch_data(minibatch = minibatch)
            
            next_opt_actions = self._get_replay_actions(old_states = next_states, old_actions = actions)
            self.train_critic(old_states = old_states, old_actions= old_actions, actions = actions, next_states = next_states, next_actions = next_opt_actions, rewards = rewards, **critic_training_kwargs)

            opt_actions, sensitivities = self._get_replay_actions_and_sensitivities(old_states = old_states, old_actions = old_actions)          

            grad = self._get_closed_loop_gradient(old_states = old_states, old_actions = old_actions, actions = opt_actions, sensitivities = sensitivities)
            
            self.old_closed_loop_cost = closed_loop_cost

        update = self._get_update(gradient = grad)
        
        self.old_update = update
        self.old_grad = grad

        self._update_parameters(update = update)

        continue_bool = self._continue_check(grad)

        return continue_bool



class AC_MPC_agent_NN_QuasiNewton(AC_MPC_agent_NN):

    def __init__(self, actor_mpc = None, critic = None, **kwargs):
        super().__init__(actor_mpc = actor_mpc, critic = critic, **kwargs)

        self.mpc_differentiator = SecondOrderDifferentiator(self.replay_mpc)
        self.mpc_differentiator.settings.check_SC = False
        self.mpc_differentiator.settings.check_LICQ = False
        self.mpc_differentiator.settings.lin_solver = "lstsq"

    def _get_replay_actions_and_sensitivities(self, old_states: np.ndarray, old_actions: np.ndarray):

        n_a = self.actor.model._u.shape[0]

        n_x0 = self.replay_mpc.opt_p(0)["_x0"].shape[0]
        n_tvp = self.replay_mpc.opt_p(0)["_tvp", 0].shape[0]
        n_p = self.replay_mpc.opt_p(0)["_p", 0].shape[0]
        n_u_prev = self.replay_mpc.opt_p(0)["_u_prev"].shape[0]

        opt_actions = np.empty(old_actions.shape) # (batch_size, n_actions, 1)
        sensitivities = np.empty((old_actions.shape[0], n_a, n_p)) # (batch_size, n_actions, n_parameters)
        second_order_sensitivities = np.empty((old_actions.shape[0], n_a, n_p, n_p)) # (batch_size, n_actions, n_parameters, n_parameters)
        for idx, (s_old, a_old) in enumerate(zip(old_states, old_actions)):
            bar_len = 60
            filled_len = int(round(bar_len * (idx + 1) / old_states.shape[0]))
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            stdout.write(f'Optimal action and Sensitivity:\t[' + bar  + f'] {idx + 1}/{old_states.shape[0]} {(100.0 * (idx + 1)/old_states.shape[0]):.1f}%\r')
            
            self.replay_mpc.u0 = a_old.copy()

            action_opt_local = self.replay_mpc.make_step(s_old.copy())
            opt_actions[idx, :, :]  = action_opt_local

            # dx_dp_num, dlam_dp_num, residuals_1, d2x_dp_num2, d2lam_dp_num2, residuals_2, LICQ_status, SC_status, where_cons_active = self.mpc_differentiator.differentiate_twice(self.replay_mpc)
            dx_dp_num, dlam_dp_num, d2x_dp_num2, d2lam_dp_num2 = self.mpc_differentiator.differentiate_twice(self.replay_mpc)


            dx_dp_num = dx_dp_num[:, n_x0 + n_tvp: - n_u_prev].full()
            relevant_sens = []
            for col in dx_dp_num.T:
                u_sens = self.actor.opt_x(col)["_u", 0, 0].full()
                relevant_sens.append(u_sens)
            sensitivities[idx, :, :] = np.concatenate(relevant_sens, axis = -1)

            d2x_dp_num2 = d2x_dp_num2[:, n_x0 + n_tvp: - n_u_prev, n_x0 + n_tvp: - n_u_prev]
            for dim_2 in range(n_p):
                for dim_3 in range(n_p):
                    element = d2x_dp_num2[:, dim_2, dim_3]
                    element = self.actor.opt_x(element)["_u", 0, 0].full().flatten()
                    second_order_sensitivities[idx, :, dim_2, dim_3] = element
        
        self.replay_mpc.reset_history()
        print()
        return opt_actions, sensitivities, second_order_sensitivities
    
    def _get_closed_loop_hessian_part_1(self, old_states: np.ndarray, old_actions: np.ndarray, actions: np.ndarray, second_order_sensitivities: np.ndarray):

        old_states_tf = tf.convert_to_tensor(old_states[:, :, 0], dtype = tf.float32)
        old_actions_tf = tf.convert_to_tensor(old_actions[:, :, 0], dtype = tf.float32)
        actions_tf = tf.convert_to_tensor(actions[:, :, 0], dtype = tf.float32)

        critic_input = tf.concat([old_states_tf, old_actions_tf, actions_tf], axis = 1)
        critic_input = self.critic_scaler.transform_input(critic_input)

        with tf.GradientTape(persistent = False) as tape:
            tape.watch(critic_input)
            scaled_Q_values = self.critic(critic_input)

        scaled_grad_Q_a = tape.gradient(scaled_Q_values, critic_input).numpy() # This is the scaled one. It must be rescaled to get the real gradient
        grad_Q_a = self.critic_scaler.reverse_tranform_gradient(scaled_Q_values, scaled_grad_Q_a)
        grad_Q_a = np.expand_dims(grad_Q_a, axis = -1)
        grad_Q_a = np.expand_dims(grad_Q_a, axis = -1)
        

        H1 = second_order_sensitivities * grad_Q_a
        H1 = np.sum(H1, axis = 1)
        H1 = np.mean(H1, axis = 0)
        return H1

    def _get_closed_loop_hessian_part_2(self, old_states: np.ndarray, old_actions: np.ndarray, actions: np.ndarray, sensitivities: np.ndarray):
        
        old_states_tf = tf.convert_to_tensor(old_states[:, :, 0], dtype = tf.float32)
        old_actions_tf = tf.convert_to_tensor(old_actions[:, :, 0], dtype = tf.float32)
        actions_tf = tf.convert_to_tensor(actions[:, :, 0], dtype = tf.float32)

        critic_input = tf.concat([old_states_tf, old_actions_tf, actions_tf], axis = 1)
        critic_input = self.critic_scaler.transform_input(critic_input)

        with tf.GradientTape(persistent = False) as second_order_tape:
            second_order_tape.watch(critic_input)
            with tf.GradientTape(persistent = False) as tape:
                tape.watch(critic_input)
                scaled_Q_values = self.critic(critic_input)
        
            scaled_grad_Q_input = tape.gradient(scaled_Q_values, critic_input)
        scaled_hess_Q_input = second_order_tape.batch_jacobian(scaled_grad_Q_input, critic_input)
        hess_Q_input = self.critic_scaler.reverse_transform_hessian(scaled_Q_values, scaled_grad_Q_input, scaled_hess_Q_input)
        d2Qda2 = hess_Q_input[:, -actions_tf.shape[-1]:, -actions_tf.shape[-1]:]

        H2 = np.transpose(sensitivities, axes = [0, 2, 1]) @ d2Qda2 @ sensitivities
        H2 = np.mean(H2, axis = 0)
        return H2
    
    def _get_closed_loop_hessian(self, old_states: np.ndarray, old_actions: np.ndarray, actions: np.ndarray, sensitivities: np.ndarray, second_order_sensitivities: np.ndarray):
        
        H1 = self._get_closed_loop_hessian_part_1(old_states, old_actions, actions, second_order_sensitivities)
        H2 = self._get_closed_loop_hessian_part_2(old_states, old_actions, actions, sensitivities)

        hessian = H1 + H2
        return hessian
    

    def _regularize_hessian(self, hessian: np.ndarray, beta: float = 1e-3):
        eigenvalues, eigenvectors = np.linalg.eig(hessian)
        min_eigenvalue = np.min(np.amin((eigenvalues, np.zeros(eigenvalues.shape[0])), axis = 0))
        
        if min_eigenvalue < 0:
            eigenvalues = np.abs(eigenvalues)
            hessian = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        return hessian
    
    def _get_update(self, gradient: np.ndarray, hessian: np.ndarray, fallback: str = "gradient"):
        hessian = self._regularize_hessian(hessian)

        try:
            update = np.linalg.solve(hessian, -gradient) # NOTE: Theoretically this should be -gradient, but because of the definitiion of the update rule, the signs must be switched.
        except:
            if fallback == "gradient":
                print("Hessian is not invertible. Using gradient instead.")
                update = gradient
            elif fallback == "lstsq":
                print("Hessian is not invertible. Using gradient instead.")
                update = np.linalg.lstsq(hessian, gradient, rcond = None)[0]
            else:
                raise ValueError("Fallback method {} not known. You can choose between gradient and lstsq.".format(fallback))
            
        update *= self.settings.learning_rate
        return update
    
    def replay(self,
        critic_training_kwargs: dict = {},
        normalize: bool = False,
        normalize_value: float = 1.0,
        ):

        batch = self._get_batch(n_samples = "all")
        old_states, old_actions, actions, next_states, rewards = self._get_minibatch_data(minibatch = batch)
        
        next_opt_actions = self._get_replay_actions(old_states = next_states, old_actions = actions)
        self.train_critic(old_states = old_states, old_actions= old_actions, actions = actions, next_states = next_states, next_actions = next_opt_actions, rewards = rewards, **critic_training_kwargs)

        opt_actions, sensitivities, second_order_sensitivities = self._get_replay_actions_and_sensitivities(old_states = old_states, old_actions = old_actions)
        grad = self._get_closed_loop_gradient(old_states = old_states, old_actions = old_actions, actions = opt_actions, sensitivities = sensitivities)
        hessian = self._get_closed_loop_hessian(old_states = old_states, old_actions = old_actions, actions = opt_actions, sensitivities = sensitivities, second_order_sensitivities = second_order_sensitivities)
        
        update = self._get_update(gradient = grad, hessian = hessian)
        if normalize:
            update = self._normalize_update(update = update, normalize_value = normalize_value)
        
        self._update_parameters(update = update)
        return True




@dataclass
class TR_settings:
    """
    This is a dataclass for the settings of the AC_MPC_agent.
    """
    delta_max: float = 1
    delta: float = delta_max
    delta_min: float = 1e-6
    eta: float = 0 # 0 means actually every update is performed which points into decending direction. eta = 0.25 means that only "good" updates are performed


class AC_MPC_agent_NN_QuasiNewton_TR(AC_MPC_agent_NN_QuasiNewton):

    def __init__(self, actor_mpc = None, critic = None, TR_kwargs: dict = {}, **kwargs):
        super().__init__(actor_mpc = actor_mpc, critic = critic, **kwargs)
        self.old_closed_loop_cost = None

        self.TR_settings = TR_settings()
        for key, value in TR_kwargs.items():
            setattr(self.TR_settings, key, value)

        # Fix all hyperparameters, which are not needed
        self.settings.learning_rate = 1.0
        self.settings.batch_size = None

    def _get_update(self, gradient: np.ndarray, hessian: np.ndarray):
        reg_hessian = self._regularize_hessian(hessian)
        update = np.linalg.lstsq(reg_hessian, -gradient, rcond = None)[0] # NOTE: Theoretically this should be -gradient, but because of the definitiion of the update rule, the signs must be switched.

        if np.linalg.norm(update) < self.TR_settings.delta:
            return update, reg_hessian, 0
        
        lam = cd.SX.sym("lam", 1)
        eigenvalues, eigenvectors = np.linalg.eig(reg_hessian)
        p = cd.simplify(-(eigenvectors @ (cd.diag(1/(eigenvalues + lam/self.TR_settings.delta)))@ eigenvectors.T) @ gradient)
        p_norm = cd.sqrt(p.T @ p)
        func = cd.simplify(1/self.TR_settings.delta -  1/p_norm) 
        TR_lam_finder_func = cd.Function("TR_lam_finder_func", [lam], [func])
        TR_lam_finder = cd.rootfinder("TR_lam_finder", "newton", TR_lam_finder_func, {"error_on_fail": True})

        init_guess = (self.TR_settings.delta - min(eigenvalues)) * self.TR_settings.delta/2   
        lam_sol = TR_lam_finder(init_guess)
        lam_sol = float(lam_sol / self.TR_settings.delta)

        if lam_sol < 0:
            return

        p_func = cd.Function("p_func", [lam], [p])
        update = p_func(lam_sol * self.TR_settings.delta)
        
        descent_direction = gradient.T @ update
        if descent_direction > 0:
            update *= -1
        return update, reg_hessian, lam_sol
    
    def _revert_update(self):
        self._update_parameters(update = -self.old_update)
        return 
    
    def _update_TR_settings(self, rho: float):
        if rho < 0.25:
            self.TR_settings.delta *= 0.25
            if self.TR_settings.delta < self.TR_settings.delta_min:
                self.TR_settings.delta = self.TR_settings.delta_min
        else:
            if rho > 0.75 and np.linalg.norm(self.old_update) <= self.TR_settings.delta:
                self.TR_settings.delta = min(2 * self.TR_settings.delta, self.TR_settings.delta_max)
        return
    
    def _continue_check(self, gradient: np.ndarray):
        continue_bool = True
        if np.linalg.norm(gradient) < self.TR_settings.delta_min:
            continue_bool = False
        return continue_bool
    
    def replay(self, critic_training_kwargs: dict = {}):

        closed_loop_cost = self._get_closed_loop_cost()


        if self.old_closed_loop_cost is None:

            minibatch = self._get_batch(n_samples = "all")
            old_states, old_actions, actions, next_states, rewards = self._get_minibatch_data(minibatch = minibatch)
            
            next_opt_actions = self._get_replay_actions(old_states = next_states, old_actions = actions)
            self.train_critic(old_states = old_states, old_actions= old_actions, actions = actions, next_states = next_states, next_actions = next_opt_actions, rewards = rewards, **critic_training_kwargs)

            opt_actions, sensitivities, second_order_sensitivities = self._get_replay_actions_and_sensitivities(old_states = old_states, old_actions = old_actions)          

            grad = self._get_closed_loop_gradient(old_states = old_states, old_actions = old_actions, actions = opt_actions, sensitivities = sensitivities)
            hessian = self._get_closed_loop_hessian(old_states = old_states, old_actions = old_actions, actions = opt_actions, sensitivities = sensitivities, second_order_sensitivities = second_order_sensitivities)
            
            update, hessian, lam = self._get_update(gradient = grad, hessian = hessian)
            self.old_update = update
            self._update_parameters(update = update)

            self.old_closed_loop_cost = closed_loop_cost
            self.old_grad = grad
            self.old_hessian = hessian
            self.old_lam = lam

            continue_bool = self._continue_check(grad)

            return continue_bool
        
        numerator = float(self.old_closed_loop_cost - closed_loop_cost)
        denominator = -float(self.old_grad.T @ self.old_update + 0.5 * self.old_update.T @ self.old_hessian @ self.old_update)
        rho = numerator / denominator
        
        self._update_TR_settings(rho)
        
            
        if (rho <= self.TR_settings.eta) and (np.linalg.norm(self.old_update) > self.TR_settings.delta_min or not np.isclose(np.linalg.norm(self.old_update), self.TR_settings.delta_min)):
            self._revert_update()
            grad = self.old_grad
            hessian = self.old_hessian
        else:

            minibatch = self._get_batch(n_samples = "all")
            old_states, old_actions, actions, next_states, rewards = self._get_minibatch_data(minibatch = minibatch)
            
            next_opt_actions = self._get_replay_actions(old_states = next_states, old_actions = actions)
            self.train_critic(old_states = old_states, old_actions= old_actions, actions = actions, next_states = next_states, next_actions = next_opt_actions, rewards = rewards, **critic_training_kwargs)

            opt_actions, sensitivities, second_order_sensitivities = self._get_replay_actions_and_sensitivities(old_states = old_states, old_actions = old_actions)          

            grad = self._get_closed_loop_gradient(old_states = old_states, old_actions = old_actions, actions = opt_actions, sensitivities = sensitivities)
            hessian = self._get_closed_loop_hessian(old_states = old_states, old_actions = old_actions, actions = opt_actions, sensitivities = sensitivities, second_order_sensitivities = second_order_sensitivities)
            
            self.old_closed_loop_cost = closed_loop_cost

        update, hessian, lam = self._get_update(gradient = grad, hessian = hessian)
        
        if lam < 0:
            lam
        self.old_update = update
        self.old_grad = grad
        self.old_hessian = hessian
        self.old_lam = lam

        self._update_parameters(update = update)

        continue_bool = self._continue_check(grad)

        return continue_bool
