
import random
import pickle
import copy

import numpy as np
from collections import deque
from dataclasses import dataclass

import casadi.tools as castools

from ..tools.RL_AC_MPC import RL_AC_MPC

@dataclass
class AC_MPC_agent_settings:
    """
    This is a dataclass for the settings of the AC_MPC_agent.
    """
    gamma: float = 1
    epsilon: float = 1
    epsilon_decay: float = 0.99
    epsilon_lower: float = 0.01
    exploration_std: float = 0.1
    learning_rate: float = 1e-3
    memory_size: int = 100
    behave_rng_seed: int = None
    exploration_rng_seed: int = None


class AC_MPC_agent():
    """
    This is an abstract class and must be inherited from.
    """

    def __init__(self, actor_mpc = None, **kwargs):

        self.actor = actor_mpc
        self.actor_behavior = copy.deepcopy(actor_mpc)
        self.replay_mpc = copy.deepcopy(actor_mpc)

        self.mpc_differentiator = None 

        
        self.settings = AC_MPC_agent_settings(**kwargs)

        self.behave_rng = np.random.default_rng(seed = self.settings.behave_rng_seed)
        self.exploration_rng = np.random.default_rng(seed = self.settings.exploration_rng_seed)

        self.memory = deque(maxlen = self.settings.memory_size)

        self.flags = {
            "differentiator_initialized": False,
        }
    
    def act(self, state: np.ndarray, old_action: np.ndarray = None):
        if old_action is not None:
            self.actor.u0 = old_action.copy()
        action = self.actor.make_step(state.copy())
        return action

    def behave(self, state: np.ndarray, old_action: np.ndarray = None):
        d = np.zeros(self.actor.opt_x["_u", 0, 0].shape)
        if self.behave_rng.uniform() <= self.settings.epsilon:
            d += self.exploration_rng.normal(loc = 0, scale = self.settings.exploration_std, size = self.actor.opt_x["_u", 0, 0].shape)
            r_factor = self.actor_behavior.rterm_factor.master.full()
            if np.allclose(r_factor, 0):
                r_factor = 1
            d *= r_factor

        
        self.actor_behavior.nlp["f"] = self.actor_behavior._nlp_obj + d.T @ (self.actor_behavior.opt_x["_u", 0, 0] - self.actor_behavior.opt_x["_u", 1, 0])
        self.actor_behavior.S = castools.nlpsol('S', 'ipopt', self.actor_behavior.nlp, self.actor_behavior.settings.nlpsol_opts)
        if old_action is not None:
            self.actor_behavior.u0 = old_action.copy()
        action = self.actor_behavior.make_step(state.copy())
        self.actor_behavior.reset_history()
        return action
    
    def remember(self, old_state: np.ndarray, old_action: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done_flag: bool):
        self.memory.append((old_state, old_action, action, reward, next_state, done_flag))

    def decay_epsilon(self):
        if self.settings.epsilon > self.settings.epsilon_lower:
            self.settings.epsilon *= self.settings.epsilon_decay
        return


    # Training section
    def replay(self, batch_size: int):
        """
        This is an abstract method and must be implemented in the child class.
        """
        raise NotImplementedError("This is an abstract method and must be implemented in the child class.")
        

    # All the saving functions
    def save_actor(self, path: str):
        self.actor.save(path)
        with open(path + "\\parameters.pkl", "wb") as f:
            pickle.dump(self.actor.p_fun(0).master, f)

    def save(self, path: str):
        if not path.endswith("\\"):
            path += "\\"

        self.save_actor(path + "actor")

        with open(path + "memory.pkl", "wb") as f:
            pickle.dump(self.memory, f)
        
        with open(path + "settings.pkl", "wb") as f:
            pickle.dump(self.settings, f)
    


    # All the loading functions
    @staticmethod
    def load_actor(path):
        if not path.endswith("\\"):
            path += "\\"

        mpc_model = RL_AC_MPC()
        mpc_model.load(path)
        print("Model is loaded.")

        with open(path + "parameters.pkl", "rb") as f:
            parameters = pickle.load(f)
        p_template = mpc_model.get_p_template(1)
        p_template.master = parameters
        mpc_model.set_p_fun(lambda t_now: p_template)

        mpc_model.setup()
        return mpc_model

    @classmethod
    def load(cls, path):
        # NOTE This is only experimental and not tested yet
        if not path.endswith("\\"):
            path += "\\"

        actor = cls.load_actor(path + "actor")

        with open(path + "settings.pkl", "rb") as f:
            settings = pickle.load(f)

        agent = cls(actor)
        agent.settings = settings
        return agent
        
