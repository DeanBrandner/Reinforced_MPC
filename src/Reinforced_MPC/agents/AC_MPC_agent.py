
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

    Attributes:
        gamma (float): Discount factor for future rewards. Must be in the range [0, 1].
        epsilon (float): Initial value for the exploration rate. Must be in the range [0, 1].
        epsilon_decay (float): Decay rate for the exploration rate. Must be in the range [0, 1].
        epsilon_lower (float): Lower bound for the exploration rate. Must be in the range [0, 1].
        exploration_std (float): Standard deviation for exploration noise. Must be positive. If it is a true standard deviation depends on the exploration method. 
        learning_rate (float): Learning rate. It is used by standard gradient descent methods or Quasi-Newton updates. Must be positive.
        memory_size (int): Maximum size of the replay buffer.
        behave_rng_seed (int): Seed for the random number generator used during behavior policy sampling. This RNG determines the sampled value from exploration_std.
        exploration_rng_seed (int): Seed for the random number generator used during exploration. This RNG determines if behavior policy explores or exploits.
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
    A class representing an Actor-Critic MPC agent.

    Attributes:
    -----------
    actor : object
        The actor MPC object.
    actor_behavior : object
        A copy of the actor MPC object. This MPC will be used when the behave method is called.
    replay_mpc : object
        A copy of the actor MPC object. This MPC will be used when the replay method is called.
    mpc_differentiator : None or object
        The MPC differentiator object. It must be initialized in the child class.
    settings : object
        The AC_MPC_agent_settings object. It contains the settings of the agent.
    behave_rng : object
        The random number generator for behavior. It is used to sample the exploration noise.
    exploration_rng : object
        The random number generator for exploration. It is used to determine if the behavior policy explores or exploits.
    memory : deque
        The memory buffer (replay buffer) for the agent.
    flags : dict
        A dictionary of flags for the agent.
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
        """
        Given a state, returns an action using the actor MPC.
        If an old action is provided, it sets the previous action to that value before making a step.
        
        Args:
        - state (np.ndarray): The current state of the environment.
        - old_action (np.ndarray): The previous action taken by the agent.
        
        Returns:
        - action (np.ndarray): The action to take in the environment.
        """
        if old_action is not None:
            self.actor.u0 = old_action.copy()
        action = self.actor.make_step(state.copy())
        return action

    def behave(self, state: np.ndarray, old_action: np.ndarray = None):
        """
            Given a state, returns an action using the behaviour MPC based on the exploration policy.
            If an old action is provided, it sets the previous action to that value before making a step.

            The behaviour is motivated from section 5 of the following paper:
            Gros, S. and Zanon, M: Towards safe reinforcement learning using nmpc and policy gradients: Part ii-deterministic case., 2019, arXiv preprint arXiv:1906.04034
            
            Args:
            - state (np.ndarray): The current state of the environment.
            - old_action (np.ndarray): The previous action taken by the agent.
            
            Returns:
            - action (np.ndarray): The action to take in the environment.
            """
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
        """
        Store the experience tuple (old_state, old_action, action, reward, next_state, done_flag) in the agent's memory.

        Args:
            old_state (np.ndarray): The state before taking the action.
            old_action (np.ndarray): The previous action taken to reach the old state.
            action (np.ndarray): The action taken in the current state.
            reward (float): The reward/stage cost received after taking the action in the old state.
            next_state (np.ndarray): The state after taking the action.
            done_flag (bool): Whether the episode has ended after taking the action.
        """
        self.memory.append((old_state, old_action, action, reward, next_state, done_flag))

    def decay_epsilon(self):
        """
        Decreases the value of exploration noise epsilon by multiplying it with the epsilon_decay factor, as long as it is above the lower exploration rate 
        epsilon_lower threshold.
        """
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
        """
        Saves the actor MPC and its parameters to disk.

        Args:
            path (str): The path where the actor and its parameters will be saved.
        """
        self.actor.save(path)
        with open(path + "\\parameters.pkl", "wb") as f:
            pickle.dump(self.actor.p_fun(0).master, f)


    def save(self, path: str):
        """
        Saves the actor, memory, and settings of the AC_MPC_agent to the specified path.

        Args:
            path (str): The path to save the agent to.
        """
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
        """
        Loads an RL_AC_MPC model (actor) and its associated parameters from the specified path.

        Args:
            path (str): The path to the directory containing the saved model and parameters.

        Returns:
            RL_AC_MPC: The loaded RL_AC_MPC model.
        """
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
        """
        Loads a saved agent from the given path.

        Args:
            path (str): The path where the agent is saved.

        Returns:
            AC_MPC_Agent: The loaded agent.
        """
        if not path.endswith("\\"):
            path += "\\"

        actor = cls.load_actor(path + "actor")

        with open(path + "settings.pkl", "rb") as f:
            settings = pickle.load(f)

        agent = cls(actor)
        agent.settings = settings
        return agent
        
