"""PettingZoo Parallel API wrapper for the multi-agent aviaries.

`BaseRLAviary` subclasses such as `MultiHoverAviary` expose all drones through
a single Gymnasium interface: one stacked observation of shape (NUM_DRONES, ...),
one stacked action of shape (NUM_DRONES, ...) and one scalar reward. This module
re-exposes them through PettingZoo's Parallel API, with one agent per drone
(`drone_0`, `drone_1`, ...). That lets multi-agent RL libraries that speak
PettingZoo (e.g. AgileRL, BenchMARL, RLlib, SuperSuit) train on them directly.

Example
-------
    from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
    from gym_pybullet_drones.utils.PettingZooWrapper import PettingZooWrapper

    env = PettingZooWrapper(MultiHoverAviary(num_drones=2))
    observations, infos = env.reset(seed=42)
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()

"""
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType


class PettingZooWrapper(ParallelEnv):
    """Exposes a multi-drone `BaseRLAviary` as a PettingZoo `ParallelEnv`.

    Each drone becomes an agent named `drone_<i>`. Agent `drone_i` receives row
    `i` of the aviary's stacked observation and controls row `i` of its stacked
    action.

    The aviaries compute a single team reward and episode-level
    terminated/truncated flags. By default, every agent receives the same team
    reward and the same flags (a fully cooperative setting). An aviary may
    instead return per-drone values as an array of length NUM_DRONES (or a
    dict keyed by agent name), and they are passed through unchanged.

    """

    metadata = {"name": "gym_pybullet_drones_v0",
                "render_modes": ["human"],
                "is_parallelizable": True,
                }

    ################################################################################

    def __init__(self,
                 aviary: BaseRLAviary
                 ):
        """Initialization of the wrapper.

        Parameters
        ----------
        aviary : BaseRLAviary
            An already constructed aviary, e.g. `MultiHoverAviary(num_drones=3)`.
            The wrapper takes ownership of it and closes it in `close()`.

        """
        if not isinstance(aviary, BaseRLAviary):
            raise TypeError("PettingZooWrapper expects a BaseRLAviary subclass, got "+type(aviary).__name__)
        self.aviary = aviary
        self.NUM_DRONES = aviary.NUM_DRONES
        self.possible_agents = ["drone_"+str(i) for i in range(self.NUM_DRONES)]
        self.agents = []
        self.render_mode = "human" if aviary.GUI else None
        #### Split the stacked spaces into per-drone spaces ########
        self._observation_spaces = {a: self._agentSpace(aviary.observation_space, i)
                                    for i, a in enumerate(self.possible_agents)}
        self._action_spaces = {a: self._agentSpace(aviary.action_space, i)
                               for i, a in enumerate(self.possible_agents)}
        #### Global state (stacked kinematic obs.) for centralised critics
        if aviary.OBS_TYPE == ObservationType.KIN:
            kin = aviary.observation_space
            self.state_space = spaces.Box(low=kin.low.flatten(),
                                          high=kin.high.flatten(),
                                          dtype=np.float32
                                          )
        self._last_obs = None

    ################################################################################

    @staticmethod
    def _agentSpace(space, i):
        """Returns the slice of a stacked (NUM_DRONES, ...) space belonging to drone `i`."""
        if isinstance(space, spaces.Dict):
            return spaces.Dict({k: PettingZooWrapper._agentSpace(s, i) for k, s in space.spaces.items()})
        return spaces.Box(low=space.low[i], high=space.high[i], dtype=space.dtype)

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    ################################################################################

    def _splitObs(self, obs):
        """Splits the aviary's stacked observation into a dict of per-agent observations."""
        out = {}
        for i, a in enumerate(self.possible_agents):
            space = self._observation_spaces[a]
            if isinstance(space, spaces.Dict):
                out[a] = {k: np.asarray(v[i], dtype=space[k].dtype) for k, v in obs.items()}
            else:
                out[a] = np.asarray(obs[i], dtype=space.dtype)
        return out

    def _perAgent(self, value, agents):
        """Broadcasts a team-level value to all agents, or passes through per-agent values."""
        if isinstance(value, dict):
            return {a: value[a] for a in agents}
        if np.ndim(value) == 0:
            return {a: value.item() if hasattr(value, "item") else value for a in agents}
        value = np.asarray(value)
        if value.shape[0] != self.NUM_DRONES:
            raise ValueError("Expected a scalar or one value per drone, got shape "+str(value.shape))
        return {a: value[self.possible_agents.index(a)].item() for a in agents}

    ################################################################################

    def reset(self,
              seed=None,
              options=None
              ):
        """Resets the aviary and returns per-agent observations and infos."""
        obs, info = self.aviary.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self._last_obs = obs
        return self._splitObs(obs), {a: dict(info) for a in self.agents}

    ################################################################################

    def step(self,
             actions
             ):
        """Steps all drones at once.

        Parameters
        ----------
        actions : dict[str, ndarray]
            One action per live agent, each drawn from `action_space(agent)`.

        """
        if not self.agents:
            raise RuntimeError("step() called on a finished episode, call reset() first")
        missing = [a for a in self.agents if a not in actions]
        if missing:
            raise KeyError("Missing actions for agents: "+", ".join(missing))
        act_shape = self._action_spaces[self.possible_agents[0]].shape
        stacked = np.zeros((self.NUM_DRONES,)+act_shape, dtype=np.float32)
        for i, a in enumerate(self.possible_agents):
            stacked[i] = actions[a]
        obs, reward, terminated, truncated, info = self.aviary.step(stacked)
        self._last_obs = obs
        agents = self.agents
        observations = self._splitObs(obs)
        rewards = self._perAgent(reward, agents)
        terminations = {a: bool(v) for a, v in self._perAgent(terminated, agents).items()}
        truncations = {a: bool(v) for a, v in self._perAgent(truncated, agents).items()}
        infos = {a: dict(info) for a in agents}
        #### All drones share one simulation: the episode ends for everyone
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    ################################################################################

    def state(self):
        """Returns the global state: all drones' kinematic observations, flattened.

        Only available with `ObservationType.KIN`.

        """
        if not hasattr(self, "state_space"):
            raise NotImplementedError("state() is only available with ObservationType.KIN")
        return np.asarray(self._last_obs, dtype=np.float32).flatten()

    def render(self):
        return self.aviary.render()

    def close(self):
        self.aviary.close()
