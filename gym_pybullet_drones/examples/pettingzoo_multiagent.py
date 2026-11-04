"""Script demonstrating `gym_pybullet_drones`'s PettingZoo (Parallel API) interface.

Class MultiHoverAviary is wrapped with PettingZooWrapper so that each drone is
a separate agent (`drone_0`, `drone_1`, ...). A single PPO policy, shared by
all drones, is trained on the PettingZoo environment and then replayed.

Example
-------
In a terminal, run as:

    $ python pettingzoo_multiagent.py

Notes
-----
This is a minimal working example of multi-agent learning through the
PettingZoo wrapper, using `stable-baselines3`'s PPO with parameter sharing:
every drone is one slot of a vectorized environment and acts with the same
network. A one-hot agent index is appended to each observation so that the
shared policy can tell the drones (and their different target heights) apart.
On a laptop CPU, training reaches the same team reward as `learn.py
--multiagent true` in a few minutes.

"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.PettingZooWrapper import PettingZooWrapper
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('one_d_rpm')
DEFAULT_AGENTS = 2
TARGET_REWARD = 949.5 # same team reward threshold as learn.py --multiagent true

################################################################################

class SharedPolicyVecEnv(VecEnv):
    """Presents a PettingZoo ParallelEnv to stable-baselines3 as a VecEnv.

    Each agent is one slot of the vectorized environment, so one policy is
    trained on, and shared by, all agents (parameter sharing). All agents of
    the wrapped aviary share one simulation, so they reset together.

    """

    def __init__(self, env):
        self.env = env
        self.agent_names = env.possible_agents
        num_agents = len(self.agent_names)
        obs_space = env.observation_space(self.agent_names[0])
        #### Append a one-hot agent index to each observation ######
        observation_space = spaces.Box(low=np.concatenate([obs_space.low.flatten(), np.zeros(num_agents)]),
                                       high=np.concatenate([obs_space.high.flatten(), np.ones(num_agents)]),
                                       dtype=np.float32
                                       )
        super().__init__(num_agents, observation_space, env.action_space(self.agent_names[0]))
        self._agent_ids = np.eye(num_agents, dtype=np.float32)
        self._actions = None
        self._seed = None

    def _stack(self, observations):
        return np.stack([np.concatenate([np.asarray(observations[a], dtype=np.float32).flatten(), self._agent_ids[i]])
                         for i, a in enumerate(self.agent_names)])

    def reset(self):
        observations, _ = self.env.reset(seed=self._seed)
        self._seed = None
        return self._stack(observations)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        actions = {a: self._actions[i] for i, a in enumerate(self.agent_names)}
        observations, rewards, terminations, truncations, _ = self.env.step(actions)
        rewards = np.array([rewards[a] for a in self.agent_names], dtype=np.float32)
        done = any(terminations.values()) or any(truncations.values())
        infos = [{} for _ in self.agent_names]
        obs = self._stack(observations)
        if done:
            for i, a in enumerate(self.agent_names):
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["TimeLimit.truncated"] = truncations[a] and not terminations[a]
            obs = self.reset()
        return obs, rewards, np.full(self.num_envs, done), infos

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self._seed = seed
        return [seed] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

################################################################################

def make_env(gui=False, record=False):
    return PettingZooWrapper(MultiHoverAviary(num_drones=DEFAULT_AGENTS,
                                              obs=DEFAULT_OBS,
                                              act=DEFAULT_ACT,
                                              gui=gui,
                                              record=record
                                              ))

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    train_env = SharedPolicyVecEnv(make_env())
    eval_env = SharedPolicyVecEnv(make_env())

    print('[INFO] Agents:', train_env.agent_names)
    print('[INFO] Per-agent action space:', train_env.action_space)
    print('[INFO] Per-agent observation space (with agent index):', train_env.observation_space)

    #### Train one PPO policy shared by all drones #############
    model = PPO('MlpPolicy',
                train_env,
                seed=0,
                verbose=1)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=TARGET_REWARD,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(1e6) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=100)

    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        timesteps = data['timesteps']
        results = data['results'][:, 0]
        print("Data from evaluations.npz")
        for j in range(timesteps.shape[0]):
            print(f"{timesteps[j]},{results[j]}")
        if local:
            plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
            plt.xlabel('Training Steps (all agents)')
            plt.ylabel('Team Episode Reward')
            plt.grid(True, alpha=0.6)
            plt.show()

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        raise FileNotFoundError("no model under the specified path: " + filename)
    model = PPO.load(path)

    #### Evaluate the shared policy ############################
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean team reward ", mean_reward, " +- ", std_reward, "\n\n")
    train_env.close()
    eval_env.close()

    #### Replay the policy through the PettingZoo API ##########
    test_env = make_env(gui=gui, record=record_video)
    agent_ids = np.eye(DEFAULT_AGENTS, dtype=np.float32)
    logger = Logger(logging_freq_hz=int(test_env.aviary.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS,
                    output_folder=output_folder,
                    colab=colab
                    )

    observations, infos = test_env.reset(seed=42)
    team_return = 0.0
    start = time.time()
    for i in range((test_env.aviary.EPISODE_LEN_SEC+2)*test_env.aviary.CTRL_FREQ):
        actions = {}
        for d, agent in enumerate(test_env.possible_agents):
            obs = np.concatenate([observations[agent].flatten(), agent_ids[d]])
            actions[agent], _ = model.predict(obs, deterministic=True)
        observations, rewards, terminations, truncations, infos = test_env.step(actions)
        team_return += rewards[test_env.possible_agents[0]]
        for d, agent in enumerate(test_env.possible_agents):
            obs = observations[agent]
            logger.log(drone=d,
                       timestamp=i/test_env.aviary.CTRL_FREQ,
                       state=np.hstack([obs[0:3],
                                        np.zeros(4),
                                        obs[3:12],
                                        np.resize(actions[agent], 4)
                                        ]),
                       control=np.zeros(12)
                       )
        test_env.render()
        sync(i, start, test_env.aviary.CTRL_TIMESTEP)
        if not test_env.agents:
            print("[INFO] Episode finished, team return:", round(team_return, 2))
            team_return = 0.0
            observations, infos = test_env.reset(seed=42)
    test_env.close()

    if plot:
        logger.plot()

    return mean_reward

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning example script using the PettingZoo wrapper')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
