"""PPO example using a physics-prior underactuated hover controller.

This is a training-oriented copy of the baseline learning example, but the action
space is interpreted as a wrench correction that is projected into rotor RPMs using
standard quadrotor mixer math and a degraded rotor factor.
"""

import os
import time
from datetime import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.UnderactuatedHoverPriorAviary import UnderactuatedHoverPriorAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_AGENTS = 2
DEFAULT_MA = False


def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, underactuated=True):
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

    if underactuated:
        train_env = make_vec_env(UnderactuatedHoverPriorAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, degraded_motor=0, motor_reduction=0.35),
                                 n_envs=1,
                                 seed=0)
        eval_env = UnderactuatedHoverPriorAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, degraded_motor=0, motor_reduction=0.35)
    elif not multiagent:
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0)
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0)
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    model = PPO('MlpPolicy', train_env, verbose=1)

    target_reward = 474. if not multiagent else 949.5
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename + '/',
                                 log_path=filename + '/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)

    model.learn(total_timesteps=int(1e6) if local else int(1e2), callback=eval_callback, log_interval=100)
    model.save(filename + '/final_model.zip')

    with np.load(filename + '/evaluations.npz') as data:
        timesteps = data['timesteps']
        results = data['results'][:, 0]
        print('Data from evaluations.npz')
        for j in range(timesteps.shape[0]):
            print(f'{timesteps[j]},{results[j]}')
        if local:
            plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
            plt.xlabel('Training Steps')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.6)
            plt.show()

    if os.path.isfile(filename + '/best_model.zip'):
        path = filename + '/best_model.zip'
    else:
        raise FileNotFoundError(f'No model found under {filename}')

    model = PPO.load(path)

    if underactuated:
        test_env = UnderactuatedHoverPriorAviary(gui=gui,
                                                obs=DEFAULT_OBS,
                                                act=DEFAULT_ACT,
                                                record=record_video,
                                                degraded_motor=0,
                                                motor_reduction=0.35)
        test_env_nogui = UnderactuatedHoverPriorAviary(obs=DEFAULT_OBS,
                                                      act=DEFAULT_ACT,
                                                      degraded_motor=0,
                                                      motor_reduction=0.35)
    elif not multiagent:
        test_env = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                   num_drones=DEFAULT_AGENTS if multiagent else 1,
                   output_folder=output_folder,
                   colab=colab)

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print('\n\n\nMean reward ', mean_reward, ' +- ', std_reward, '\n\n')

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print('Obs:', obs, '\tAction', action, '\tReward:', reward, '\tTerminated:', terminated, '\tTruncated:', truncated)

        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                           timestamp=i / test_env.CTRL_FREQ,
                           state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                           control=np.zeros(12))
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                               timestamp=i / test_env.CTRL_FREQ,
                               state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                               control=np.zeros(12))

        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Physics-prior RL training example for underactuated hover.')
    parser.add_argument('--multiagent', default=DEFAULT_MA, type=str2bool, help='Whether to use multiple drones (default: False)', metavar='')
    parser.add_argument('--underactuated', default=True, type=str2bool, help='Whether to use the degraded underactuated environment (default: True)', metavar='')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help='Whether example is being run by a notebook', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
