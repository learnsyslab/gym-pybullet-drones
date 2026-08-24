import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
import argparse
import numpy as np
import gymnasium as gym
import sys
from stable_baselines3 import PPO
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.UnderactuatedHoverAviary import UnderactuatedHoverAviary
from gym_pybullet_drones.envs.UnderactuatedHoverPriorAviary import UnderactuatedHoverPriorAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_MODEL_PATH = "results/save-08.21.2026_12.32.02/best_model.zip"
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('one_d_rpm')
DEFAULT_AGENTS = 2
DEFAULT_MA = False


def _bool_arg(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _match_env_to_model(model, gui, underactuated=False, multiagent=False):
    """Build the environment using the same action and observation contract as the loaded policy."""
    action_shape = getattr(model.action_space, "shape", ())
    action_dim = 1 if len(action_shape) == 1 else int(action_shape[-1])
    obs_shape = getattr(model.observation_space, "shape", ())
    obs_dim = 27 if len(obs_shape) == 2 else int(obs_shape[-1]) if len(obs_shape) == 1 else 0
    action_type = ActionType.ONE_D_RPM if action_dim == 1 else ActionType.RPM
    obs_type = ObservationType.KIN

    # A literal z=0.0 start places the collision cylinder's bottom exactly on the
    # ground plane, which produces an unstable/penetrating contact at reset and
    # sends the drone flying sideways regardless of the policy. Use a small but
    # safe clearance instead (verified stable in isolation with zero net RPM delta).
    if underactuated:
        start_xyz = np.array([[0.0, 0.0, 0.02]], dtype=np.float32)
        if action_type == ActionType.ONE_D_RPM:
            return UnderactuatedHoverAviary(gui=gui, obs=obs_type, act=action_type, initial_xyzs=start_xyz, disabled_motor=0, motor_reduction=0.35)
        return UnderactuatedHoverPriorAviary(gui=gui, obs=obs_type, act=action_type, initial_xyzs=start_xyz, degraded_motor=0, motor_reduction=0.35)
    if not multiagent:
        return HoverAviary(gui=gui, obs=obs_type, act=action_type, initial_xyzs=np.array([[0.0, 0.0, 0.02]], dtype=np.float32))
    return MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=obs_type, act=action_type)


def play(model_path=DEFAULT_MODEL_PATH, multiagent=DEFAULT_MA, gui=DEFAULT_GUI, underactuated=False):
    #### Load saved model ####
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    model = PPO.load(model_path)
    print(f"[INFO] Loaded model from {model_path}")
    print(f"[INFO] Model action space: {model.action_space}")
    print(f"[INFO] Model observation space: {model.observation_space}")

    #### Create test environment matching the loaded policy ####
    env = _match_env_to_model(model, gui=gui, underactuated=underactuated, multiagent=multiagent)

    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder="logs_playback/",
                    colab=False)

    #### Run the simulation ####
    obs, _ = env.reset(seed=42, options={})
    start = time.time()

    for i in range((env.EPISODE_LEN_SEC+2)*env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        obs2 = obs.squeeze()
        act2 = action.squeeze()

        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                     np.zeros(4),
                                     obs2[3:15],
                                     act2]),
                    control=np.zeros(12))
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                         np.zeros(4),
                                         obs2[d][3:15],
                                         act2[d]]),
                        control=np.zeros(12))

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            break

    env.close()
    logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained PPO policy in PyBullet drones environment.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to saved policy zip file')
    parser.add_argument('--multiagent', type=_bool_arg, default=DEFAULT_MA, help='Whether to use MultiHoverAviary')
    parser.add_argument('--underactuated', type=_bool_arg, default=False, help='Disable one motor to make the drone underactuated')
    parser.add_argument('--gui', type=_bool_arg, default=DEFAULT_GUI, help='Enable GUI rendering')
    args = parser.parse_args()

    play(**vars(args))
