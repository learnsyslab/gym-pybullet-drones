import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
from pathlib import Path

from gym_pybullet_drones.envs.UnderactuatedHoverPriorAviary import UnderactuatedHoverPriorAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def _load_learn_test_module():
    module_path = Path(__file__).with_name('learn-test.py')
    spec = importlib.util.spec_from_file_location('learn_test_module', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_underactuated_training_defaults_use_prior_rpm_action_space():
    learn_test = _load_learn_test_module()

    assert learn_test.DEFAULT_ACT == ActionType.RPM
    assert learn_test.DEFAULT_OBS == ObservationType.KIN


def test_prior_action_keeps_hover_trim_with_degraded_motor():
    env = UnderactuatedHoverPriorAviary(
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        degraded_motor=0,
        motor_reduction=0.35,
    )

    action = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    processed = env._preprocessAction(action)
    thrust = float(np.sum(processed[0] ** 2) * env.KF)

    assert processed.shape == (1, 4)
    assert np.all(processed >= 0.0)
    # env.GRAVITY is already the drone's weight (M * g); hover trim thrust should match it.
    assert np.isclose(thrust, env.GRAVITY, rtol=0.03)
    assert processed[0, 0] < processed[0, 1]
    assert processed[0, 0] < processed[0, 2]
    assert processed[0, 0] < processed[0, 3]

    env.close()


def test_prior_action_uses_physics_backbone_and_reduces_degraded_motor():
    env = UnderactuatedHoverPriorAviary(
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        degraded_motor=0,
        motor_reduction=0.35,
    )

    action = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    processed = env._preprocessAction(action)

    assert processed.shape == (1, 4)
    assert np.all(processed >= 0.0)
    assert processed[0, 0] < processed[0, 1]
    assert processed[0, 0] < processed[0, 2]
    assert processed[0, 0] < processed[0, 3]
    assert processed[0, 0] < env.HOVER_RPM

    env.close()


def test_hover_reward_penalizes_large_acceleration_after_grace_period():
    env = UnderactuatedHoverPriorAviary(
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        degraded_motor=0,
        motor_reduction=0.35,
    )

    env.step_counter = int(env.PYB_FREQ * (env.FAILURE_GRACE_SEC + 0.5))
    env.prev_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def state_for(vel):
        return np.array([
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            *vel,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=np.float32)

    smooth_state = state_for(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    jittery_state = state_for(np.array([0.5, 0.0, 0.0], dtype=np.float32))

    env._getDroneStateVector = lambda drone: smooth_state
    smooth_reward = env._computeReward()

    env._getDroneStateVector = lambda drone: jittery_state
    jittery_reward = env._computeReward()

    assert smooth_reward > jittery_reward
    assert jittery_reward < 1.0

    env.close()


def test_hover_reward_is_positive_during_climb_to_target():
    env = UnderactuatedHoverPriorAviary(
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        degraded_motor=0,
        motor_reduction=0.35,
    )

    env.step_counter = int(env.PYB_FREQ * 0.5)
    env.prev_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    state = np.array([
        0.0, 0.0, 0.5,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.5,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ], dtype=np.float32)
    env._getDroneStateVector = lambda drone: state

    assert env._computeReward() > 0.0

    env.close()


def test_success_termination_waits_for_grace_window():
    env = UnderactuatedHoverPriorAviary(
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        degraded_motor=0,
        motor_reduction=0.35,
    )

    env.step_counter = int(env.PYB_FREQ * 0.5)
    state = np.array([
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.05,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ], dtype=np.float32)
    env._getDroneStateVector = lambda drone: state

    assert not env._computeTerminated()

    env.close()
