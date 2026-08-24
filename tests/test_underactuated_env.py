import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from gym_pybullet_drones.envs.UnderactuatedHoverAviary import UnderactuatedHoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def test_underactuated_motor_is_degraded():
    env = UnderactuatedHoverAviary(
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        motor_reduction=0.35,
    )
    action = np.array([[0.5, 0.25, -0.3, 0.1]], dtype=np.float32)

    processed = env._preprocessAction(action)

    assert processed.shape == (1, 4)
    assert 0.0 < processed[0, 0] < processed[0, 1]
    assert processed[0, 0] < env.HOVER_RPM
    assert processed[0, 1] > 0.0
    assert processed[0, 2] > 0.0
    assert processed[0, 3] > 0.0

    env.close()


def test_reward_penalizes_high_velocity():
    env = UnderactuatedHoverAviary(gui=False, obs=ObservationType.KIN, act=ActionType.RPM, ctrl_freq=30)
    base_state = np.zeros(16, dtype=np.float32)
    base_state[0:3] = np.array([0.0, 0.0, 1.0])
    base_state[7:10] = np.array([0.0, 0.0, 0.0])
    base_state[10:13] = np.array([0.0, 0.0, 0.0])

    with patch.object(env, "_getDroneStateVector", return_value=base_state):
        reward_low = env._computeReward()

    high_velocity_state = base_state.copy()
    high_velocity_state[10:13] = np.array([0.0, 0.0, 3.0])

    with patch.object(env, "_getDroneStateVector", return_value=high_velocity_state):
        reward_high = env._computeReward()

    assert reward_high < reward_low

    env.close()


def test_reward_prefers_safe_near_hover_over_out_of_envelope_state():
    env = UnderactuatedHoverAviary(gui=False, obs=ObservationType.KIN, act=ActionType.RPM, ctrl_freq=30)
    safe_state = np.zeros(16, dtype=np.float32)
    safe_state[0:3] = np.array([0.0, 0.0, 1.0])
    safe_state[7:10] = np.array([0.0, 0.0, 0.0])
    safe_state[10:13] = np.array([0.0, 0.0, 0.0])

    unsafe_state = safe_state.copy()
    unsafe_state[0:3] = np.array([0.5, 0.0, 1.0])
    unsafe_state[7:10] = np.array([0.6, 0.0, 0.0])

    with patch.object(env, "_getDroneStateVector", return_value=safe_state):
        reward_safe = env._computeReward()

    with patch.object(env, "_getDroneStateVector", return_value=unsafe_state):
        reward_unsafe = env._computeReward()

    assert reward_safe > reward_unsafe

    env.close()


def test_reward_is_bounded_when_outside_envelope():
    env = UnderactuatedHoverAviary(gui=False, obs=ObservationType.KIN, act=ActionType.RPM, ctrl_freq=30)
    bad_state = np.zeros(16, dtype=np.float32)
    bad_state[0:3] = np.array([2.0, 0.0, 0.0])
    bad_state[7:10] = np.array([1.0, 0.0, 0.0])
    bad_state[10:13] = np.array([2.0, 0.0, 0.0])

    with patch.object(env, "_getDroneStateVector", return_value=bad_state):
        reward = env._computeReward()

    assert -50.0 < reward < 0.0
    env.close()


def test_success_requires_hover_stability_not_just_position():
    env = UnderactuatedHoverAviary(gui=False, obs=ObservationType.KIN, act=ActionType.RPM, ctrl_freq=30)
    env.step_counter = 100

    hovering_state = np.zeros(16, dtype=np.float32)
    hovering_state[0:3] = np.array([0.0, 0.0, 1.0])
    hovering_state[10:13] = np.array([0.0, 0.0, 0.0])

    moving_state = hovering_state.copy()
    moving_state[0:3] = np.array([0.05, 0.0, 1.0])
    moving_state[10:13] = np.array([0.5, 0.0, 0.0])

    with patch.object(env, "_getDroneStateVector", return_value=hovering_state):
        assert env._computeTerminated() is True

    with patch.object(env, "_getDroneStateVector", return_value=moving_state):
        assert env._computeTerminated() is False

    env.close()


