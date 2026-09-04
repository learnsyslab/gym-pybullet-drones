import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def _make_env():
    return HoverAviary(
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.ONE_D_RPM,
        pyb_freq=240,
        ctrl_freq=30,
    )


def test_one_d_rpm_keeps_urdf_hover_trim_by_default():
    env = _make_env()
    try:
        baseline_mass = float(p.getDynamicsInfo(env.DRONE_IDS[0], -1, physicsClientId=env.CLIENT)[0])
        action = np.zeros((1, 1), dtype=np.float32)
        rpm = env._preprocessAction(action)[0]

        np.testing.assert_allclose(rpm, env.HOVER_RPM)
        thrust = float(np.sum(rpm**2) * env.KF)
        np.testing.assert_allclose(thrust, env.G * baseline_mass, rtol=1e-6)
    finally:
        env.close()


def test_one_d_rpm_tracks_runtime_mass_randomization():
    env = _make_env()
    try:
        baseline_mass = float(p.getDynamicsInfo(env.DRONE_IDS[0], -1, physicsClientId=env.CLIENT)[0])
        baseline_hover = env.HOVER_RPM
        scale = 1.75
        p.changeDynamics(
            env.DRONE_IDS[0],
            -1,
            mass=baseline_mass * scale,
            physicsClientId=env.CLIENT,
        )

        action = np.zeros((1, 1), dtype=np.float32)
        rpm = env._preprocessAction(action)[0]
        expected_hover = np.sqrt(env.G * baseline_mass * scale / (4 * env.KF))

        np.testing.assert_allclose(rpm, expected_hover, rtol=1e-6)
        thrust = float(np.sum(rpm**2) * env.KF)
        np.testing.assert_allclose(thrust, env.G * baseline_mass * scale, rtol=1e-6)
        # The legacy URDF-derived attribute remains a stable reference value.
        np.testing.assert_allclose(env.HOVER_RPM, baseline_hover)
    finally:
        env.close()


def test_rpm_action_uses_each_drone_runtime_mass():
    env = MultiHoverAviary(
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_drones=2,
        pyb_freq=240,
        ctrl_freq=30,
    )
    try:
        baseline_mass = [
            float(p.getDynamicsInfo(body_id, -1, physicsClientId=env.CLIENT)[0])
            for body_id in env.DRONE_IDS
        ]
        scales = np.array([0.6, 1.4])
        for body_id, mass, scale in zip(env.DRONE_IDS, baseline_mass, scales):
            p.changeDynamics(
                int(body_id),
                -1,
                mass=mass * scale,
                physicsClientId=env.CLIENT,
            )

        action = np.full((2, 4), 0.5, dtype=np.float32)
        rpm = env._preprocessAction(action)
        expected_hover = np.sqrt(env.G * np.asarray(baseline_mass) * scales / (4 * env.KF))
        expected_rpm = np.broadcast_to((expected_hover * 1.025)[:, None], rpm.shape)
        np.testing.assert_allclose(rpm, expected_rpm, rtol=1e-6)
    finally:
        env.close()
