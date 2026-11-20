import os

import numpy as np
import pybullet as p
import pytest

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.CameraRecorder import (CameraRecorder,
                                                      camera_pose_to_view_matrix,
                                                      depth_buffer_to_metric,
                                                      view_matrix_to_camera_pose)


def _quat_to_R(q):
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


@pytest.fixture
def env():
    env = CtrlAviary(num_drones=1, gui=False, obstacles=False)
    env.reset(seed=0)
    yield env
    env.close()


def test_view_matrix_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(p.getQuaternionFromEuler(rng.uniform(-np.pi, np.pi, 3)))
        T[:3, 3] = rng.uniform(-5, 5, 3)
        np.testing.assert_allclose(view_matrix_to_camera_pose(camera_pose_to_view_matrix(T)), T, atol=1e-9)


def test_pose_from_pybullet_look_at():
    """A pose recovered from p.computeViewMatrix() looks at the target, with y pointing down."""
    eye, target = np.array([1., 2., 3.]), np.array([4., -1., 2.])
    T = view_matrix_to_camera_pose(p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1]))
    np.testing.assert_allclose(T[:3, 3], eye, atol=1e-6)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    np.testing.assert_allclose(T[:3, 2], fwd, atol=1e-6)
    assert T[2, 1] < 0  # image "down" has a negative world-z component


@pytest.mark.parametrize("rpy_deg,pitch_deg", [((0, 0, 0), 0), ((25, -15, 70), 0), ((-30, 10, -120), 20)])
def test_logged_pose_matches_rendered_pixels(env, tmp_path, rpy_deg, pitch_deg):
    """Projecting a known sphere with the logged pose and intrinsics lands on its rendered pixels."""
    W, H, radius = 320, 240, 0.05
    rec = CameraRecorder(env, str(tmp_path), width=W, height=H, fov_deg=70, pitch_deg=pitch_deg, save_depth=True)
    drone_pos = np.array([0.3, -0.2, 1.0])
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], drone_pos.tolist(),
                                      p.getQuaternionFromEuler(np.deg2rad(rpy_deg).tolist()),
                                      physicsClientId=env.CLIENT)
    T_wc = rec.camera_pose()
    #### Place a sphere 1.5 m ahead of the camera, offset in the image
    target = T_wc[:3, :3] @ np.array([0.2, -0.1, 1.5]) + T_wc[:3, 3]
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, physicsClientId=env.CLIENT)
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=env.CLIENT)
    sphere = p.createMultiBody(0, col, shape, target.tolist(), physicsClientId=env.CLIENT)

    rgb, T_logged = rec.capture()
    rec.close()
    np.testing.assert_allclose(T_logged, T_wc)

    #### Segmentation rendered with the recorder's own matrices
    _, _, _, _, seg = p.getCameraImage(W, H, viewMatrix=camera_pose_to_view_matrix(T_wc),
                                       projectionMatrix=rec.PROJ, renderer=p.ER_TINY_RENDERER,
                                       physicsClientId=env.CLIENT)
    seg = np.reshape(seg, (H, W))
    v, u = np.nonzero(seg == sphere)
    assert len(u) > 20
    observed = np.array([u.mean(), v.mean()])

    #### Re-read everything from disk, as a downstream user would
    fx, fy, cx, cy = np.loadtxt(tmp_path / "intrinsics.txt")
    row = np.genfromtxt(tmp_path / "poses.csv", delimiter=",", names=True)
    R = _quat_to_R([row["qx"], row["qy"], row["qz"], row["qw"]])
    t = np.array([row["tx"], row["ty"], row["tz"]])
    x_c = R.T @ (target - t)
    predicted = np.array([fx * x_c[0] / x_c[2] + cx, fy * x_c[1] / x_c[2] + cy])
    np.testing.assert_allclose(observed, predicted, atol=0.25)

    #### Metric depth at the sphere's centre is the near surface of the sphere
    depth = np.load(tmp_path / "depth" / "000000.npy")
    d = depth[int(round(predicted[1])), int(round(predicted[0]))]
    assert abs(d - (x_c[2] - radius * x_c[2] / np.linalg.norm(x_c))) < 0.01

    #### The camera is body-fixed: it rolls/pitches with the drone
    R_wb = _quat_to_R(p.getQuaternionFromEuler(np.deg2rad(rpy_deg).tolist()))
    pitch = np.deg2rad(pitch_deg)
    expected_fwd = R_wb @ np.array([np.cos(pitch), 0., -np.sin(pitch)])
    np.testing.assert_allclose(T_wc[:3, 2], expected_fwd, atol=1e-6)


def test_sequence_files(env, tmp_path):
    from gym_pybullet_drones.examples.record_camera import run
    out = run(gui=False, duration_sec=1, output_folder=str(tmp_path), save_depth=False)
    images = sorted(os.listdir(os.path.join(out, "images")))
    poses = np.genfromtxt(os.path.join(out, "poses.csv"), delimiter=",", names=True)
    assert len(images) == len(poses) > 5
    assert np.all(np.diff(poses["timestamp"]) > 0)
    q = np.stack([poses[k] for k in ("qx", "qy", "qz", "qw")], 1)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1., atol=1e-6)
    assert np.ptp(poses["tz"]) > 0.05  # the drone actually moved
