"""Record onboard camera frames together with the exact camera pose.

`CameraRecorder` renders a camera rigidly mounted on one drone of a
`BaseAviary` environment and writes a posed image sequence to disk:

    <output_folder>/
        images/000000.png, 000001.png, ...
        depth/000000.npy, ...       (optional, metric depth in metres, float32)
        poses.csv                   timestamp,tx,ty,tz,qx,qy,qz,qw
        intrinsics.txt              fx fy cx cy

Each row of `poses.csv` is the camera-to-world transform of the matching
image, with the camera axes in the OpenCV convention (x right, y down,
z forward) and the world frame of the PyBullet simulation (z up).
Timestamps are simulation time in seconds. The pose is obtained by
inverting the same view matrix that is used to render the image, so pose
and pixels are consistent by construction.

This layout (TUM-style pose columns, OpenCV camera, pinhole intrinsics)
can be read by most visual odometry, SLAM, NeRF / Gaussian-splatting and
world-model training pipelines.

Images are always rendered with `p.ER_TINY_RENDERER`, so the intrinsics
are the same with and without the GUI.

Unlike `BaseAviary._getDroneImages()`, whose camera keeps the world z axis
as its up vector, the camera here is fixed to the drone body, so it rolls
and pitches with the airframe like a real FPV camera.

Example
-------
    env = CtrlAviary(...)
    rec = CameraRecorder(env, "results/flight_0", drone=0, width=256, height=256)
    for i in range(steps):
        env.step(action)
        if i % 5 == 0:
            rec.capture()
    rec.close()

"""
import csv
import os

import numpy as np
import pybullet as p
from PIL import Image

#### OpenCV camera axes expressed in the drone body frame (x fwd, y left, z up)
#### camera x (right) = -body y, camera y (down) = -body z, camera z (fwd) = body x
R_BODY_FROM_CV = np.array([[0., 0., 1.],
                           [-1., 0., 0.],
                           [0., -1., 0.]])

#### OpenGL eye axes (x right, y up, z backward) from OpenCV axes
_GL_FROM_CV = np.diag([1., -1., -1., 1.])

################################################################################

def camera_pose_to_view_matrix(T_wc: np.ndarray) -> list:
    """Converts an OpenCV camera-to-world pose into a PyBullet view matrix.

    Parameters
    ----------
    T_wc : ndarray
        (4, 4)-shaped camera-to-world transform, OpenCV camera axes.

    Returns
    -------
    list
        16 floats, column-major, as expected by `p.getCameraImage()`.

    """
    T_wc_gl = T_wc @ _GL_FROM_CV
    view = np.linalg.inv(T_wc_gl)
    return view.flatten(order='F').tolist()

################################################################################

def view_matrix_to_camera_pose(view_matrix) -> np.ndarray:
    """Converts a PyBullet view matrix into an OpenCV camera-to-world pose.

    Works with any view matrix, e.g. one from `p.computeViewMatrix()`.

    Parameters
    ----------
    view_matrix : list | ndarray
        16 floats, column-major (PyBullet / OpenGL convention).

    Returns
    -------
    ndarray
        (4, 4)-shaped camera-to-world transform, OpenCV camera axes.

    """
    view = np.asarray(view_matrix, dtype=float).reshape(4, 4, order='F')
    return np.linalg.inv(view) @ _GL_FROM_CV

################################################################################

def projection_matrix_to_intrinsics(projection_matrix, width: int, height: int, tiny_renderer: bool=True):
    """Extracts pinhole intrinsics and clipping planes from a PyBullet projection matrix.

    PyBullet's TinyRenderer (`p.ER_TINY_RENDERER`, the default without GUI)
    samples pixels half a pixel away from the ideal OpenGL convention; its
    principal point is at (W/2, H/2 - 1) instead of ((W-1)/2, (H-1)/2).

    Parameters
    ----------
    projection_matrix : list | ndarray
        16 floats, column-major, e.g. from `p.computeProjectionMatrixFOV()`.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    tiny_renderer : bool, optional
        Whether the images are rendered with `p.ER_TINY_RENDERER`.

    Returns
    -------
    ndarray
        (3, 3)-shaped intrinsics matrix K (pixel centres at integer coordinates).
    float
        Near clipping plane distance.
    float
        Far clipping plane distance.

    """
    P = np.asarray(projection_matrix, dtype=float).reshape(4, 4, order='F')
    fx = P[0, 0] * width / 2.
    fy = P[1, 1] * height / 2.
    cx = (1. - P[0, 2]) * width / 2. - .5
    cy = (1. + P[1, 2]) * height / 2. - .5
    if tiny_renderer:
        cx, cy = cx + .5, cy - .5
    near = P[2, 3] / (P[2, 2] - 1.)
    far = P[2, 3] / (P[2, 2] + 1.)
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    return K, near, far

################################################################################

def depth_buffer_to_metric(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    """Converts a PyBullet depth buffer (values in [0, 1]) into metric z-depth.

    Parameters
    ----------
    depth_buffer : ndarray
        Depth buffer returned by `p.getCameraImage()`.
    near : float
        Near clipping plane distance.
    far : float
        Far clipping plane distance.

    Returns
    -------
    ndarray
        Distance along the camera's optical axis, in metres.

    """
    depth_buffer = np.asarray(depth_buffer, dtype=np.float64)
    return far * near / (far - (far - near) * depth_buffer)

################################################################################

class CameraRecorder(object):
    """Saves posed onboard camera images from a `BaseAviary` environment."""

    ################################################################################

    def __init__(self,
                 env,
                 output_folder: str,
                 drone: int=0,
                 width: int=256,
                 height: int=256,
                 fov_deg: float=90.0,
                 pitch_deg: float=0.0,
                 offset_m=None,
                 near: float=None,
                 far: float=1000.0,
                 save_depth: bool=False,
                 shadow: bool=True
                 ):
        """Initialization.

        Parameters
        ----------
        env : BaseAviary
            The environment (any `BaseAviary` subclass).
        output_folder : str
            Directory where the sequence is written (created if missing).
        drone : int, optional
            Index of the drone carrying the camera.
        width : int, optional
            Image width in pixels.
        height : int, optional
            Image height in pixels.
        fov_deg : float, optional
            Vertical field of view in degrees.
        pitch_deg : float, optional
            Downward tilt of the camera from the body x axis, in degrees.
        offset_m : ndarray, optional
            (3,)-shaped camera position in the body frame, in metres.
            Defaults to `[env.L, 0, 0]` (front of the frame).
        near : float, optional
            Near clipping plane, defaults to `env.L`.
        far : float, optional
            Far clipping plane.
        save_depth : bool, optional
            Whether to also save metric depth maps as .npy files.
        shadow : bool, optional
            Whether to render shadows.

        """
        self.ENV = env
        self.DRONE = drone
        self.WIDTH = int(width)
        self.HEIGHT = int(height)
        self.SAVE_DEPTH = save_depth
        self.SHADOW = shadow
        self.OFFSET = np.array([env.L, 0., 0.]) if offset_m is None else np.asarray(offset_m, dtype=float)
        pitch = np.deg2rad(pitch_deg)
        #### Rotation about body y (left); positive pitch_deg tilts the view down
        R_pitch = np.array([[np.cos(pitch), 0., np.sin(pitch)],
                            [0., 1., 0.],
                            [-np.sin(pitch), 0., np.cos(pitch)]])
        self.T_BODY_CAM = np.eye(4)
        self.T_BODY_CAM[:3, :3] = R_pitch @ R_BODY_FROM_CV
        self.T_BODY_CAM[:3, 3] = self.OFFSET
        self.PROJ = p.computeProjectionMatrixFOV(fov=fov_deg,
                                                 aspect=self.WIDTH/self.HEIGHT,
                                                 nearVal=env.L if near is None else near,
                                                 farVal=far
                                                 )
        self.K, self.NEAR, self.FAR = projection_matrix_to_intrinsics(self.PROJ, self.WIDTH, self.HEIGHT)
        #### Output files ##########################################
        self.OUTPUT_FOLDER = output_folder
        os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
        if self.SAVE_DEPTH:
            os.makedirs(os.path.join(output_folder, "depth"), exist_ok=True)
        np.savetxt(os.path.join(output_folder, "intrinsics.txt"),
                   self.K[[0, 1, 0, 1], [0, 1, 2, 2]][None],
                   fmt="%.6f",
                   header="fx fy cx cy"
                   )
        self._csv_file = open(os.path.join(output_folder, "poses.csv"), "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        self.frame_count = 0

    ################################################################################

    def camera_pose(self) -> np.ndarray:
        """Returns the current camera-to-world transform (OpenCV camera axes).

        Returns
        -------
        ndarray
            (4, 4)-shaped transform.

        """
        pos, quat = p.getBasePositionAndOrientation(self.ENV.DRONE_IDS[self.DRONE],
                                                    physicsClientId=self.ENV.CLIENT
                                                    )
        T_wb = np.eye(4)
        T_wb[:3, :3] = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        T_wb[:3, 3] = pos
        return T_wb @ self.T_BODY_CAM

    ################################################################################

    def capture(self, timestamp: float=None):
        """Renders one frame, writes it to disk and appends its pose.

        Parameters
        ----------
        timestamp : float, optional
            Time stamp in seconds, defaults to the simulation time.

        Returns
        -------
        ndarray
            (h, w, 3)-shaped uint8 RGB image.
        ndarray
            (4, 4)-shaped camera-to-world transform (OpenCV camera axes).

        """
        T_wc = self.camera_pose()
        if timestamp is None:
            timestamp = self.ENV.step_counter / self.ENV.PYB_FREQ
        [w, h, rgb, dep, _] = p.getCameraImage(width=self.WIDTH,
                                               height=self.HEIGHT,
                                               viewMatrix=camera_pose_to_view_matrix(T_wc),
                                               projectionMatrix=self.PROJ,
                                               shadow=int(self.SHADOW),
                                               flags=p.ER_NO_SEGMENTATION_MASK,
                                               renderer=p.ER_TINY_RENDERER,
                                               physicsClientId=self.ENV.CLIENT
                                               )
        rgb = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
        name = "{:06d}".format(self.frame_count)
        Image.fromarray(rgb, 'RGB').save(os.path.join(self.OUTPUT_FOLDER, "images", name + ".png"))
        if self.SAVE_DEPTH:
            depth = depth_buffer_to_metric(np.reshape(dep, (h, w)), self.NEAR, self.FAR)
            np.save(os.path.join(self.OUTPUT_FOLDER, "depth", name + ".npy"), depth.astype(np.float32))
        quat = _rotation_matrix_to_quat_xyzw(T_wc[:3, :3])
        self._csv.writerow(["{:.6f}".format(timestamp)] + ["{:.9f}".format(v) for v in T_wc[:3, 3]]
                           + ["{:.9f}".format(v) for v in quat])
        self._csv_file.flush()
        self.frame_count += 1
        return rgb, T_wc

    ################################################################################

    def close(self):
        """Closes the pose file."""
        if not self._csv_file.closed:
            self._csv_file.close()

    ################################################################################

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

################################################################################

def _rotation_matrix_to_quat_xyzw(R: np.ndarray) -> list:
    """Converts a rotation matrix into a unit quaternion [x, y, z, w] with w >= 0."""
    m = np.asarray(R, dtype=float)
    tr = np.trace(m)
    if tr > 0.:
        s = 2. * np.sqrt(tr + 1.)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, .25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2. * np.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2])
        q = [.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2. * np.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, .25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2. * np.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, .25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return (q if q[3] >= 0. else -q).tolist()
