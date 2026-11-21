"""Script recording a posed onboard camera sequence.

A single drone, controlled by `DSLPIDControl` in a `CtrlAviary` with
obstacles, flies a climbing circle while facing its direction of travel.
A body-fixed camera is recorded with `CameraRecorder`, which writes
images, camera-to-world poses (OpenCV camera axes) and intrinsics.

The output can be used to train or evaluate visual odometry, SLAM,
novel-view synthesis or action-conditioned world models.

Example
-------
In a terminal, run as:

    $ python record_camera.py --duration_sec 20 --camera_hz 10

"""
import argparse
import os
import time
from datetime import datetime

import numpy as np

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.CameraRecorder import CameraRecorder
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool, sync

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_CAMERA_HZ = 12
DEFAULT_DURATION_SEC = 12
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_FOV_DEG = 90.
DEFAULT_PITCH_DEG = 15.
DEFAULT_SAVE_DEPTH = True
DEFAULT_OUTPUT_FOLDER = 'results'

def run(
        drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        camera_hz=DEFAULT_CAMERA_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        fov_deg=DEFAULT_FOV_DEG,
        pitch_deg=DEFAULT_PITCH_DEG,
        save_depth=DEFAULT_SAVE_DEPTH,
        output_folder=DEFAULT_OUTPUT_FOLDER
        ):
    if control_freq_hz % camera_hz != 0:
        raise ValueError("camera_hz must divide control_freq_hz")

    #### Climbing circle, facing the direction of travel #######
    R, H0, H1 = 1.0, 0.5, 1.2
    INIT_XYZS = np.array([[R, 0., H0]])
    INIT_RPYS = np.array([[0., 0., np.pi/2]])
    env = CtrlAviary(drone_model=drone,
                     num_drones=1,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=Physics.PYB,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     obstacles=True
                     )
    ctrl = DSLPIDControl(drone_model=drone)

    folder = os.path.join(output_folder, 'camera_' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    rec = CameraRecorder(env, folder, drone=0, width=width, height=height,
                         fov_deg=fov_deg, pitch_deg=pitch_deg, save_depth=save_depth)

    action = np.zeros((1, 4))
    steps = int(duration_sec * env.CTRL_FREQ)
    every = int(control_freq_hz // camera_hz)
    START = time.time()
    for i in range(steps):
        obs, _, _, _, _ = env.step(action)
        if i % every == 0:
            rec.capture()

        s = (i + 1) / steps
        theta = 2 * np.pi * s
        target_pos = np.array([R * np.cos(theta), R * np.sin(theta), H0 + (H1 - H0) * np.sin(np.pi * s)])
        target_rpy = np.array([0., 0., theta + np.pi/2])
        action[0, :], _, _ = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                          state=obs[0],
                                                          target_pos=target_pos,
                                                          target_rpy=target_rpy
                                                          )
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    rec.close()
    env.close()
    print("[INFO] saved", rec.frame_count, "posed frames to", folder)
    return folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record a posed onboard camera sequence')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,              type=DroneModel, help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,                type=str2bool,   help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,        help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,    type=int,        help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--camera_hz',          default=DEFAULT_CAMERA_HZ,          type=int,        help='Camera frame rate in Hz, must divide control_freq_hz (default: 12)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,       type=int,        help='Duration of the simulation in seconds (default: 12)', metavar='')
    parser.add_argument('--width',              default=DEFAULT_WIDTH,              type=int,        help='Image width (default: 256)', metavar='')
    parser.add_argument('--height',             default=DEFAULT_HEIGHT,             type=int,        help='Image height (default: 256)', metavar='')
    parser.add_argument('--fov_deg',            default=DEFAULT_FOV_DEG,            type=float,      help='Vertical field of view in degrees (default: 90)', metavar='')
    parser.add_argument('--pitch_deg',          default=DEFAULT_PITCH_DEG,          type=float,      help='Camera down-tilt in degrees (default: 15)', metavar='')
    parser.add_argument('--save_depth',         default=DEFAULT_SAVE_DEPTH,         type=str2bool,   help='Whether to save metric depth maps (default: True)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,      type=str,        help='Folder where to save the sequence (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
