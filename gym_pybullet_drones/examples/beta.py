"""Control + Betaflight. 

Setup
-----
Use script `gym_pybullet_drones/assets/clone_bfs.sh` to create
executables for as many drones as needed (e.g. 2):
    $ ./gym_pybullet_drones/assets/clone_bfs.sh 2

Note
-------
This example will automatically start as many SITL Betaflight as drones
in the simulation in separate terminals:

    $ cd betaflight_sitl/bf0/
    $ ./obj/main/betaflight_SITL.elf

    $ cd betaflight_sitl/bf1/
    $ ./obj/main/betaflight_SITL.elf

    $ ..

Run as:

    $ cd gym-pybullet-drones/gym_pybullet_drones/examples/
    $ python beta.py

"""
import time
import argparse
import numpy as np
import csv

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from transforms3d.utils import normalized_vector

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.BetaAviary import BetaAviary
from gym_pybullet_drones.control.CTBRControl import CTBRControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("racer")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 500
DEFAULT_CONTROL_FREQ_HZ = 500
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_DRONES = 2

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        ):
    #### Create the environment with or without video capture ##
    INIT_XYZ = np.array([[.3*i, .3*i, .1] for i in range(1,num_drones+1)])
    INIT_RPY = np.array([[.0, .0, .0] for _ in range(num_drones)])
    env = BetaAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZ,
                        initial_rpys=INIT_RPY,
                        physics=physics,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        user_debug_gui=user_debug_gui
                        )

    ctrl = CTBRControl(drone_model=drone)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    )

    #### Run the simulation ####################################
    with open("../assets/beta-traj.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        trajectory1 = iter([{
            "pos": np.array([
                float(row["p_x"]),
                float(row["p_y"]),
                float(row["p_z"]),
            ]),
            "vel": np.array([
                float(row["v_x"]),
                float(row["v_y"]),
                float(row["v_z"]),
            ]),
        } for row in csv_reader])
    with open("../assets/beta-traj.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        trajectory2 = iter(reversed([{
            "pos": np.array([
                float(row["p_x"]),
                float(row["p_y"]),
                float(row["p_z"]),
            ]),
            "vel": np.array([
                float(row["v_x"]),
                float(row["v_y"]),
                float(row["v_z"]),
            ]),
        } for row in csv_reader]))
    action = np.zeros((num_drones,4))
    ARM_TIME = 1.
    TRAJ_TIME = 1.5
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        t = i/env.CTRL_FREQ
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action, i)
        
        if t > env.TRAJ_TIME:
            for j in range(num_drones):
                try:
                    target = next(trajectory1) if j%2 == 0 else next(trajectory2)
                    action[j,:] = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                    state=obs[j],
                                                    target_pos=target["pos"]+[INIT_XYZ[j][0], INIT_XYZ[j][1], 0],
                                                    target_vel=target["vel"]
                                                    )
                except Exception:
                    break
        

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                        timestamp=i/env.CTRL_FREQ,
                        state=obs[j]
                        )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            pass
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("beta") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Test flight script using SITL Betaflight')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: BETA)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
