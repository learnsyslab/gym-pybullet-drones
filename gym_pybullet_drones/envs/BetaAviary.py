import numpy as np
from gymnasium import spaces
import socket
import struct 
import os
import subprocess
import time

from transforms3d.quaternions import rotate_vector, qconjugate

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

BASE_PORT_PWM=9002 # In
BASE_PORT_STATE=9003 # Out
BASE_PORT_RC=9004 # Out

class BetaAviary(BaseAviary):
    """Multi-drone environment class for use of BetaFlight controller."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 udp_ip="127.0.0.1"
                 ):
        """Initialization of an aviary environment for use of BetaFlight controller.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        udp_ip : base ip for betaflight controller emulator 

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )
        
        # Spawn SITL Betaflight instances (must have been created with assets/clone_bfs/sh first)
        for i in range(num_drones):
            FOLDER = os.path.dirname(os.path.abspath(__file__))+'/../../betaflight_sitl/bf'+str(i)+'/'
            cmd = f"bash -c 'cd {FOLDER} && ./obj/main/betaflight_SITL.elf; exec bash'"
            subprocess.Popen(cmd, shell=True)
        time.sleep(2)
        
        # Initialize connection to BetaFlight Controller 
        self.UDP_IP = udp_ip
        self.ARM_TIME = 1
        self.TRAJ_TIME = 1.5

        self.sock = []
        self.sock_pwm = []
        for i in range(self.NUM_DRONES):
            self.sock.append(socket.socket(socket.AF_INET,    # Internet
                                socket.SOCK_DGRAM)) # UDP
            self.sock_pwm.append(socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM))  # UDP
            self.sock_pwm[i].bind((self.UDP_IP, BASE_PORT_PWM + 10 * (i))) # drone0 will be on 9002, drone1 on 9012, etc.
            self.sock_pwm[i].settimeout(0.0)

        self.beta_action = np.zeros((self.NUM_DRONES, 4))
    
    ################################################################################

    def step(self, action, i):
        obs, reward, terminated, truncated, info = super().step(self.beta_action)

        t = i/self.CTRL_FREQ

        for j in range(self.NUM_DRONES): #TODO: add multi-drone support

            #### State message to Betaflight ###########################
            o = obs[j,:] # p, q, euler, v, w, rpm (all in world frame)
            p = o[:3]
            q = np.array([o[6], o[3], o[4], o[5]]) # w, x, y, z
            v = o[10:13]
            w = o[13:16] # world frame
            w_body = rotate_vector(w, qconjugate(q)) # local frame
            
            fdm_packet = struct.pack(
                '@dddddddddddddddddd', # t, w, a, q, v, p, pressure
                t, # double timestamp in seconds
                # minus signs due to ENU to NED conversion
                w_body[0], -w_body[1], -w_body[2], # double imu_angular_velocity_rpy[3]
                0, 0, 0, # double imu_linear_acceleration_xyz[3]
                1., .0, .0, 0., # double imu_orientation_quat[4] w, x, y, z
                0, 0, 0, # double velocity_xyz[3]
                0, 0, 0, # double position_xyz[3]
                1.0 # double pressure;
            )
            self.sock[j].sendto(fdm_packet, (self.UDP_IP, BASE_PORT_STATE + 10 * (j))) # drone0 will be on 9003, drone1 on 9013, etc.

            # Action is (thro, roll, pitch, yaw) 
            thro = 1000 # Positive up
            yaw = 1500 # Positive CCW
            pitch = 1500 # Positive forward in x
            roll = 1500 # Positive right/forward in y

            if t > self.TRAJ_TIME: 
                thro, roll, pitch, yaw = self.ctbr2beta(*action[j,:])

            #### RC message to Betaflight ##############################
            aux1 = 1000 if t < self.ARM_TIME else 1500 # Arm
            rc_packet = struct.pack(
                '@dHHHHHHHHHHHHHHHH', 
                t, # datetime.now().timestamp(), # double timestamp; // in seconds
                round(roll), round(pitch), round(thro), round(yaw),              # roll, pitch, throttle, yaw
                aux1, 1000, 1000, 1000,              # aux 1, ..
                1000, 1000, 1000, 1000, 
                1000, 1000, 1000, 1000
            )
            # print("rc", struct.unpack('@dHHHHHHHHHHHHHHHH', rc_packet))
            self.sock[j].sendto(rc_packet, (self.UDP_IP, BASE_PORT_RC + 10 * (j))) # drone0 will be on 9004, drone1 on 9014, etc.

            #### PWMs message from Betaflight ##########################
            try:
                data, addr = self.sock_pwm[j].recvfrom(16) # buffer size is 100 bytes (servo_packet size 16)
            except socket.error as msg:
                _action = self.beta_action[j,:]
                pass
            else:
                # print("received message: ", data)
                _action = np.array(struct.unpack('@ffff', data)).reshape((1,4))
            self.beta_action[j,:] = _action

        return obs, reward, terminated, truncated, info

    ################################################################################
    
    def ctbr2beta(self, thrust, roll, pitch, yaw):
        MIN_CHANNEL = 1000
        MAX_CHANNEL = 2000
        MAX_RATE = 360
        MAX_THRUST = 40.9
        mid = (MAX_CHANNEL + MIN_CHANNEL) / 2
        d = (MAX_CHANNEL - MIN_CHANNEL) / 2
        thrust = thrust / MAX_THRUST * d * 2 + MIN_CHANNEL
        rates = np.array([roll, pitch, -yaw])
        rates = rates / np.pi * 180 / MAX_RATE * d + mid
        thrust = np.clip(thrust, MIN_CHANNEL, MAX_CHANNEL)
        rates = np.clip(rates, MIN_CHANNEL, MAX_CHANNEL)
        return thrust, *rates

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded CTBR to Betaflight SITL.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        remapped_input = np.array([[
                                    action[i][2], # 0
                                    action[i][1], # 1
                                    action[i][3], # 2
                                    action[i][0]  # 3
                                ] for i in range(self.NUM_DRONES)]) # Betaflight SITL motor mapping

        ret = np.array(np.sqrt(self.MAX_THRUST / 4 / self.KF * remapped_input))
        assert(ret.shape == (self.NUM_DRONES, 4)), "Error in preprocess action"
        return ret

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
