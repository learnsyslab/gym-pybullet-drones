import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
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
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
        self.FAILURE_GRACE_SEC = 3.0
        self.HOVER_SAFE_RADIUS = 0.25
        self.HOVER_SAFE_Z_MARGIN = 0.20
        self.HOVER_SAFE_TILT = 0.35
        self.HOVER_SAFE_YAW = 2.5
        self.prev_vel = np.zeros(3, dtype=np.float32)
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):
        """Reward a physically viable climb-to-hover objective.

        The agent receives a positive reward while it is moving toward the target and
        then a stable hover bonus once it is close to the setpoint. Large acceleration
        is penalized only after the initial grace window so the agent is not punished
        during the lift-off phase.
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        roll, pitch, yaw = state[7:10]

        # A flip-and-rest-on-the-ground state must be clearly worse than any
        # botched climb attempt, otherwise it becomes a low-variance local
        # optimum PPO settles into instead of trying to hover.
        if abs(roll) > self.HOVER_SAFE_TILT or abs(pitch) > self.HOVER_SAFE_TILT:
            return -20.0

        horizontal_error = np.linalg.norm(pos[0:2] - self.TARGET_POS[0:2])
        vertical_error = abs(pos[2] - self.TARGET_POS[2])
        pos_error = np.linalg.norm(self.TARGET_POS - pos)
        velocity_penalty = 0.5 * np.linalg.norm(vel)**2

        # Signed speed along the direction to the target: positive while closing
        # the gap, negative while drifting away. Used so that climbing quickly
        # toward the target is rewarded instead of being penalized just for
        # having speed.
        if pos_error > 1e-6:
            approach_speed = float(np.dot(vel, self.TARGET_POS - pos) / pos_error)
        else:
            approach_speed = 0.0

        if self.step_counter / self.PYB_FREQ >= self.FAILURE_GRACE_SEC:
            acceleration = np.linalg.norm(vel - self.prev_vel) / max(self.CTRL_TIMESTEP, 1e-6)
            acceleration_penalty = 1.5 * acceleration
        else:
            acceleration_penalty = 0.0
        self.prev_vel = vel.copy()

        hover_bonus = 3.0 if (horizontal_error <= self.HOVER_SAFE_RADIUS and vertical_error <= self.HOVER_SAFE_Z_MARGIN) else 0.0
        climb_bonus = 1.5 * max(0.0, 1.0 - pos_error)
        lift_bonus = 1.0 * max(0.0, pos[2] - 0.2)

        inside_safe_envelope = (
            horizontal_error <= self.HOVER_SAFE_RADIUS and
            vertical_error <= self.HOVER_SAFE_Z_MARGIN
        )

        if not inside_safe_envelope:
            progress_reward = 1.5 * max(0.0, approach_speed)
            drift_penalty = 0.75 * max(0.0, -approach_speed)
            overshoot_penalty = 2.5 * max(pos_error - 0.35, 0.0) + drift_penalty + acceleration_penalty
            return float(np.clip(-1.0 + climb_bonus + lift_bonus + progress_reward - overshoot_penalty, -50.0, 6.0))

        ret = 2.0 + hover_bonus + climb_bonus + lift_bonus - velocity_penalty - acceleration_penalty
        return float(np.clip(ret, -10.0, 6.0))

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        roll, pitch, yaw = state[7:10]
        # Fail fast on a flip, even during the grace window, so the agent can't
        # ride out a crash as a "safe" low-penalty resting state.
        if abs(roll) > self.HOVER_SAFE_TILT or abs(pitch) > self.HOVER_SAFE_TILT:
            return True
        hover_error = np.linalg.norm(self.TARGET_POS - pos)
        hover_velocity = np.linalg.norm(vel)
        if self.step_counter / self.PYB_FREQ < self.FAILURE_GRACE_SEC:
            return False
        if hover_error < 0.08 and hover_velocity < 0.10:
            return True
        if (np.linalg.norm(pos[0:2] - self.TARGET_POS[0:2]) > 0.75 or
            abs(pos[2] - self.TARGET_POS[2]) > 0.75):
            #abs(roll) > 1.0 or abs(pitch) > 1.0): #or abs(yaw) > 3.5):
            return True
        return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        roll, pitch, yaw = state[7:10]
        if self.step_counter / self.PYB_FREQ < self.FAILURE_GRACE_SEC:
            return False
        if (abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5 or pos[2] > 2.0):
            # or abs(roll) > .4 or abs(pitch) > .4):
            return True
        #if abs(yaw) > 3.5:
        #    return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
