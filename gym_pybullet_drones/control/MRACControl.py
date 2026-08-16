import math
import numpy as np
import pybullet as p
import control as ct
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_lyapunov

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


class MRACControl(BaseControl):
    """Model Reference Adaptive Controller class for Crazyflies.

        Based on the implementation of https://github.com/caoty777/Quadcoptor-Adaptive-Flight-Control
    """

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
            print("[ERROR] MRAC requires DroneModel.CF2X or DroneModel.CF2P or DroneModel.RACE")
            exit()
        self.Ixx = self._getURDFParameter("ixx")
        self.Iyy = self._getURDFParameter("iyy")
        self.Izz = self._getURDFParameter("izz")
        self.J = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.mass = self._getURDFParameter("m")
        self.l = self._getURDFParameter("arm")
        self.g = g
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.Ka = self.KF
        self.Km = self.KM

        if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])

        self.Kx, self.Kr = self._compute_K()
        self.Xm = np.zeros((12))
        self.reset()

    def _compute_K(self, psi=0):
        """x = x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r
            u = [w1^2, w2^2, w3^2, w4^2] or [thrust, tx, ty, tz]
        """
        g = self.g
        m = self.mass
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        l = self.l
        Ka = self.Ka
        Km = self.Km

        a_sub = np.array([[0, 0, 0, g*np.sin(psi), g*np.cos(psi), 0],
                          [0, 0, 0, -g*np.cos(psi), g*np.sin(psi), 0]])
        a_sub = np.vstack((a_sub, np.zeros((4, 6))))
        
        A = np.block([[np.zeros((6,6)), np.eye(6)],
                      [a_sub, np.zeros((6,6))]])
        
        b_sub = np.array([[1/m, 0, 0, 0],
                          [0, 1/Ixx, 0, 0],
                          [0, 0, 1/Iyy, 0],
                          [0, 0, 0, 1/Izz]])    
        # b_sub = np.array([[Ka/m, Ka/m, Ka/m, Ka/m],
        #             [0, -Ka*l/Ixx, 0, Ka*l/Ixx],
        #             [Ka*l/Iyy, 0, -Ka*l/Iyy, 0],
        #             [Km/Izz, -Km/Izz, Km/Izz, -Km/Izz]]) # For direct rpm
        
        B = np.vstack((np.zeros((8, 4)), b_sub))
        Q = np.eye(12)*600
        # Q = np.diag((700, 700, 700, 500, 500, 500, 500, 500, 500, 500, 500, 500))
        # R = np.eye(4)*10
        # K, self.P, _ = ct.lqr(A, B, Q, R)

        desired_poles = -np.linspace(1, 12, 12)
        K = ct.place(A, B, desired_poles)
        self.Kr_ref_gain = np.linalg.pinv(B) @ (A - B @ K)

        self.Am = A - B@K
        self.Bm = np.copy(B)
        self.P = solve_lyapunov(self.Am.T, -Q)

        self.Gamma_x = np.eye(12) * 5e-3
        self.Gamma_r = np.eye(4) * 5e-3

        Kx = -K.T
        Kr = np.eye(4) 
        return Kx, Kr

    def reset(self):
        super().reset()

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        cur_ang_vel = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_ang_vel) # Convert angular velocity to body frame
        
        if self.control_counter == 0:
            self.Xm = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel)).reshape(12, 1)
        self.control_counter += 1

        r = np.hstack((target_pos, target_rpy, target_vel, target_rpy_rates)).reshape(12, 1)
        rt = -self.Kr_ref_gain @ r

        X_actual = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel)).reshape(12, 1)
        u = self.Kx.T @ X_actual + self.Kr.T @ rt
        e = X_actual - self.Xm # TODO plot X_actual and Xm
        Kx_dot = -self.Gamma_x @ X_actual @ e.T @ self.P @ self.Bm
        Kr_dot = -self.Gamma_r @ rt @ e.T @ self.P @ self.Bm

        self.Kx += Kx_dot * control_timestep
        self.Kr += Kr_dot * control_timestep

        thrust, tx, ty, tz = u.squeeze()
        thrust = np.maximum(0, thrust)
        target_torques = np.hstack((tx, ty, tz))
        target_torques = np.clip(target_torques, -3200, 3200)

        thrust = (math.sqrt(thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        
        pos_e = target_pos - cur_pos
        rpy_e = target_rpy - cur_rpy

        Xm_dot = self.Am @ self.Xm + self.Bm @ rt
        self.Xm += Xm_dot*control_timestep

        return rpm, pos_e, rpy_e
