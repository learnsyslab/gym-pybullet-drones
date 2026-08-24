import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class UnderactuatedHoverPriorAviary(HoverAviary):
    """Hover task using a standard quadrotor wrench-to-RPM backbone with one degraded motor."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
        degraded_motor: int = 0,
        motor_reduction: float = 0.35,
    ):
        self.DEGRADED_MOTOR = int(degraded_motor)
        self.MOTOR_REDUCTION = float(motor_reduction)
        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

    def _mixer_to_rpm(self, thrust, roll_torque, pitch_torque, yaw_torque):
        """Map a desired thrust and torque vector to rotor RPMs using the standard quadrotor mixer.

        The degraded rotor can only ever deliver ``MOTOR_REDUCTION`` of whatever RPM it is
        commanded, so simply rescaling every rotor's output to restore the total collective
        thrust (as a naive fix would) corrects thrust but leaves the roll/pitch/yaw torque
        that rotor was supposed to contribute uncompensated -- that residual torque is what
        was flipping the drone almost immediately. Instead, treat the degraded rotor's actual
        (capped) output as fixed and re-solve a least-squares correction across the three
        healthy rotors so the full 4-DOF wrench (thrust + roll + pitch + yaw) is matched as
        closely as three actuators can manage.
        """
        l = self.L / np.sqrt(2.0)
        mixer = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
            [-1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            [1.0, -1.0, 1.0, -1.0],
        ], dtype=np.float32)
        body_wrench = np.array([
            thrust,
            roll_torque * l,
            pitch_torque * l,
            yaw_torque,
        ], dtype=np.float32)

        rotor_sq_nominal = np.linalg.pinv(mixer) @ body_wrench
        rotor_sq_nominal = np.clip(rotor_sq_nominal, 0.0, None)

        degraded_actual = self.MOTOR_REDUCTION * rotor_sq_nominal[self.DEGRADED_MOTOR]
        deficit = mixer[:, self.DEGRADED_MOTOR] * (rotor_sq_nominal[self.DEGRADED_MOTOR] - degraded_actual)

        # Three healthy rotors can't cancel a 4-DOF (thrust/roll/pitch/yaw) deficit
        # exactly -- some residual error is unavoidable. Roll/pitch error is what
        # tips the drone over, so weight the least-squares fit to cancel that
        # first, and let yaw (and, to a lesser extent, thrust) absorb the leftover
        # error instead.
        weights = np.array([8.0, 12.0, 12.0, 0.3], dtype=np.float32)
        sqrt_w = np.sqrt(weights)
        healthy = [i for i in range(4) if i != self.DEGRADED_MOTOR]
        weighted_A = sqrt_w[:, None] * mixer[:, healthy]
        weighted_b = sqrt_w * deficit
        correction = np.linalg.pinv(weighted_A) @ weighted_b

        rotor_sq = rotor_sq_nominal.copy()
        rotor_sq[self.DEGRADED_MOTOR] = degraded_actual
        for idx, h in enumerate(healthy):
            rotor_sq[h] = rotor_sq_nominal[h] + correction[idx]

        rotor_sq = np.clip(rotor_sq, 0.0, None)
        rpm = np.sqrt(np.clip(rotor_sq / self.KF, 0.0, None))
        return np.clip(rpm, 0.0, self.MAX_RPM)

    def _preprocessAction(self, action):
        """Interpret the action as a nominal hover correction in wrench-space, then project it through the quadrotor mixer."""
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.shape[-1] == 4:
            for k in range(action.shape[0]):
                thrust_cmd = self.GRAVITY * (1.0 + 0.25 * action[k, 0])
                roll_cmd = 0.12 * action[k, 1]
                pitch_cmd = 0.12 * action[k, 2]
                yaw_cmd = 0.08 * action[k, 3]
                rpm_k = self._mixer_to_rpm(
                    thrust=thrust_cmd,
                    roll_torque=roll_cmd,
                    pitch_torque=pitch_cmd,
                    yaw_torque=yaw_cmd,
                )
                if k == 0:
                    rpm = np.zeros((action.shape[0], 4), dtype=np.float32)
                rpm[k, :] = rpm_k
            return rpm
        return super()._preprocessAction(action)
