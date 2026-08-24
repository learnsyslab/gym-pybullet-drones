import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class UnderactuatedHoverAviary(HoverAviary):
    """Single-agent hover task with one motor disabled to make the drone underactuated."""

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
        disabled_motor: int = 0,
        motor_reduction: float = 0.35,
    ):
        self.DISABLED_MOTOR = disabled_motor
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

    def _preprocessAction(self, action):
        """Reduce one motor's output instead of turning it completely off."""
        rpm = super()._preprocessAction(action)
        if rpm.ndim == 2:
            rpm[:, self.DISABLED_MOTOR] *= self.MOTOR_REDUCTION
        return rpm
