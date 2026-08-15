import os
import numpy as np
import xml.etree.ElementTree as etxml
from importlib.resources import files

from gym_pybullet_drones.utils.enums import DroneModel

class BaseControl(object):
    """Base class for control.

    Implements `__init__()`, `reset(), and interface `computeControlFromState()`,
    the main method `computeControl()` should be implemented by its subclasses.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        #### Set general use constants #############################
        self.DRONE_MODEL = drone_model
        """DroneModel: The type of drone to control."""
        self.GRAVITY = g*self._getURDFParameter('m')
        """float: The gravitational force (M*g) acting on each drone."""
        self.KF = self._getURDFParameter('kf')
        """float: The coefficient converting RPMs into thrust."""
        self.KM = self._getURDFParameter('km')
        """float: The coefficient converting RPMs into torque."""
        self.reset()

    ################################################################################

    def reset(self):
        """Reset the control classes.

        A general use counter is set to zero.

        """
        self.control_counter = 0

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_rpy_rates=np.zeros(3)
                                ):
        """Interface method using `computeControl`.

        It can be used to compute a control action directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step().

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        """
        return self.computeControl(control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=state[3:7],
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   target_pos=target_pos,
                                   target_rpy=target_rpy,
                                   target_vel=target_vel,
                                   target_rpy_rates=target_rpy_rates
                                   )

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Abstract method to compute the control action for a single drone.

        It must be implemented by each subclass of `BaseControl`.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        """
        raise NotImplementedError

################################################################################

    def setPIDCoefficients(self,
                           p_coeff_pos=None,
                           i_coeff_pos=None,
                           d_coeff_pos=None,
                           p_coeff_att=None,
                           i_coeff_att=None,
                           d_coeff_att=None
                           ):
        """Sets the coefficients of a PID controller.

        This method throws an error message and exist is the coefficients
        were not initialized (e.g. when the controller is not a PID one).

        Parameters
        ----------
        p_coeff_pos : ndarray, optional
            (3,1)-shaped array of floats containing the position control proportional coefficients.
        i_coeff_pos : ndarray, optional
            (3,1)-shaped array of floats containing the position control integral coefficients.
        d_coeff_pos : ndarray, optional
            (3,1)-shaped array of floats containing the position control derivative coefficients.
        p_coeff_att : ndarray, optional
            (3,1)-shaped array of floats containing the attitude control proportional coefficients.
        i_coeff_att : ndarray, optional
            (3,1)-shaped array of floats containing the attitude control integral coefficients.
        d_coeff_att : ndarray, optional
            (3,1)-shaped array of floats containing the attitude control derivative coefficients.

        """
        ATTR_LIST = ['P_COEFF_FOR', 'I_COEFF_FOR', 'D_COEFF_FOR', 'P_COEFF_TOR', 'I_COEFF_TOR', 'D_COEFF_TOR']
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[ERROR] in BaseControl.setPIDCoefficients(), not all PID coefficients exist as attributes in the instantiated control class.")
            exit()
        else:
            self.P_COEFF_FOR = self.P_COEFF_FOR if p_coeff_pos is None else p_coeff_pos
            self.I_COEFF_FOR = self.I_COEFF_FOR if i_coeff_pos is None else i_coeff_pos
            self.D_COEFF_FOR = self.D_COEFF_FOR if d_coeff_pos is None else d_coeff_pos
            self.P_COEFF_TOR = self.P_COEFF_TOR if p_coeff_att is None else p_coeff_att
            self.I_COEFF_TOR = self.I_COEFF_TOR if i_coeff_att is None else i_coeff_att
            self.D_COEFF_TOR = self.D_COEFF_TOR if d_coeff_att is None else d_coeff_att

    ################################################################################
    
    def _getURDFParameter(self,
                          parameter_name: str
                          ):
        """Reads a parameter from a drone's URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to read.

        Returns
        -------
        float
            The value of the parameter.

        """
        #### Get the XML tree of the drone model to control ########
        URDF = self.DRONE_MODEL.value + ".urdf"
        path = str(files('gym_pybullet_drones') / 'assets' / URDF)
        URDF_TREE = etxml.parse(path).getroot()
        #### Find and return the desired parameter #################
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff', 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]
