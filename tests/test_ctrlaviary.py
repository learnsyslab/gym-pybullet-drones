import pytest
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

def test_ctrlaviary_vision_attributes():
    env = CtrlAviary(vision_attributes=True)
    env.reset()
    rgb, dep, seg = env._getDroneImages(0)
    assert rgb.shape == (48, 64, 4)
    assert dep.shape == (48, 64)
    env.close()

def test_get_images_without_vision_raises():
    env = CtrlAviary()            # vision_attributes=False
    env.reset()
    with pytest.raises(ValueError):
        env._getDroneImages(0)
    env.close()