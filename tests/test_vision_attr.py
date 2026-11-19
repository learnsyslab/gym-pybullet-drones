import pytest
from gym_pybullet_drones.envs import CtrlAviary, VelocityAviary


@pytest.mark.parametrize("env_cls", [CtrlAviary, VelocityAviary])
def test_vision_attributes(env_cls):
    env = env_cls(vision_attributes=True)
    env.reset()
    rgb, dep, seg = env._getDroneImages(0)
    assert rgb.shape == (48, 64, 4)
    assert dep.shape == (48, 64)
    env.close()


@pytest.mark.parametrize("env_cls", [CtrlAviary, VelocityAviary])
def test_get_images_without_vision_raises(env_cls):
    env = env_cls()  # vision_attributes=False
    env.reset()
    with pytest.raises(ValueError):
        env._getDroneImages(0)
    env.close()
