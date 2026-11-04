import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ActionType
from gym_pybullet_drones.utils.PettingZooWrapper import PettingZooWrapper

@pytest.mark.parametrize("act", [ActionType.RPM, ActionType.PID, ActionType.VEL, ActionType.ONE_D_RPM])
def test_parallel_api(act):
    env = PettingZooWrapper(MultiHoverAviary(num_drones=2, act=act))
    parallel_api_test(env, num_cycles=300)
    env.close()

def test_matches_underlying_aviary():
    """Per-agent observations are the rows of the aviary's stacked observation,
    and the team reward is shared by every agent."""
    env = PettingZooWrapper(MultiHoverAviary(num_drones=2))
    observations, _ = env.reset(seed=0)
    for i, agent in enumerate(env.possible_agents):
        np.testing.assert_allclose(observations[agent], env._last_obs[i], rtol=1e-6)
        assert env.observation_space(agent).contains(observations[agent])
    actions = {a: env.action_space(a).sample() for a in env.agents}
    observations, rewards, terminations, truncations, _ = env.step(actions)
    assert len(set(rewards.values())) == 1
    assert set(rewards) == set(env.possible_agents)
    np.testing.assert_allclose(env.state(), env._last_obs.flatten(), rtol=1e-6)
    assert env.state_space.contains(env.state())
    env.close()

def test_actions_reach_the_right_drone():
    """Throttling only drone_1 up should lift drone_1, not drone_0."""
    env = PettingZooWrapper(MultiHoverAviary(num_drones=2, act=ActionType.RPM))
    env.reset(seed=0)
    for _ in range(20):
        env.step({"drone_0": -np.ones(4, dtype=np.float32), "drone_1": np.ones(4, dtype=np.float32)})
    z = env.aviary.pos[:, 2]
    init_z = env.aviary.INIT_XYZS[:, 2]
    assert z[1] > init_z[1] and z[0] <= init_z[0] + 1e-3
    env.close()
