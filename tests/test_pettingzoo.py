import numpy as np
import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.PettingZooWrapper import PettingZooWrapper


@pytest.mark.parametrize("act", [ActionType.RPM, ActionType.PID, ActionType.VEL, ActionType.ONE_D_RPM])
def test_parallel_api(act):
    env = PettingZooWrapper(MultiHoverAviary(num_drones=2, act=act))
    parallel_api_test(env, num_cycles=300)
    env.close()


def test_three_drones_api():
    env = PettingZooWrapper(MultiHoverAviary(num_drones=3))
    parallel_api_test(env, num_cycles=300)
    env.close()


def test_seed():
    parallel_seed_test(lambda: PettingZooWrapper(MultiHoverAviary(num_drones=2)))


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


def test_episode_ends_for_all_agents():
    env = PettingZooWrapper(MultiHoverAviary(num_drones=2))
    env.reset(seed=0)
    steps = 0
    while env.agents:
        _, _, terminations, truncations, _ = env.step({a: env.action_space(a).sample() for a in env.agents})
        steps += 1
        assert len(set(terminations.values())) == 1 and len(set(truncations.values())) == 1
    assert steps > 0 and env.agents == []
    with pytest.raises(RuntimeError):
        env.step({})
    env.close()


def test_single_drone_aviary():
    env = PettingZooWrapper(HoverAviary())
    assert env.possible_agents == ["drone_0"]
    parallel_api_test(env, num_cycles=100)
    env.close()


def test_rejects_non_aviary():
    with pytest.raises(TypeError):
        PettingZooWrapper(object())


def test_vision_observations():
    """RGB observations are split into one camera image per drone."""
    env = PettingZooWrapper(MultiHoverAviary(num_drones=2, obs=ObservationType.RGB, ctrl_freq=48))
    parallel_api_test(env, num_cycles=20)
    observations, _ = env.reset(seed=0)
    assert observations["drone_0"].shape == env.observation_space("drone_0").shape
    assert observations["drone_0"].ndim == 3
    with pytest.raises(NotImplementedError):
        env.state()
    env.close()


def test_example():
    from gym_pybullet_drones.examples.pettingzoo_multiagent import run
    returns = run(num_drones=2, gui=False)
    assert set(returns) == {"drone_0", "drone_1"}
