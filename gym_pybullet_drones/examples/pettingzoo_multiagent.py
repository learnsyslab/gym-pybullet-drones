"""Script demonstrating `gym_pybullet_drones`'s PettingZoo (Parallel API) interface.

Class MultiHoverAviary is wrapped with PettingZooWrapper so that each drone is
a separate agent (`drone_0`, `drone_1`, ...), as expected by multi-agent RL
libraries that use PettingZoo.

Example
-------
In a terminal, run as:

    $ python pettingzoo_multiagent.py
    $ python pettingzoo_multiagent.py --num_drones 3 --gui true

Notes
-----
Every drone sends the "hover" action (zero offset from hover RPM). The script
only illustrates the dictionary-based API; plug in your own policies instead.

"""
import argparse

import numpy as np

from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.PettingZooWrapper import PettingZooWrapper
from gym_pybullet_drones.utils.utils import str2bool

DEFAULT_NUM_DRONES = 2
DEFAULT_GUI = False

def run(num_drones=DEFAULT_NUM_DRONES, gui=DEFAULT_GUI):
    env = PettingZooWrapper(MultiHoverAviary(num_drones=num_drones,
                                             obs=ObservationType.KIN,
                                             act=ActionType.RPM,
                                             gui=gui
                                             ))
    print("[INFO] Agents:", env.possible_agents)
    print("[INFO] Per-agent observation space:", env.observation_space("drone_0"))
    print("[INFO] Per-agent action space:", env.action_space("drone_0"))

    observations, infos = env.reset(seed=42)
    returns = {agent: 0.0 for agent in env.possible_agents}
    steps = 0
    while env.agents:
        actions = {agent: np.zeros(env.action_space(agent).shape, dtype=np.float32) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent, r in rewards.items():
            returns[agent] += r
        steps += 1
    print("[INFO] Episode finished after", steps, "steps, returns:", {a: round(r, 2) for a, r in returns.items()})
    env.close()
    return returns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PettingZoo multi-agent example")
    parser.add_argument("--num_drones", default=DEFAULT_NUM_DRONES, type=int, help="Number of drones (default: 2)", metavar="")
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, help="Whether to use PyBullet GUI (default: False)", metavar="")
    ARGS = parser.parse_args()
    run(**vars(ARGS))
