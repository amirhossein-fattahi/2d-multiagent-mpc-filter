import pytest
import numpy as np
from src.env import MultiAgentEnv

@ pytest.mark.parametrize("n", [2,3])

def test_reset_shape(n):
    env = MultiAgentEnv(n_agents=n)
    obs = env.reset()
    assert obs.shape == (4*n,)

@ pytest.mark.parametrize("pos1,pos2,rad,expected", [
    ([0,0],[0.5,0],0.3,True),
    ([0,0],[1,1],0.3,False)
])
def test_collision_detection(pos1, pos2, rad, expected):
    env = MultiAgentEnv(n_agents=2, radius=rad)
    env.pos = np.array([pos1, pos2])
    d = np.linalg.norm(env.pos[0]-env.pos[1])
    assert (d < 2*rad) == expected