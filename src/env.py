import gym
from gym import spaces
import numpy as np

class MultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, n_agents=2, grid_size=10, dt=0.1, radius=0.5, max_steps=200):
        super().__init__()
        self.n = n_agents
        self.grid_size = grid_size
        self.dt = dt
        self.radius = radius
        self.max_steps = max_steps

        # Observation: [pos1, pos2, ..., goal1, goal2, ...]
        dim = 4 * self.n
        self.observation_space = spaces.Box(
            low=0, high=grid_size, shape=(dim,), dtype=np.float32
        )
        # Action: concatenated velocity vectors
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2*self.n,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.pos = np.random.uniform(0, self.grid_size, (self.n, 2))
        self.goals = np.random.uniform(0, self.grid_size, (self.n, 2))
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.pos.flatten(), self.goals.flatten()])

    def step(self, action):
        act = action.reshape(self.n, 2)
        self.pos += self.dt * np.clip(act, -1, 1)
        self.steps += 1

        dists = np.linalg.norm(self.pos - self.goals, axis=1)
        rewards = -dists.copy()
        # collision penalty
        for i in range(self.n):
            for j in range(i+1, self.n):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < 2*self.radius:
                    rewards[i] -= 10
                    rewards[j] -= 10

        done = self.steps >= self.max_steps or np.all(dists < 0.1)
        return self._get_obs(), rewards, done, {}

    def render(self, mode='console'):
        print(f"Step {self.steps}: positions={self.pos}")

    def close(self):
        pass