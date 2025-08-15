import gym
from gym import spaces
import numpy as np

DEFAULT_GOAL_THRESHOLD = 0.3


class MultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, n_agents=2, grid_size=10, dt=0.1, radius=0.5, max_steps=200, goal_threshold=DEFAULT_GOAL_THRESHOLD):
        super().__init__()
        self.n = n_agents
        self.grid_size = grid_size
        self.dt = dt
        self.radius = radius
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold

        dim = 4 * self.n
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2*self.n,), dtype=np.float32)

        self.reset()

    def reset(self, starts=None, goals=None):
        if starts is not None:
            self.pos = np.array(starts, dtype=float).reshape(self.n, 2)
        else:
            self.pos = np.random.uniform(0, self.grid_size, (self.n, 2))
        if goals is not None:
            self.goals = np.array(goals, dtype=float).reshape(self.n, 2)
        else:
            self.goals = np.random.uniform(0, self.grid_size, (self.n, 2))
        self.steps = 0
        self.reached_mask = np.zeros(self.n, dtype=bool)
        dists = np.linalg.norm(self.pos - self.goals, axis=1)
        self.prev_dists = dists.copy()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.pos.flatten(), self.goals.flatten()])

    def step(self, action):
        act = action.reshape(self.n, 2)
        self.pos += self.dt * np.clip(act, -1, 1)
        self.pos = np.clip(self.pos, 0.0, self.grid_size)
        self.steps += 1

        # freeze agents that already reached their goal
        if hasattr(self, "reached_mask"):
            act[self.reached_mask] = 0.0
        else:
            self.reached_mask = np.zeros(self.n, dtype=bool)

        # apply dynamics and clamp to the box
        

        # distances and progress
        dists = np.linalg.norm(self.pos - self.goals, axis=1)
        if not hasattr(self, "prev_dists"):
            self.prev_dists = dists.copy()
        progress = self.prev_dists - dists  # >0 if moving closer
        self.prev_dists = dists.copy()

        # after computing progress
        weights = np.ones(self.n)
        if self.reached_mask.any():
            # boost the incentive for the not-yet-reached agents
            weights[~self.reached_mask] = 2.0   # try 2–3x
        rewards = weights * (8.0 * progress) - 0.02

        # reward: progress bonus + small step cost
        #rewards = 10.0 * progress - 0.05  # tweak 5.0 and 0.05 as needed

        collisions_step = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < 2*self.radius:
                    rewards[i] -= 10.0
                    rewards[j] -= 10.0
                    collisions_step += 1

        # goal handling: one-time bonus + mark reached
        reached = dists < self.goal_threshold
        rewards[reached] += 100.0  # one-time per-step bonus is okay in our simple setup

        all_reached = bool(np.all(reached))
        done = self.steps >= self.max_steps or all_reached

        info = {
            "all_reached": all_reached,
            "reached_mask": reached.copy(),
            "collisions_step": collisions_step
        }
        return self._get_obs(), rewards, done, info

    def render(self, mode='console'):
        print(f"Step {self.steps}: positions={self.pos}")

    def close(self):
        pass