import gym
from gym import spaces
import numpy as np

DEFAULT_GOAL_THRESHOLD = 0.3


class MultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, n_agents=2, grid_size=10, obs_mode="abs", normalize_obs=True, dt=0.1, radius=0.5, max_steps=200,
             goal_threshold=DEFAULT_GOAL_THRESHOLD,
             terminal_bonus=200.0, progress_scale=5.0, step_cost=0.05,
             sticky_on_goal=True):
        super().__init__()
        self.n = n_agents
        self.grid_size = grid_size
        self.dt = dt
        self.radius = radius
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        self.obs_mode = obs_mode
        if obs_mode == "abs": dim = 4*self.n
        elif obs_mode == "abs+rel": dim = 6*self.n
        
        # new knobs
        self.terminal_bonus = terminal_bonus
        self.progress_scale = progress_scale
        self.step_cost = step_cost
        self.sticky_on_goal = sticky_on_goal
        self.normalize_obs = normalize_obs
        dim = 4 * self.n
        self.observation_space = spaces.Box(
            low=0.0 if normalize_obs else 0.0,
            high=1.0 if normalize_obs else self.grid_size,
            shape=(dim,), dtype=np.float32
        )
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

        if getattr(self, "sample_short_tasks", False):
            # re-sample until distances are within [3, 6]
            for _ in range(100):
                self.pos = np.random.uniform(0, self.grid_size, (self.n, 2))
                self.goals = np.random.uniform(0, self.grid_size, (self.n, 2))
                d0 = np.linalg.norm(self.pos - self.goals, axis=1)
                if np.all((d0 > 3.0) & (d0 < 6.0)):
                    break
                
        return self._get_obs()

    def _get_obs(self):
        pos = self.pos / self.grid_size if self.normalize_obs else self.pos
        goals = self.goals / self.grid_size if self.normalize_obs else self.goals
        if self.obs_mode == "abs":
            obs = [pos, goals]
        else:  # "abs+rel"
            rel = (self.goals - self.pos) / self.grid_size
            obs = [pos, goals, rel]
        return np.concatenate([a.flatten() for a in obs]).astype(np.float32)
    
    def step(self, action):
        act = action.reshape(self.n, 2)

        # ensure mask exists
        if not hasattr(self, "reached_mask"):
            self.reached_mask = np.zeros(self.n, dtype=bool)

        # 1) Freeze reached BEFORE moving
        act[self.reached_mask] = 0.0

        # 2) Move and clamp
        act = np.clip(act, -1, 1)
        self.pos += self.dt * act
        self.pos = np.clip(self.pos, 0.0, self.grid_size)
        self.steps += 1

        # 3) Distances and progress
        dists = np.linalg.norm(self.pos - self.goals, axis=1)
        if not hasattr(self, "prev_dists"):
            self.prev_dists = dists.copy()
        progress = self.prev_dists - dists
        self.prev_dists = dists.copy()

        # 4) Progress reward (+ close-range magnet)
        weights = np.ones(self.n)
        if self.reached_mask.any():
            weights[~self.reached_mask] = 2.0
        rewards = weights * (8.0 * progress) - 0.01

        capture_radius = max(self.goal_threshold, 0.7)  # wider pull early in training
        inside = dists < capture_radius
        rewards += 5.0 * (capture_radius - dists) * inside.astype(float)

        # 5) Collisions
        collisions_step = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < 2*self.radius:
                    rewards[i] -= 5.0
                    rewards[j] -= 5.0
                    collisions_step += 1

        # 6) Terminal bonus + sticky goal
        newly_reached = (~self.reached_mask) & (dists < self.goal_threshold)
        if np.any(newly_reached):
            rewards[newly_reached] += 400.0
            # snap & freeze to avoid overshoot/jitter
            self.pos[newly_reached] = self.goals[newly_reached]
            self.reached_mask |= newly_reached

        all_reached = bool(np.all(self.reached_mask))
        done = (self.steps >= self.max_steps) or all_reached
        info = {
            "all_reached": all_reached,
            "reached_mask": self.reached_mask.copy(),
            "collisions_step": collisions_step
        }
        return self._get_obs(), rewards, done, info





    def render(self, mode='console'):
        print(f"Step {self.steps}: positions={self.pos}")

    def close(self):
        pass