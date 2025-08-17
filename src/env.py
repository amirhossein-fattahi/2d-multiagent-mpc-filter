import gym
from gym import spaces
import numpy as np

DEFAULT_GOAL_THRESHOLD = 0.3


class MultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, n_agents=2, grid_size=10, normalize_obs=True, dt=0.1, radius=0.5, max_steps=200,
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
        return self._get_obs()

    def _get_obs(self):
        if self.normalize_obs:
            pos = self.pos / self.grid_size
            goals = self.goals / self.grid_size
        else:
            pos = self.pos
            goals = self.goals
        return np.concatenate([pos.flatten(), goals.flatten()]).astype(np.float32)
    
    
    def step(self, action):
        # reshape & clamp action
        act = np.clip(action.reshape(self.n, 2), -1, 1)

        # ensure mask exists
        if not hasattr(self, "reached_mask"):
            self.reached_mask = np.zeros(self.n, dtype=bool)

        # 1) freeze finished agents BEFORE moving
        act[self.reached_mask] = 0.0

        # distances BEFORE move (for progress)
        old_dists = np.linalg.norm(self.pos - self.goals, axis=1)

        # apply dynamics and clamp to the box
        self.pos = np.clip(self.pos + self.dt * act, 0.0, self.grid_size)
        self.steps += 1

        # distances AFTER move
        dists = np.linalg.norm(self.pos - self.goals, axis=1)
        progress = old_dists - dists  # >0 if we moved closer

        # ---------- rewards ----------
        # distance-weighted progress: stronger signal near goal
        # near_gain ~ 2.5 when on top of goal, ~1 when far
        near_gain = 1.0 + 1.5 * np.exp(-dists)
        rewards = 3.5 * near_gain * progress  # progress_scale ~= 3.5

        # smaller time penalty, only for agents not yet done
        active = (~self.reached_mask).astype(float)
        #rewards -= 0.01 * active  # step_cost ~= 0.02

        # collision penalty
        collisions_step = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < 2 * self.radius:
                    rewards[i] -= 10.0
                    rewards[j] -= 10.0
                    collisions_step += 1

        # NEW: gentle "close-range" magnet (helps final commit)
        capture_radius = max(self.goal_threshold, 0.7)  # curriculum-friendly
        inside = dists < capture_radius
        rewards += 0.5 * (capture_radius - dists) * inside.astype(float)

        # terminal bonus & sticky goal
        newly_reached = (~self.reached_mask) & (dists < self.goal_threshold)
        if np.any(newly_reached):
            rewards[newly_reached] += 300.0  # terminal_bonus ~300
            # snap to goal to avoid jitter
            self.pos[newly_reached] = self.goals[newly_reached]
            self.reached_mask |= newly_reached

        # update for next step
        self.prev_dists = dists.copy()

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