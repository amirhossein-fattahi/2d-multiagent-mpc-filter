import gym
from gym import spaces
import numpy as np

DEFAULT_GOAL_THRESHOLD = 0.3


class MultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, n_agents=2, grid_size=10, obs_mode="abs", normalize_obs=True,
                 dt=0.1, radius=0.5, max_steps=200, goal_threshold=DEFAULT_GOAL_THRESHOLD,
                 terminal_bonus=200.0, progress_scale=5.0, step_cost=0.05, sticky_on_goal=True):
        super().__init__()
        self.n = n_agents
        self.grid_size = grid_size
        self.dt = dt
        self.radius = radius
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        self.obs_mode = obs_mode
        self.normalize_obs = normalize_obs

        # ---- FIX: correct observation dimension (no override later) ----
        if self.obs_mode == "abs":
            dim = 4 * self.n           # [x,y] for all agents + [gx,gy] for all
        elif self.obs_mode == "abs+rel":
            dim = 6 * self.n           # add [gx-x, gy-y] for all
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

        # knobs (unchanged)
        self.terminal_bonus = terminal_bonus
        self.progress_scale = progress_scale
        self.step_cost = step_cost
        self.sticky_on_goal = sticky_on_goal

        # ---- FIX: set Box bounds to match actual value ranges ----
        if self.normalize_obs:
            low = -np.ones(dim, dtype=np.float32)   # centered features in [-1,1]
            high =  np.ones(dim, dtype=np.float32)
        else:
            low = np.zeros(dim, dtype=np.float32)
            high = np.full(dim, self.grid_size, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, shape=(dim,), dtype=np.float32)
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

        # optional curriculum
        if getattr(self, "sample_short_tasks", False):
            for _ in range(100):
                self.pos = np.random.uniform(0, self.grid_size, (self.n, 2))
                self.goals = np.random.uniform(0, self.grid_size, (self.n, 2))
                d0 = np.linalg.norm(self.pos - self.goals, axis=1)
                if np.all((d0 > 3.0) & (d0 < 6.0)):
                    break

        # ---- FIX: prev_dists must match the final (pos, goals) ----
        dists = np.linalg.norm(self.pos - self.goals, axis=1)
        self.prev_dists = dists.copy()
        return self._get_obs()

    def _get_obs(self):
        if self.normalize_obs:
            pos   = (self.pos   / self.grid_size) * 2.0 - 1.0
            goals = (self.goals / self.grid_size) * 2.0 - 1.0
        else:
            pos, goals = self.pos, self.goals

        if self.obs_mode == "abs":
            parts = [pos, goals]
        else:
            rel = (self.goals - self.pos) / self.grid_size  # in [-1,1]
            parts = [pos, goals, rel]

        return np.concatenate([p.flatten() for p in parts]).astype(np.float32)
    
    def step(self, action):
        act = action.reshape(self.n, 2)
        act = np.clip(act, -1, 1)

        # freeze agents that already reached
        if not hasattr(self, "reached_mask"):
            self.reached_mask = np.zeros(self.n, dtype=bool)
        act[self.reached_mask] = 0.0

        # apply dynamics and clamp to the box
        self.pos += self.dt * act
        self.pos = np.clip(self.pos, 0.0, self.grid_size)
        self.steps += 1

        # distances & progress (potential-based shaping)
        dists = np.linalg.norm(self.pos - self.goals, axis=1)
        if not hasattr(self, "prev_dists"):
            self.prev_dists = dists.copy()
        progress = self.prev_dists - dists          # > 0 when moving closer
        self.prev_dists = dists.copy()

        # --- Reward terms (tuneable constants) ---
        k_progress = 8.0           # main "get closer" drive
        k_time     = 0.02          # small per-step cost
        k_back     = 4.0           # penalty when moving away (progress < 0)
        k_goal     = 300.0         # one-time reach bonus
        k_close    = 2.0           # close-range magnet strength
        k_col      = 25.0          # collision penalty scale

        # Base: progress reward (and extra penalty if moving away)
        rewards = k_progress * progress - k_time
        moving_away = progress < 0.0
        rewards[moving_away] -= k_back * (-progress[moving_away])

        # Close-range "magnet" so it commits at the end (quadratic well)
        # Use a capture radius >= goal_threshold so attraction starts earlier.
        capture_radius = max(self.goal_threshold, 0.7)  # works with curriculum
        gap = np.maximum(0.0, capture_radius - dists)
        rewards += k_close * (gap ** 2)

        # Collisions: scale by overlap depth (smoother than fixed -10)
        collisions_step = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dij = np.linalg.norm(self.pos[i] - self.pos[j])
                overlap = max(0.0, 2 * self.radius - dij)
                if overlap > 0.0:
                    penalty = k_col * (overlap / (2 * self.radius))
                    rewards[i] -= penalty
                    rewards[j] -= penalty
                    collisions_step += 1

        # Reaching: big one-time bonus + "sticky" goal (snap & freeze)
        reached_now = (dists < self.goal_threshold) & (~self.reached_mask)
        if np.any(reached_now):
            rewards[reached_now] += k_goal
            # Snap to goal to avoid jitter/overshoot, then freeze
            self.pos[reached_now] = self.goals[reached_now]
            self.reached_mask[reached_now] = True

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