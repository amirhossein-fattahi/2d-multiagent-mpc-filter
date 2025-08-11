# 2D Multi-Agent Navigation with MPC Safety Filter

Small, reproducible project for **multi-agent navigation** in a 2D square world.  
Each agent learns a policy (PPO) to reach its own goal. A tiny **MPC-style safety filter** projects proposed actions to avoid collisions. We compare baseline vs. filtered on collisions and success.

> Status: working baseline + MPC filter; reward shaping and feasibility slacks added; early successes observed with filter.

---

## Features

- **Gym-style env** with N point-mass agents in a shared square.
- **Independent PPO** policies (optionally switch to shared weights).
- **MPC safety filter**: 1-step convex QP with optional slacks for feasibility.
- **Reward shaping**: progress toward goal + one-time goal bonus.
- **Metrics**: collisions/episode, success (all goals reached), time-to-goal, return.
- **Windows-friendly** setup (cvxpy + OSQP).

---

## Project structure

