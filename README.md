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

2d-multiagent-mpc-filter/
├── src/
│ ├── env.py # MultiAgentEnv
│ ├── agents/
│ │ ├── init.py
│ │ └── ppo_agent.py # Actor-Critic (continuous 2D)
│ ├── filter/
│ │ ├── init.py
│ │ └── mpc_filter.py # QP safety projection (with slacks)
│ ├── train_baseline.py # PPO without filter
│ └── train_with_filter.py # PPO + MPC filter
├── tests/
│ └── test_env.py # basic sanity tests
├── requirements.txt
├── setup.py # optional; pip install -e .
└── .github/workflows/ci.yml # optional; GitHub Actions CI


Run
Baseline (no filter):
python src/train_baseline.py --n-agents 2 --episodes 50 --horizon 300

With MPC filter:
python src/train_with_filter.py --n-agents 2 --episodes 50 --horizon 300

Useful flags you can add (if present in your scripts):
--grid-size 10    # world size
--dt 0.2          # time step (speed)
--radius 0.5      # agent radius (safety)
--epochs 8        # PPO epochs per update
--lr 1e-4         # learning rate

Environment details
State: concat of all agent positions and all goals (global observation).

Action: per-agent 2D velocity; env clips to bounds.

Dynamics: pos ← clamp(pos + dt * action, 0, grid_size).

Rewards (shaped):

progress toward goal (k * (prev_dist - dist)) minus a small step cost,

one-time +100 when an agent reaches (dist < threshold, e.g., 0.3),

collision penalty if two agents are closer than 2*radius.

reached agents are optionally frozen.

Episode ends when all agents reach or max_steps is hit.

MPC safety filter
At each step, we solve a quadratic program:

Objective: keep actions close to the policy proposals.

Constraints: linearized pairwise safety — don’t decrease inter-agent distance below a margin in the next step.

Slack variables: ensure feasibility; penalized strongly in the objective.

Solvers: OSQP (preferred), CLARABEL/SCS/ECOS as fallbacks.

Example results (typical)
Baseline: frequent collision spikes; episodes often time out (no early finish).

Filtered: near-zero collisions; clear single-agent reaches; some episodes end early (both reached).
With slightly larger dt and looser success radius, success rate increases.

Tips to increase success
Increase speed (--dt 0.2) and relax reach threshold (e.g., 0.3–0.4).

Add a small team bonus when all agents reach.

Boost progress weight for the not-yet-reached agents once one has arrived.

Consider shared policy (one Actor-Critic used by all agents) for better sample efficiency.

Normalize observations: divide by grid_size before feeding the net.

Troubleshooting
cvxpy/OSQP missing: pip install cvxpy osqp, then python -c "import cvxpy as cp; print(cp.installed_solvers())".

Gym + NumPy 2 warning: either ignore, pin gym==0.26.2 and numpy<2, or migrate to Gymnasium.

Episodes always stop at 200: set max_steps=args.horizon when constructing the env.

Development
Run tests:
pytest -q

Editable install:
pip install -e .


License
This project is licensed under the MIT License. See LICENSE for details.

Citation
If you use this project in academic work or demos, please consider citing the repository.

@misc{2d-multiagent-mpc-filter,
  title  = {2D Multi-Agent Navigation with an MPC Safety Filter},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/<your-username>/2d-multiagent-mpc-filter}
}