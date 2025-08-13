import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

import torch

# Local imports from your project
from env import MultiAgentEnv
from agents.ppo_agent import ActorCritic
try:
    from filter.mpc_filter import mpc_filter
    HAS_FILTER = True
except Exception:
    HAS_FILTER = False


def count_collisions(positions: np.ndarray, radius: float) -> int:
    n = positions.shape[0]
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < 2 * radius:
                c += 1
    return c


def try_load_policies(nets, load_dir: str):
    """Attempt to load policy weights for each agent from a directory.
    Expected filenames include 'agent{i}' (any prefix/suffix)."""
    if not load_dir or not os.path.isdir(load_dir):
        print("[viewer] No valid --load-dir provided; running with current weights.")
        return
    for i, net in enumerate(nets):
        # find a file matching *agent{i}*.pt
        pattern = os.path.join(load_dir, f"*agent{i}*.pt")
        matches = sorted(glob.glob(pattern))
        if matches:
            path = matches[-1]
            try:
                state = torch.load(path, map_location="cpu")
                net.load_state_dict(state)
                print(f"[viewer] Loaded agent {i} weights from: {path}")
            except Exception as e:
                print(f"[viewer] Failed loading {path}: {e}")
        else:
            print(f"[viewer] No checkpoint found for agent {i} in {load_dir} (pattern: {pattern})")


def build_view(ax, grid_size):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])
    # draw boundary
    ax.add_patch(Rectangle((0,0), grid_size, grid_size, fill=False, lw=1.5))


def main():
    ap = argparse.ArgumentParser(description="2D multi-agent viewer")
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--grid-size", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mode", choices=["random", "policy"], default="policy",
                    help="random: random actions; policy: ActorCritic nets (loadable)")
    ap.add_argument("--filter", action="store_true", help="apply MPC safety filter before stepping")
    ap.add_argument("--goal-threshold", type=float, default=0.3, help="for reach display only")
    ap.add_argument("--fps", type=int, default=30, help="animation frames per second")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-dir", type=str, default="", help="directory with saved *.pt checkpoints (optional)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Environment
    env = MultiAgentEnv(
        n_agents=args.n_agents,
        grid_size=args.grid_size,
        dt=args.dt,
        radius=args.radius,
        max_steps=args.horizon,
    )
    obs = env.reset().astype(np.float32)

    # Agents (Actor-Critic) if using policy mode
    obs_dim = env.observation_space.shape[0]
    act_dim = 2
    nets = None
    if args.mode == "policy":
        nets = [ActorCritic(obs_dim, act_dim) for _ in range(args.n_agents)]
        try_load_policies(nets, args.load_dir)
        for net in nets:
            net.eval()

    # Figure + artists
    fig, ax = plt.subplots(figsize=(6, 6))
    build_view(ax, env.grid_size)

    # goals (static)
    goals_plot = ax.scatter(env.goals[:, 0], env.goals[:, 1], marker="x", s=80, linewidths=2, c="k", label="goals")

    # agents (circles)
    agent_colors = plt.cm.tab10(np.linspace(0, 1, args.n_agents))
    agent_patches = []
    for i in range(args.n_agents):
        circ = Circle((env.pos[i, 0], env.pos[i, 1]), env.radius, fc=agent_colors[i], ec="k", lw=1.0, alpha=0.9)
        ax.add_patch(circ)
        agent_patches.append(circ)

    # HUD text
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10,
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # state
    frame = {"t": 0, "collisions_total": 0}

    def step_once():
        nonlocal obs
        # propose actions
        if args.mode == "random":
            u_prop = np.random.uniform(-1, 1, size=(args.n_agents, 2)).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32)
                actions = []
                for i in range(args.n_agents):
                    a, _, _ = nets[i].act(obs_t)
                    actions.append(a.cpu().numpy())
                u_prop = np.stack(actions, axis=0).astype(np.float32)

        # optional safety filter
        u = u_prop
        if args.filter:
            if not HAS_FILTER:
                print("[viewer] filter requested but not available; skipping.")
            else:
                try:
                    u = mpc_filter(u_prop, env.pos.copy(), dt=env.dt, radius=env.radius)
                except Exception as e:
                    print(f"[viewer] mpc_filter error: {e}")
                    u = u_prop

        # env step
        action_vec = u.reshape(-1).astype(np.float32)
        obs, rewards, done, _ = env.step(action_vec)

        # per-step collisions (tick count)
        c_now = count_collisions(env.pos, env.radius)
        frame["collisions_total"] += c_now

        # compute display reach mask (env may also track it internally)
        reach_mask = (np.linalg.norm(env.pos - env.goals, axis=1) < args.goal_threshold)

        return done, c_now, reach_mask

    def update(_):
        # one env step
        done, c_now, reach_mask = step_once()
        frame["t"] += 1

        # update agent circles
        for i, circ in enumerate(agent_patches):
            circ.center = (env.pos[i, 0], env.pos[i, 1])
            # highlight agent if currently colliding
            colliding = False
            for j in range(args.n_agents):
                if i != j and np.linalg.norm(env.pos[i] - env.pos[j]) < 2 * env.radius:
                    colliding = True
                    break
            circ.set_edgecolor("r" if colliding else "k")
            # dim if reached
            circ.set_alpha(0.5 if reach_mask[i] else 0.9)

        # update HUD
        hud.set_text(
            f"step: {frame['t']}/{args.horizon}\n"
            f"collisions (tick): {frame['collisions_total']}\n"
            f"reached: {int(reach_mask.sum())}/{args.n_agents}\n"
            f"mode: {args.mode}{' + filter' if args.filter else ''}"
        )

        # finish episode early if env finished
        if done or frame["t"] >= args.horizon:
            anim.event_source.stop()

        return agent_patches + [hud]

    interval_ms = int(1000 / max(1, args.fps))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
