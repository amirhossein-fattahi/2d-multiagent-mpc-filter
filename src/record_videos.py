import argparse, os, json
import numpy as np
import torch
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import MultiAgentEnv
from agents.ppo_agent import ActorCritic
try:
    from filter.mpc_filter import mpc_filter
except Exception:
    mpc_filter = None

def load_policy(ckpt_dir, obs_dim, act_dim, n_agents, device):
    nets = []
    for i in range(n_agents):
        net = ActorCritic(obs_dim, act_dim).to(device)
        state = torch.load(os.path.join(ckpt_dir, f"agent_{i}.pt"), map_location=device)
        net.load_state_dict(state)
        net.eval()
        nets.append(net)
    return nets

def policy_action(nets, obs, device, deterministic=True):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    acts = []
    for net in nets:
        with torch.no_grad():
            dist, _ = net(obs_t)
            a = dist.mean if deterministic else dist.sample()
        acts.append(a.cpu().numpy())
    return np.concatenate(acts, axis=-1)

def render_frame(env, fig, ax):
    ax.clear()
    L = env.grid_size
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect('equal')
    # goals
    ax.scatter(env.goals[:,0], env.goals[:,1], marker='*', s=180, label='Goals')
    # agents
    ax.scatter(env.pos[:,0], env.pos[:,1], s=80, label='Agents')
    # safety radii
    for p in env.pos:
        c = plt.Circle((p[0], p[1]), env.radius, fill=False, linewidth=1.0)
        ax.add_patch(c)
    ax.set_title(f"Step {env.steps}")
    ax.legend(loc="upper right", fontsize=8)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    try:
        # Preferred in modern Matplotlib
        buf = fig.canvas.buffer_rgba()
        img = (np.frombuffer(buf, dtype=np.uint8)
                 .reshape(h, w, 4)[..., :3]  # RGBA -> RGB (drop alpha)
                 .copy())
    except AttributeError:
        # Fallback for older versions
        buf = fig.canvas.tostring_argb()
        argb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        # ARGB -> RGB by dropping alpha and reordering channels
        img = argb[:, :, 1:4].copy()

    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, required=True, help="checkpoints/<tag>")
    ap.add_argument("--out-dir", type=str, default="videos")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--filtered", action="store_true", help="use MPC safety filter at eval")
    ap.add_argument("--deterministic", action="store_true", help="use mean action")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read meta or default to 2 agents
    meta_path = os.path.join(args.ckpt_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, "r"))
        n_agents = int(meta.get("n_agents", 2))
    else:
        n_agents = 2

    env = MultiAgentEnv(n_agents=n_agents, max_steps=args.horizon)
    obs_dim = env.observation_space.shape[0]
    act_dim = 2

    nets = load_policy(args.ckpt_dir, obs_dim, act_dim, n_agents, device)

    if args.filtered and mpc_filter is None:
        raise RuntimeError("filtered=True but mpc_filter is not available/importable.")

    rng = np.random.default_rng(args.seed)

    for ep in range(1, args.n_episodes + 1):
        # new random episode every time
        obs = env.reset()
        frames = []
        fig, ax = plt.subplots(figsize=(5,5), dpi=150)

        done = False
        while not done and env.steps < args.horizon:
            a = policy_action(nets, obs, device, deterministic=args.deterministic)
            if args.filtered:
                u_prop = a.reshape(n_agents,2)
                u_safe = mpc_filter(u_prop, env.pos.copy(), dt=env.dt, radius=env.radius)
                a = u_safe.reshape(-1)
            obs, rew, done, info = env.step(a)
            frame = render_frame(env, fig, ax)
            frames.append(frame)

        plt.close(fig)
        out_path = os.path.join(args.out_dir, f"ep_{ep:03d}.mp4")
        with imageio.get_writer(out_path, fps=int(1.0/env.dt)) as w:
            for fr in frames:
                w.append_data(fr)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
