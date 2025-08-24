# record_videos.py — MP4 + README-friendly GIFs
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

# optional, for high-quality resizing (recommended)
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

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
    ax.scatter(env.goals[:,0], env.goals[:,1], marker='*', s=180, label='Goals')
    ax.scatter(env.pos[:,0], env.pos[:,1], s=80, label='Agents')
    for p in env.pos:
        c = plt.Circle((p[0], p[1]), env.radius, fill=False, linewidth=1.0)
        ax.add_patch(c)
    ax.set_title(f"Step {env.steps}")
    ax.legend(loc="upper right", fontsize=8)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    try:
        buf = fig.canvas.buffer_rgba()
        img = (np.frombuffer(buf, dtype=np.uint8)
                 .reshape(h, w, 4)[..., :3]  # RGBA -> RGB
                 .copy())
    except AttributeError:
        buf = fig.canvas.tostring_argb()
        argb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        img = argb[:, :, 1:4].copy()
    return img

def to_gif(frames, out_path, gif_fps=12, max_width=512):
    """Downsample + resize then write a small looping GIF for README."""
    if len(frames) == 0:
        return
    # Downsample frames to target fps
    # Original fps ~= 1/dt (e.g., dt=0.1 -> 10 fps). We’ll simple-step down.
    # If you want exact timing, pass durations=... instead.
    step = max(1, int(len(frames) * (gif_fps / max(1, len(frames)))))
    sampled = frames[::step] if step > 1 else frames

    # Resize to max_width while keeping aspect
    if max_width and PIL_OK:
        resized = []
        for fr in sampled:
            h, w = fr.shape[:2]
            if w > max_width:
                new_h = int(round(h * (max_width / w)))
                im = Image.fromarray(fr)
                im = im.resize((max_width, new_h), Image.BILINEAR)
                fr = np.array(im)
            resized.append(fr)
        sampled = resized

    imageio.mimsave(out_path, sampled, fps=gif_fps, loop=0)  # loop=0: infinite
    print(f"Saved {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, required=True, help="checkpoints/<tag>")
    ap.add_argument("--out-dir", type=str, default="videos")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--filtered", action="store_true", help="use MPC safety filter at eval")
    ap.add_argument("--deterministic", action="store_true", help="use mean action")
    ap.add_argument("--seed", type=int, default=0)
    # README-friendly media options
    ap.add_argument("--gif", action="store_true", help="also save a small GIF per episode")
    ap.add_argument("--gif-fps", type=int, default=12, help="GIF frame rate")
    ap.add_argument("--gif-max-width", type=int, default=512, help="max GIF width (px)")
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
        obs = env.reset()
        frames = []
        fig, ax = plt.subplots(figsize=(5,5), dpi=140)

        done = False
        while not done and env.steps < args.horizon:
            a = policy_action(nets, obs, device, deterministic=args.deterministic)
            if args.filtered:
                u_prop = a.reshape(n_agents,2)
                u_safe = mpc_filter(u_prop, env.pos.copy(), dt=env.dt, radius=env.radius)
                a = u_safe.reshape(-1)
            obs, rew, done, info = env.step(a)
            frames.append(render_frame(env, fig, ax))

        plt.close(fig)

        # Save MP4 (good for tweets, local viewing)
        mp4_path = os.path.join(args.out_dir, f"ep_{ep:03d}.mp4")
        with imageio.get_writer(mp4_path, fps=int(round(1.0/env.dt))) as w:
            for fr in frames:
                w.append_data(fr)
        print(f"Saved {mp4_path}")

        # Save small looping GIF (good for GitHub README)
        if args.gif:
            gif_path = os.path.join(args.out_dir, f"ep_{ep:03d}.gif")
            to_gif(frames, gif_path, gif_fps=args.gif_fps, max_width=args.gif_max_width)

if __name__ == "__main__":
    main()
