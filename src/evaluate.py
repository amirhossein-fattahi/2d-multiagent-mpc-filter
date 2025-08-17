import argparse, os, json, csv
import numpy as np
import torch

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, required=True)
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--filtered", action="store_true")
    ap.add_argument("--normalize-obs", action="store_true")
    ap.add_argument("--out", type=str, default="results.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read metadata (e.g. number of agents)
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

    results = []
    for ep in range(1, args.n_episodes + 1):
        obs = env.reset()
        total_rew = np.zeros(n_agents)
        collisions = 0
        done = False

        while not done and env.steps < args.horizon:
            if args.normalize_obs:
                obs = (obs - obs.mean()) / (obs.std() + 1e-8)

            a = policy_action(nets, obs, device, deterministic=args.deterministic)
            if args.filtered:
                u_prop = a.reshape(n_agents, 2)
                u_safe = mpc_filter(u_prop, env.pos.copy(), dt=env.dt, radius=env.radius)
                a = u_safe.reshape(-1)

            obs, rew, done, info = env.step(a)
            total_rew += rew
            collisions += info.get("collisions_step", 0)

        success = int(info.get("all_reached", False))
        mean_return = total_rew.mean()

        print(f"[Eval] Episode {ep:04d} | Return mean {mean_return:.2f} | Collisions {collisions} | Steps {env.steps} | Success {success}")

        results.append([ep, mean_return, collisions, env.steps, success])

    # Write CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return_mean", "collisions", "steps", "success"])
        writer.writerows(results)

    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
