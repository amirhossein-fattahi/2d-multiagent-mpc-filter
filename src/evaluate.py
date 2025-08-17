# src/evaluate.py
import argparse, os, numpy as np, torch, pandas as pd
from multiagent_policy import load_policy  # the same loader used by your train/record scripts
from env import MultiAgentEnv

def run_one_episode(env, policy, deterministic=True):
    obs = env.reset()
    done = False
    steps = 0
    collisions = 0
    ret_sum = 0.0
    ever_all_reached = False
    while not done:
        with torch.no_grad():
            act = policy.act(obs, deterministic=deterministic)  # same API as record_videos
        obs, rew, done, info = env.step(act)
        steps += 1
        ret_sum += float(np.mean(rew))  # mean over agents; adjust if you prefer sum
        collisions += int(info.get("collisions_step", 0))
        ever_all_reached = ever_all_reached or bool(info.get("all_reached", False))
    return {
        "success": int(ever_all_reached),
        "steps": steps,
        "collisions": collisions,
        "return_mean": ret_sum,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--normalize-obs", action="store_true")
    p.add_argument("--grid-size", type=float, default=10.0)
    p.add_argument("--goal-threshold", type=float, default=0.3)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--out", default=None)  # CSV path
    args = p.parse_args()

    # env matches training settings
    env = MultiAgentEnv(
        n_agents=2, grid_size=args.grid_size, max_steps=args.horizon,
        goal_threshold=args.goal_threshold, normalize_obs=args.normalize_obs
    )
    policy = load_policy(args.ckpt_dir)  # same function used by record_videos.py

    rows = []
    for ep in range(args.n_episodes):
        rows.append(run_one_episode(env, policy, deterministic=args.deterministic))
    df = pd.DataFrame(rows)
    print(df.describe(include="all"))
    print("\nSuccess rate:", df["success"].mean())
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print("Saved:", args.out)

if __name__ == "__main__":
    main()
