import argparse
import numpy as np
import torch
import torch.nn.functional as F
import csv, os, time, numpy as np
import json, torch
from env import MultiAgentEnv
from agents.ppo_agent import ActorCritic

# ---------- Helpers ----------

tag = "baseline"  # in filtered script set to "filtered"
logdir = os.path.join("logs", tag)
os.makedirs(logdir, exist_ok=True)
csv_path = os.path.join(logdir, f"run_{int(time.time())}.csv")
writer = csv.writer(open(csv_path, "w", newline=""))
writer.writerow(["episode", "ep_return_mean", "ep_return_std", "collisions", "steps"])


def set_seed(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def count_collisions(positions: np.ndarray, radius: float) -> int:
    n = positions.shape[0]
    c = 0
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < 2*radius:
                c += 1
    return c

# ---------- PPO Core ----------

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = torch.zeros(T)
    last_adv = 0.0
    for t in reversed(range(T)):
        next_value = values[t+1] if t < T-1 else torch.tensor(0.0)
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_adv = delta + gamma * lam * next_nonterminal * last_adv
        adv[t] = last_adv
    returns = adv + values[:-1] if len(values) == T+1 else adv + values
    return adv, returns



def set_curriculum(env, ep):
    # Phase 0: 0–999 (easy)
    if ep < 1000:
        env.goal_threshold = 0.7
        env.radius = 0.45
        env.max_steps = 3000
        # (optional) keep starts/goals a bit closer:
        # env.sample_short_tasks = True  # if you add such logic to reset()
    # Phase 1: 1000–2999 (medium)
    elif ep < 3000:
        env.goal_threshold = 0.5
        env.radius = 0.45
        env.max_steps = 3000
    # Phase 2: 3000+ (target)
    else:
        env.goal_threshold = 0.3
        env.radius = 0.5
        env.max_steps = 3000


# ---------- Training ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-agents', type=int, default=2)
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--horizon', type=int, default=3000, help='max steps per episode')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=4, help='PPO epochs per episode')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    
    run_tag = f"{'filtered' if 'with_filter' in __file__ else 'baseline'}_{int(time.time())}"
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", run_tag + ".csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "return_mean", "collisions", "steps", "success"])  # header

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = MultiAgentEnv(n_agents=args.n_agents, dt=0.25, max_steps=args.horizon)    
    obs_dim = env.observation_space.shape[0]
    act_dim = 2  # per-agent action is 2D

    # One independent policy per agent (each sees the full global state)
    nets = [ActorCritic(obs_dim, act_dim).to(device) for _ in range(args.n_agents)]
    opts = [torch.optim.Adam(net.parameters(), lr=args.lr) for net in nets]

    for ep in range(1, args.episodes + 1):
        # in your training loop, at the start of each episode:
        if ep < 100:
            env.goal_threshold = 0.7
        elif ep < 200:
            env.goal_threshold = 0.5
        else:
            env.goal_threshold = 0.3

        set_curriculum(env, ep)
        obs = env.reset().astype(np.float32)
        done = False
        ep_return = np.zeros(args.n_agents, dtype=np.float32)
        ep_collisions = 0
        step = 0

        # Rollout storage per agent
        obs_buf = [[] for _ in range(args.n_agents)]
        act_buf = [[] for _ in range(args.n_agents)]
        logp_buf = [[] for _ in range(args.n_agents)]
        val_buf = [[] for _ in range(args.n_agents)]
        rew_buf = [[] for _ in range(args.n_agents)]
        done_buf = [[] for _ in range(args.n_agents)]

        while not done and step < args.horizon:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            actions = []
            logps = []
            values = []

            # Each agent proposes its own 2D action from the same global observation
            for i in range(args.n_agents):
                a, logp, v = nets[i].act(obs_t, deterministic=False)  # each sees same global obs
                actions.append(a.squeeze(0).cpu().numpy())   # [2]
                logps.append(logp.cpu())
                values.append(v.cpu())

            action_vec = np.concatenate(actions, axis=0)  # shape (2*n_agents,)
            next_obs, rewards, done, info = env.step(action_vec)



            ep_return += np.array(rewards, dtype=np.float32)
            ep_collisions += info.get("collisions_step", 0)


            # Save to buffers
            for i in range(args.n_agents):
                obs_buf[i].append(obs.copy())
                act_buf[i].append(actions[i].copy())
                logp_buf[i].append(logps[i].item())
                val_buf[i].append(values[i].item())
                rew_buf[i].append(float(rewards[i]))
                done_buf[i].append(bool(done))

            obs = next_obs.astype(np.float32)
            step += 1

        # PPO Update per agent
        for i in range(args.n_agents):
            obs_t = torch.tensor(np.array(obs_buf[i]), dtype=torch.float32, device=device)
            act_t = torch.tensor(np.array(act_buf[i]), dtype=torch.float32, device=device)
            old_logp_t = torch.tensor(np.array(logp_buf[i]), dtype=torch.float32, device=device)
            val_t = torch.tensor(np.array(val_buf[i]), dtype=torch.float32, device=device)
            rew_t = torch.tensor(np.array(rew_buf[i]), dtype=torch.float32, device=device)
            done_t = torch.tensor(np.array(done_buf[i]), dtype=torch.bool, device=device)

            # bootstrap value for last state (0 if episode ended)
            with torch.no_grad():
                last_v = torch.tensor(0.0, device=device)
            values_with_bootstrap = torch.cat([val_t, last_v.unsqueeze(0)], dim=0)

            adv, ret = compute_gae(rew_t.cpu(), values_with_bootstrap.cpu(), done_t.cpu(), args.gamma, args.lam)
            adv = adv.to(device)
            ret = ret.to(device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for _ in range(args.epochs):
                dist, value = nets[i](obs_t)
                new_logp = dist.log_prob(act_t).sum(-1)
                ratio = (new_logp - old_logp_t).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret)
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                opts[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nets[i].parameters(), 0.5)
                opts[i].step()

        writer.writerow([ep, float(np.mean(ep_return)), float(np.std(ep_return)), int(ep_collisions), int(step)])

        #print(f"[Baseline] Episode {ep:04d} | Return per agent {ep_return} | Collisions {ep_collisions} | Steps {step}")
        success = 1 if (step < args.horizon or (info and info.get("all_reached", False))) else 0
        ret_mean = float(np.mean(ep_return))
        csv_writer.writerow([ep, ret_mean, int(ep_collisions), int(step), success])
        csv_file.flush()
        print(f"[{'Filtered' if 'with_filter' in __file__ else 'Baseline'}] "
            f"Episode {ep:04d} | Return mean {ret_mean:.2f} | Collisions {ep_collisions} | "
            f"Steps {step} | Success {success}")

    
    tag = f"{'filtered' if 'with_filter' in __file__ else 'baseline'}_{int(time.time())}"
    ckpt_dir = os.path.join("checkpoints", tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    for i, net in enumerate(nets):
        torch.save(nets[i].state_dict(), os.path.join(ckpt_dir, f"agent_{i}.pt"))
    meta = dict(n_agents=args.n_agents, horizon=args.horizon, seed=args.seed)
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved checkpoints to {ckpt_dir}")

    print("Training finished.")
    csv_file.close()
    print(f"Saved metrics to: {csv_path}")


if __name__ == '__main__':
    main()