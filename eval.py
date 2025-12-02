import torch
import copy
import numpy as np
from gym_env import PointsEnv
from network import AttentionNet
import time
from typing import List


# Parameter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model/greedy_best.pt"     # Model path, or: model/best.pt
GRAPH_TASKS = 100                       # Number of collection tasks in each episode
MAX_DIST = 3.0
ENVS_NUMBER = 10
K = 16

time_start = time.time()
policy = AttentionNet().to(DEVICE)
VISIUALIZE_TEST = 1                     # 0:No visualization  1：greedy  2：best

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

# Retrieve parameter dictionary
if isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

new_state = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state[k[len("module."):]] = v
    else:
        new_state[k] = v

# Load
missing, unexpected = policy.load_state_dict(new_state, strict=False)
if missing or unexpected:
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)
policy.eval()
print(f"Model loaded: {MODEL_PATH}")

@torch.no_grad()
def run_greedy(env):   # Greedy
    env.reset(env.world.rewards, env.world.start_depot_index)
    R = 0.0
    done = False
    while not done:
        points, agent_idx, remaining_distance, valid_mask = env.observe()
        points = torch.tensor(points, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        agent_idx = torch.tensor([agent_idx], dtype=torch.int64, device=DEVICE)
        remaining_distance = torch.tensor([[remaining_distance]], dtype=torch.float32, device=DEVICE)
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
        dist = policy(points, agent_idx, remaining_distance, valid_mask)
        action = torch.argmax(dist.probs, dim=-1).item()
        _, r, done = env.step(action)
        R += r
    return R, env.remaining_distance

@torch.no_grad()
def run_sample(env):
    best_reward = -1e9
    best_remaining = -1e9

    for _ in range(K):
        env.reset(env.world.rewards, env.world.start_depot_index)
        R = 0.0
        done = False
        while not done:
            pts, ag, rem, m = env.observe()

            points = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            agent_idx = torch.tensor([ag], dtype=torch.long, device=DEVICE)
            remaining_distance = torch.tensor([[rem]], dtype=torch.float32, device=DEVICE)
            valid_mask = torch.tensor(m, dtype=torch.bool, device=DEVICE).unsqueeze(0)

            dist = policy(points, agent_idx, remaining_distance, valid_mask)
            action = dist.sample().item()

            _, r, done = env.step(int(action))
            R += float(r)

        if R > best_reward:
            best_reward = R
            best_remaining = env.remaining_distance

    return best_reward, best_remaining


def main():
    single_env = PointsEnv(tasks_number=GRAPH_TASKS, max_distance=MAX_DIST)
    multy_envs = [PointsEnv(tasks_number=GRAPH_TASKS, max_distance=MAX_DIST) for _ in range(ENVS_NUMBER)]

    print("\n===== Single Environment Test =====")
    Rg, rem_g = run_greedy(single_env)
    print(f"Greedy Reward = {Rg:.2f}, Remaining Distance = {rem_g:.2f}")
    if VISIUALIZE_TEST == 1:
        single_env.render()
    Rb, rem_b = run_sample(single_env)
    print(f"Sample(Best-of-{K}) Reward = {Rb:.2f}, Remaining Distance = {rem_b:.2f}")
    if VISIUALIZE_TEST == 2:
        single_env.render()

    print("\n===== Multiple Environments Test =====")
    rewards_g = []
    rewards_b = []

    total_envs = len(multy_envs)
    for idx, env in enumerate(multy_envs, start=1):
        Rg, rem_g = run_greedy(env)
        rewards_g.append(Rg)
        Rb, rem_b = run_sample(env)
        rewards_b.append(Rb)
        print(f"[{idx}/{total_envs}] " f"Greedy Reward = {Rg:.2f}, Remaining Distance = {rem_g:.2f}\n\tSample(Best-of-{K}) Reward = {Rb:.2f} Remaining Distance = {rem_b:.2f}")

    # Final summary
    print("\n===== Evaluation Summary =====")
    print(f"Greedy Avg Reward = {np.mean(rewards_g):.2f} ± {np.std(rewards_g):.2f}")
    print(f"Sample(Best-of-{K}) Avg Reward = {np.mean(rewards_b):.2f} ± {np.std(rewards_b):.2f}")


if __name__ == "__main__":
    main()