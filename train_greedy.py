import os
import time
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from gym_env import PointsEnv
from network import AttentionNet
from valid import build_multi_fixed_envs
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# Parameter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAPH_TASKS = 100           # Number of tasks
MAX_DIST = 3.0              # Max distance
BATCH_SIZE = 64            # Number of graphs for each epoch
N_EPOCHS = 2000             # Number of epoch
UPDATES_PER_EPOCH = 10      # Number of backpropagations
LR = 1e-4                   # Learning rate
MAX_GRAD_NORM = 2
VAL_EPISODES = 100          # Sample size in validation
SAVE_DIR = "checkpoint_greedy"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load
MODEL_DIR = "checkpoint_greedy"         # Folder containing trained models
RESUME = False                                     # True:Continue training; False:From the beginning
RESUME_PATH = "checkpoint_greedy/last.pt"

# Import Model
def find_latest_ckpt(model_dir: str) -> str:
    cand = glob.glob(os.path.join(model_dir, "*.pt")) + glob.glob(os.path.join(model_dir, "*.pth"))
    if not cand:
        return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]

def try_resume_from_ckpt(policy: nn.Module,
                         optimizer: optim.Optimizer,
                         path: str,
                         device: str = "cpu"):
    # Restore from checkpoint
    print(f"Loading checkpoint (CPU-safe): {path}")
    ckpt = torch.load(path, map_location="cuda")
    if isinstance(ckpt, dict) and ("model" not in ckpt) and \
       ("state_dict" in ckpt or any(isinstance(v, torch.Tensor) for v in ckpt.values())):
        state = ckpt.get("state_dict", ckpt)
        policy.load_state_dict(state)
        policy.to(device)
        print("Loaded model weights (state_dict only).")
        return 0, -1e9
    # Save
    if "model" in ckpt:
        policy.load_state_dict(ckpt["model"])
        policy.to(device)
        print("Loaded model weights.")
    # Optimizer
    if "opt" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["opt"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            for pg in optimizer.param_groups:
                pg["lr"] = LR
            print(f"Restored optimizer (moved to {device}, lr={LR}).")
        except Exception as e:
            print(f"Optimizer state not loaded: {e}")

    start_epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", -1e9))
    print(f"Resume from epoch={start_epoch}, best_val={best_val:.4f}")
    return start_epoch, best_val

# Draw
def init_result_dir():
    tag = datetime.now().strftime("%m_%d_%H%M")
    base_dir = os.path.join("result_cur", f"train_{tag}")
    os.makedirs(base_dir, exist_ok=True)
    print(f"results will be saved under: {base_dir}  (tag={tag})")
    return base_dir

def save_excels(base_dir, train_rows, eval_rows):
    df_train = pd.DataFrame(train_rows)  # Column: epoch, update, reward, loss
    df_eval  = pd.DataFrame(eval_rows)   # Column: epoch, val_return, eval_time, train_time
    # Two Excels：Evaluation results and time consumption; Training reward/loss
    eval_xlsx  = os.path.join(base_dir, "eval.xlsx")
    train_xlsx = os.path.join(base_dir, "train.xlsx")
    df_eval.to_excel(eval_xlsx, index=False)
    df_train.to_excel(train_xlsx, index=False)
    return eval_xlsx, train_xlsx

def plot_curves(base_dir, train_rows, eval_rows):
    if len(eval_rows) > 0:
        df_eval = pd.DataFrame(eval_rows)
        # val_return
        plt.figure()
        plt.plot(df_eval["epoch"], df_eval["val_return"])
        plt.xlabel("epoch"); plt.ylabel("val_return (greedy on fixed-100)")
        plt.title("Validation Return over Epochs")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_dir, "val_return_curve.png"), dpi=150)
        plt.close()

        # eval_time
        plt.figure()
        plt.plot(df_eval["epoch"], df_eval["eval_time"])
        plt.xlabel("epoch"); plt.ylabel("eval_time (s)")
        plt.title("Evaluation Time over Epochs")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_dir, "eval_time_curve.png"), dpi=150)
        plt.close()

    if len(train_rows) > 0:
        df_train = pd.DataFrame(train_rows)
        df_ep = df_train.groupby("epoch", as_index=False).agg(
            mean_reward=("reward", "mean"),
            mean_loss=("loss", "mean")
        )

        # mean_reward per epoch
        plt.figure()
        plt.plot(df_ep["epoch"], df_ep["mean_reward"])
        plt.xlabel("epoch"); plt.ylabel("mean_reward (per-epoch avg)")
        plt.title("Training Mean Reward per Epoch")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_dir, "train_mean_reward_curve.png"), dpi=150)
        plt.close()

        # mean_loss per epoch
        plt.figure()
        plt.plot(df_ep["epoch"], df_ep["mean_loss"])
        plt.xlabel("epoch"); plt.ylabel("mean_loss (per-epoch avg)")
        plt.title("Training Mean Loss per Epoch")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_dir, "train_mean_loss_curve.png"), dpi=150)
        plt.close()

# Sample a batch and calculate the REINFORCE loss
def reinforce_update(policy: AttentionNet, optimizer) -> Tuple[float, float]:
    policy.train()
    envs = [PointsEnv(tasks_number=GRAPH_TASKS, max_distance=MAX_DIST) for _ in range(BATCH_SIZE)]

    # Cumulative Track Rewards
    sum_logp = torch.zeros(BATCH_SIZE, device=DEVICE)
    ep_return = torch.zeros(BATCH_SIZE, device=DEVICE)
    done = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)

    # Initialize observation cache
    obs = []
    for e in envs:
        e.reset(e.world.rewards, e.world.start_depot_index)
        obs.append(e.observe())

    # Rollout
    step_cap = GRAPH_TASKS + 1
    for _ in range(step_cap):
        idxs = [i for i, d in enumerate(done.tolist()) if not d]
        if not idxs:
            break

        points_t = torch.stack([torch.as_tensor(obs[i][0], dtype=torch.float32, device=DEVICE) for i in idxs],dim=0)
        agent_t = torch.as_tensor([obs[i][1] for i in idxs], dtype=torch.long, device=DEVICE)
        rem_t = torch.as_tensor([[obs[i][2]] for i in idxs], dtype=torch.float32, device=DEVICE)
        mask_t = torch.stack([torch.as_tensor(obs[i][3], dtype=torch.bool, device=DEVICE) for i in idxs],dim=0)
        dist = policy(points_t, agent_t, rem_t, mask_t)   # One forward
        actions = dist.sample()                           # [B_active]
        logps   = dist.log_prob(actions)                  # [B_active]

        for j, i in enumerate(idxs):
            a = int(actions[j].item())
            _, r, d = envs[i].step(a)
            ep_return[i] += float(r)
            sum_logp[i]  += logps[j]
            done[i] = d
            if not d:
                obs[i] = envs[i].observe()

        if done.all():
            break

    # returns / seq_logps
    returns_t   = ep_return              # [B]
    seq_logps_t = sum_logp               # [B]

    # Baseline
    with torch.inference_mode():
        obs_b = []
        for e in envs:
            e.reset(e.world.rewards, e.world.start_depot_index)
            obs_b.append(e.observe())

        done_b = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)
        b = torch.zeros(BATCH_SIZE, device=DEVICE)

        range_step = range(GRAPH_TASKS + 1)
        for _ in range_step:
            idxs_t = (~done_b).nonzero(as_tuple=False).squeeze(-1)
            if idxs_t.numel() == 0:
                break
            idxs = idxs_t.tolist()

            points_t = torch.stack([torch.as_tensor(obs_b[i][0], dtype=torch.float32, device=DEVICE) for i in idxs],dim=0)
            agent_t = torch.as_tensor([obs_b[i][1] for i in idxs], dtype=torch.long, device=DEVICE)
            rem_t = torch.as_tensor([[obs_b[i][2]] for i in idxs], dtype=torch.float32, device=DEVICE)
            mask_t = torch.stack([torch.as_tensor(obs_b[i][3], dtype=torch.bool, device=DEVICE) for i in idxs],dim=0)

            dist = policy(points_t, agent_t, rem_t, mask_t)

            greedy_actions = torch.argmax(dist.logits, dim=-1)

            for j, i in enumerate(idxs):
                a = int(greedy_actions[j].item())
                _, r, d = envs[i].step(a)
                b[i] += float(r)
                done_b[i] = d
                if not d:
                    obs_b[i] = envs[i].observe()

    # Advance and loss
    adv = returns_t - b
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    loss_vec = -(adv.detach() * seq_logps_t)    # [B]
    valid = torch.isfinite(loss_vec)
    reinforce_loss = loss_vec[valid].mean()
    loss = reinforce_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return float(returns_t.mean().item()), float(loss.item())

# Greedy evaluation
@torch.no_grad()
def evaluate_greedy_fixed(policy: AttentionNet, eval_envs) -> float:
    policy.eval()
    M = len(eval_envs)
    device = DEVICE

    # Reset env
    obs = []
    for env in eval_envs:
        env.reset(env.world.rewards, env.world.start_depot_index)
        obs.append(env.observe())

    done = torch.zeros(M, dtype=torch.bool, device=device)
    ep_ret = torch.zeros(M, dtype=torch.float32, device=device)

    step_cap = GRAPH_TASKS + 1
    for _ in range(step_cap):
        idxs_t = (~done).nonzero(as_tuple=False).squeeze(-1)  # [B_active]
        if idxs_t.numel() == 0:
            break
        idxs = idxs_t.tolist()
        points_t = torch.stack([torch.as_tensor(obs[i][0], dtype=torch.float32, device=device) for i in idxs], dim=0)
        agent_t = torch.as_tensor([obs[i][1] for i in idxs], dtype=torch.long, device=device)
        rem_t = torch.as_tensor([[obs[i][2]] for i in idxs], dtype=torch.float32, device=device)
        mask_t = torch.stack([torch.as_tensor(obs[i][3], dtype=torch.bool, device=device) for i in idxs], dim=0)

        # Action
        dist = policy(points_t, agent_t, rem_t, mask_t)
        greedy_actions = torch.argmax(dist.logits, dim=-1)

        # Update environment
        for j, i in enumerate(idxs):
            a = int(greedy_actions[j].item())
            _, r, d = eval_envs[i].step(a)
            ep_ret[i] += float(r)
            done[i] = d
            if not d:
                obs[i] = eval_envs[i].observe()

        if done.all():
            break

    return float(ep_ret.mean().item())


def main():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    BASE_DIR = init_result_dir()                     # Results Directory
    ckpt_best = os.path.join(SAVE_DIR, "best.pt")
    ckpt_last = os.path.join(SAVE_DIR, "last.pt")
    policy = AttentionNet().to(DEVICE)               # Initialize component
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    eval_envs = build_multi_fixed_envs(tasks=GRAPH_TASKS, max_dist=MAX_DIST, n_envs=VAL_EPISODES, base_seed=42)

    best_val = -1e9
    if RESUME:
        load_path = RESUME_PATH if RESUME_PATH is not None else find_latest_ckpt(MODEL_DIR)
        if load_path is not None and os.path.exists(load_path):
            start_epoch, best_val = try_resume_from_ckpt(policy, optimizer, load_path, device=DEVICE)
        else:
            print(f"No checkpoint found to resume (looked at {RESUME_PATH or MODEL_DIR}). Start fresh.")

    # Log caching
    train_rows = []  # {epoch, update, reward, loss}
    eval_rows  = []  # {epoch, val_return, eval_time, train_time}

    try:
        for epoch in range(1, N_EPOCHS + 1):
            t0_train = time.time()
            loss_meter = []
            reward_meter = []

            # Update in each epoch
            for u in range(1, UPDATES_PER_EPOCH + 1):
                mean_reward, loss_val = reinforce_update(policy, optimizer)
                loss_meter.append(loss_val)
                reward_meter.append(mean_reward)

                train_rows.append({
                    "epoch": epoch,
                    "update": u,
                    "reward": float(mean_reward),
                    "loss": float(loss_val),
                })

                if u % 1 == 0:
                    print(f"update {u:4d}/{UPDATES_PER_EPOCH}, reward={mean_reward:.4f}  loss={loss_val:.4f}")

            train_dur = time.time() - t0_train

            # Evaluation
            t0_eval = time.time()
            val_ret = evaluate_greedy_fixed(policy, eval_envs)
            eval_dur = time.time() - t0_eval

            eval_rows.append({
                "epoch": epoch,
                "val_return": float(val_ret),
                "eval_time": float(eval_dur),
                "train_time": float(train_dur),
            })

            print(
                f"[Epoch {epoch:03d}/{N_EPOCHS}] "
                f"updates={UPDATES_PER_EPOCH}  "
                f"train_loss(avg)={np.mean(loss_meter):.4f}  "
                f"train_reward(avg)={np.mean(reward_meter):.4f}  "
                f"val_return(greedy)={val_ret:.4f}  "
            )

            print(f"train_time={train_dur:.2f}s  eval_time={eval_dur:.2f}s")

            # Save
            if val_ret > best_val:
                best_val = val_ret
                torch.save({
                    "model": policy.state_dict(),
                    "opt": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                }, ckpt_best)
                print(f"Saved best to {ckpt_best} (val_return={best_val:.4f})")

            print(f"best_val={best_val:.4f}")

            save_excels(BASE_DIR, train_rows, eval_rows)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (KeyboardInterrupt). Saving partial results...")

    finally:
        torch.save({
            "model": policy.state_dict(),
            "opt": optimizer.state_dict(),
            "epoch": min(len(eval_rows), N_EPOCHS),
            "best_val": best_val,
        }, ckpt_last)
        print(f"Last checkpoint saved to {ckpt_last}")

        eval_xlsx, train_xlsx = save_excels(BASE_DIR, train_rows, eval_rows)
        print(f"Excel saved: \n  - {eval_xlsx}\n  - {train_xlsx}")

        plot_curves(BASE_DIR, train_rows, eval_rows)
        print(f"Curves saved under: {BASE_DIR}")

if __name__ == "__main__":
    main()
