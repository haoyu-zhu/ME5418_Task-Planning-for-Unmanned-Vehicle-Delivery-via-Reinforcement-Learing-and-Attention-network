import torch
import random
import numpy as np
from gym_env import PointsEnv

GRAPH_TASKS = 100
MAX_DIST = 3

def build_fixed_env(tasks=100, max_dist=3.0, seed=42) -> PointsEnv:
    np.random.seed(seed)
    random.seed(seed)

    env = PointsEnv(tasks_number=GRAPH_TASKS, max_distance=MAX_DIST)

    return env

def build_multi_fixed_envs(tasks=100, max_dist=3.0, n_envs=100, base_seed=42):
    rng = random.Random(base_seed)  # independent RNG for seed generation
    envs = []

    for i in range(n_envs):

        seed_i = rng.randint(0, 2**31 - 1)

        random.seed(seed_i)
        np.random.seed(seed_i)

        env = PointsEnv(tasks_number=tasks, max_distance=max_dist)
        envs.append(env)

    return envs

if __name__ == '__main__':
    envs = build_multi_fixed_envs()
    for env in envs:
        obs = env.observe()
        print(obs[0][10])