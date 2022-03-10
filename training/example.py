import csv
import json
import statistics
import sys

import gym
import numpy as np
import pkg_resources
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gym_vehicle
import gym_symmetric


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        print("initial state: ", env.reset())
        print("Action Space: ", env.observation_space)
        print("Observation Space: ", env.action_space)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    id = sys.argv[1]
    layer_n = int(sys.argv[2])
    layer_l = int(sys.argv[3])
    mil_steps = int(sys.argv[4])
    eval_n = int(sys.argv[5])
    eval_m = int(sys.argv[6])
    eval_k = int(sys.argv[7])

    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env(id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    layers = [layer_n for _ in range(layer_l)]
    policy_kwargs = {"net_arch": [{"vi": layers, "vf": layers}]}
    network_type = '-'.join(list(map(str, layers)))
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0)

    # model = PPO.load("./data/1mil")
    # model.set_env(env)

    for i in range(mil_steps):
        model.learn(total_timesteps=1_000_000)
        model.save(f"./data/n8v16/{network_type}/{id}/{i + 1}")
        accu = 0

        list_sums = []
        for _ in range(eval_n):
            sums = 0
            j = 0
            obs = env.reset()
            for _ in range(eval_m * eval_k):
                j += 1
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                sums = sums + rewards
            list_sums.extend((sums / eval_m).tolist())
        with open(f"./data/n8v16/{network_type}/{id}.tsv", 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(list_sums)
        print(f"{network_type}/{id}/{i + 1}: average return: ", statistics.mean(list_sums), ", stdev = ",
              statistics.stdev(list_sums))
