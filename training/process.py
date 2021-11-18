import math

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gym_vehicle
import value_iteration


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        print("initial state: ", env.reset())
        return env

    set_random_seed(seed)
    return _init


def make_vi():
    """
    :return: Pre-trained Solver instance
    """
    number_of_vehicles = 2
    number_of_nodes = 2
    queue_size = 2
    price_discretization = 5
    poisson_cap = 1
    poisson_parameter = 0.5
    operating_cost = 0.1
    waiting_penalty = 0.2
    overflow_penalty = 100
    converge_discount = 0.99

    solv = value_iteration.Solver(number_of_vehicles,
                                  number_of_nodes,
                                  queue_size,
                                  price_discretization,
                                  poisson_parameter,
                                  poisson_cap,
                                  operating_cost,
                                  waiting_penalty,
                                  overflow_penalty,
                                  converge_discount)

    solv.read_json('./data/test.json')

    return solv


def convert(solv, obs):
    """
    :param solv: Solver instance
    :param obs: observation
    :return: action: action

    This converts DeepRL observation into value-iteration observation
    and then produce action from the state-action map,
    and then convert the action into DeepRL action
    """
    st = solv.State([obs[0]], [min(2, obs[2]), min(2, obs[3])])
    q_space = solv.get_state(st)
    act_ind = q_space.max_ind
    veh = q_space.veh_states[act_ind[0]]
    action = env.action_space.sample()
    action[0] = st.veh_vec[0] - veh[0]
    action[1] = veh[0]
    action[2] = veh[1]
    action[3] = solv.n_veh - st.veh_vec[0] - veh[1]
    action[4] = math.floor(math.fmod(act_ind[1], 6)) / 5
    action[5] = math.floor(act_ind[1] / 6) / 5
    return action


if __name__ == "__main__":
    env_id = "vehicle-v0"
    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment
    env = make_env(env_id, 12345)()

    solv = make_vi()

    obs = env.reset()
    sums = 0
    qs = 0
    for _ in range(1000):
        action = convert(solv, obs)
        obs, rewards, dones, info = env.step(action)
        sums = sums + rewards
        qs = qs + obs[2] + obs[3]
    sums = sums / 1000
    qs = qs / 1000
    print("average return: ", sums)
    print("average queue length: ", qs)
