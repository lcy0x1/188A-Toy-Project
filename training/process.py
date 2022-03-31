import math
import statistics

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import value_iteration
import sys
sys.path.insert(0, 'C:/Users/Soulget/Desktop/Temp HW files/ECE 188A/GITHUB/188A-Toy-Project/gym_vehicle')
import gym_vehicle


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
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

    solv.read_json('./data_n2_v2/test.json')

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


def compare():
    model = PPO.load("./data_n2_v2/3mil")
    model.set_env(env)

    list_sums = []
    list_qs = []
    for trial in range(100):
        obs = env.reset()
        sums = 0
        qs = 0
        for _ in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            sums = sums + rewards
            qs = qs + obs[2] + obs[3]
        sums = sums / 1000
        qs = qs / 1000
        list_sums.append(sums)
        list_qs.append(qs)
    print("DeepRL: average return: ", statistics.mean(list_sums), ", stdev = ", statistics.stdev(list_sums))
    print("DeepRL: average queue length: ", statistics.mean(list_qs), ", stdev = ", statistics.stdev(list_qs))

    solv = make_vi()

    list_sums = []
    list_qs = []
    for trial in range(100):
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
        list_sums.append(sums)
        list_qs.append(qs)
    print("Value Iteration: average return: ", statistics.mean(list_sums), ", stdev = ", statistics.stdev(list_sums))
    print("Value Iteration: average queue length: ", statistics.mean(list_qs), ", stdev = ", statistics.stdev(list_qs))


def plot(n):
    ret_list = []
    q_list = []
    for i in range(50):
        model = PPO.load(f"./traveling_time_data/demo_n4_v4_revised/{i + 1}mil")
        model.set_env(env)

        list_sums = []
        list_qs = []
        for trial in range(100):
            obs = env.reset()
            sums = 0
            qs = 0
            for _ in range(1000):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                sums = sums + rewards
                qs = qs + obs[2] + obs[3]
            sums = sums / 1000
            qs = qs / 1000
            list_sums.append(sums)
            list_qs.append(qs)
        print(f"DeepRL {i + 1}: average return: ", statistics.mean(list_sums), ", stdev = ",
              statistics.stdev(list_sums))
        print(f"DeepRL {i + 1}: average queue length: ", statistics.mean(list_qs), ", stdev = ",
              statistics.stdev(list_qs))
        ret_list.append(statistics.mean(list_sums))
        q_list.append(statistics.mean(list_qs))
    print("return curve: ", ret_list)
    print("queue curve: ", q_list)


if __name__ == "__main__":
    env_id = "vehicle-v0"
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = make_env(env_id, 12345)()
    # compare()
    plot(1)
