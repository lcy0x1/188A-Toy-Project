import gym
import math
from gym import error, spaces, utils
from gym.utils import seeding
import json


class VehicleAction:

    def __init__(self, env):
        self.motion = [[0 for _ in range(env.node)] for _ in range(env.node)]
        self.price = [[0 for _ in range(env.node)] for _ in range(env.node)]
        pass


class VehicleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed):
        self.random = seeding.np_random(seed)
        self.config = json.load(open("./setup.py"))
        self.node = self.config["node"]
        self.vehicle = self.config["vehicle"]
        self.poisson_param = self.config["poisson_param"]
        self.operating_cost = self.config["operating_cost"]
        self.waiting_penalty = self.config["waiting_penalty"]
        self.vehicles = [0 for _ in range(self.node)]
        self.queue = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.reset()

    def step(self, action: VehicleAction):
        utility = 0
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                veh_motion = action.motion[i][j]
                self.vehicles[i] = self.vehicles[i] - veh_motion
                self.vehicles[j] = self.vehicles[j] + veh_motion
                self.queue[i][j] = max(0, self.queue[i][j] - veh_motion)
                utility = utility - veh_motion * self.operating_cost
                utility = utility - self.queue[i][j] * self.waiting_penalty
                request = self.random.poisson(self.poisson_param * (1 - action.price[i][j]))
                self.queue[i][j] = self.queue[i][j] + request
                utility = utility + request * action.price[i][j]
        return utility

    def reset(self):
        for i in range(self.node):
            self.vehicles[i] = 0
            for j in range(self.node):
                self.queue[i][j] = 0
        for i in range(self.vehicle):
            pos = self.random.randint(0, self.node)
            self.vehicles[pos] = self.vehicles[pos] + 1

    def render(self, mode='human'):
        pass

    def close(self):
        pass
