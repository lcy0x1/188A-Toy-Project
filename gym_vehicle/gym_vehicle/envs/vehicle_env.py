import gym
import math
import pkg_resources
from gym import error, spaces, utils
from gym.utils import seeding
import json


class VehicleAction:

    def __init__(self, env, arr):
        self.motion = [[0 for _ in range(env.node)] for _ in range(env.node)]
        self.price = [[0 for _ in range(env.node)] for _ in range(env.node)]
        ind = 0
        for i in range(env.node):
            for j in range(env.node):
                if i == j:
                    continue
                self.motion[i][j] = arr[ind]
                ind = ind + 1
        for i in range(env.node):
            for j in range(env.node):
                if i == j:
                    continue
                self.price[i][j] = arr[ind]
                ind = ind + 1


class VehicleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.config = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
        self.node = self.config["node"]
        self.vehicle = self.config["vehicle"]
        self.poisson_param = self.config["poisson_param"]
        self.operating_cost = self.config["operating_cost"]
        self.waiting_penalty = self.config["waiting_penalty"]
        self.price_discretization = self.config["price_discretization"]
        self.queue_size = self.config["queue_size"]
        self.overflow = self.config["overflow"]
        self.vehicles = [0 for _ in range(self.node)]
        self.queue = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.random = None
        self.observation_space = spaces.MultiDiscrete(
            [self.vehicle + 1 for _ in range(self.node)] +
            [self.queue_size + 1 for _ in range(self.node * (self.node - 1))])
        self.action_space = spaces.MultiDiscrete(
            [self.vehicle + 1 for _ in range(self.node * (self.node - 1))] +
            [self.price_discretization + 1 for _ in range(self.node * (self.node - 1))])

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)

    def step(self, act):
        action = VehicleAction(self, act)
        utility = 0
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                veh_motion = min(action.motion[i][j], self.vehicles[i])  # TODO truncate invalid action
                self.vehicles[i] = self.vehicles[i] - veh_motion
                self.vehicles[j] = self.vehicles[j] + veh_motion
                self.queue[i][j] = max(0, self.queue[i][j] - veh_motion)
                utility = utility - veh_motion * self.operating_cost
                utility = utility - self.queue[i][j] * self.waiting_penalty
                price = action.price[i][j] / self.price_discretization
                request = self.random.poisson(self.poisson_param * (1 - price))
                act_req = min(request, self.queue_size - self.queue[i][j])
                utility = utility - (request - act_req) * self.overflow
                self.queue[i][j] = self.queue[i][j] + act_req
                utility = utility + act_req * action.price[i][j]
        return self.to_observation(), utility, False, {}

    def reset(self):
        for i in range(self.node):
            self.vehicles[i] = 0
            for j in range(self.node):
                self.queue[i][j] = 0
        for i in range(self.vehicle):
            pos = self.random.randint(0, self.node)
            self.vehicles[pos] = self.vehicles[pos] + 1
        return self.to_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def to_observation(self):
        arr = [0 for _ in range(self.node * self.node)]
        for i in range(self.node):
            arr[i] = self.vehicles[i]
        ind = self.node
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                arr[ind] = self.queue[i][j]
                ind = ind + 1
        return arr
