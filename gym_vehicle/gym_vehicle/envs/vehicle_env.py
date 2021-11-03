import gym
from gym import error, spaces, utils
from gym.utils import seeding
import json


class VehicleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed):
        self.random = seeding.np_random(seed)
        self.config = json.load(open("./setup.py"))
        self.node = self.config["node"]
        self.vehicle = self.config["vehicle"]
        self.vehicles = [0 for _ in range(self.node)]
        self.queue = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.reset()

    def step(self, action):
        pass

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
