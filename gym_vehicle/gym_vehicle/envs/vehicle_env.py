import gym
import math
import pkg_resources
from gym import error, spaces, utils
from gym.utils import seeding
import json
import sys


class VehicleAction:

    def __init__(self, env, arr):
        self.motion = [[0 for _ in range(env.node)] for _ in range(env.node)]
        self.price = [[0.0 for _ in range(env.node)] for _ in range(env.node)]
        ind = 0
        tmp = [0 for _ in range(env.node)]
        for i in range(env.node):
            rsum = 0
            for j in range(env.node):
                tmp[j] = arr[ind]
                rsum = rsum + arr[ind]
                ind = ind + 1
            rsum = max(1e-5, rsum)
            rem: int = env.vehicles[i]
            for j in range(env.node):
                tmp[j] = env.vehicles[i] * tmp[j] / rsum
                rem = rem - math.floor(tmp[j])
            random = env.random.rand(1)
            rem = rem - 1
            for j in range(env.node):
                mrem = tmp[j] - math.floor(tmp[j])
                if (random > 0) and (random < mrem):
                    self.motion[i][j] = math.floor(tmp[j]) + 1
                    if rem > 0:
                        random = random + env.random.rand(1)
                        rem = rem - 1
                else:
                    self.motion[i][j] = math.floor(tmp[j])
                random = random - mrem

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
        # Number of actual vehicles
        self.vehicle = self.config["vehicle"]
        #self.poisson_param = self.config["poisson_param"]
        self.operating_cost = self.config["operating_cost"]
        self.waiting_penalty = self.config["waiting_penalty"]
        self.queue_size = self.config["queue_size"]
        self.overflow = self.config["overflow"]
        self.poisson_cap = self.config["poisson_cap"]
        self.poisson_param_cap = self.config["poisson_param_cap"] #used to adjust the cap pf poisson_param
        # self.vehicles != self.vehicle -> this variable defines # of vehicles at a specific node
        self.vehicles = [0 for _ in range(self.node)]
        self.queue = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.random = None
        self.time = 0 #time initiation, it indicates the number of steps
        self.time_unit = self.config["time_unit"]

        # Attempt at edge initialization
        # Edge matrix: self.edge(0) = 1->2 , self.edge(1) = 2->1     for 2 node case (2 edges)
        # n nodes: self.edge(0) = 1->2 , 1->3 , ... 1->n , 2->1 , 2->3, ... 2->n , ... n->n-2 , n->n-1  (? edges)
        edge_matrix = self.config["edge_lengths"]
        edge_num = len(edge_matrix)
        if (self.node * (self.node-1)) != edge_num:
            print("Incorrect edge_lengths parameter. Total nodes and edges do not match!")
            sys.exit()
        self.edge = [[0 for _ in range(self.node)] for _ in range(self.node)]
        tmp = 0
        extra_obs_space = 0
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                else:
                    self.edge[i][j] = edge_matrix[tmp]
                    if edge_matrix[tmp] > 1:
                        extra_obs_space += 1
                    tmp = tmp + 1
        # Initializing "travel" matrix; each position indicates a vehicle is traveling from i to j
        # for self.travel[i][j] more time steps
        self.travel = [(0 for _ in range(self.node)) for _ in range(self.node)]

        self.observation_space = spaces.MultiDiscrete(
            [self.vehicle + 1 for _ in range(self.node)] +
            [self.queue_size + 1 for _ in range(self.node * (self.node - 1))] + [self.time_unit]) ###########################################
        # Either traveling or not -> For each node length > 1, add an additional space
        # Ignore for now? Errors with Stable_baselines
            # + [extra_obs_space])
        self.action_space = spaces.Box(0, 1, (self.node * self.node + self.node * (self.node - 1),))

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)

    def step(self, act):
        action = VehicleAction(self, act)
        op_cost = 0
        wait_pen = 0
        overf = 0
        rew = 0
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                # Traveling state condition here: if "traveling," progress 1 unit of length and continue
                # 2 node implementation ONLY for now

                veh_motion = action.motion[i][j]
                self.vehicles[i] = self.vehicles[i] - veh_motion
                self.vehicles[j] = self.vehicles[j] + veh_motion
                self.queue[i][j] = max(0, self.queue[i][j] - veh_motion)
                op_cost += veh_motion * self.operating_cost
                wait_pen += self.queue[i][j] * self.waiting_penalty
                price = action.price[i][j]
                #self.poisson_param = math.floor(self.poisson_param_cap *(0.5676 + (-0.2124)*math.cos(w*t) + (-0.4349)*math.sin(w*t) + (0.0187)*math.cos(2*w*t) + (-0.0543)*math.sin(2*w*t)))
                demand_calculator()
                request = min(self.poisson_cap, self.random.poisson(self.poisson_param * (1 - price)))
                act_req = min(request, self.queue_size - self.queue[i][j])
                overf += (request - act_req) * self.overflow
                self.queue[i][j] = self.queue[i][j] + act_req
                rew += act_req * action.price[i][j]
                self.time = (self.time + 24/self.time_unit) % 24
        debuf_info = {'reward': rew, 'operating_cost': op_cost, 'wait_penalty': wait_pen, 'overflow': overf}
        return self.to_observation(), rew - op_cost - wait_pen - overf, False, debuf_info

    #######################################################
    def demand_calculator(self):
        t = self.time * (24/self.time_unit) #step number * hours per step
        w = 2 * math.pi / 24 #make the fourier fitiing periodical
        self.poisson_param = math.floor(self.poisson_param_cap * (
                    0.5676 + (-0.2124) * math.cos(w * t) + (-0.4349) * math.sin(w * t) + (0.0187) *
                    math.cos(2 * w * t) + (-0.0543) * math.sin(2 * w * t)))
        return self.poisson_param
    #######################################################

    def reset(self):
        for i in range(self.node):
            self.vehicles[i] = 0
            for j in range(self.node):
                self.queue[i][j] = 0
        for i in range(self.vehicle):
            pos = self.random.randint(0, self.node)
            self.vehicles[pos] = self.vehicles[pos] + 1
        # Reset all edge lengths to 1

        return self.to_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def to_observation(self):
        arr = [0 for _ in range(self.node * self.node + 1)]
        for i in range(self.node):
            arr[i] = self.vehicles[i]
        ind = self.node
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                arr[ind] = self.queue[i][j]
                ind = ind + 1
        arr[ind] = self.time % self.time_unit ####################################
        return arr
