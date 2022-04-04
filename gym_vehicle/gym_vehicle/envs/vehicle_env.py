import gym
import math
import pkg_resources
from gym import error, spaces, utils
from gym.utils import seeding
import json
import sys


class VehicleAction:

    # Define and allow charging action here
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
        self.poisson_param = self.config["poisson_param"]
        self.operating_cost = self.config["operating_cost"]
        self.waiting_penalty = self.config["waiting_penalty"]
        self.queue_size = self.config["queue_size"]
        self.overflow = self.config["overflow"]
        self.poisson_cap = self.config["poisson_cap"]
        self.queue = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.random = None
        # Charging variables
        self.charge_price = self.config["charging_price"]
        self.max_charge = self.config["total_charge"]
        # self.battery matrix represents charge of vehicles at a node. If the value is greater than the max_charge, this
        # indicates that there is no car present
        # self.battery has dimensions self.node by self.vehicle
        self.battery = [[(self.max_charge + 1) for _ in range(self.vehicle)] for _ in range(self.node)]

        self.edge_list = self.config["edge_lengths"]
        self.edge_matrix = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.bounds = [0 for _ in range(self.node)]
        self.fill_edge_matrix()

        # self.vehicles != self.vehicle -> this variable defines # of vehicles at a specific node
        self.vehicles = [0 for _ in range(self.node)]

        self.observation_space = spaces.MultiDiscrete(
            [self.vehicle + 1 for _ in range(sum(self.bounds))] +
            [self.queue_size + 1 for _ in range(self.node * (self.node - 1))]
            # Added obs space for vehicle battery states
            + [self.max_charge + 1 for _ in range(self.max_charge + 1)])

        # Added self.node to allow for charging action
        self.action_space = spaces.Box(0, 1, (self.node * self.node + self.node * (self.node - 1) + self.node,))

        # Stores number of vehicles at mini node between i and j
        self.mini_vehicles = [[[0 for _ in range(self.edge_matrix[i][j] - 1)]
                               for j in range(self.node)] for i in range(self.node)]

    def fill_edge_matrix(self):
        edge_num = len(self.edge_list)
        if (self.node * (self.node - 1)) != edge_num:
            print("Incorrect edge_lengths parameter. Total nodes and edges do not match!")
            sys.exit()
        # Creating 2D matrix for easier access
        tmp = 0
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                self.edge_matrix[i][j] = self.edge_list[tmp]
                self.bounds[j] = max(self.bounds[j], self.edge_matrix[i][j])
                tmp += 1
                if self.edge_matrix[i][j] < 1:
                    print("Error! Edge length too short (minimum length 1).")
                    sys.exit()
                if self.edge_matrix[i][j] % 1 != 0:
                    print("Error! Edge length must be integer value.")
                    sys.exit()

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)

    # ANDY
    def step(self, act):
        action = VehicleAction(self, act)
        op_cost = 0
        wait_pen = 0
        overf = 0
        rew = 0

        # Move cars in mini-nodes ahead
        # TO DO: Adjust battery levels every time step a vehicle moves
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                # Sweeping BACKWARDS to avoid pushing vehicles multiple times in same time step
                for m in range(self.edge_matrix[i][j] - 1):
                    if m == 0:
                        # Stop tracking mini-node behavior and push cars to main node
                        self.vehicles[j] += self.mini_vehicles[i][j][m]
                    else:
                        # Vehicles still in mini nodes (traveling)
                        # Shifting vehicles further along path
                        self.mini_vehicles[i][j][m - 1] = self.mini_vehicles[i][j][m]
                    op_cost += self.mini_vehicles[i][j][m] * self.operating_cost
                    self.mini_vehicles[i][j][m] = 0

        # Change self.mini_vehicles s.t. [i][j] = nodes, [m] = distance left, value of matrix = # of cars

        # TO DO: Adjust battery level of each vehicle as its moves 1 time step
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                veh_motion = action.motion[i][j]
                # Statement to feed to mini-nodes
                # Only feed to mini nodes if required (edge length > 1)   ->   Feed to first mini-node
                if self.edge_matrix[i][j] > 1:
                    # for distance 2, it feeds to the 1st mininode (index 0)
                    # for distance 5, it feeds to the 4th mininode (index 3)
                    self.mini_vehicles[i][j][self.edge_matrix[i][j] - 2] += veh_motion
                else:
                    # Cars arriving at node j (for length 1 case)
                    self.vehicles[j] += veh_motion

                # Cars leaving node i
                self.vehicles[i] -= veh_motion
                self.queue[i][j] = max(0, self.queue[i][j] - veh_motion)
                # May need to adjust op_cost
                op_cost += veh_motion * self.operating_cost
                wait_pen += self.queue[i][j] * self.waiting_penalty
                price = action.price[i][j]
                request = min(self.poisson_cap, self.random.poisson(self.poisson_param * (1 - price)))
                act_req = min(request, self.queue_size - self.queue[i][j])
                overf += (request - act_req) * self.overflow
                self.queue[i][j] = self.queue[i][j] + act_req
                rew += act_req * action.price[i][j] * self.edge_matrix[i][j]
        debuf_info = {'reward': rew, 'operating_cost': op_cost, 'wait_penalty': wait_pen, 'overflow': overf}
        return self.to_observation(), rew - op_cost - wait_pen - overf, False, debuf_info

    # TO DO: Reset vehicle batteries to full   (finished?)
    def reset(self):
        # Reset queue, vehicles at nodes AND in travel
        for i in range(self.node):
            self.vehicles[i] = 0
            for j in range(self.node):
                self.queue[i][j] = 0
                for k in range(self.edge_matrix[i][j] - 1):
                    self.mini_vehicles[i][j][k] = 0

        for i in range(self.vehicle):
            pos = self.random.randint(0, self.node)
            self.vehicles[pos] = self.vehicles[pos] + 1
            # Use same matrix as initialization to reset battery state to full?
            self.battery = [[(self.max_charge + 1) for _ in range(self.vehicle)] for _ in range(self.node)]

        return self.to_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # Added battery states to function
    def to_observation(self):
        arr = [0 for _ in range(self.node * (self.node - 1) + sum(self.bounds) + self.max_charge + 1)]
        ind = 0
        for i in range(self.node):
            arr[ind] = self.vehicles[i]
            ind += 1
        for j in range(self.node):
            sums = [0 for _ in range(self.bounds[j] - 1)]
            for i in range(self.node):
                for k in range(self.edge_matrix[i][j] - 1):
                    sums[k] += self.mini_vehicles[i][j][k]
            for k in range(self.bounds[j] - 1):
                arr[ind] += sums[k]
                ind += 1
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                arr[ind] = self.queue[i][j]
                ind += 1
        return arr
