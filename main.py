import math
import random
import time
import json
import array_heap


class State(object):

    def __init__(self, vs, mat):
        self.veh_vec: list[int] = vs
        self.queue_vec: list[int] = mat

    def transition(self, env, veh_act, queue_act):
        ind = 0
        n = len(self.queue_vec)

        ans = 0

        for x in range(env.n_node):
            for y in range(env.n_node):
                if x == y:
                    continue
                val = veh_act[ind]
                ans = ans - val * env.cost
                if x < env.n_node - 1:
                    self.veh_vec[x] = self.veh_vec[x] - val
                if y < env.n_node - 1:
                    self.veh_vec[y] = self.veh_vec[y] + val
                rem = self.queue_vec[ind] - val
                if rem < 0:
                    rem = 0
                self.queue_vec[ind] = rem
                ans = ans - self.queue_vec[ind] * env.penalty
                ind = ind + 1

        ind = 0
        for x in range(env.n_node):
            for y in range(env.n_node):
                if x == y:
                    continue
                price = math.floor(math.fmod(queue_act, env.n_price + 1))
                price_f = price / env.n_price
                poi = env.get_poi(price)
                queue_act = math.floor(queue_act / (env.n_price + 1))
                grow = self.queue_vec[n - ind - 1] + poi
                if grow > env.n_queue:
                    ans = ans - (grow - env.n_queue) * env.overflow
                    grow = env.n_queue
                ans = ans + (grow - self.queue_vec[n - ind - 1]) * price_f
                self.queue_vec[n - ind - 1] = grow
                ind = ind + 1
        return ans


class StateStorage(object):

    def __init__(self, env, state):
        self.env: Solver = env
        self.state: State = state
        self.veh_state_count = 0
        self.veh_states: list[list[int]] = []
        self.veh_state_map: list[int] = []
        self.gen_vehicle_state()
        self.q_space = array_heap.ArrHeap(env.action_size * self.veh_state_count, 0)
        self.count = [[0 for _ in range(env.action_size)] for _ in range(self.veh_state_count)]

    # since it's difficult to calculate compact vehicle motion action ID,
    # I find a sparse ID (product of (vehicle at each node + 1)) and map it to compact ID
    def gen_vehicle_state(self):
        act_veh_vec = [0 for _ in range(self.env.n_node * (self.env.n_node - 1))]
        full_veh_pos = [0 for _ in range(self.env.n_node)]
        remain = self.env.n_veh
        for k in range(self.env.n_node - 1):
            full_veh_pos[k] = self.state.veh_vec[k]
            remain = remain - full_veh_pos[k]
        full_veh_pos[self.env.n_node - 1] = remain

        n_sparse = 1
        for k in full_veh_pos:
            n_sparse = n_sparse * (k + 1) ** (self.env.n_node - 1)
        self.veh_state_map = [0 for _ in range(n_sparse)]

        def rec_i(state_veh_i: int, ind_i: int):
            if ind_i == self.env.n_node:
                self.veh_states.append(act_veh_vec.copy())
                self.veh_state_map[state_veh_i] = self.veh_state_count
                self.veh_state_count = self.veh_state_count + 1
                return
            veh = full_veh_pos[ind_i]

            def rec_j(state_veh_j: int, remain_j: int, ind_j: int):
                if ind_j == self.env.n_node - 1:
                    rec_i(state_veh_j, ind_i + 1)
                    return
                for i in range(remain_j + 1):
                    act_veh_vec[ind_i * (self.env.n_node - 1) + ind_j] = i
                    rec_j(state_veh_j * (veh + 1) + i, remain_j - i, ind_j + 1)

            rec_j(state_veh_i, veh, 0)

        rec_i(0, 0)

    def update(self, car: int, ind: int, val: float):
        index = ind + car * self.env.action_size
        old = self.q_space.get(index)
        n = self.count[car][ind]
        self.count[car][ind] = n + 1
        alpha = self.env.get_moving_average(n)
        q_val = old * (1 - alpha) + val * alpha
        self.q_space.update(index, q_val)

    def get_max(self):
        return self.q_space.get_max()

    def get_optimal_action(self):
        """return vehicle state and then queue state of the heuristic"""
        val = self.q_space.get_index_of_max()
        return math.floor(val / self.env.action_size), math.floor(math.fmod(val, self.env.action_size))

    def to_obj(self):
        return {'q': self.q_space.to_obj(), 'c': self.count}

    def read(self, data):
        self.q_space.read(data['q'])
        self.count = data['c']


class Solver:
    def __init__(self, nv: int, nn: int, nq: int, np: int, poi: float, npoi: int, cost: float, penalty: float,
                 overflow: float, decay: float, moving_average: float):
        self.n_veh = nv
        self.n_node = nn
        self.n_queue = nq
        self.n_price = np
        self.poi = poi
        self.n_poi = npoi
        self.cost = cost
        self.penalty = penalty
        self.overflow = overflow
        self.decay = decay
        self.moving_average = moving_average
        self.car_state_count = 0
        self.car_state_map = [0 for _ in range((nv + 1) ** (nn - 1))]
        self.car_states: list[list[int]] = []
        self.gen_car_state()
        self.prob_cache = [[0 for _ in range(npoi + 1)] for _ in range(np + 1)]
        self.gen_prob_cache()
        self.queue_size = (self.n_queue + 1) ** (self.n_node * (self.n_node - 1))
        self.action_size = (self.n_price + 1) ** (self.n_node * (self.n_node - 1))
        self.transition_size = (self.n_poi + 1) ** (self.n_node * (self.n_node - 1))
        self.queue_states: list[list[int]] = [[] for _ in range(self.queue_size)]
        self.gen_queue_states()
        self.state_space = self.gen_state_space()

    # calculate how large is the problem
    # returns state size, action size, and average transition size
    def get_total_q_size(self):
        ans = 0
        count = 0
        for s0 in self.state_space:
            for s1 in s0:
                count = count + 1
                ans = ans + s1.veh_state_count
        return count, ans * self.action_size, ans * self.action_size * self.transition_size

    def get_car_id(self, vec):
        ans = 0
        rem = self.n_veh
        for i in range(self.n_node - 1):
            rem = rem - vec[i]
            ans = ans * (self.n_veh + 1) + vec[i]
        return ans

    def get_queue_id(self, q_mat):
        ans = 0
        for i in q_mat:
            ans = ans * (self.n_queue + 1) + i
        return ans

    # get StateStorage from State
    def get_state(self, state: State):
        car_id = self.car_state_map[self.get_car_id(state.veh_vec)]
        queue_id = self.get_queue_id(state.queue_vec)
        return self.state_space[car_id][queue_id]

    # get probability to get number of new passengers 0~self.npoi with price level 0~self.np
    def get_prob(self, price: int, request: int):
        return self.prob_cache[price][request]

    def gen_car_state(self):
        vec = [0 for _ in range(self.n_node - 1)]

        def rec(val, ind, rem):
            if ind == self.n_node - 1:
                self.car_state_map[val] = self.car_state_count
                self.car_states.append(vec.copy())
                self.car_state_count = self.car_state_count + 1
                return
            for i in range(rem + 1):
                vec[ind] = i
                rec(val * (self.n_veh + 1) + i, ind + 1, rem - i)

        rec(0, 0, self.n_veh)

    def get_poi(self, price):
        rand = random.random()
        for req in range(self.n_poi + 1):
            prob = self.get_prob(price, req)
            if rand < prob:
                return req
            rand = rand - prob
        return self.n_poi

    def gen_prob_cache(self):
        def get_prob(price, v):
            def fact(k):
                if k < 2:
                    return 1
                return k * fact(k - 1)

            lbd = self.poi * (self.n_price - price) / self.n_price
            return lbd ** v * math.exp(-lbd) / fact(v)

        for p in range(self.n_price + 1):
            rem = 1
            for req in range(self.n_poi):
                val = get_prob(p, req)
                rem = rem - val
                self.prob_cache[p][req] = val
            self.prob_cache[p][self.n_poi] = rem

    def gen_queue_states(self):
        vec = [0 for _ in range(self.n_node * (self.n_node - 1))]

        def rec(ind):
            if ind == self.n_node * (self.n_node - 1):
                self.queue_states[self.get_queue_id(vec)] = vec.copy()
                return
            for i in range(self.n_queue + 1):
                vec[ind] = i
                rec(ind + 1)

        rec(0)

    def gen_state_space(self):
        ans = [[StateStorage(self, State(self.car_states[i], self.queue_states[j])) for j in
                range(self.queue_size)] for i in range(self.car_state_count)]
        return ans

    def train(self, step, epsilon):
        iss = self.state_space[random.randrange(self.car_state_count)][random.randrange(self.queue_size)].state
        state = State(iss.veh_vec.copy(), iss.queue_vec.copy())
        ss0 = self.get_state(state)
        for i in range(step):
            rand = random.random()
            if rand < epsilon:
                action = (random.randrange(ss0.veh_state_count), random.randrange(self.action_size))
            else:
                action = ss0.get_optimal_action()
            reward = state.transition(self, ss0.veh_states[action[0]], action[1])
            ss1 = self.get_state(state)
            ss0.update(action[0], action[1], reward + self.decay * ss1.get_max())
            ss0 = ss1

    def write_json(self, filename):
        data = []
        for v0 in self.state_space:
            for v1 in v0:
                data.append(v1.to_obj())
        json.dump(data, open(filename, 'w'))

    def read_json(self, filename):
        data = json.load(open(filename, 'r'))
        vi = 0
        for v0 in self.state_space:
            for v1 in v0:
                v1.read(data[vi])
                vi = vi + 1

    def get_moving_average(self, n):
        return self.moving_average


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    number_of_vehicles = 1
    number_of_nodes = 3
    queue_size = 1
    price_discretization = 3
    poisson_cap = 1
    poisson_parameter = 0.5
    operating_cost = 0.1
    waiting_penalty = 0.2
    overflow_penalty = 100
    converge_discount = 0.99
    moving_average = 0.01
    macro_step = 100
    middle_step = 40000
    micro_step = 25
    eps_decay = 0.95

    print("Start")
    solv = Solver(number_of_vehicles,
                  number_of_nodes,
                  queue_size,
                  price_discretization - 1,
                  poisson_parameter,
                  poisson_cap,
                  operating_cost,
                  waiting_penalty,
                  overflow_penalty,
                  converge_discount,
                  moving_average)
    print("State Machine Initialized")
    print("\t(state size, q size, transition:): ", solv.get_total_q_size())
    print("\tprobability matrix: ", solv.prob_cache)

    filename = f"./data/{number_of_vehicles}-{number_of_nodes}-{queue_size}-{price_discretization}-{poisson_cap}/{poisson_parameter}-{operating_cost}-{waiting_penalty}-{overflow_penalty}-{converge_discount}/{macro_step * middle_step * micro_step}.json"

    eps = 1
    for i in range(macro_step):
        for j in range(middle_step):
            solv.train(micro_step, eps)
        eps = eps * eps_decay
        print(f'{i + 1}%')
    solv.write_json(filename)
    # solv.read_json(filename)

    for a0 in solv.state_space:
        for a1 in a0:
            act = a1.get_optimal_action()
            print(a1.state.veh_vec, a1.state.queue_vec,
                  f'{a1.veh_states[act[0]]}, {(math.floor(act[1] / (solv.n_price + 1)) / solv.n_price, math.floor(math.fmod(act[1], solv.n_price + 1)) / solv.n_price)}',
                  a1.get_max())

    print("complete")
