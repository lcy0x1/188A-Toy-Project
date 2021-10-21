import math
import time


class Solver:
    class State:
        def __init__(self, vs, mat):
            self.veh_vec: list[int] = vs
            self.queue_vec: list[int] = mat

    class StateStorage:

        def __init__(self, env, state):
            self.env: Solver = env
            self.state: Solver.State = state
            self.veh_state_count = 0
            self.veh_states: list[list[int]] = []
            self.veh_state_map: list[int] = []
            self.gen_vehicle_state()
            self.q_space = [[0 for _ in range(env.action_size)] for _ in range(self.veh_state_count)]
            self.max: float = 0

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

        def update(self, car: int, ind: int, val: int):
            if val > self.max:
                self.max = val
            self.q_space[car][ind] = val

        def get_max(self):
            return self.max

        # 1. iterate through all possible motion of vehicle
        # 2. iterate through all possible price
        # 3. iterate through all possible transition
        def iterate(self):
            self.max = float('-inf')
            # number of OD pairs
            n_od = self.env.n_node * (self.env.n_node - 1)
            # current queue state vector modified throughout recursion
            cur_queue_vec = self.state.queue_vec.copy()
            # current action vector of price modified throughout recursion
            act_price = [0 for _ in range(n_od)]
            # current vehicle movement vector modified throughout recursion
            act_veh = [0 for _ in range(self.env.n_node * (self.env.n_node - 1))]
            # full vehicle position vector, immutable
            full_veh_vec = [0 for _ in range(self.env.n_node)]
            remain = self.env.n_veh
            for k in range(self.env.n_node - 1):
                full_veh_vec[k] = self.state.veh_vec[k]
                remain = remain - full_veh_vec[k]
            full_veh_vec[self.env.n_node - 1] = remain
            # current vehicle position vector, modified throught recursion
            cur_veh_vec = full_veh_vec.copy()

            # iterate through transition
            def rec_poisson(ind_od):
                if ind_od == n_od:
                    return self.env.decay * self.env.get_state(self.env.State(cur_veh_vec, cur_queue_vec)).get_max()
                ans = 0
                for i in range(self.env.n_poi + 1):
                    coming = min(self.env.n_queue - cur_queue_vec[ind_od], i)
                    cur_queue_vec[ind_od] = cur_queue_vec[ind_od] + coming
                    gain = rec_poisson(ind_od + 1) + coming * act_price[ind_od] - (i - coming) * self.env.overflow
                    ans = ans + self.env.get_prob(act_price[ind_od], i) * gain
                    cur_queue_vec[ind_od] = cur_queue_vec[ind_od] - coming
                return ans

            # iterate through price setting action
            def rec_act_price(state_act_veh, state_act_queue, ind_od, cost):
                if ind_od == n_od:
                    self.update(state_act_veh, state_act_queue, rec_poisson(0) - cost)
                    return
                for i in range(self.env.n_price + 1):
                    act_price[ind_od] = i
                    rec_act_price(state_act_veh, state_act_queue * (self.env.n_price + 1) + i, ind_od + 1, cost)

            def rec_act_veh_i(state_act_veh_i, ind_veh_i, cost_i):
                if ind_veh_i == self.env.n_node:
                    rec_act_price(self.veh_state_map[state_act_veh_i], 0, 0, cost_i)
                    return
                veh = full_veh_vec[ind_veh_i]

                def rec_act_veh_j(state_act_veh_j, rem_j, ind_veh_j, cost_j):
                    if ind_veh_j == self.env.n_node - 1:
                        rec_act_veh_i(state_act_veh_j, ind_veh_i + 1, cost_j)
                        return
                    for i in range(rem_j + 1):
                        node_i = ind_veh_i
                        node_j = ind_veh_j
                        if ind_veh_j >= ind_veh_i:
                            node_j = node_j + 1
                        ind_od = ind_veh_i * (self.env.n_node - 1) + ind_veh_j
                        act_veh[ind_od] = i
                        cur_veh_vec[node_i] = cur_veh_vec[node_i] - i
                        cur_veh_vec[node_j] = cur_veh_vec[node_j] + i
                        vmi = min(i, cur_queue_vec[ind_od])
                        cur_queue_vec[ind_od] = cur_queue_vec[ind_od] - vmi
                        t_cost = cost_j + i * self.env.cost + cur_queue_vec[ind_od] * self.env.penalty
                        rec_act_veh_j(state_act_veh_j * (veh + 1) + i, rem_j - i, ind_veh_j + 1, t_cost)
                        cur_veh_vec[node_i] = cur_veh_vec[node_i] + i
                        cur_veh_vec[node_j] = cur_veh_vec[node_j] - i
                        cur_queue_vec[ind_od] = cur_queue_vec[ind_od] + vmi

                rec_act_veh_j(state_act_veh_i, veh, 0, cost_i)

            rec_act_veh_i(0, 0, 0)

    def __init__(self, nv: int, nn: int, nq: int, np: int, poi: float, npoi: int, cost: float, penalty: float,
                 overflow: float, decay: float):
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
        self.old_state_space = self.gen_state_space()
        self.new_state_space = self.gen_state_space()

    # calculate how large is the problem
    # returns state size, action size, and average transition size
    def get_total_q_size(self):
        ans = 0
        count = 0
        for s0 in self.old_state_space:
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
        return self.old_state_space[car_id][queue_id]

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

    def gen_prob_cache(self):
        def get_prob(price, v):
            def fact(k):
                if k < 2:
                    return 1
                return k * fact(k - 1)

            lbd = self.poi * (self.n_price - price) / (self.n_price + 1)
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
        ans = [[self.StateStorage(self, self.State(self.car_states[i], self.queue_states[j])) for j in
                range(self.queue_size)] for i in range(self.car_state_count)]
        return ans


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    number_of_vehicles = 3
    number_of_nodes = 3
    queue_size = 2
    price_discretization = 3
    poisson_parameter = 0.5
    poisson_cap = 1
    operating_cost = 0.1
    waiting_penalty = 0.2
    overflow_penalty = 100
    converge_discount = 0.99
    epsilon = 1e-4
    stats = True

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
                  converge_discount)
    print("State Machine Initialized")
    print("\t(state size, q size, transition:): ", solv.get_total_q_size())
    print("\tprobability matrix: ", solv.prob_cache)

    prev = -1
    ite = 0
    while True:
        maxv = 0

        ix = 0

        for v0 in solv.new_state_space:
            for v1 in v0:

                t0 = time.time()

                v1.iterate()
                if maxv < v1.get_max():
                    maxv = v1.get_max()

                if stats:
                    t1 = time.time()
                    ix = ix + 1
                    print(ix, '/', solv.car_state_count * solv.queue_size, ", time: ", t1 - t0)

        temp = solv.new_state_space
        solv.new_state_space = solv.old_state_space
        solv.old_state_space = temp
        ite = ite + 1
        print("step training, iteration: ", ite, ",max: ", maxv)
        if maxv - prev < epsilon:
            break
        prev = maxv
    for v0 in solv.old_state_space:
        for v1 in v0:
            print(v1.q_space)
