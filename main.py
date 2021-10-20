import math
import time


class Solver:
    class State:
        def __init__(self, vs, mat):
            self.vs = vs
            self.mat = mat

    class StateStorage:

        def __init__(self, env):
            self.env = env
            self.state = None
            self.car_state_count = 0
            self.car_states = []
            self.car_state_map = []
            self.q_space = []

        def gen_car_state(self):
            car = [0 for _ in range(self.env.nn * (self.env.nn - 1))]

            vs = [0 for _ in range(self.env.nn)]
            rem = self.env.nv
            for k in range(self.env.nn - 1):
                vs[k] = self.state.vs[k]
                rem = rem - vs[k]
            vs[self.env.nn - 1] = rem

            n = 1
            for k in vs:
                n = n * (k + 1) ** (self.env.nn - 1)
            self.car_state_map = [0 for _ in range(n)]

            def rec_v(val, ind):
                if ind == self.env.nn:
                    self.car_states.append(car.copy())
                    self.car_state_map[val] = self.car_state_count
                    self.car_state_count = self.car_state_count + 1
                    return
                veh = vs[ind]

                def rec_vv(vjl, rjm, jnd):
                    if jnd == self.env.nn - 1:
                        rec_v(vjl, ind + 1)
                        return
                    for i in range(rjm + 1):
                        car[ind * (self.env.nn - 1) + jnd] = i
                        rec_vv(vjl * (veh + 1) + i, rjm - i, jnd + 1)

                rec_vv(val, veh, 0)

            rec_v(0, 0)

        def update(self, car, ind, val):
            self.q_space[car][ind] = val

        def get_max(self):
            ans = self.q_space[0][0]
            for i0 in self.q_space:
                for i1 in i0:
                    if ans < i1:
                        ans = i1
            return ans

        def iterate(self):
            n = self.env.nn * (self.env.nn - 1)
            cmat = self.state.mat.copy()
            action = [0 for _ in range(n)]
            car = [0 for _ in range(self.env.nn * (self.env.nn - 1))]
            vs = [0 for _ in range(self.env.nn)]
            rem = self.env.nv
            for k in range(self.env.nn - 1):
                vs[k] = self.state.vs[k]
                rem = rem - vs[k]
            vs[self.env.nn - 1] = rem
            cvs = vs.copy()  # vehicle of next state

            # iterate through transition
            def rec_poi(ind):
                if ind == n:
                    return self.env.decay * self.env.get_state(self.env.State(cvs, cmat)).get_max()
                ans = 0
                for i in range(self.env.npoi + 1):
                    val = min(self.env.nq - cmat[ind], i)
                    cmat[ind] = cmat[ind] + val
                    gain = rec_poi(ind + 1) + val * action[ind] - (i - val) * self.env.overflow
                    ans = ans + self.env.get_prob(action[ind], i) * gain
                    cmat[ind] = cmat[ind] - val
                return ans

            # iterate through price setting action
            def rec_q(car_ind, act_ind, ind, cost):
                if ind == n:
                    self.update(car_ind, act_ind, rec_poi(0) - cost)
                    return
                for i in range(self.env.np + 1):
                    action[ind] = i
                    rec_q(car_ind, act_ind * (self.env.np + 1) + i, ind + 1, cost)

            def rec_v(car_ind, ind, cost):
                if ind == self.env.nn:
                    rec_q(self.car_state_map[car_ind], 0, 0, cost)
                    return
                veh = vs[ind]

                def rec_vv(vjl, rjm, jnd, cost):
                    if jnd == self.env.nn - 1:
                        rec_v(vjl, ind + 1, cost)
                        return
                    for i in range(rjm + 1):
                        ci = ind
                        cj = jnd
                        if jnd >= ind:
                            cj = cj + 1
                        mi = ind * (self.env.nn - 1) + jnd
                        car[mi] = i
                        cvs[ci] = cvs[ci] - i
                        cvs[cj] = cvs[cj] + i
                        vmi = min(i, cmat[mi])
                        cmat[mi] = cmat[mi] - vmi
                        t_cost = cost + i * self.env.cost + cmat[mi] * self.env.penalty
                        rec_vv(vjl * (veh + 1) + i, rjm - i, jnd + 1, t_cost)
                        cvs[ci] = cvs[ci] + i
                        cvs[cj] = cvs[cj] - i
                        cmat[mi] = cmat[mi] + vmi

                rec_vv(car_ind, veh, 0, cost)

            rec_v(0, 0, 0)

    def __init__(self, nv, nn, nq, np, poi, npoi, cost, penalty, overflow, decay):
        self.nv = nv
        self.nn = nn
        self.nq = nq
        self.np = np
        self.poi = poi
        self.npoi = npoi
        self.cost = cost
        self.penalty = penalty
        self.overflow = overflow
        self.decay = decay
        self.car_state_count = 0
        self.car_state_map = [0 for _ in range((nv + 1) ** (nn - 1))]
        self.car_states = []
        self.gen_car_state()
        self.prob_cache = [[0 for _ in range(npoi + 1)] for _ in range(np + 1)]
        self.gen_prob_cache()
        self.queue_size = (self.nq + 1) ** (self.nn * (self.nn - 1))
        self.action_size = (self.np + 1) ** (self.nn * (self.nn - 1))
        self.transition_size = (self.npoi + 1) ** (self.nn * (self.nn - 1))
        self.queue_states = [0 for _ in range(self.queue_size)]
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
                ans = ans + s1.car_state_count
        return count, ans * self.action_size, ans * self.action_size * self.transition_size

    def get_car_id(self, vec):
        ans = 0
        rem = self.nv
        for i in range(self.nn - 1):
            rem = rem - vec[i]
            ans = ans * (self.nv + 1) + vec[i]
        return ans

    def get_queue_id(self, q_mat):
        ans = 0
        for i in q_mat:
            ans = ans * (self.nq + 1) + i
        return ans

    # get StateStorage from State
    def get_state(self, state: State):
        car_id = self.car_state_map[self.get_car_id(state.vs)]
        queue_id = self.get_queue_id(state.mat)
        return self.old_state_space[car_id][queue_id]

    # get probability to get number of new passengers 0~self.npoi with price level 0~self.np
    def get_prob(self, price, request):
        return self.prob_cache[price][request]

    def gen_car_state(self):
        vec = [0 for _ in range(self.nn - 1)]

        def rec(val, ind, rem):
            if ind == self.nn - 1:
                self.car_state_map[val] = self.car_state_count
                self.car_states.append(vec.copy())
                self.car_state_count = self.car_state_count + 1
                return
            for i in range(rem + 1):
                vec[ind] = i
                rec(val * (self.nv + 1) + i, ind + 1, rem - i)

        rec(0, 0, self.nv)

    def gen_prob_cache(self):
        def get_prob(price, v):
            def fact(k):
                if k < 2:
                    return 1
                return k * fact(k - 1)

            lbd = self.poi * (self.np - price) / (self.np + 1)
            return lbd ** v * math.exp(-lbd) / fact(v)

        for p in range(self.np + 1):
            rem = 1
            for req in range(self.npoi):
                val = get_prob(p, req)
                rem = rem - val
                self.prob_cache[p][req] = val
            self.prob_cache[p][self.npoi] = rem

    def gen_queue_states(self):
        vec = [0 for _ in range(self.nn * (self.nn - 1))]

        def rec(ind):
            if ind == self.nn * (self.nn - 1):
                self.queue_states[self.get_queue_id(vec)] = vec.copy()
                return
            for i in range(self.nq + 1):
                vec[ind] = i
                rec(ind + 1)

        rec(0)

    def gen_state_space(self):
        ans = [[self.StateStorage(self) for _ in range(self.queue_size)] for _ in
               range(self.car_state_count)]
        for i in range(self.car_state_count):
            for j in range(self.queue_size):
                ans[i][j].state = self.State(self.car_states[i], self.queue_states[j])
                ans[i][j].gen_car_state()
                ans[i][j].q_space = [[0 for _ in range(self.action_size)] for _ in range(ans[i][j].car_state_count)]
        return ans


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    number_of_vehicles = 2
    number_of_nodes = 2
    queue_size = 3
    price_discretization = 3
    poisson_parameter = 0.5
    poisson_cap = 1
    operating_cost = 0.1
    waiting_penalty = 0.2
    overflow_penalty = 100
    converge_discount = 0.99
    epsilon = 1e-4

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
        for s0 in solv.new_state_space:
            for s1 in s0:
                s1.iterate()
                if maxv < s1.get_max():
                    maxv = s1.get_max()
        temp = solv.new_state_space
        solv.new_state_space = solv.old_state_space
        solv.old_state_space = solv.new_state_space
        ite = ite + 1
        print("step training, iteration: ", ite, ",max: ", maxv)
        if maxv - prev < epsilon:
            break
        prev = maxv
    for s0 in solv.old_state_space:
        for s1 in s0:
            print(s1.q_space)
