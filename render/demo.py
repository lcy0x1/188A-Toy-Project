import math
import math
from typing import List
from graphics import *
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

import sys
sys.path.append('../gym_vehicle')

from gym_vehicle.envs.vehicle_env import VehicleAction


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


class Container(object):

    def __init__(self):
        self.env = make_env("vehicle-v0", 12345)()
        self.model = PPO.load(f"../data_n4_v4/100mil")
        self.model.set_env(self.env)
        self.state = self.env.reset()

    def get_queue(self, state):
        ans = [[0 for _ in range(4)] for _ in range(4)]
        index = 4
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                ans[i][j] = state[index]
                index = index + 1
        return ans


class RenderObject(object):

    def __init__(self):
        self.x = 0
        self.y = 0

    def render(self, time):
        pass

    def update(self):
        pass


class Customer(RenderObject):

    def __init__(self, x, y, i, j):
        super().__init__()
        self.sprite = Image(Point(20, - 20), f"./gostation{i + 1}.gif")
        self.position = j
        self.dst = i
        self.x = x + 20 * j
        self.y = y
        self.old_x = self.x
        self.old_y = self.y
        self.sprite.move(self.x, self.y)
        self.current_x = self.x
        self.current_y = self.y
        self.target_x = self.x
        self.target_y = self.y
        self.status = 0

    def render(self, time):
        nx = (self.target_x - self.old_x) * time + self.old_x
        ny = (self.target_y - self.old_y) * time + self.old_y
        self.sprite.move(nx - self.current_x, ny - self.current_y)
        self.current_x = nx
        self.current_y = ny

    def set_enter(self, index):
        self.old_x = self.x - (stations[index].x - 100)
        self.old_y = self.y - (stations[index].y - 100)
        self.sprite.move(self.old_x - self.current_x, self.old_y - self.current_y)
        self.current_x = self.old_x
        self.current_y = self.old_y

    def set_wait(self, advance):
        self.old_x = self.x + 20 * advance
        self.old_y = self.y
        self.sprite.move(self.old_x - self.current_x, self.old_y - self.current_y)
        self.current_x = self.old_x
        self.current_y = self.old_y

    def update(self):
        self.old_x = self.x
        self.old_y = self.y
        self.sprite.move(self.old_x - self.current_x, self.old_y - self.current_y)
        self.current_x = self.x
        self.current_y = self.y

    def draw(self, window):
        self.sprite.draw(window)

    def undraw(self):
        self.sprite.undraw()


# class small_node(RenderObject):
#     def __init__(self, startpoint, destpoint, index, n):
#         super().__init__()
#         self.x = startpoint.x + (destpoint.x - startpoint.x) / (n) * (index + 1)
#         self.y = startpoint.y + (destpoint.y - startpoint.y) / (n) * (index + 1)
#         self.sprite = Circle(Point(self.x, self.y), 5)
#         self.sprite.draw(win)
#         self.next_vehicle_index = 0
#         self.vehicles: List[Cars] = []
#         self.next_vehicle_index = 0

class Station(RenderObject):

    def __init__(self, index, n):
        super().__init__()
        # 100 100, 900 900 , 100 900, 900 100
        # 100 100, 100 900, 900 100, 900 900
        # 100 100, 900 100, 100 900, 900 900
        x1index = [-1, 1, -1, 1]
        y1index = [-1, -1, 1, 1]
        self.x = 500 + 300 * x1index[index]
        self.y = 500 + 300 * y1index[index]
        self.vehicles: List[Cars] = []
        self.sprite = Circle(Point(self.x, self.y), 15)
        self.sprite.draw(win)

        self.queue_sprite = [[Customer(self.x, self.y, i, j) for j in range(15)] for i in range(4)]

        # self.smallstations: list(List[small_node])=[]
        self.dis = list()

        self.needmove = 0
        self.targetmem = list()

        self.old_queue = [0, 0, 0, 0]
        self.target_queue = [0, 0, 0, 0]
        self.nowindex = index
        self.next_vehicle_index = 0
        self.queue_leave = [0, 0, 0, 0]
        num = Text(Point(self.x, self.y), f"{index + 1}")
        num.setSize(18)
        num.draw(win)

    def draw_queue(self, start):
        for qs in self.queue_sprite:
            for q in qs:
                q.update()
        if start == 1:
            old_sprite_ind = 0
            for dst in range(4):
                if self.nowindex == dst:
                    continue
                for q in range(self.old_queue[dst]):
                    self.queue_sprite[dst][old_sprite_ind].undraw()
                    old_sprite_ind = old_sprite_ind + 1
        sprite_ind = 0
        advance = 0
        for dst in range(4):
            if self.nowindex == dst:
                continue
            to_leave = self.queue_leave[dst]
            advance += to_leave
            for q in range(self.target_queue[dst]):
                if q < self.old_queue[dst] - to_leave:
                    self.queue_sprite[dst][sprite_ind].set_wait(advance)
                else:
                    self.queue_sprite[dst][sprite_ind].set_enter(self.nowindex)
                self.queue_sprite[dst][sprite_ind].draw(win)
                sprite_ind = sprite_ind + 1
        self.old_queue = self.target_queue

    def render(self, time):
        for qs in self.queue_sprite:
            for q in qs:
                q.render(time)

    def remove_vehicle(self, dst, n):
        self.next_vehicle_index -= n
        self.queue_leave[dst] = min(self.old_queue[dst], n)


    def move_vehicle(self, target, n, mid_node):
        for _ in range(n):
            self.vehicles[0].set_index(target, mid_node)
            # index1 = 1 if (target - self.nowindex) == 1 else 0
            # print(index1)
            # print(target)
            # print(target - self.nowindex)
            # totallen = len(self.smallstations[index1])
            # if distance != 0:
            #     self.smallstations[index1][totallen-distance].next_vehicle_index = 0


    # def set_smallnode(self, smallstation, dist):
    #     self.smallstations.append(smallstation)
    #     self.dis.append(dist)


class Cars(RenderObject):
    def __init__(self):
        super().__init__()
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
        self.current_x = 0
        self.current_y = 0
        self.index = 0
        self.sprite = Image(Point(0, 0), "./car.gif")
        self.sprite.draw(win)


    def init_pos(self, i):
        self.index = i
        self.x = stations[self.index].x
        self.y = stations[self.index].y + 30 + len(stations[self.index].vehicles) * 20
        self.target_x = self.x
        self.target_y = self.y
        stations[self.index].vehicles.append(self)
        self.sprite.move(self.x, self.y)
        self.current_x = self.x
        self.current_y = self.y

    def set_index(self, i, mid_node):
        old_station = self.index
        self.index = i

        s0 = stations[old_station]
        # if dist != 0:
        #     currentdist = stations[old_station].dis[index]
        #     s1 = stations[old_station].smallstations[index][currentdist - dist - 1]
        # else:
        if mid_node == 1:
            if (old_station == 0 and i == 3) or (old_station == 3 and i == 0):
                s1 = stations[1]
                s1.needmove += 1
                s1.targetmem.append(i)
            else:
                s1 = stations[0]
                s1.needmove += 1
                s1.targetmem.append(i)
        else:
            s1 = stations[self.index]

        self.target_x = s1.x

        # if index == 0:
        self.target_y = s1.y + 30 + s1.next_vehicle_index * 20
        # else:
        #     self.target_y = s1.y - 30 - s1.next_vehicle_index * 20
        s1.next_vehicle_index += 1
        s0.vehicles.remove(self)
        s0.needmove -= 1
        s1.vehicles.append(self)

    def render(self, time):
        nx = (self.target_x - self.x) * time + self.x
        ny = (self.target_y - self.y) * time + self.y
        self.sprite.move(nx - self.current_x, ny - self.current_y)
        self.current_x = nx
        self.current_y = ny

    def update(self):
        self.x = self.target_x
        self.y = self.target_y


class Price(object):
    def __init__(self):
        self.price: List[List] = [[None for _ in range(4)] for _ in range(4)]
        self.lines: List[List] = [[None for _ in range(4)] for _ in range(4)]

        a = 0.1  # space between ends of arrow and stations
        b = 0.03  # arrow offset
        c = 0.05  # price offset

        for src in range(4):
            for dst in range(4):
                if src == dst:
                    continue
                ssrc = stations[src]
                sdst = stations[dst]
                psrc = Point(ssrc.x * (1 - a) + sdst.x * a, ssrc.y * (1 - a) + sdst.y * a)
                pdst = Point(sdst.x * (1 - a) + ssrc.x * a, sdst.y * (1 - a) + ssrc.y * a)
                dy = sdst.y - ssrc.y
                dx = sdst.x - ssrc.x
                line = Line(psrc, pdst)
                line.move(-dy * b, dx * b)
                if (src == 2 and dst == 3) or (src == 3 and dst == 0) or (src == 3 and dst == 2) or (
                        src == 0 and dst == 3):
                    line.setArrow("first")
                else:
                    line.setArrow("last")

                if (src == 0 and dst == 3) or (src == 3 and dst == 0):
                    line.setFill('blue')

                if (src == 1 and dst == 2) or (src == 2 and dst == 1):
                    line.setOutline('orange')

                line.draw(win)

                self.lines[src][dst] = line

                lc = line.getCenter()
                p = Text(Point(lc.x - dy * c, lc.y + dx * c), "$")
                if (src == 0 and dst == 3) or (src == 3 and dst == 0):
                    p.setTextColor('blue')

                if (src == 1 and dst == 2) or (src == 2 and dst == 1):
                    p.setOutline('orange')
                p.draw(win)

                self.price[src][dst] = p

    def drawprice(self, price, _):
        for src in range(4):
            for dst in range(4):
                if src == dst:
                    continue
                if (src == 2 and dst == 3) or (src == 3 and dst == 0) or (src == 3 and dst == 2) or (
                        src == 0 and dst == 3):
                    p: Text = self.price[dst][src]
                else:
                    p: Text = self.price[src][dst]
                p.setSize(18)
                p.setText("$" + str(round(price[src][dst], 2)))


class Rewards(object):
    def __init__(self):
        self.total_profit = 0
        self.profit = 0
        profit_title = Text(Point(300, 50), "Profit:")
        profit_title.setSize(18)
        profit_title.draw(win)
        total_profit_title = Text(Point(300, 80), "Total Profit:")
        total_profit_title.setSize(18)
        total_profit_title.draw(win)
        self.step_title = Text(Point(100, 50), "Step: ")
        self.step_title.setSize(18)
        self.step_title.draw(win)
        gain = Text(Point(600, 50), "Gain: ")
        gain.setSize(18)
        gain.setTextColor("green")
        gain.draw(win)
        wait_pen = Text(Point(600, 80), "Waiting Penalty: ")
        wait_pen.setSize(18)
        wait_pen.setTextColor("red")
        wait_pen.draw(win)
        opcost = Text(Point(600, 110), "Operating Cost:")
        opcost.setSize(18)
        opcost.setTextColor("red")
        opcost.draw(win)
        overflow = Text(Point(600, 140), "Overflow Penalty:")
        overflow.setTextColor("red")
        overflow.setSize(18)
        overflow.draw(win)
        self.gain = Text(Point(820, 50), 0)
        self.gain.setSize(18)
        self.gain.setTextColor("green")
        self.wait_pen = Text(Point(820, 80), 0)
        self.wait_pen.setSize(18)
        self.wait_pen.setTextColor("red")
        self.op_cost = Text(Point(820, 110), 0)
        self.op_cost.setSize(18)
        self.op_cost.setTextColor("red")
        self.overflow = Text(Point(820, 140), 0)
        self.overflow.setSize(18)
        self.overflow.setTextColor("red")
        self.profit_text = Text(Point(400, 50), round(self.profit, 2))
        self.profit_text.setSize(18)
        self.total_profit_text = Text(Point(400, 80), round(self.total_profit, 2))
        self.total_profit_text.setSize(18)
        self.step_text = Text(Point(150, 50), 0)
        self.step_text.setSize(18)

    def drawreward(self, reward, start, step, info):
        if start == 0:
            self.profit = reward
            self.total_profit = self.total_profit + reward
            self.profit_text.setText(round(self.profit, 2))
            self.total_profit_text.setText(round(self.total_profit, 2))
            self.gain.setText(round(info['reward'], 2))
            self.wait_pen.setText(round(info['wait_penalty'], 2))
            self.op_cost.setText(round(info['operating_cost'], 2))
            self.overflow.setText(round(info['overflow'], 2))
            self.gain.draw(win)
            self.wait_pen.draw(win)
            self.op_cost.draw(win)
            self.overflow.draw(win)
            self.profit_text.draw(win)
            self.total_profit_text.draw(win)
            self.step_text.draw(win)


        else:
            self.profit = reward
            self.total_profit = self.total_profit + reward
            self.profit_text.setText(round(self.profit, 2))
            self.total_profit_text.setText(round(self.total_profit, 2))
            self.gain.setText(round(info['reward'], 2))
            self.wait_pen.setText(round(info['wait_penalty'], 2))
            self.op_cost.setText(round(info['operating_cost'], 2))
            self.overflow.setText(round(info['overflow'], 2))
            self.step_text.setText(step)


def moving(stations):
    _action, _state = container.model.predict(container.state)
    action = VehicleAction(container.env, _action)
    container.state, reward, _, info = container.env.step(_action)
    print('state:',container.state)
    for i in range(4):
        stations[i].next_vehicle_index = len(stations[i].vehicles)
        stations[i].queue_leave = [0, 0, 0, 0]
        for j in range(4):
            if i == j or action.motion[i][j] == 0:
                continue
            stations[i].remove_vehicle(j, action.motion[i][j])
            for z in range(stations[i].needmove, 0, -1):
                stations[i].remove_vehicle(stations[i].targetmem[z-1], 1, 0)


    for i in range(4):
        for j in range(4):
            mid_node = 0
            if (i == 0 and j == 3) or (i == 3 and j == 0) or (i == 2 and j == 1) or (i == 1 and j == 2):
                mid_node = 1
            if i == j or action.motion[i][j] == 0:
                continue
            stations[i].move_vehicle(j, action.motion[i][j], mid_node)
            for z in range(stations[i].needmove, 0, -1):
                stations[i].move_vehicle(stations[i].targetmem[z-1], 1, 0)
                stations[i].targetmem.pop()
    return action.price, action.motion, reward, info


if __name__ == "__main__":
    container = Container()
    start = 0
    step = 0
    win = GraphWin("My demo", 1000, 1000)

    stations = [Station(i, 4) for i in range(4)]
    # smallnode = list(list())
    #
    # smallnode.append([small_node(stations[0], stations[3], i, 4) for i in range(3)])
    # smallnode.append([small_node(stations[0], stations[1], i, 3) for i in range(2)])
    # smallnode.append([small_node(stations[1], stations[0], i, 3) for i in range(2)])
    # smallnode.append([small_node(stations[1], stations[2], i, 2) for i in range(1)])
    # smallnode.append([small_node(stations[2], stations[1], i, 2) for i in range(1)])
    # smallnode.append([small_node(stations[2], stations[3], i, 4) for i in range(3)])
    # smallnode.append([small_node(stations[3], stations[2], i, 4) for i in range(3)])
    # smallnode.append([small_node(stations[3], stations[0], i, 4) for i in range(3)])

    # print(smallnode)
    # map = [[0, 3, 0, 4],
    #        [3, 0, 2, 0],
    #        [0, 2, 0, 4],
    #        [4, 0, 4, 0]]

    # for i in range(4):
    #     if (i - 1) < 0:
    #         stations[i].set_smallnode(smallnode[2*i], map[i][3])
    #     else:
    #         stations[i].set_smallnode(smallnode[2*i], map[i][i - 1])
    #     if i + 1 > 3:
    #         stations[i].set_smallnode(smallnode[2*i+1], map[i][0])
    #     else:
    #         stations[i].set_smallnode(smallnode[2*i+1], map[i][i + 1])

    cars = [Cars() for i in range(4)]
    prices = Price()
    Reward = Rewards()
    ind = 0
    for i in range(4):
        n = container.state[i]
        for j in range(n):
            cars[ind].init_pos(i)
            ind = ind + 1

    while True:
        try:
            win.getMouse()
        except Exception as inst:
            break

        price, motion, reward, info = moving(stations)
        print('motion:',motion)

        Reward.drawreward(reward, start, step, info)
        prices.drawprice(price, start)
        allqueue = container.get_queue(container.state)
        print('queue:', allqueue)
        for x in range(4):
            stations[x].target_queue = allqueue[x]
            stations[x].draw_queue(start)

        n = 30
        for i in range(n):
            t = (i + 1) / n
            for c in cars:
                c.render(t)
            for s in stations:
                s.render(t)
            time.sleep(1 / 30)
        for c in cars:
            c.update()
        start = 1
        step += 1
        # pause for click in window
