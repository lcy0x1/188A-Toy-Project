import math
from typing import List
from graphics import *

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

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
        self.model = PPO.load(f"../training/data_n3_v3_set1/9mil")
        self.model.set_env(self.env)
        self.state = self.env.reset()

    def get_queue(self, state):
        ans = [[0 for _ in range(3)] for _ in range(3)]
        index = 3
        for i in range(3):
            for j in range(3):
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
        self.old_x = self.x - (stations[index].x - 500)
        self.old_y = self.y - (stations[index].y - 500)
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


class Station(RenderObject):

    def __init__(self, index, n):
        super().__init__()
        self.x = 500 + 400 * math.cos(index / n * 2 * math.pi)
        self.y = 500 + 400 * math.sin(index / n * 2 * math.pi)
        self.vehicles: List[Cars] = []
        self.sprite = Circle(Point(self.x, self.y), 15)
        self.sprite.draw(win)

        self.queue_sprite = [[Customer(self.x, self.y, i, j) for j in range(8)] for i in range(3)]

        self.old_queue = [0, 0, 0]
        self.target_queue = [0, 0, 0]
        self.nowindex = index
        self.next_vehicle_index = 0
        self.queue_leave = [0, 0, 0]
        num = Text(Point(self.x, self.y), f"{index + 1}")
        num.setSize(18)
        num.draw(win)

    def draw_queue(self, start):
        for qs in self.queue_sprite:
            for q in qs:
                q.update()
        if start == 1:
            old_sprite_ind = 0
            for dst in range(3):
                if self.nowindex == dst:
                    continue
                for q in range(self.old_queue[dst]):
                    self.queue_sprite[dst][old_sprite_ind].undraw()
                    old_sprite_ind = old_sprite_ind + 1
        sprite_ind = 0
        advance = 0
        for dst in range(3):
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

    def move_vehicle(self, target, n):
        for _ in range(n):
            self.vehicles[0].set_index(target)


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

    def set_index(self, i):
        old_station = self.index
        self.index = i
        s0 = stations[old_station]
        s1 = stations[self.index]
        self.target_x = s1.x
        self.target_y = s1.y + 30 + s1.next_vehicle_index * 20
        s1.next_vehicle_index += 1
        s0.vehicles.remove(self)
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
        self.price: List[List] = [[None for _ in range(3)] for _ in range(3)]
        self.lines: List[List] = [[None for _ in range(3)] for _ in range(3)]

        a = 0.1  # space between ends of arrow and stations
        b = 0.03  # arrow offset
        c = 0.05  # price offset

        for src in range(3):
            for dst in range(3):
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
                line.setArrow("last")
                line.draw(win)
                self.lines[src][dst] = line

                lc = line.getCenter()
                p = Text(Point(lc.x - dy * c, lc.y + dx * c), "$")
                p.draw(win)
                self.price[src][dst] = p

    def drawprice(self, price, _):
        for src in range(3):
            for dst in range(3):
                if src == dst:
                    continue
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
        gain = Text(Point(700, 50), "Gain: ")
        gain.setSize(18)
        gain.setTextColor("green")
        gain.draw(win)
        wait_pen = Text(Point(700,80), "Waiting Penalty: ")
        wait_pen.setSize(18)
        wait_pen.setTextColor("red")
        wait_pen.draw(win)
        opcost = Text(Point(700,110), "Operating Cost:")
        opcost.setSize(18)
        opcost.setTextColor("red")
        opcost.draw(win)
        overflow = Text(Point(700,140), "Overflow Penalty:")
        overflow.setTextColor("red")
        overflow.setSize(18)
        overflow.draw(win)
        self.gain = Text(Point(820,50),0)
        self.gain.setSize(18)
        self.gain.setTextColor("green")
        self.wait_pen = Text(Point(820,80),0)
        self.wait_pen.setSize(18)
        self.wait_pen.setTextColor("red")
        self.op_cost = Text(Point(820,110),0)
        self.op_cost.setSize(18)
        self.op_cost.setTextColor("red")
        self.overflow = Text(Point(820,140),0)
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
            self.gain.setText(round(info['reward'],2))
            self.wait_pen.setText(round(info['wait_penalty'],2))
            self.op_cost.setText(round(info['operating_cost'],2))
            self.overflow.setText(round(info['overflow'],2))
            self.gain.draw(win)
            self.wait_pen.draw(win)
            self.op_cost.draw(win)
            self.overflow.draw(win)
            self.profit_text.draw(win)
            self.total_profit_text.draw(win)
            self.step_text.draw(win)
            print(info['operating_cost'])

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


def moving():
    _action, _state = container.model.predict(container.state)
    action = VehicleAction(container.env, _action)
    container.state, reward, _, info = container.env.step(_action)
    for i in range(3):
        stations[i].next_vehicle_index = len(stations[i].vehicles)
        stations[i].queue_leave = [0, 0, 0]
        for j in range(3):
            if i == j or action.motion[i][j] == 0:
                continue
            stations[i].remove_vehicle(j, action.motion[i][j])

    for i in range(3):
        for j in range(3):
            if i == j or action.motion[i][j] == 0:
                continue
            stations[i].move_vehicle(j, action.motion[i][j])
    return action.price, action.motion, reward, info


if __name__ == "__main__":
    container = Container()
    start = 0
    step = 0
    win = GraphWin("My demo", 1000, 900)

    stations = [Station(i, 3) for i in range(3)]
    cars = [Cars() for i in range(3)]
    prices = Price()
    Reward = Rewards()
    ind = 0
    for i in range(3):
        n = container.state[i]
        for j in range(n):
            cars[ind].init_pos(i)
            ind = ind + 1

    while True:
        try:
            win.getMouse()
        except Exception as inst:
            break

        price, motion, reward, info = moving()
        Reward.drawreward(reward, start, step, info)
        prices.drawprice(price, start)
        allqueue = container.get_queue(container.state)

        for x in range(3):
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
