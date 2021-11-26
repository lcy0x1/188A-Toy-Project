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


class Station(RenderObject):

    def __init__(self, index, n):
        super().__init__()
        self.x = 400 + 300 * math.cos(index / n * 2 * math.pi)
        self.y = 400 + 300 * math.sin(index / n * 2 * math.pi)
        self.vehicles: List[Cars] = []
        self.sprite = Circle(Point(self.x, self.y), 10)
        self.sprite.draw(win)

        self.queue_sprite = [[Image(Point((self.x + 20 * (j + 1)), (self.y - 20)),
                                    f"./gostation{i + 1}.gif") for j in range(4)] for i in range(3)]
        self.Queue = []
        self.newQueue = []
        self.nowindex = index
        self.to_leave = 0

        Text(Point(self.x, self.y), f"{index + 1}").draw(win)

    def drawQueue(self, start):

        if start == 1:
            old_sprite_ind = 0
            for dst in range(3):
                if self.nowindex == dst:
                    continue
                for q in range(self.Queue[dst]):
                    self.queue_sprite[dst][old_sprite_ind].undraw()
                    old_sprite_ind = old_sprite_ind + 1

        sprite_ind = 0
        for dst in range(3):
            if self.nowindex == dst:
                continue
            for q in range(self.newQueue[dst]):
                self.queue_sprite[dst][sprite_ind].draw(win)
                sprite_ind = sprite_ind + 1

        self.Queue = self.newQueue

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

    def set_index(self, i):
        old_station = self.index
        self.index = i
        s0 = stations[old_station]
        s1 = stations[self.index]
        self.target_x = s1.x
        self.target_y = s1.y + 30 + (len(s1.vehicles) - s1.to_leave) * 20
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
                p.setText("$" + str(round(price[src][dst], 2)))


def moving():
    _action, _state = container.model.predict(container.state)
    action = VehicleAction(container.env, _action)
    container.state, reward, _, _ = container.env.step(_action)
    for i in range(3):
        stations[i].to_leave = 0
        for j in range(3):
            if i == j or action.motion[i][j] == 0:
                continue
            stations[i].to_leave = stations[i].to_leave + action.motion[i][j]
    for i in range(3):
        for j in range(3):
            if i == j or action.motion[i][j] == 0:
                continue
            stations[i].move_vehicle(j, action.motion[i][j])
    return action.price, action.motion


if __name__ == "__main__":
    container = Container()
    start = 0

    win = GraphWin("My Circle", 800, 800)

    stations = [Station(i, 3) for i in range(3)]
    cars = [Cars() for i in range(3)]
    prices = Price()

    ind = 0
    for i in range(3):
        n = container.state[i]
        for j in range(n):
            cars[ind].init_pos(i)
            ind = ind + 1

    while True:
        price, motion = moving()
        prices.drawprice(price, start)
        allqueue = container.get_queue(container.state)
        for x in range(3):
            stations[x].newQueue = allqueue[x]
            stations[x].drawQueue(start)

        start = 1
        n = 30
        for i in range(n):
            for c in cars:
                c.render((i + 1) / n)
            time.sleep(1 / 30)
        for c in cars:
            c.update()
        try:
            win.getMouse()  # pause for click in window
        except Exception as inst:
            break
