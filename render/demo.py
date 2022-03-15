from graphics import *
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from gym_vehicle.envs import VehicleEnv
from gym_vehicle.envs.vehicle_env import VehicleAction
from render.graphics import GraphWin, Point, Circle, Text, Image
from typing import List

import sys

sys.path.append('../gym_vehicle')


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


class StationIndex(object):

    def __init__(self, main: int, src: int, dst: int, dis: int):
        self.main = main
        self.src = src
        self.dst = dst
        self.dis = dis


class Container(object):

    def __init__(self, win: GraphWin):
        self.win = win
        self.env: VehicleEnv = make_env("vehicle-v0", 12345)()
        self.n = self.env.node
        self.v = self.env.vehicle
        self.b = sum(self.env.bounds)
        self.model = PPO.load(f"../training/data_nonsym_n4v8/50.zip")
        self.model.set_env(self.env)
        self.state = self.env.reset()

        x1index = [-1, 1, -1, 1]
        y1index = [-1, -1, 1, 1]
        x2index = [[[], [], [], [1]], [[], [], [1], []], [[], [-1], [], []], [[-1], [], [], []]]
        y2index = [[[], [], [], [-1]], [[], [], [1], []], [[], [-1], [], []], [[1], [], [], []]]

        self.stations = [Station(StationIndex(i, 0, 0, 0), self,
                                 500 + 200 * x1index[i],
                                 500 + 200 * y1index[i])
                         for i in range(self.n)]
        self.mini_node = [[[MiniStation(StationIndex(-1, i, j, k), self,
                                        500 + 300 * x2index[i][j][k],
                                        500 + 300 * y2index[i][j][k])
                            for k in range(self.env.edge_matrix[i][j] - 1)]
                           for j in range(self.n)]
                          for i in range(self.n)]

        self.cars = [Cars(self) for _ in range(self.v)]

        ind = 0
        for i in range(self.n):
            for j in range(self.state[i]):
                self.cars[ind].init_pos(StationIndex(i, 0, 0, 0))
                ind = ind + 1

    def get_queue(self, state):
        ans = [[0 for _ in range(self.n)] for _ in range(self.n)]
        index = self.b
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                ans[i][j] = state[index]
                index = index + 1
        return ans

    def get_station(self, index: StationIndex):
        if index.main >= 0:
            return self.stations[index.main]
        return self.mini_node[index.src][index.dst][index.dis]

    def moving(self):
        _action, _state = self.model.predict(self.state)
        print("State:      ", self.state)
        self.state, reward, _, info = self.env.step(_action)
        action = info['action']
        print("Veh Action: ", action.motion)
        print("Next State: ", self.state)

        for i in range(self.n):
            node = self.stations[i]
            node.next_vehicle_index = len(node.vehicles)
            node.queue_leave = [0 for _ in range(self.n)]
            for j in range(self.n):
                if i == j or action.motion[i][j] == 0:
                    continue
                node.remove_vehicle(j, action.motion[i][j])
            node.reorder_vehicle()

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                for m in range(self.env.edge_matrix[i][j] - 1):
                    node = self.mini_node[i][j][m]
                    node.next_vehicle_index = 0
                    if m == 0:
                        # Stop tracking mini-node behavior and push cars to main node
                        node.move_vehicle(StationIndex(j, 0, 0, 0), len(node.vehicles))
                        pass
                    else:
                        # Vehicles still in mini nodes (traveling)
                        # Shifting vehicles further along path
                        node.move_vehicle(StationIndex(-1, i, j, m - 1), len(node.vehicles))
                        pass

        for i in range(4):
            node = self.stations[i]
            for j in range(4):
                if i == j or action.motion[i][j] == 0:
                    continue
                m = self.env.edge_matrix[i][j]
                if m > 1:
                    # for distance 2, it feeds to the 1st mininode (index 0)
                    # for distance 5, it feeds to the 4th mininode (index 3)
                    node.move_vehicle(StationIndex(-1, i, j, m - 2), action.motion[i][j])
                else:
                    # Cars arriving at node j (for length 1 case)
                    node.move_vehicle(StationIndex(j, 0, 0, 0), action.motion[i][j])

        return action.price, action.motion, reward, info


class RenderObject(object):

    def __init__(self):
        self.x = 0
        self.y = 0

    def render(self, time):
        pass

    def update(self):
        pass


class Cars(RenderObject):
    def __init__(self, cont: Container):
        super().__init__()
        self.cont = cont
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
        self.current_x = 0
        self.current_y = 0
        self.index = StationIndex(0, 0, 0, 0)
        self.sprite = Image(Point(0, 0), f"./car.gif")
        self.sprite.draw(cont.win)

    def init_pos(self, i: StationIndex):
        self.index = i
        self.x = self.cont.get_station(self.index).x
        self.y = self.cont.get_station(self.index).y + 30 + len(self.cont.get_station(self.index).vehicles) * 20
        self.target_x = self.x
        self.target_y = self.y
        self.cont.get_station(self.index).vehicles.append(self)
        self.sprite.move(self.x, self.y)
        self.current_x = self.x
        self.current_y = self.y

    def set_index(self, i: StationIndex):
        old_station = self.index
        self.index = i

        s0 = self.cont.get_station(old_station)
        s1 = self.cont.get_station(i)

        self.target_x = s1.x
        self.target_y = s1.y + 30 + s1.next_vehicle_index * 20
        s1.next_vehicle_index += 1
        s0.vehicles.remove(self)
        s1.vehicles.append(self)

    def reset_position(self, s1, ind: int):
        self.target_x = s1.x
        self.target_y = s1.y + 30 + ind * 20

    def render(self, time):
        nx = (self.target_x - self.x) * time + self.x
        ny = (self.target_y - self.y) * time + self.y
        self.sprite.move(nx - self.current_x, ny - self.current_y)
        self.current_x = nx
        self.current_y = ny

    def update(self):
        self.x = self.target_x
        self.y = self.target_y


class AbstractStation(RenderObject):

    def __init__(self, index: StationIndex, cont: Container, x: int, y: int):
        super().__init__()
        self.cont = cont
        self.x = x
        self.y = y
        self.index = index
        self.vehicles: List[Cars] = []
        self.next_vehicle_index = 0

    def move_vehicle(self, target: StationIndex, n):
        """move vehicles to target node"""
        for _ in range(n):
            self.vehicles[0].set_index(target)


class Station(AbstractStation):
    radius = 15
    text_size = 18
    max_queue_length = 15

    def __init__(self, index: StationIndex, cont: Container, x, y):
        super().__init__(index, cont, x, y)

        self.sprite = Circle(Point(self.x, self.y), Station.radius)
        self.sprite.draw(cont.win)

        self.queue_sprite = [[Customer(cont, self.x, self.y, i, j) for j in range(Station.max_queue_length)]
                             for i in range(cont.n)]

        self.old_queue = [0 for _ in range(cont.n)]
        self.target_queue = [0 for _ in range(cont.n)]
        self.queue_leave = [0 for _ in range(cont.n)]
        num = Text(Point(self.x, self.y), f"{index.main + 1}")
        num.setSize(Station.text_size)
        num.draw(cont.win)

    def draw_queue(self, start):
        for qs in self.queue_sprite:
            for q in qs:
                q.update()
        if start == 1:
            old_sprite_ind = 0
            for dst in range(4):
                if self.index == dst:
                    continue
                for q in range(self.old_queue[dst]):
                    self.queue_sprite[dst][old_sprite_ind].undraw()
                    old_sprite_ind = old_sprite_ind + 1
        sprite_ind = 0
        advance = 0
        for dst in range(4):
            if self.index == dst:
                continue
            to_leave = self.queue_leave[dst]
            advance += to_leave
            for q in range(self.target_queue[dst]):
                if q < self.old_queue[dst] - to_leave:
                    self.queue_sprite[dst][sprite_ind].set_wait(advance)
                else:
                    self.queue_sprite[dst][sprite_ind].set_enter(self.index.main)
                self.queue_sprite[dst][sprite_ind].draw()
                sprite_ind = sprite_ind + 1
        self.old_queue = self.target_queue

    def render(self, time):
        for qs in self.queue_sprite:
            for q in qs:
                q.render(time)

    def remove_vehicle(self, dst, n):
        """Mark vehicles ready to leave the station"""
        self.next_vehicle_index -= n
        self.queue_leave[dst] = min(self.old_queue[dst], n)

    def reorder_vehicle(self):
        for i in range(self.next_vehicle_index):
            self.vehicles[i + len(self.vehicles) - self.next_vehicle_index].reset_position(self, i)


class MiniStation(AbstractStation):
    radius = 10
    text_size = 12

    def __init__(self, index: StationIndex, cont: Container, x, y):
        super().__init__(index, cont, x, y)

        self.sprite = Circle(Point(self.x, self.y), MiniStation.radius)
        self.sprite.draw(cont.win)

        num = Text(Point(self.x, self.y), f"{index.dst + 1}")
        num.setSize(MiniStation.text_size)
        num.draw(cont.win)


class Customer(RenderObject):

    def __init__(self, cont: Container, x: int, y: int, i: int, j: int):
        super().__init__()
        self.cont = cont
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

    def set_enter(self, index: int):
        self.old_x = self.x - (self.cont.stations[index].x - 100)
        self.old_y = self.y - (self.cont.stations[index].y - 100)
        self.sprite.move(self.old_x - self.current_x, self.old_y - self.current_y)
        self.current_x = self.old_x
        self.current_y = self.old_y

    def set_wait(self, advance: int):
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

    def draw(self):
        self.sprite.draw(self.cont.win)

    def undraw(self):
        self.sprite.undraw()


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
                ssrc = container.stations[src]
                sdst = container.stations[dst]
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

        for src in range(container.n):
            for dst in range(container.n):
                if src == dst:
                    continue
                edge = container.env.edge_matrix[src][dst]
                if edge <= 1:
                    continue
                old = container.get_station(StationIndex(src, 0, 0, 0))
                for m in range(edge - 1):
                    sdst = container.get_station(StationIndex(-1, src, dst, m))
                    self.draw_arrow(old, sdst, a, -b)
                    old = sdst
                sdst = container.get_station(StationIndex(dst, 0, 0, 0))
                self.draw_arrow(old, sdst, a, -b)

    def draw_arrow(self, ssrc, sdst, a, b):
        psrc = Point(ssrc.x * (1 - a) + sdst.x * a, ssrc.y * (1 - a) + sdst.y * a)
        pdst = Point(sdst.x * (1 - a) + ssrc.x * a, sdst.y * (1 - a) + ssrc.y * a)
        dy = sdst.y - ssrc.y
        dx = sdst.x - ssrc.x
        line = Line(psrc, pdst)
        line.move(-dy * b, dx * b)
        line.draw(win)
        line.setArrow("last")
        line.setFill("gray")
        return line

    def drawprice(self, price, _):
        for src in range(4):
            for dst in range(4):
                if src == dst:
                    continue
                if (src == 0 and dst == 3) or (src == 3 and dst == 0) or (src == 1 and dst == 2) or (
                        src == 2 and dst == 1):
                    p: Text = self.price[dst][src]
                    p.setSize(18)
                    p.setText("$" + str(round(price[src][dst] * 2, 2)))
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


if __name__ == "__main__":
    win = GraphWin("My demo", 1000, 1000)

    container = Container(win)
    start = 0
    step = 0

    prices = Price()
    Reward = Rewards()

    while True:
        try:
            win.getMouse()
        except Exception as inst:
            break

        price, motion, reward, info = container.moving()
        Reward.drawreward(reward, start, step, info)
        prices.drawprice(price, start)
        allqueue = container.get_queue(container.state)
        for x in range(4):
            container.stations[x].target_queue = allqueue[x]
            container.stations[x].draw_queue(start)
        n = 30
        for i in range(n):
            t = (i + 1) / n
            for c in container.cars:
                c.render(t)
            for s in container.stations:
                s.render(t)
            time.sleep(1 / 30)
        for c in container.cars:
            c.update()
        start = 1
        step += 1
        # pause for click in window
