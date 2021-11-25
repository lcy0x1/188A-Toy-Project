import math
from typing import List
from graphics import *

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from gym_vehicle.envs.vehicle_env import VehicleAction

samestaion = 0
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
        index = 0
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
        self.Queue = []
        self.newQueue = []
        self.Q1 = Image(Point((self.x+20), (self.y-20)), "./gostation1.png")
        self.Q2 = Image(Point((self.x+40), (self.y-20)), "./gostation1.png")
        self.Q3 = Image(Point((self.x+60), (self.y-20)), "./gostation1.png")
        self.Q4 = Image(Point((self.x + 20), (self.y - 20)), "./gostation2.png")
        self.Q5 = Image(Point((self.x + 40), (self.y - 20)), "./gostation2.png")
        self.Q6 = Image(Point((self.x + 60), (self.y - 20)), "./gostation2.png")
        self.Q7 = Image(Point((self.x + 20), (self.y - 20)), "./gostation3.png")
        self.Q8 = Image(Point((self.x + 40), (self.y - 20)), "./gostation3.png")
        self.Q9 = Image(Point((self.x + 60), (self.y - 20)), "./gostation3.png")
        self.nowindex = 0
    def move_vehicle(self, target, n):
        for _ in range(n):
            self.vehicles[0].set_index(target)
    def drawQueue(self,start):
        if start == 0:
            if self.nowindex == 1:
                if self.newQueue[1] == 2:
                    self.Q4.draw(win)
                    self.Q5.draw(win)
                    if self.newQueue[2] == 1:
                        self.Q9.draw(win)
                elif self.newQueue[1] == 1:
                    self.Q4.draw(win)
                    if self.newQueue[2] == 2:
                        self.Q8.draw(win)
                        self.Q9.draw(win)
                    elif self.newQueue[2] == 1:
                        self.Q8.draw(win)
                else:
                    if self.newQueue[2] == 2:
                        self.Q7.draw(win)
                        self.Q8.draw(win)
                    elif self.newQueue[2] ==1:
                        self.Q7.draw(win)
            elif self.nowindex == 2:
                if self.newQueue[0] == 2:
                    self.Q1.draw(win)
                    self.Q2.draw(win)
                    if self.newQueue[2] == 1:
                        self.Q9.draw(win)
                elif self.newQueue[0] == 1:
                    self.Q1.draw(win)
                    if self.newQueue[2] == 2:
                        self.Q8.draw(win)
                        self.Q9.draw(win)
                    elif self.newQueue[2] == 1:
                        self.Q8.draw(win)
                else:
                    if self.newQueue[2] == 2:
                        self.Q7.draw(win)
                        self.Q8.draw(win)
                    elif self.newQueue[2] == 1:
                        self.Q7.draw(win)
            elif self.nowindex == 3:
                if self.newQueue[0] == 2:
                    self.Q1.draw(win)
                    self.Q2.draw(win)
                    if self.newQueue[1] == 1:
                        self.Q6.draw(win)
                elif self.newQueue[0] == 1:
                    self.Q1.draw(win)
                    if self.newQueue[1] == 2:
                        self.Q5.draw(win)
                        self.Q6.draw(win)
                    elif self.newQueue[1] == 1:
                        self.Q5.draw(win)
                else:
                    if self.newQueue[1] == 2:
                        self.Q4.draw(win)
                        self.Q5.draw(win)
                    elif self.newQueue[1] == 1:
                        self.Q4.draw(win)
        elif start == 1:
            if self.nowindex == 1:
                if self.Queue[1] == 2:
                    self.Q4.undraw()
                    self.Q5.undraw()
                    if self.Queue[2] == 1:
                        self.Q9.undraw()
                elif self.Queue[1] == 1:
                    self.Q4.undraw()
                    if self.Queue[2] == 2:
                        self.Q8.undraw()
                        self.Q9.undraw()
                    elif self.Queue[2] == 1:
                        self.Q8.undraw()
                else:
                    if self.Queue[2] == 2:
                        self.Q7.undraw()
                        self.Q8.undraw()
                    elif self.Queue[2] ==1:
                        self.Q7.undraw()
            elif self.nowindex == 2:
                if self.Queue[0] == 2:
                    self.Q1.undraw()
                    self.Q2.undraw()
                    if self.Queue[2] == 1:
                        self.Q9.undraw()
                elif self.Queue[0] == 1:
                    self.Q1.undraw()
                    if self.Queue[2] == 2:
                        self.Q8.undraw()
                        self.Q9.undraw()
                    elif self.Queue[2] == 1:
                        self.Q8.undraw()
                else:
                    if self.Queue[2] == 2:
                        self.Q7.undraw()
                        self.Q8.undraw()
                    elif self.Queue[2] == 1:
                        self.Q7.undraw()
            elif self.nowindex == 3:
                if self.Queue[0] == 2:
                    self.Q1.undraw()
                    self.Q2.undraw()
                    if self.Queue[1] == 1:
                        self.Q6.undraw()
                elif self.Queue[0] == 1:
                    self.Q1.undraw()
                    if self.Queue[1] == 2:
                        self.Q5.undraw()
                        self.Q6.undraw()
                    elif self.Queue[1] == 1:
                        self.Q5.undraw()
                else:
                    if self.Queue[1] == 2:
                        self.Q4.undraw()
                        self.Q5.undraw()
                    elif self.Queue[1] == 1:
                        self.Q4.undraw()

            if self.nowindex == 1:
                if self.newQueue[1] == 2:
                    self.Q4.draw(win)
                    self.Q5.draw(win)
                    if self.newQueue[2] == 1:
                        self.Q9.draw(win)
                elif self.newQueue[1] == 1:
                    self.Q4.draw(win)
                    if self.newQueue[2] == 2:
                        self.Q8.draw(win)
                        self.Q9.draw(win)
                    elif self.newQueue[2] == 1:
                        self.Q8.draw(win)
                else:
                    if self.newQueue[2] == 2:
                        self.Q7.draw(win)
                        self.Q8.draw(win)
                    elif self.newQueue[2] ==1:
                        self.Q7.draw(win)
            elif self.nowindex == 2:
                if self.newQueue[0] == 2:
                    self.Q1.draw(win)
                    self.Q2.draw(win)
                    if self.newQueue[2] == 1:
                        self.Q9.draw(win)
                elif self.newQueue[0] == 1:
                    self.Q1.draw(win)
                    if self.newQueue[2] == 2:
                        self.Q8.draw(win)
                        self.Q9.draw(win)
                    elif self.newQueue[2] == 1:
                        self.Q8.draw(win)
                else:
                    if self.newQueue[2] == 2:
                        self.Q7.draw(win)
                        self.Q8.draw(win)
                    elif self.newQueue[2] == 1:
                        self.Q7.draw(win)
            elif self.nowindex == 3:
                if self.newQueue[0] == 2:
                    self.Q1.draw(win)
                    self.Q2.draw(win)
                    if self.newQueue[1] == 1:
                        self.Q6.draw(win)
                elif self.newQueue[0] == 1:
                    self.Q1.draw(win)
                    if self.newQueue[1] == 2:
                        self.Q5.draw(win)
                        self.Q6.draw(win)
                    elif self.newQueue[1] == 1:
                        self.Q5.draw(win)
                else:
                    if self.newQueue[1] == 2:
                        self.Q4.draw(win)
                        self.Q5.draw(win)
                    elif self.newQueue[1] == 1:
                        self.Q4.draw(win)
        self.Queue = self.newQueue





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
        stations[self.index].vehicles.append(self)
        if len(stations[self.index].vehicles) == 1:
            self.x = stations[self.index].x
            self.y = stations[self.index].y + 50
        elif len(stations[self.index].vehicles) == 2:
            self.x = stations[self.index].x
            self.y = stations[self.index].y + 70
        else:
            self.x = stations[self.index].x
            self.y = stations[self.index].y + 30

        self.target_x = self.x
        self.target_y = self.y


    def set_index(self, i):
        old_station = self.index
        self.index = i
        stations[old_station].vehicles.remove(self)
        stations[self.index].vehicles.append(self)
        if len(stations[self.index].vehicles) == 2:
            self.target_x = stations[self.index].x
            self.target_y = stations[self.index].y + 50
        elif len(stations[self.index].vehicles) == 3:
            self.target_x = stations[self.index].x
            self.target_y = stations[self.index].y + 70
        else:
            self.target_x = stations[self.index].x
            self.target_y = stations[self.index].y+ 30


    def render(self, time):
        nx = (self.target_x - self.x) * time + self.x
        ny = (self.target_y - self.y) * time + self.y
        self.sprite.move(nx - self.current_x, ny - self.current_y)
        self.current_x = nx
        self.current_y = ny

    def update(self):
        self.x = self.target_x
        self.y = self.target_y


def moving():
    _action, _state = container.model.predict(container.state)

    action = VehicleAction(container.env, _action)
    container.state, reward, _, _ = container.env.step(_action)
    a = container.get_queue(container.state)
    print(container.get_queue(container.state))

    for i in range(3):
        for j in range(3):
            if i == j or action.motion[i][j] == 0:
                continue
            stations[i].move_vehicle(j, action.motion[i][j])






if __name__ == "__main__":
    container = Container()
    start = 0
    win = GraphWin("My Circle", 800, 800)

    stations = [Station(i, 3) for i in range(3)]
    stations[0].nowindex = 1
    stations[1].nowindex = 2
    stations[2].nowindex = 3
    message1 = Text(Point(stations[0].x,stations[0].y),"1")
    message2 = Text(Point(stations[1].x,stations[1].y),"2")
    message3 = Text(Point(stations[2].x,stations[2].y),"3")
    message1.draw(win)
    message2.draw(win)
    message3.draw(win)
    aline = Line(Point(stations[0].x-30, stations[0].y+18), Point(stations[1].x+30,stations[1].y-18))
    aline.setArrow("both")
    bline = Line(Point(stations[1].x, stations[1].y-25), Point(stations[2].x,stations[2].y+25))
    bline.setArrow("both")
    cline = Line(Point(stations[0].x-30, stations[0].y-18), Point(stations[2].x+30, stations[2].y+18))
    cline.setArrow("both")
    aline.draw(win)
    bline.draw(win)
    cline.draw(win)
    cars = [Cars() for i in range(3)]


    ind = 0
    for i in range(3):
        n = container.state[i]
        for j in range(n):
            cars[ind].init_pos(i)
            ind = ind + 1

    while True:

        moving()
        n = 30
        allqueue = container.get_queue(container.state)


        for x in range(3):

            stations[x].newQueue = allqueue[x]
            stations[x].drawQueue(start)

        start = 1
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
