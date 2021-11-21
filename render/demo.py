import math
import turtle
from typing import List

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


class Station(turtle.Turtle):
    def __init__(self, index, n):
        turtle.Turtle.__init__(self)
        self.penup()
        self.shape("circle")
        self.color("yellow")
        self.shapesize(stretch_wid=5)
        self.speed(0)
        self.x = 300 * math.cos(index / n * 2 * math.pi)
        self.y = 300 * math.sin(index / n * 2 * math.pi)
        self.setposition(self.x, self.y)
        self.vehicles: List[Cars] = []

    def move_vehicle(self, target, n):
        for _ in range(n):
            self.vehicles[0].set_index(target)


class Cars(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.penup()
        self.shape("car.gif")
        self.speed(0)
        self.index = 0

    def init_pos(self, i):
        self.index = i
        self.setposition(stations[self.index].x, stations[self.index].y)
        stations[self.index].vehicles.append(self)
        self.speed(2)

    def set_index(self, i):
        old_station = self.index
        self.index = i
        self.goto(stations[self.index].x, stations[self.index].y)
        stations[old_station].vehicles.remove(self)
        stations[self.index].vehicles.append(self)


def moving():
    _action, _state = container.model.predict(container.state)
    action = VehicleAction(container.env, _action)
    container.state, reward, _, _ = container.env.step(_action)
    for i in range(3):
        for j in range(3):
            if i == j or action.motion[i][j] == 0:
                continue
            stations[i].move_vehicle(j, action.motion[i][j])


if __name__ == "__main__":
    container = Container()

    wn = turtle.Screen()
    wn.title("demo")
    wn.register_shape("car.gif")

    stations = [Station(i, 3) for i in range(3)]
    cars = [Cars() for i in range(3)]

    ind = 0
    for i in range(3):
        n = container.state[i]
        for j in range(n):
            cars[ind].init_pos(i)
            ind = ind + 1

    print([cars[i].index for i in range(3)])

    wn.listen()
    wn.onkeypress(moving, "w")

    while True:
        wn.update()

# import turtle
# import tkinter as tk
#
# def show_cat():
#     turtle.ht()
#     turtle.penup()
#     turtle.goto (15, 220)
#     turtle.color("black")
#     turtle.write("CAT", move=False, align="center", font=("Times New Roman", 120, "bold"))
#
# screen = turtle.Screen()
# screen.setup(800,800)
#
# canvas = screen.getcanvas()
#
# button = tk.Button(canvas.master, text="Click Me", command=show_cat)
# canvas.create_window(0, 0, window=button)
#
# #canvas.create_rectangle((100, 100, 700, 300))
#
# turtle.mainloop()

# import turtle
# import time
#
# wn = turtle.Screen()
# wn.title("Animation")
# wn.bgcolor("black")
#
# player = turtle.Turtle()
#
#
# wn.register_shape("car.gif")
#
#
#
# class Player(turtle.Turtle):
#     def __init__(self):
#         turtle.Turtle.__init__(self)
#         self.penup()
#         self.shape("car.gif")
#         self.color("green")
#         self.frame = 0
#         self.frames = ["car.gif", "circle"]
#
#     def animation(self):
#         self.frame +=1
#         if self.frame >= len(self.frames):
#             self.frame = 0
#         self.shape(self.frames[player.frame])
#         wn.ontimer(self.animation, 500)
#
#
# player = Player()
#
# player.animation()
#
# player2 = Player()
# player2.goto(-100,0)
# player2.animation()
#
#
#
# while True:
#     wn.update()
#
#
# wn.mainloop()


# import turtle
#
# wn = turtle.Screen()
# wn.tracer(0)
#
# paddle = turtle.Turtle()
# paddle.speed(0)
# paddle.shape("square")
# paddle.shapesize(stretch_wid=5,stretch_len=1)
# paddle.penup()
# paddle.goto(0,0)
#
# def paddle_up():
#     y = paddle.ycor()
#     y+=10
#     paddle.sety(y)
#
# wn.listen()
# wn.onkeypress(paddle_up,"w")
#
# while True:
#     wn.update()
