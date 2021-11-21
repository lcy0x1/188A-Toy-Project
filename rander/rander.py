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


import turtle

wn = turtle.Screen()
wn.title("demo")
wn.register_shape("car.gif")


class Station(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.penup()
        self.shape("circle")
        self.color("yellow")
        self.shapesize(stretch_wid=5)
        self.speed(0)

    def setloction(self, x, y):
        self.goto(x, y)


class Cars(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.penup()
        self.shape("car.gif")
        self.speed(2)


station1 = Station()
station1.setloction(-300, 100)

station2 = Station()
station2.setloction(300, 100)

station3 = Station()
station3.setloction(0, -300)

car1 = Cars()
car1.goto(-300, 160)
car1.index = 0


def moving():
    if car1.index == 0:
        car1.goto(300, 160)
        car1.index = 1
    elif car1.index == 1:
        car1.goto(0, -240)
        car1.index = 2
    elif car1.index == 2:
        car1.goto(-300, 160)
        car1.index = 0


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
