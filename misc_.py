# # -*- coding: utf-8 -*-
# """
# Created on Thu Dec 15 16:44:34 2022

# @author: lukej
# """
# from datetime import datetime
# from matplotlib import pyplot
# from matplotlib.animation import FuncAnimation
# from random import randrange

# x_data = []

# figure = pyplot.figure()
# line, = pyplot.plot(x_data)

# def update(frame):
#     x_data.append(3)
#     #y_data.append(randrange(0, 100))
#     line.set_data(x_data)
#     figure.gca().relim()
#     figure.gca().autoscale_view()
#     return line,

# animation = FuncAnimation(figure, update, interval=200)

# pyplot.show()

import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np

def make_fig():
    plt.plot(y)  # I think you meant this

plt.ion()  # enable interactivity
fig = plt.figure()  # make a figure

# x = list()
y = list()

for i in range(1000):
    # temp_y = np.random.random()
    # x.append(i)
    y.append(np.random.random())  # or any arbitrary update to your figure's data
    i += 1
    drawnow(make_fig)