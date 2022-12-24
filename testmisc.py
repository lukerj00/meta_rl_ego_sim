# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:42:30 2022

@author: lukej
"""
import pygame
import numpy as np
import matplotlib.pyplot as plt

R = 5
num_points = 10000

np.random.seed(1)
theta = np.random.uniform(0,2*np.pi, num_points)
radius = np.random.uniform(0,300**2, num_points) ** 0.5

x = radius * np.cos(theta)
y = radius * np.sin(theta)

# visualize the points:
plt.scatter(x,y, s=1)

GRID_SZ = 600
POS_DOT_NO = 3
NEG_DOT_NO = 1
DOT_SZ = 5
# DOT_COORD = GRID_SZ*np.random.random([DOT_NO, 2])
DOT_CLR = ['red', 'blue', 'green','magenta']
BACKGRND_CLR = (255, 255, 255)
AGENT_CLR = (5,5,5)
FPS = 30
VEL = 5
AGENT_SZ = 15
POS_VALUE = 100
NEG_VALUE = -100


agent = pygame.Rect(GRID_SZ/2,GRID_SZ/2,AGENT_SZ,AGENT_SZ)

# dot_str = ''
# for i in range(DOT_NO):
#         exec('dot_' + str(i) + ' = pygame.Rect(DOT_COORD[i, 0], DOT_COORD[i, 1], DOT_SZ, DOT_SZ)')
#         dot_str += 'dot_' + str(i) + ', '

def create_dot(grid_sz,dot_sz,value):
    dot = pygame.Rect(grid_sz*np.random.random(), grid_sz*np.random.random(), dot_sz, dot_sz)
    # dot.value = value
    return(dot)

dots = {}
for i in range(POS_DOT_NO):
    dot_i = "posdot_%d" % i
    dots[dot_i] = create_dot(GRID_SZ, DOT_SZ, POS_VALUE)
    
for i in range(NEG_DOT_NO):
    dot_i = "negdot_%d" % i
    dots[dot_i] = create_dot(GRID_SZ, DOT_SZ, NEG_VALUE)
    
    print(len(dots))
    print(dots)
    
    i = 0
    for key in dots:
        print(type(dots[key]))
        i += 1
for dot in dots.values():
    print(type(agent))
    print(type(dot))
    
r = np.random.uniform(0, 300) ** 0.5
x = r*np.cos(np.random.uniform(0, 2*np.pi))
y = r*np.sin(np.random.uniform(0, 2*np.pi))
print(np.random.uniform(0, 300) ** 0.5)
print(np.random.uniform(0, 2*np.pi))
print(r,x,y)
