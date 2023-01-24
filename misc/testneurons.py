# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:54:31 2022

@author: lukej
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# create 2d grid of polar coords
## define reeceptive field for single point
## define points corresponding to each neurons
## use g's formula to detremine firing (intensity) of neuron

N = 11
RF_RAD = 0.5
MARK_SZ = 0.5
N_TCKS = 5
SPHERE_RAD = 1

i = j = np.linspace(0,N,N)
th_i = np.reshape((2*np.pi/N)*i,(1,i.size))
print(th_i)
th_j = np.reshape((2*np.pi/N)*j,(1,j.size))
arr_i = np.tile(np.array(th_i).transpose(),(1,N)) # varies in y
arr_j = np.tile(np.array(th_j),(N,1)) # varies in x
r = (arr_j.T).ravel()

del_1 = np.zeros([1,N])
del_2 = np.zeros([N,1]).reshape([N,1])
sigma = 1
f_test = np.exp((np.cos(np.tile(del_1,(N,1))) + np.cos(np.tile(del_2,(1,N))) - 2)/sigma**2)
f_r =(f_test.T).ravel()

# print(r)
print(np.floor(N/2))
print(th_i[0,np.int32(np.floor(N/2))])
fig,ax = plt.subplots(1)
ax.set_aspect('equal')
plt.scatter(arr_j,arr_i,s=MARK_SZ,color='black')
plt.scatter(th_i[0,np.int32(np.floor(N/2))],th_j[0,np.int32(np.floor(N/2))],s=MARK_SZ*20,marker='x',color='black')
plt.xticks(ticks=np.linspace(0,2*np.pi,N_TCKS),labels=['-\u03C0','-\u03C0 / 2','0','\u03C0 / 2','\u03C0'])
plt.yticks(ticks=np.linspace(0,2*np.pi,N_TCKS),labels=['-\u03C0','-\u03C0 / 2','0','\u03C0 / 2','\u03C0'])
plt.xlabel('\u03B8_1 (j)') #\u2081
plt.ylabel('\u03B8_2 (i)') #\u2082
# for i in arr_i[:,1]:
    # for j in arr_j[1,:]:
        # ax.add_patch(Circle((i,j),RF_RAD,edgecolor='blue',fill=False))
plt.show()

# make function to return neuron activations r, g, b
# r_ij = sum over all dots: redness * f(angle_distance(dot_i),angle_distance(dot_j))

#call r_test on all dots, neuron array
def r_test(p_dots,n_dots,p_colors,arr_j,arr_i,sigma):
    r_arr = []
    for dot in p_dots.values():
        r,g,b = p_colors[dot]
        r_arr += r/255 * f(dot,arr_j,arr_i,sigma)
    return r_arr

#call f on particular dot, returning values to each corresponding neuron in array
def f(dot,arr_j,arr_i,sigma):
    th_1 = (dot.x-GRID_RAD)*(THETA_SEG/GRID_RAD)
    #(np.pi/GRID_RAD)/SPHERE_RAD
    th_2 = (GRID_RAD-dot.y)*(THETA_SEG/GRID_RAD)
    #(np.pi/GRID_RAD)/SPHERE_RAD
    del_1 = th_1*np.ones(1,N) - np.array(arr_j[1,:])
    del_2 = th_2*np.ones(N,1) - np.array(arr_i[:,1])
    # need to edit; should be N x N
    f_ = np.exp((np.cos(np.tile(del_1,(N,1))) + np.cos(np.tile(del_2,(1,N))) - 2)/sigma^2)
    return f_


