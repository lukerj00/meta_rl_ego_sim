import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax
# import jax._src.device_array
# import jax._src.abstract_arrays
# import jax
import numpy as np
from numpy import genfromtxt
import pickle
import pathlib
from pathlib import Path
import os
import sys

# import

R_arr_ = 'R_arr_01_02-1947'#'R_arr01_02-1829.pkl'
std_arr_ = 'std_arr_01_02-1947'#'std_arr01_02-1829.pkl'
path_ = str(Path(__file__).resolve().parents[1]) # C:\Users\lukej\Documents\MPhil\meta_rl_ego_sim
path_pkl = path_ + "\\pkl\\"
path_csv = path_ + "\\csv_plotter\\"
R_arr = genfromtxt(path_csv+R_arr_, delimiter=',')
print(R_arr.shape)
# R_arr_ = open(path_pkl + R_arr_path,'rb')
# R_arr = pickle.load(R_arr_)
std_arr = genfromtxt(path_csv+std_arr_, delimiter=',')
print(std_arr.shape)
# std_arr_ = open(path_pkl + std_arr_path,'rb')
# std_arr = pickle.load(std_arr_)
# colors_ = open(path_pkl + colors_path,'rb') # (remember to open as binary)
# SELECT = pickle.load(select_)
# print(type(R_arr))

EPOCHS = R_arr.shape[0]
IT = 25
VMAPS = 200
UPDATE = 0.001
SIGMA_A = 1
SIGMA_R = 0.5
SIGMA_N = 1.8

# plot
plt.figure()
plt.errorbar(jax.numpy.arange(EPOCHS),R_arr,yerr=std_arr/2,ecolor="black",elinewidth=0.5,capsize=1.5) # jnp.arange(EPOCHS),
plt.show(block=False)
title__ = f'epochs={EPOCHS}, it={IT}, vmaps={VMAPS}, update={UPDATE:.3f}, SIGMA_A={SIGMA_A:.1f}, SIGMA_R={SIGMA_R:.1f}, SIGMA_N={SIGMA_N:.1f}' # \n colors={jnp.array_str(COLORS[0][:]) + jnp.array_str(COLORS[1][:]) + jnp.array_str(COLORS[2][:]) + jnp.array_str(COLORS[3][:]) + jnp.array_str(COLORS[4][:])}'
plt.title(title__,fontsize=8)
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.show()