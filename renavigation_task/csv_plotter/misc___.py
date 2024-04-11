import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as np
from numpy import genfromtxt
import pickle
import pathlib
from pathlib import Path
import os
import sys
from datetime import datetime

key = rnd.PRNGKey(1)
e_t_1 = jnp.arange(6,dtype=jnp.float32).reshape(3,2)

print('e_t_1',e_t_1,e_t_1.shape)

# other e_t_1 values to relative positions:
print('e_t_1',e_t_1[0,:],e_t_1[1:,:])
del_ = e_t_1[1:,:] - e_t_1[0,:] # [N_DOTS-1,2] v
print('del_',del_)
theta_ = rnd.uniform(key,minval=-jnp.pi,maxval=jnp.pi)
# assemble matrix of relative positions (th_ab = th_b-th_a):
e_t_th = jnp.arctan(del_[:,1]/del_[:,0]) # [N_DOTS-1,]
print('e_t_th',e_t_th)
e_t_abs = jnp.linalg.norm(del_,axis=1) # [N_DOTS-1,]
print('e_t_abs',e_t_abs)

diag_transform = jnp.diag(e_t_abs)
print('diag',diag_transform)
theta_transform = jnp.vstack((jnp.cos(e_t_th+theta_),jnp.sin(e_t_th+theta_))).T
print('theta',theta_transform)
e_t_1 = e_t_1.at[0,:].set(rnd.uniform(key,shape=[2,],minval=-jnp.pi,maxval=jnp.pi,dtype=jnp.float32))
e_t_1 = e_t_1.at[1:,:].set(e_t_1[0,:] + jnp.matmul(diag_transform, theta_transform)) ### ASSEMBLE RIGHT THETA !
print('e_t_1',e_t_1,e_t_1.shape)