# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 11:46:39 2022

@author: lukej
"""

import torch
import jax
import jax.numpy as jnp
import numpy as np
from jax import random as rnd

def sigmoid(x):
  return 1 / (1 + jnp.exp(-x))

key_init = rnd.PRNGKey(0)
ki = rnd.split(key_init,num=50)
key_eps = rnd.PRNGKey(1)
key_dot = rnd.PRNGKey(2)

# e_t = state of env; h_t = hidden state of GRU (v_x,v_y); theta = [psi,(env params)]

# parameters
# theta: W_f, U_f, b_f, W_h, U_h, b_h, C(?)
# env params: no_dots, reward, sigma_n
    
# create theta parameters (pytree)

### env parameters, add as required
REWARD = 100
SIGMA_N = 1
SIGMA_T = 1
APERTURE = jnp.pi/3
NEURONS = 10
N_DOT_NO = 1

### GRU parameter initialisations
M = 50
N = 3*NEURONS**2
W_f0 = rnd.uniform(ki[0],(M,N))
U_f0 = rnd.uniform(ki[1],(M,N))
b_f0 = rnd.uniform(ki[2],(N,1))
W_h0 = rnd.uniform(ki[3],(M,N))
U_h0 = rnd.uniform(ki[4],(M,N))
b_h0 = rnd.uniform(ki[5],(N,1))
C0 = rnd.uniform(ki[6],(M,N))

key_dot, subkey_dot = rnd.split(key_dot)
d = rnd.uniform(subkey_dot,minval=0.0,maxval=2*jnp.pi)
print(d)

def create_dot():
    key_dot_ , subkey_dot = rnd.split(key_dot)
    d_j = rnd.uniform(subkey_dot,minval=0.0,maxval=2*jnp.pi)
    #(grid_sz/(2*np.pi))*
    d_i = rnd.uniform(subkey_dot,minval=0.0,maxval=2*jnp.pi)
    #d = pygame.Rect(d_j, d_i, dot_sz, dot_sz)
    return (d_j,d_i)

t = create_dot()
print(type(t))

theta = { "GRU_params" : {
        "W_f" : W_f0,
        "U_f" : U_f0,
        "b_f" : b_f0,
        "W_h" : W_h0,
        "U_h" : U_h0,
        "b_h" : b_h0,
        "C"   : C0
    },
          "env_params" : {
        "APERTURE" : APERTURE,
        "N_DOT_NO" : N_DOT_NO,
        "NEURONS"  : NEURONS,
        "REWARD"   : REWARD,
        "SIGMA_N"  : SIGMA_N,
        "SIGMA_T"  : SIGMA_T
    }
         }

b_ftest = theta["GRU_params"]["b_f"]
print(b_ftest.shape)

k = rnd.PRNGKey(0)
karr = rnd.split(k, num=2)
print((karr[1]))
ftest1 = rnd.uniform(k,(1,1))
ftest2 = rnd.uniform(k,(1,1))
print(ftest1, ftest2)

def f_(a):
    b = a + 2
    return a, b

print(f_(5)[0])

testdict = {"val": (3,4)}
print([x for x in testdict.values()][0])
print(type(testdict.values()))

print(float(np.asarray([x for x in testdict.values()])))
dot_loc = []
dot_loc.append([x for x in testdict.values()][0])
print(dot_loc)

M = 50.327926
print(f'M={M:3.2f}')

# state = TrainState.create(
        # apply_fn=model.apply,
        # params=variables['params'],
        # tx=tx)
        # grad_fn = jax.grad(make_loss_fn(state.apply_fn))
        # for batch in data:
        #     grads = grad_fn(state.params, batch)
        #     state = state.apply_gradients(grads=grads)