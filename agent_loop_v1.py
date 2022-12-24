# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:05:20 2022

@author: lukej
"""

# import torch
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as rnd

# e_t = state of env; h_t = hidden state of GRU (v_x,v_y); theta = [psi,(env params)]
    
# global parameters:

## global env parameters (add as required)
REWARD = 100
SIGMA_N = 0.1
SIGMA_T = 5
APERTURE = jnp.pi/3
NEURONS = 10
N_DOT_NO = 1
N_COLORS = [[255,0,0]]
key_dot = rnd.PRNGKey(0)

## global GRU parameters
M = 50
N = 3*NEURONS**2
key_init = rnd.PRNGKey(1)
key_eps = rnd.PRNGKey(2)
step = 0.001

# fnc definitions

def sigmoid(x):
  return 1 / (1 + jnp.exp(-x))

def gen_neurons_j(NEURONS,APERTURE):
    th_j = jnp.reshape(jnp.linspace(-APERTURE,APERTURE,NEURONS),(1,NEURONS))
    arr_j = jnp.tile(jnp.array(th_j),(NEURONS,1)) # varies in 'x'
    return arr_j
    
def gen_neurons_i(NEURONS,APERTURE):
    th_i = jnp.reshape(jnp.linspace(-APERTURE,APERTURE,NEURONS),(NEURONS,1))
    arr_i = jnp.tile(jnp.array(th_i),(1,NEURONS)) # varies in 'y'
    return arr_i

def create_dot():
    key_dot_, subkey_dot = rnd.split(key_dot)
    d_j = rnd.uniform(subkey_dot,minval=-jnp.pi,maxval=jnp.pi)
    key_dot_, subkey_dot = rnd.split(key_dot_)
    d_i = rnd.uniform(subkey_dot,minval=-jnp.pi,maxval=jnp.pi)
    #d = pygame.Rect(d_j, d_i, dot_sz, dot_sz)
    d_ji = jnp.array([d_j,d_i]).reshape((2,1))
    return d_ji

def gen_dots_n(n_dot_no):
    n_dots = {}
    for i in range(n_dot_no):
        dot_i = "ndot_%d" % i
        n_dots[dot_i] = create_dot()
    return n_dots

def f(dot,n_j,n_i,SIGMA_T,NEURONS):
    del_j = jnp.ones([1,NEURONS])*dot[0] - n_j[1,:]
    del_i = jnp.ones([NEURONS,1])*dot[1] - jnp.array(n_i[:,1]).reshape(NEURONS,1)
    f_ = jnp.exp((jnp.cos(jnp.tile(del_j,(NEURONS,1))) + jnp.cos(jnp.tile(del_i,(1,NEURONS)) - 2))/SIGMA_T**2)
    f_ = (f_.T).ravel().reshape([f_.size,1])
    return f_

def neuron_act(e_t_1,neurons_j,neurons_i,sigma_t,neurons,n_colors): # e_t_1: DOT OBJECTS (th_j,th_i coords),n_dots,n_colors,arr_j,arr_i,sigma_t
    # ignore p_dots, p_colors
    act_r = act_g = act_b = jnp.zeros([neurons_i.size,1])
    i = 0
    for dot in e_t_1.values():
        r,g,b = n_colors[i]
        f_ = f(dot,neurons_j,neurons_i,SIGMA_T,NEURONS)
        act_r += r/255 * f_.reshape([NEURONS**2,1])
        act_g += g/255 * f_.reshape([NEURONS**2,1])
        act_b += b/255 * f_.reshape([NEURONS**2,1])
        i += 1
    s_t = jnp.concatenate((act_r,act_g,act_b),axis=0)
    return s_t

def obj(s_t):
    R_t = jnp.sum(s_t)
    return R_t

def new_env(e_t_1,h_t,v_t,theta):
    # parameter for step size?
    for dot in e_t_1.values():
        #print(f'dot_1={dot:3.2f}')
        print('dot_1 =',dot)
        dot = dot.at[0].add(-v_t[0])
        dot = dot.at[1].add(-v_t[1])
        print('dot =',dot)
    return e_t_1

def single_step(e_t_1,h_t_1,theta): 
    
    # extract theta data from pytree
    W_f = theta["GRU_params"]["W_f"]
    U_f = theta["GRU_params"]["U_f"]
    b_f = theta["GRU_params"]["b_f"]
    W_h = theta["GRU_params"]["W_h"]
    U_h = theta["GRU_params"]["U_h"]
    b_h = theta["GRU_params"]["b_h"]
    C = theta["GRU_params"]["C"]
    
    NEURONS_J = theta["env_params"]["NEURONS_J"]
    NEURONS_I = theta["env_params"]["NEURONS_I"]
    SIGMA_T = theta["env_params"]["SIGMA_T"]
    SIGMA_N = theta["env_params"]["SIGMA_N"]
    NEURONS = theta["env_params"]["NEURONS"]
    N_COLORS = theta["env_params"]["N_COLORS"]
    
    # neuron activations from dots
    s_t = neuron_act(e_t_1,NEURONS_J,NEURONS_I,SIGMA_T,NEURONS,N_COLORS)
    
    # reward from neurons
    R_t = obj(s_t)
    print('R_t = ',R_t)
    
    # minimal GRU equations
    f_t = sigmoid( jnp.matmul(W_f,s_t) + jnp.matmul(U_f,h_t_1) + b_f)
    hhat_t = jnp.tanh( jnp.matmul(W_h,s_t) + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h)
    h_t = jnp.multiply((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t)
    
    # v_t = C*h_t + eps
    key_eps_, subkey_eps = rnd.split(key_eps)
    v_t = step * (jnp.matmul(C,h_t) + SIGMA_N*rnd.normal(subkey_eps,(2,1)))
    print('v_t = ',v_t)
    
    # new env data from sim dynamics; agent moves wrt dots, generating new dot locations
    e_t = new_env(e_t_1,h_t,v_t,theta)
    
    return (R_t,e_t,h_t)

def tot_reward(e0,h0,theta,it):
    
    R_tot = 0
    R_arr = np.zeros((it,1))
    dot_loc = []
    # R_tot = jax.lax.scan( single_step(.....) ), or loop: 
    e_t_1, h_t_1 = e0, h0
    for t in range(it):
        R_t,e_t,h_t = single_step(e_t_1,h_t_1,theta) 
        R_tot += R_t
        R_arr[t] = R_t # for debug, can't return
        dot_loc.append([x for x in e_t_1.values()][0]) # for debug, can't return
        e_t_1, h_t_1 = e_t, h_t
    return R_tot

# main routine

def main():
        
    epochs = 5
    it = 10
    
    # generate initial values
    ki = rnd.split(key_init,num=50)
    
    h0 = rnd.uniform(ki[0],(N,1))
    
    W_f0 = rnd.uniform(ki[1],(N,N))
    U_f0 = rnd.uniform(ki[2],(N,N))
    b_f0 = rnd.uniform(ki[3],(N,1))
    W_h0 = rnd.uniform(ki[4],(N,N))
    U_h0 = rnd.uniform(ki[5],(N,N))
    b_h0 = rnd.uniform(ki[6],(N,1))
    C0 = rnd.uniform(ki[7],(2,N))
    NEURONS_I = gen_neurons_i(NEURONS,APERTURE)
    NEURONS_J = gen_neurons_j(NEURONS,APERTURE)
    
    # generate theta tree
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
            "APERTURE"     : APERTURE,
            "NEURONS"      : NEURONS,
            "NEURONS_I"    : NEURONS_I,
            "NEURONS_J"    : NEURONS_J,
            "N_DOT_NO"     : N_DOT_NO,
            "N_COLORS"     : N_COLORS,
            "REWARD"       : REWARD,
            "SIGMA_N"      : SIGMA_N,
            "SIGMA_T"      : SIGMA_T
        }
             }
    
    for e in range(epochs):
    
        # generate target dot(s)
        e0 = gen_dots_n(theta["env_params"]["N_DOT_NO"]) 
            
        # run epoch
        R_tot = tot_reward(e0,h0,theta,it)
        #print(R_arr)
        #print(dot_loc)
        
        # get gradients
        G = jax.grad(tot_reward,argnums=2)
        print('G = ',G)    
        
        # update theta using tree_map
        
if __name__ == "__main__":
    main()
        
        
        

# use jax.lax.scan(single_step,...)
# jax treewrap (?)
# G = grad(tot_reward,argnum=2)
