# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:18:06 2022

@author: lukej
"""

# import torch
# import numpy as np
import jax
import jax.numpy as jnp
from jax import random as rnd
import matplotlib.pyplot as plt
jax.config.update('jax_platform_name', 'cpu')

# fnc definitions
def sigmoid(x):
  return 1 / (1 + jnp.exp(-x))

def gen_neurons_j(NEURONS,APERTURE):
    th_j = jnp.reshape(jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32"),([1,NEURONS]))
    arr_j = jnp.tile(jnp.array(th_j),(NEURONS,1))
    return arr_j
    
def gen_neurons_i(NEURONS,APERTURE):
    th_i = jnp.reshape(jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32"),([NEURONS,1]))
    arr_i = jnp.tile(jnp.array(th_i),(1,NEURONS))
    return arr_i

def create_dot(key_dot):
    key_dot_, subkey_dot = rnd.split(key_dot)
    p = jnp.float32(jnp.pi)
    d_j = rnd.uniform(subkey_dot,minval=-p,maxval=p,dtype="float32")
    key_dot__, subkey_dot = rnd.split(key_dot_)
    d_i = rnd.uniform(subkey_dot,minval=-p,maxval=p,dtype="float32")
    d_ji = jnp.array([d_j,d_i],dtype="float32").reshape([2,1])
    return d_ji

def gen_dots_n(n_dot_no,key_dot):
    n_dots = {}
    kd = rnd.split(key_dot,num=n_dot_no)
    for i in range(n_dot_no):
        dot_i = "ndot_%d" % i
        n_dots[dot_i] = create_dot(kd[i])
    return n_dots

def f(dot,n_j,n_i,sigma_t,neurons):
    del_j = jnp.ones([1,neurons],dtype="float32")*dot[0] - jnp.array(n_j[0,:]).reshape([1,neurons])
    del_i = jnp.ones([neurons,1],dtype="float32")*dot[1] - jnp.array(n_i[:,0]).reshape([neurons,1])
    f_ = jnp.exp(((jnp.cos(jnp.tile(del_j,(neurons,1))) + jnp.cos(jnp.tile(del_i,(1,neurons))) - jnp.float32(2))/sigma_t**2))
    f_ = f_.ravel().reshape([f_.size,1])
    return f_

def neuron_act(e_t_1,neurons_j,neurons_i,sigma_t,neurons,n_colors): # e_t_1: DOT OBJECTS (th_j,th_i coords),n_dots,n_colors,arr_j,arr_i,sigma_t # ignore p_dots, p_colors
    act_r = act_g = act_b = jnp.zeros([neurons_i.size,1],dtype="float32")
    for i,(k,dot) in enumerate(e_t_1.items()):
        r,g,b = n_colors[i]
        f_ = f(dot,neurons_j,neurons_i,sigma_t,neurons)
        act_r += jnp.float32(r/255) * f_.reshape([neurons**2,1])
        act_g += jnp.float32(g/255) * f_.reshape([neurons**2,1])
        act_b += jnp.float32(b/255) * f_.reshape([neurons**2,1])
    s_t = jnp.concatenate((act_r,act_g,act_b),axis=0)
    return s_t

def obj(s_t):
    R_t = jnp.sum(s_t)
    return R_t

def new_env(e_t_1,v_t):
    e_t_1 = {k:v - v_t for k,v in e_t_1.items()}
    # for i,(k,dot) in enumerate(e_t_1.items()): # use tree map instead
    #     # print(f'dot_1={dot:3.2f}')
    #     # print('dot[0] = ',dot[0])
    #     # print('dot[1] = ',dot[1])        
    #     dot_ = e_t_1[k].at[:].add(-v_t)
    #     e_t_1[k] = dot_
    return e_t_1
    # return jax.tree_util.tree_map(alpha: e, v: e = e + v, e_t_1, v_t)
    
def single_step(e_t_1,h_t_1,theta,psi,eps_t): 
    
    # extract data from theta
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
    N_COLORS = theta["env_params"]["N_COLORS"]
    STEP = theta["env_params"]["STEP"]
    
    NEURONS = psi["NEURONS"]
    
    # neuron activations
    s_t = neuron_act(e_t_1,NEURONS_J,NEURONS_I,SIGMA_T,NEURONS,N_COLORS)
    
    # reward from neurons
    R_t = obj(s_t)
    
    # minimal GRU equations
    f_t = sigmoid( jnp.matmul(W_f,s_t) + jnp.matmul(U_f,h_t_1) + b_f)
    hhat_t = jnp.tanh( jnp.matmul(W_h,s_t) + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply((jnp.float32(1)-f_t),h_t_1) + jnp.multiply(f_t,hhat_t)
    
    # v_t = C*h_t + eps
    v_t = STEP * (jnp.matmul(C,h_t) + SIGMA_N * eps_t) #rnd.normal(subkey_eps,(2,1))) #key_eps_, subkey_eps = rnd.split(key_eps)
    # print('v_t = ',v_t)
    
    # new env from sim dynamics (agent fixed, dots move)
    e_t = new_env(e_t_1,v_t)
    
    return (R_t,e_t,h_t)

def tot_reward(e0,h0,theta,psi):
    
    it = psi["IT"]
    key_eps = psi["KEY_EPS"]
    R_tot = jnp.float32(0)
    e_t_1, h_t_1 = e0, h0
    
    # R_tot = jax.lax.scan( single_step(.....) ), or loop: 
    for t in range(it):
        eps = rnd.normal(key_eps,shape=(it,2),dtype="float32")
        eps_t = eps[t,:].reshape([2,1])
        R_t,e_t,h_t = single_step(e_t_1,h_t_1,theta,psi,eps_t) 
        e_t_1, h_t_1 = e_t, h_t
        R_tot += R_t
        # dot_loc.append([x for x in e_t_1.values()][0]) # for debug (can't return)
    return R_tot

# main routine
def main():
    
    # env parameters
    SIGMA_N = jnp.float32(0.01)
    SIGMA_T = jnp.float32(5)
    STEP = jnp.float32(0.0001)
    APERTURE = jnp.float32(jnp.pi/3)
    N_COLORS = jnp.float32([[255,0,0]])
    KEY_DOT = rnd.PRNGKey(0)
    NEURONS = 11
    N_DOT_NO = 1
    
    # GRU parameters
    N = 3*NEURONS**2
    KEY_INIT = rnd.PRNGKey(1)
    KEY_EPS = rnd.PRNGKey(2)
    
    # main() params
    EPOCHS = 20
    IT = 10
    INIT = jnp.float32(0.2)
    UPDATE = jnp.float32(0.001)
    R_arr = jnp.zeros(EPOCHS)
    
    # generate initial values
    ki = rnd.split(KEY_INIT,num=50)
    h0 = rnd.normal(ki[0],(N,1),dtype="float32")
    W_f0 = (INIT/N*N)*rnd.normal(ki[1],(N,N),dtype="float32")
    U_f0 = (INIT/N*N)*rnd.normal(ki[2],(N,N),dtype="float32")
    b_f0 = (INIT/N)*rnd.normal(ki[3],(N,1),dtype="float32")
    W_h0 = (INIT/N*N)*rnd.normal(ki[4],(N,N),dtype="float32")
    U_h0 = (INIT/N*N)*rnd.normal(ki[5],(N,N),dtype="float32")
    b_h0 = (INIT/N)*rnd.normal(ki[6],(N,1),dtype="float32")
    C0 = (INIT/2*N)*rnd.normal(ki[7],(2,N),dtype="float32")
    NEURONS_I = gen_neurons_i(NEURONS,APERTURE)
    NEURONS_J = gen_neurons_j(NEURONS,APERTURE)
    
    # generate theta pytree
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
            "NEURONS_I"    : NEURONS_I,
            "NEURONS_J"    : NEURONS_J,
            "N_COLORS"     : N_COLORS,
            "SIGMA_N"      : SIGMA_N,
            "SIGMA_T"      : SIGMA_T,
            "STEP"         : STEP
        }
             }
    
    # (separate pytree for ints)
    psi = { 
            "NEURONS"      : NEURONS,
            "N_DOT_NO"     : N_DOT_NO,
            "KEY_EPS"      : KEY_EPS,
            "IT"           : IT
           }
    
    # generate dot(s) key
    ke = rnd.split(KEY_DOT, num = EPOCHS)
    
    for e in range(EPOCHS): # or use jax.lax.scan(single_step,...)
    
        print('epoch: ',e)
    
        # generate target dots
        e0 = gen_dots_n(psi["N_DOT_NO"],ke[e])
            
        # run epoch; get total reward and gradients 
        R_tot, grads = jax.value_and_grad(tot_reward,argnums=2)(e0,h0,theta,psi) # ,allow_int=True
        R_arr = R_arr.at[e].set(R_tot)
        print('R_tot =', R_arr[e])
        print(grads["GRU_params"]["W_h"])

        # just reward
        # R_tot = tot_reward(e0, h0, theta, psi)
        # print('R_tot =', R_tot)
        
        # just gradients
        #grads = jax.grad(tot_reward,argnums=2)(e0,h0,theta,psi)
        #print(grads["GRU_params"]["b_f"])
        
        # update theta (tree_map?)
        # use optax (jax optimisation framework - ADAM)
        theta["GRU_params"] = jax.tree_util.tree_map(lambda t, g: t + UPDATE * g, theta["GRU_params"], grads["GRU_params"])
    
    print(R_arr)
    plt.plot(R_arr)
    #title_ = "NO UPDATE (control) "
    title_ = "epochs = " + str(EPOCHS) + ", it = " + str(IT) + ", update = " + str(UPDATE) # 
    plt.title(title_)
    
if __name__ == "__main__":
    main() 