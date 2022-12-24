# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:18:06 2022

@author: lukej
"""

# import torch
# import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import jax.random as rnd
from jax.experimental import optimizers as jax_opt
import matplotlib.pyplot as plt
jax.config.update('jax_platform_name', 'cpu')

# fnc definitions
@jit
def sigmoid(x):
  return 1 / (1 + jnp.exp(-x))

def gen_neurons_j(NEURONS,APERTURE):
    th_j = jnp.reshape(jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32"),[1,NEURONS])
    return jnp.tile(jnp.array(th_j),[NEURONS,1])
    
def gen_neurons_i(NEURONS,APERTURE):
    th_i = jnp.reshape(jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32"),[NEURONS,1])
    return jnp.tile(jnp.array(th_i),[1,NEURONS])

@jit
def create_dot(KEY_DOT): # d_ji
    key_dot_, subkey_dot = rnd.split(KEY_DOT)
    p = jnp.pi
    d_j = rnd.uniform(subkey_dot,minval=-p,maxval=p,dtype="float32")
    key_dot__, subkey_dot = rnd.split(key_dot_)
    d_i = rnd.uniform(subkey_dot,minval=-p,maxval=p,dtype="float32")
    return jnp.array([d_j,d_i]).reshape([2,1])

def gen_dots_n(N_DOT_NO,KEY_DOT):
    # kd = rnd.split(KEY_DOT,num=N_DOT_NO)
    # n_dots = jnp.array()
    n_dots = {}
    kd = rnd.split(KEY_DOT,num=N_DOT_NO)
    for i in range(N_DOT_NO):
        dot_i = "ndot_%d" % i
        n_dots[dot_i] = create_dot(kd[i])
    return n_dots

def f(dot,n_j,n_i,SIGMA_T,NEURONS):
    del_j = dot[0] - jnp.array(n_j[0,:]).reshape([1,NEURONS])
    del_i = dot[1] - jnp.array(n_i[:,0]).reshape([NEURONS,1])
    f_ = jnp.exp((jnp.cos(jnp.tile(del_j,(NEURONS,1))) + jnp.cos(jnp.tile(del_i,(1,NEURONS))) - 2)/SIGMA_T**2)
    return f_.ravel().reshape([f_.size,1])

def neuron_act(e_t_1,n_j,n_i,SIGMA_T,NEURONS,N_COLORS): # s_t
    act_r = act_g = act_b = jnp.zeros([n_i.size,1],dtype="float32")
    for i,(k,dot) in enumerate(e_t_1.items()):
        r,g,b = N_COLORS[i]
        f_ = f(dot,n_j,n_i,SIGMA_T,NEURONS)
        act_r += r/255 * f_.reshape([NEURONS**2,1])
        act_g += g/255 * f_.reshape([NEURONS**2,1])
        act_b += b/255 * f_.reshape([NEURONS**2,1])
    return jnp.concatenate((act_r,act_g,act_b),axis=0)

#@jit
def obj(s_t,NEURONS): # R_t
    sz = NEURONS**2
    ind = jnp.int32(jnp.floor(sz/2))
    return jnp.reshape(s_t[ind] + s_t[ind+sz] + s_t[ind+2*sz],[])

def new_env(e_t_1,v_t): # e_t
    return {k:v - v_t for k,v in e_t_1.items()}
    
def single_step(e_t_1,h_t_1,theta,eps_t): 
    
    # extract data from theta
    W_f = theta["GRU_params"]["W_f"]
    U_f = theta["GRU_params"]["U_f"]
    b_f = theta["GRU_params"]["b_f"]
    W_h = theta["GRU_params"]["W_h"]
    U_h = theta["GRU_params"]["U_h"]
    b_h = theta["GRU_params"]["b_h"]
    C = theta["GRU_params"]["C"]
    
    NEURONS_J = theta["ENV_params"]["NEURONS_J"]
    NEURONS_I = theta["ENV_params"]["NEURONS_I"]
    SIGMA_T = theta["ENV_params"]["SIGMA_T"]
    SIGMA_N = theta["ENV_params"]["SIGMA_N"]
    N_COLORS = theta["ENV_params"]["N_COLORS"]
    STEP = theta["ENV_params"]["STEP"]
    NEURONS = theta["ENV_params"]["NEURONS"]
    
    # neuron activations
    s_t = neuron_act(e_t_1,NEURONS_J,NEURONS_I,SIGMA_T,NEURONS,N_COLORS)
    
    # reward from neurons
    R_t = obj(s_t,NEURONS)
    
    # minimal GRU equations
    f_t = sigmoid( jnp.matmul(W_f,s_t) + jnp.matmul(U_f,h_t_1) + b_f)
    hhat_t = jnp.tanh( jnp.matmul(W_h,s_t) + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t)
    
    # v_t = C*h_t + eps
    v_t = STEP * (jnp.matmul(C,h_t) + SIGMA_N * eps_t) #rnd.normal(subkey_eps,(2,1))) #key_eps_, subkey_eps = rnd.split(key_eps)
    #print('v_t = ',v_t)
    
    # new env from sim dynamics (agent fixed, dots move)
    e_t = new_env(e_t_1,v_t)
    
    return (R_t,e_t,h_t)

def tot_reward(e0,h0,theta):
    
    IT = theta["ENV_params"]["IT"]
    KEY_EPS = theta["ENV_params"]["KEY_EPS"]
    R_tot = jnp.float32(0)
    e_t_1, h_t_1 = e0, h0
    
    # R_tot = jax.lax.scan( single_step(.....) ), or loop: 
    for t in range(IT):
        eps = rnd.normal(KEY_EPS,shape=[IT,2],dtype="float32")
        eps_t = eps[t,:].reshape([2,1])
        R_t, e_t, h_t = single_step(e_t_1,h_t_1,theta,eps_t) 
        e_t_1, h_t_1 = e_t, h_t
        R_tot += R_t
        # dot_loc.append([x for x in e_t_1.values()][0]) # for debug (can't return)
    return R_tot

# main routine
def main():
    
    # ENV parameters
    SIGMA_N = jnp.float32(1)
    SIGMA_T = jnp.float32(5)
    STEP = jnp.float32(0.0001)
    APERTURE = jnp.pi/3
    N_COLORS = jnp.float32([[255,0,0]])
    KEY_DOT = rnd.PRNGKey(0)
    NEURONS = 11
    N_DOT_NO = 1
    
    # GRU parameters
    N = 3*NEURONS**2
    KEY_INIT = rnd.PRNGKey(1)
    KEY_EPS = rnd.PRNGKey(2)
    
    # main() params
    EPOCHS = 10
    IT = 10
    INIT = jnp.float32(0.3)
    UPDATE = jnp.float32(1)
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
              "ENV_params" : {
            "APERTURE"     : jax.lax.stop_gradient(APERTURE),
            "NEURONS_I"    : jax.lax.stop_gradient(NEURONS_I),
            "NEURONS_J"    : jax.lax.stop_gradient(NEURONS_J),
            "N_COLORS"     : jax.lax.stop_gradient(N_COLORS),
            "SIGMA_N"      : jax.lax.stop_gradient(SIGMA_N),
            "SIGMA_T"      : jax.lax.stop_gradient(SIGMA_T),
            "STEP"         : jax.lax.stop_gradient(STEP),
            "NEURONS"      : jax.lax.stop_gradient(NEURONS),
            "N_DOT_NO"     : jax.lax.stop_gradient(N_DOT_NO),
            "KEY_EPS"      : jax.lax.stop_gradient(KEY_EPS),
            "IT"           : jax.lax.stop_gradient(IT)            
        }
             }
    
    # generate dot keys
    ke = rnd.split(KEY_DOT,num=EPOCHS)
    
    for e in range(EPOCHS): # or use jax.lax.scan(single_step,...)
    
        print('epoch: ',e)
    
        # generate target dot(s)
        e0 = gen_dots_n(theta["ENV_params"]["N_DOT_NO"],ke[e])
            
        # run epoch, get total reward and gradients 
        R_tot, grads = jax.value_and_grad(tot_reward,argnums=2,allow_int=True)(e0,h0,theta)
        R_arr = R_arr.at[e].set(R_tot)
        print(theta["GRU_params"]["b_h"])
        print('R_tot =', R_arr[e])

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