# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:41:08 2022

@author: lukej
"""

# import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import jax.random as rnd
import optax
import matplotlib.pyplot as plt
jax.config.update('jax_platform_name', 'cpu')

# fnc definitions
@jit
def sigmoid(x):
  return 1 / (1 + jnp.exp(-x))

def gen_neurons_j(NEURONS,APERTURE):
    return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32")
gen_neurons_j = jit(gen_neurons_j,static_argnums=(0,1))

def gen_neurons_i(NEURONS,APERTURE):
    return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32")
gen_neurons_i = jit(gen_neurons_i,static_argnums=(0,1))

def create_dots(N_DOTS,KEY_DOT): # [d_j,d_i]
    return rnd.uniform(KEY_DOT,shape=[N_DOTS,2],minval=-jnp.pi,maxval=jnp.pi,dtype="float32")
create_dots = jit(create_dots,static_argnums=(0,))

@jit
def neuron_act_(e_t_1,n_j,n_i,SIGMA_T,N_COLORS):
    NEURONS = jnp.size(n_j)
    N_DOTS = jnp.size(e_t_1,0)
    n_j_tot = jnp.tile(jnp.array(n_j).reshape([1,NEURONS,1]),[NEURONS,1,N_DOTS])
    n_i_tot = jnp.tile(jnp.array(n_i).reshape([NEURONS,1,1]),[1,NEURONS,N_DOTS])
    del_j = e_t_1[:,0] - n_j_tot
    del_i = e_t_1[:,1] - n_i_tot
    act = jnp.repeat(jnp.exp((jnp.cos(del_j) + jnp.cos(del_i) - 2)/SIGMA_T**2),3,axis=2)
    act_rgb = act*jnp.array(N_COLORS/255).ravel()
    act_r = act_rgb[:,:,0::3].sum(axis=2).ravel()
    act_g = act_rgb[:,:,1::3].sum(axis=2).ravel()
    act_b = act_rgb[:,:,2::3].sum(axis=2).ravel()
    return (act_r,act_g,act_b)

@jit
def obj(s_t,NEURONS): # R_t
    sz = NEURONS**2
    ind = jnp.int32(jnp.floor(sz/2))
    return -(s_t[ind] + s_t[ind+sz] + s_t[ind+2*sz])

@jit
def new_env(e_t_1,v_t): # e_t
    v_t = v_t.reshape([2,]) # check shapes correct when adding more dots ([(dots),2] - [2,])
    return e_t_1 - v_t

@jit
def single_step(EHT_t_1,eps_t): 
    # unpack carry
    e_t_1,h_t_1,theta = EHT_t_1
    
    # extract data from theta
    W_f = theta["GRU"]["W_f"]
    U_f = theta["GRU"]["U_f"]
    b_f = theta["GRU"]["b_f"]
    W_h = theta["GRU"]["W_h"]
    U_h = theta["GRU"]["U_h"]
    b_h = theta["GRU"]["b_h"]
    C = theta["GRU"]["C"]
    
    NEURONS_J = theta["ENV"]["NEURONS_J"]
    NEURONS_I = theta["ENV"]["NEURONS_I"]
    SIGMA_T = theta["ENV"]["SIGMA_T"]
    SIGMA_N = theta["ENV"]["SIGMA_N"]
    N_COLORS = theta["ENV"]["N_COLORS"]
    STEP = theta["ENV"]["STEP"]
    NEURONS = theta["ENV"]["NEURONS"]
    
    # neuron activations
    s_t = neuron_act_(e_t_1,NEURONS_J,NEURONS_I,SIGMA_T,N_COLORS)
    
    # reward from neurons
    R_t = obj(s_t,NEURONS)
    
    # minimal GRU equations
    f_t = sigmoid( jnp.matmul(W_f,s_t) + jnp.matmul(U_f,h_t_1) + b_f)
    hhat_t = jnp.tanh( jnp.matmul(W_h,s_t) + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t) # dot?
    
    # v_t = C*h_t + eps
    v_t = STEP * (jnp.matmul(C,h_t) + SIGMA_N * eps_t)
    
    # new env
    e_t = new_env(e_t_1,v_t)
    
    # assemble output
    EHT_t = (e_t,h_t,theta)
    
    return (EHT_t,R_t)

def tot_reward(e0,h0,theta,eps,IT):
    EHT_0 = (e0,h0,theta)
    EHT_,R_t = jax.lax.scan(single_step,EHT_0,eps,length=IT)
    return jnp.sum(R_t)

# main routine
def main():
    # ENV parameters
    SIGMA_N = jnp.float32(1)
    SIGMA_T = jnp.float32(1)
    STEP = jnp.float32(0.001)
    APERTURE = jnp.pi/3
    N_COLORS = jnp.float32([[255,255,255]]) # must correspond to n_dots
    N_DOTS = 1
    KEY_DOT = rnd.PRNGKey(0)
    NEURONS = 11
    
    # GRU parameters
    N = 3*NEURONS**2
    KEY_INIT = rnd.PRNGKey(1)
    KEY_EPS = rnd.PRNGKey(2)
    INIT = jnp.float32(0.4) ###
    
    # generate initial values
    ki = rnd.split(KEY_INIT,num=50)
    h0 = rnd.normal(ki[0],(N,),dtype="float32")
    W_f0 = (INIT/N*N)*rnd.normal(ki[1],(N,N),dtype="float32")
    U_f0 = (INIT/N*N)*rnd.normal(ki[2],(N,N),dtype="float32")
    b_f0 = (INIT/N)*rnd.normal(ki[3],(N,),dtype="float32")
    W_h0 = (INIT/N*N)*rnd.normal(ki[4],(N,N),dtype="float32")
    U_h0 = (INIT/N*N)*rnd.normal(ki[5],(N,N),dtype="float32")
    b_h0 = (INIT/N)*rnd.normal(ki[6],(N,),dtype="float32")
    C0 = (INIT/2*N)*rnd.normal(ki[7],(2,N),dtype="float32")
    NEURONS_I = gen_neurons_i(NEURONS,APERTURE)
    NEURONS_J = gen_neurons_j(NEURONS,APERTURE)
    
    # generate theta pytree
    theta = { "GRU" : {
            "W_f" : W_f0,
            "U_f" : U_f0,
            "b_f" : b_f0,
            "W_h" : W_h0,
            "U_h" : U_h0,
            "b_h" : b_h0,
            "C"   : C0
        },
              "ENV" : {
            "APERTURE"     : APERTURE,
            "NEURONS_I"    : NEURONS_I,
            "NEURONS_J"    : NEURONS_J,
            "N_COLORS"     : N_COLORS,
            "SIGMA_N"      : SIGMA_N,
            "SIGMA_T"      : SIGMA_T,
            "STEP"         : STEP,
            "NEURONS"      : NEURONS,
            "N_DOTS"       : N_DOTS,
            "KEY_EPS"      : KEY_EPS
        }
             }
    jax.lax.stop_gradient(theta["ENV"])
    
    # main() params
    EPOCHS = 200
    IT = 20
    PARALLEL = 10
    UPDATE = jnp.float32(0.01)
    R_arr = jnp.zeros(EPOCHS)
    optimizer = optax.adam(learning_rate=UPDATE)
    opt_state = optimizer.init(theta["GRU"])
    ke = rnd.split(KEY_DOT,num=EPOCHS)
    
    for e in range(EPOCHS): #jax.lax.scan again or need loop?
    
        # generate target dot(s)
        e0 = create_dots(theta["ENV"]["N_DOTS"],ke[e])
            
        # generate noise
        KEY_EPS = theta["ENV"]["KEY_EPS"]
        eps = rnd.normal(KEY_EPS,shape=[IT,2,PARALLEL],dtype="float32")
        
        # vmap over tot_reward; find avg r_tot, grad
        val_grad_vmap = jax.vmap(jax.value_and_grad(tot_reward,argnums=2,allow_int=True),in_axes=(None,None,None,2,None),out_axes=(0,0))
        R_tot,grads = val_grad_vmap(e0,h0,theta,eps,IT)
        grads_ = jax.tree_util.tree_map(lambda g: jnp.mean(g,axis=0), grads["GRU"])
        R_arr = R_arr.at[e].set(jnp.mean(R_tot))
        print(grads_["W_h"])
        
        # update theta
        update, opt_state = optimizer.update(grads_, opt_state, theta["GRU"])
        theta["GRU"] = optax.apply_updates(theta["GRU"], update)
        print(f'epoch: {e}, R_tot: {R_arr[e]}')
            
    plt.plot(R_arr) # [::5]
    title_ = "it = " + str(IT) + ", epochs = " + str(EPOCHS) + ", it = " + str(IT) + ", update = " + str(UPDATE) #title_ = "NO UPDATE (control) "
    plt.title(title_)
    # axis stuff
    
if __name__ == "__main__":
    main() 