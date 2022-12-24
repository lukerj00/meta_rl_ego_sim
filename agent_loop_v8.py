# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:38:20 2022

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

#@jit
def gen_neurons_j(NEURONS,APERTURE):
    return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32")
    
#@jit
def gen_neurons_i(NEURONS,APERTURE):
    return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32")

#@jit
def create_dots(N_DOTS,KEY_DOT): # d_ji
    return rnd.uniform(KEY_DOT,shape=[N_DOTS,2],minval=-jnp.pi,maxval=jnp.pi,dtype="float32")

#@jit
def neuron_act_(e_t_1,n_j,n_i,SIGMA_T,NEURONS,N_COLORS):
    N_DOTS = jnp.size(e_t_1,0)
    n_j_tot = jnp.tile(jnp.array(n_j).reshape([1,NEURONS,1]),[NEURONS,1,N_DOTS])
    n_i_tot = jnp.tile(jnp.array(n_i).reshape([NEURONS,1,1]),[1,NEURONS,N_DOTS])
    del_j = e_t_1[:,0] - n_j_tot
    del_i = e_t_1[:,1] - n_i_tot
    act = jnp.repeat(jnp.exp((jnp.cos(del_j) + jnp.cos(del_i) - 2)/SIGMA_T**2),3,axis=2)
    act_rgb = act*jnp.array(N_COLORS/255).ravel()
    act_r = act_rgb[:,:,0::3].sum(axis=2).ravel('F')
    act_g = act_rgb[:,:,1::3].sum(axis=2).ravel('F')
    act_b = act_rgb[:,:,2::3].sum(axis=2).ravel('F')
    return jnp.concatenate((act_r,act_g,act_b),axis=0)

@jit
def obj(s_t,NEURONS): # R_t
    sz = NEURONS**2
    ind = jnp.int32(jnp.floor(sz/2))
    return -(s_t[ind] + s_t[ind+sz] + s_t[ind+2*sz])

@jit
def new_env(e_t_1,v_t): # e_t
    v_t = v_t.reshape([2,]) # check shapes correct when adding more dots
    return e_t_1 - v_t
    
#@jit
def single_step(e_t_1,h_t_1,theta,eps_t): 
    
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
    s_t = neuron_act_(e_t_1,NEURONS_J,NEURONS_I,SIGMA_T,NEURONS,N_COLORS)
    
    # reward from neurons
    R_t = obj(s_t,NEURONS)
    
    # minimal GRU equations
    s_t = s_t.reshape([3*NEURONS**2,1])
    f_t = sigmoid( jnp.matmul(W_f,s_t) + jnp.matmul(U_f,h_t_1) + b_f)
    hhat_t = jnp.tanh( jnp.matmul(W_h,s_t) + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t)
    
    # v_t = C*h_t + eps
    v_t = STEP * (jnp.matmul(C,h_t) + SIGMA_N * eps_t) #rnd.normal(subkey_eps,(2,1))) #key_eps_, subkey_eps = rnd.split(key_eps)
    
    # new env from sim dynamics (agent fixed, dots move)
    e_t = new_env(e_t_1,v_t)
    
    return (R_t,e_t,h_t)

def tot_reward(e0,h0,theta):
    
    IT = theta["ENV"]["IT"]
    KEY_EPS = theta["ENV"]["KEY_EPS"]
    R_tot = jnp.float32(0)
    e_t_1, h_t_1 = e0, h0
    eps = rnd.normal(KEY_EPS,shape=[2*IT,],dtype="float32")
    # R_tot = jax.lax.scan( single_step(.....) ), or loop: 
    ###R_tot = jax.lax.scan(single_step, init, xs, length=IT)
        
    
    for t in range(IT):
        eps_t = eps[2*t:2*t+2].reshape([2,1])
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
    N_DOTS = 1
    
    # GRU parameters
    N = 3*NEURONS**2
    KEY_INIT = rnd.PRNGKey(1)
    KEY_EPS = rnd.PRNGKey(2)
    
    # main() params
    EPOCHS = 400
    IT = 10
    INIT = jnp.float32(0.4)
    UPDATE = jnp.float32(0.001)
    R_arr = jnp.zeros(EPOCHS)
    optimizer = optax.adam(learning_rate=UPDATE)
    
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
            "APERTURE"     : jax.lax.stop_gradient(APERTURE),
            "NEURONS_I"    : jax.lax.stop_gradient(NEURONS_I),
            "NEURONS_J"    : jax.lax.stop_gradient(NEURONS_J),
            "N_COLORS"     : jax.lax.stop_gradient(N_COLORS),
            "SIGMA_N"      : jax.lax.stop_gradient(SIGMA_N),
            "SIGMA_T"      : jax.lax.stop_gradient(SIGMA_T),
            "STEP"         : jax.lax.stop_gradient(STEP),
            "NEURONS"      : jax.lax.stop_gradient(NEURONS),
            "N_DOTS"       : jax.lax.stop_gradient(N_DOTS),
            "KEY_EPS"      : jax.lax.stop_gradient(KEY_EPS),
            "IT"           : jax.lax.stop_gradient(IT)            
        }
             }
    
    # generate dot keys
    ke = rnd.split(KEY_DOT,num=EPOCHS)
    
    #generate optimizer
    opt_state = optimizer.init(theta["GRU"])
    
    for e in range(EPOCHS): # or use jax.lax.scan(single_step,...)
    
        # generate target dot(s)
        e0 = create_dots(theta["ENV"]["N_DOTS"],ke[e])
            
        # run epoch, get total reward and gradients 
        R_tot, grads = jax.value_and_grad(tot_reward,argnums=2,allow_int=True)(e0,h0,theta)
        grads_ = grads["GRU"]
        R_arr = R_arr.at[e].set(R_tot)
        #print(grads_["W_h"])
        
        # update theta
        update, opt_state = optimizer.update(grads_, opt_state, theta["GRU"])
        theta["GRU"] = optax.apply_updates(theta["GRU"], update)
        #print('update = ',update)
        print(f'epoch: {e}, R_tot: {R_arr[e]}')
            
    plt.plot(R_arr[::5])
    title_ = "it = " + str(IT) + ", epochs = " + str(EPOCHS) + ", it = " + str(IT) + ", update = " + str(UPDATE) #title_ = "NO UPDATE (control) "
    plt.title(title_)
    
if __name__ == "__main__":
    main() 