# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:20:45 2022

@author: lukej
"""

import jax
import jax.numpy as jnp
from jax import jit
import jax.random as rnd
import optax
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
#jax.config.update('jax_platform_name', 'cpu')

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

def create_dots(N_DOTS,KEY_DOT,VMAPS): # [d_j,d_i]
    return rnd.uniform(KEY_DOT,shape=[N_DOTS,2,VMAPS],minval=-jnp.pi,maxval=jnp.pi,dtype="float32")
#create_dots = jit(create_dots,static_argnums=(0,2))

@jit
def neuron_act_(e_t_1,th_j,th_i,SIGMA_T,SIGMA_R,N_COLORS,sel):
    ### TO DO: CHANGE TO INCLUDE DOT PROD OF EXPONENT WITH SELECT
    NEURONS = jnp.size(th_j)
    IND = jnp.int32(jnp.floor(NEURONS/2))
    N_DOTS = jnp.size(e_t_1,0)
    n_j_tot = jnp.tile(jnp.array(th_j).reshape([1,NEURONS,1]),[NEURONS,1,N_DOTS])
    n_i_tot = jnp.tile(jnp.array(th_i).reshape([NEURONS,1,1]),[1,NEURONS,N_DOTS])
    del_j = e_t_1[:,0] - n_j_tot
    del_i = e_t_1[:,1] - n_i_tot
    act = jnp.exp((jnp.cos(del_j) + jnp.cos(del_i) - 2)/SIGMA_T**2)
    act_r = jnp.exp((jnp.cos(del_j[IND,IND,:]) + jnp.cos(del_i[IND,IND,:]) - 2)/SIGMA_R**2)
    act = act.at[IND,IND,:].set(act_r)
    act = jnp.repeat(act,3,axis=2)
    act_rgb = act*jnp.array(N_COLORS/255).ravel()*jnp.repeat(sel,3)
    act_r = act_rgb[:,:,0::3].sum(axis=2).ravel()
    act_g = act_rgb[:,:,1::3].sum(axis=2).ravel()
    act_b = act_rgb[:,:,2::3].sum(axis=2).ravel()
    return (act_r,act_g,act_b)

@jit
def obj(act_r,act_g,act_b,NEURONS): # R_t
    SZ = NEURONS**2
    IND = jnp.int32(jnp.floor(SZ/2))
    return -(act_r[IND] + act_g[IND] + act_b[IND])

@jit
def new_env(e_t_1,v_t): # e_t
    return e_t_1 - v_t

@jit
def single_step(EHT_t_1,eps):
    # unpack values
    e_t_1,h_t_1,theta,sel = EHT_t_1
    #eps_t,sel_t = eps_sel_t
    
    # extract data from theta
    Wr_f = theta["GRU"]["Wr_f"]
    Wg_f = theta["GRU"]["Wg_f"]
    Wb_f = theta["GRU"]["Wb_f"]
    U_f = theta["GRU"]["U_f"]
    b_f = theta["GRU"]["b_f"]
    Wr_h = theta["GRU"]["Wr_h"]
    Wg_h = theta["GRU"]["Wg_h"]
    Wb_h = theta["GRU"]["Wb_h"]
    U_h = theta["GRU"]["U_h"]
    b_h = theta["GRU"]["b_h"]
    C = theta["GRU"]["C"]
    
    THETA_J = theta["ENV"]["THETA_J"]
    THETA_I = theta["ENV"]["THETA_I"]
    SIGMA_T = theta["ENV"]["SIGMA_T"]
    SIGMA_N = theta["ENV"]["SIGMA_N"]
    SIGMA_R = theta["ENV"]["SIGMA_R"]
    N_COLORS = theta["ENV"]["N_COLORS"]
    STEP = theta["ENV"]["STEP"]
    NEURONS = theta["ENV"]["NEURONS"]
    
    # neuron activations
    (act_r,act_g,act_b) = neuron_act_(e_t_1,THETA_J,THETA_I,SIGMA_T,SIGMA_R,N_COLORS,sel)
    
    # reward from neurons
    R_t = obj(act_r,act_g,act_b,NEURONS)
    
    # minimal GRU equations
    f_t = sigmoid( jnp.matmul(Wr_f,act_r) + jnp.matmul(Wg_f,act_g) + jnp.matmul(Wb_f,act_b) + jnp.matmul(U_f,h_t_1) + b_f )
    hhat_t = jnp.tanh( jnp.matmul(Wr_h,act_r)  + jnp.matmul(Wg_h,act_g) + jnp.matmul(Wb_h,act_b) + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t) # dot? 
    
    # v_t = C*h_t + eps
    v_t = STEP*(jnp.matmul(C,h_t) + SIGMA_N*eps)
    
    # new env
    e_t = new_env(e_t_1,v_t)
    
    # assemble output
    EHT_t = (e_t,h_t,theta,sel)
    
    return (EHT_t,R_t)

def tot_reward(e0,h0,theta,sel,eps): #sel
    EHT_0 = (e0,h0,theta,sel)
    IT = theta["ENV"]["IT"]
    EHT_,R_t = jax.lax.scan(single_step,EHT_0,eps,length=IT)
    return jnp.sum(R_t)

# main routine
def main():
    # ENV parameters
    SIGMA_N = jnp.float32(0.5)
    SIGMA_T = jnp.float32(1)
    SIGMA_R = jnp.float32(2)
    STEP = jnp.float32(0.005)
    APERTURE = jnp.pi/3
    N_COLORS = jnp.float32([[255,200,150],[150,255,200],[200,150,255]]) # [[255,0,0],[0,200,200],[100,100,100]]
    N_DOTS = jnp.int32(N_COLORS.size/3)
    KEY_DOT = rnd.PRNGKey(0)
    NEURONS = 11
    
    # GRU parameters
    N = NEURONS**2
    G = 50
    KEY_INIT = rnd.PRNGKey(1)
    KEY_EPS = rnd.PRNGKey(2)
    INIT = jnp.float32(0.2)
    
    # main() params
    EPOCHS = 50
    IT = 25
    VMAPS = 100
    UPDATE = jnp.float32(0.005)
    R_arr = jnp.empty(EPOCHS)*jnp.nan
    var_arr = jnp.empty(EPOCHS)*jnp.nan
    optimizer = optax.adam(learning_rate=UPDATE)
    ke = rnd.split(KEY_DOT,num=EPOCHS)
    
    # generate initial values
    ki = rnd.split(KEY_INIT,num=50)
    h0 = rnd.normal(ki[0],(G,),dtype="float32")
    Wr_f0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype="float32")
    Wg_f0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype="float32")
    Wb_f0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype="float32")
    U_f0 = (INIT/G*G)*rnd.normal(ki[2],(G,G),dtype="float32")
    b_f0 = (INIT/G)*rnd.normal(ki[3],(G,),dtype="float32")
    Wr_h0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype="float32")
    Wg_h0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype="float32")
    Wb_h0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype="float32")
    U_h0 = (INIT/G*G)*rnd.normal(ki[5],(G,G),dtype="float32")
    b_h0 = (INIT/G)*rnd.normal(ki[6],(G,),dtype="float32")
    C0 = (INIT/2*G)*rnd.normal(ki[7],(2,G),dtype="float32")
    THETA_I = gen_neurons_i(NEURONS,APERTURE)
    THETA_J = gen_neurons_j(NEURONS,APERTURE)
    EPS = rnd.normal(KEY_EPS,shape=[IT,2,VMAPS],dtype="float32")
    SELECT = jnp.eye(N_DOTS)[rnd.choice(ki[8],N_DOTS,(EPOCHS,))]
    
    # assemble theta pytree
    theta = { "GRU" : {
            "Wr_f" : Wr_f0,
            "Wg_f" : Wg_f0,
            "Wb_f" : Wb_f0,            
            "U_f"  : U_f0,
            "b_f"  : b_f0,
            "Wr_h" : Wr_h0,
            "Wg_h" : Wg_h0,
            "Wb_h" : Wb_h0,
            "U_h"  : U_h0,
            "b_h"  : b_h0,
            "C"    : C0
        },
              "ENV" : {
            "APERTURE"     : APERTURE,
            "THETA_I"      : THETA_I,
            "THETA_J"      : THETA_J,
            "N_COLORS"     : N_COLORS,
            "SIGMA_N"      : SIGMA_N,
            "SIGMA_T"      : SIGMA_T,
            "SIGMA_R"      : SIGMA_R,
            "STEP"         : STEP,
            "NEURONS"      : NEURONS,
            "N_DOTS"       : N_DOTS,
            "SELECT"       : SELECT,
            "IT"           : IT
        }
             }
    jax.lax.stop_gradient(theta["ENV"])
    opt_state = optimizer.init(theta["GRU"])

    plt.figure()

    def fig_():
        plt.errorbar(jnp.arange(EPOCHS),R_arr,yerr=var_arr/2,ecolor="black",elinewidth=0.5,capsize=1.5) # jnp.arange(EPOCHS),
        plt.show(block=False)
        title_ = "epochs = " + str(EPOCHS) + ", it = " + str(IT) + ", vmaps = " + str(VMAPS) + ", update = " + str(UPDATE) + "\n" + "colors = " + jnp.array_str(N_COLORS[0][:]) + "," + jnp.array_str(N_COLORS[1][:]) + "," + jnp.array_str(N_COLORS[2][:]) # N_COLORS.tolist()) #title_ = "NO UPDATE (control) "
        plt.title(title_,fontsize=10)
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
    
    for e in range(EPOCHS): #jax.lax.scan again or need loop?
    
        # generate target dot(s) # (pre-generate...)
        e0 = create_dots(theta["ENV"]["N_DOTS"],ke[e],VMAPS)

        # get select vector
        sel = SELECT[e,:]
        
        # vmap over tot_reward; find avg r_tot, grad
        val_grad_vmap = jax.vmap(jax.value_and_grad(tot_reward,argnums=2,allow_int=True),in_axes=(2,None,None,None,2),out_axes=(0,0))
        R_tot,grads = val_grad_vmap(e0,h0,theta,sel,EPS)
        grads_ = jax.tree_util.tree_map(lambda g: jnp.mean(g,axis=0), grads["GRU"])
        R_arr = R_arr.at[e].set(jnp.mean(R_tot))
        var_arr = var_arr.at[e].set(jnp.std(R_tot))
        print(grads_["C"])
        
        # update theta
        update, opt_state = optimizer.update(grads_, opt_state, theta["GRU"])
        theta["GRU"] = optax.apply_updates(theta["GRU"], update)
        print(f'epoch: {e}, R_tot: {R_arr[e]}')

        drawnow(fig_)

    plt.show()        
    # plt.plot(R_arr) # [::5]
    # title_ = "it = " + str(IT) + ", epochs = " + str(EPOCHS) + ", vmaps = " + str(VMAPS) + ", update = " + str(UPDATE) #title_ = "NO UPDATE (control) "
    # plt.title(title_)
    # plt.show()
    # axis stuff
    
if __name__ == "__main__":
    main() 