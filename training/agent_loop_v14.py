# -*- coding: utf-8 -*-
"""
Created on Fri Jan  25 2023

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
import csv
import pickle
from pathlib import Path
# import os
# from os.path import dirname, abspath
jax.config.update('jax_platform_name', 'cpu')

# fnc definitions
@jit
def sigmoid(x):
  return 1 / (1 + jnp.exp(-x))

# def gen_neurons(NEURONS,APERTURE):
#     return jnp.linspace(-APERTURE,APERTURE,NEURONS**2,dtype="float32")
# gen_neurons_j = jit(gen_neurons,static_argnums=(0,1))  

def gen_neurons_j(NEURONS,APERTURE):
    return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32")
gen_neurons_j = jit(gen_neurons_j,static_argnums=(0,1))

def gen_neurons_i(NEURONS,APERTURE):
    return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32")
gen_neurons_i = jit(gen_neurons_i,static_argnums=(0,1))

def create_dots(N_DOTS,KEY_DOT,VMAPS,EPOCHS): # [d_j,d_i]
    return rnd.uniform(KEY_DOT,shape=[EPOCHS,N_DOTS,2,VMAPS],minval=-jnp.pi,maxval=jnp.pi,dtype="float32")
create_dots = jit(create_dots,static_argnums=(0,2,3))

@jit
def neuron_act(e_t_1,th_j,th_i,SIGMA_A,COLORS):
    D_ = COLORS.shape[0]
    N_ = th_j.size
    G_0 = jnp.vstack((jnp.tile(th_j,N_),jnp.tile(th_i,N_)))
    G = jnp.tile(G_0.reshape(2,N_**2,1),(1,1,D_))
    C = (COLORS/255).transpose((1,0))
    E = G.transpose((1,0,2)) - e_t_1.T
    act = jnp.exp((jnp.cos(E[:,0,:]) + jnp.cos(E[:,1,:]) - 2)/SIGMA_A**2).reshape((D_,N_**2))
    act_C = jnp.matmul(C,act)
    return act_C # (act_r,act_g,act_b) = act_C

@jit
def obj(e_t_1,sel,SIGMA_R): # R_t
    obj = -jnp.exp(-jnp.sum((e_t_1)**2,axis=1)/SIGMA_R**2)
    return jnp.dot(obj,sel)

@jit
def new_env(e_t_1,v_t): # e_t
    return e_t_1 - v_t

@jit
def abs_dist(e_t):
    e_t_ = (e_t + jnp.pi)%(2*jnp.pi)-jnp.pi
    return jnp.sqrt(e_t_[:,0]**2+e_t_[:,1]**2)

def csv_write(data,file_): # use 'csv_test_multi.csv'
    data = data.ravel()
    path_ = str(Path(__file__).resolve().parents[1]) + '\\csv_plotter\\'
    with open(path_+file_,'a',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def save_params(params,file_):  # pickle_test.pkl
    path_ = str(Path(__file__).resolve().parents[1]) + '\\pkl\\'
    with open(path_+file_,'wb') as file:
        pickle.dump(params,file,pickle.HIGHEST_PROTOCOL)

@jit
def single_step(EHT_t_1,eps):

    # unpack values
    e_t_1,h_t_1,theta,sel = EHT_t_1  #eps_t,sel_t = eps_sel_t
    
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
    SIGMA_A = theta["ENV"]["SIGMA_A"]
    SIGMA_N = theta["ENV"]["SIGMA_N"]
    SIGMA_R = theta["ENV"]["SIGMA_R"]
    COLORS = theta["ENV"]["COLORS"]
    STEP = theta["ENV"]["STEP"]
    
    # neuron activations
    (act_r,act_g,act_b) = neuron_act(e_t_1,THETA_J,THETA_I,SIGMA_A,COLORS) # (act_r_,act_g_,act_b_) = neuron_act__(e_t_1,THETA_J,THETA_I,SIGMA_A,N_COLORS)
    print(act_r)
    
    # reward from neurons
    R_t = obj(e_t_1,sel,SIGMA_R) # R_t_ = obj(act_r,act_g,act_b,NEURONS)
    
    # minimal GRU equations
    f_t = sigmoid( jnp.matmul(Wr_f,act_r) + jnp.matmul(Wg_f,act_g) + jnp.matmul(Wb_f,act_b) + jnp.matmul(U_f,h_t_1) + b_f )
    hhat_t = jnp.tanh( jnp.matmul(Wr_h,act_r)  + jnp.matmul(Wg_h,act_g) + jnp.matmul(Wb_h,act_b) + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t) # dot? 
    
    # v_t = C*h_t + eps
    v_t = STEP*(jnp.matmul(C,h_t) + SIGMA_N*eps) # 'motor noise'
    
    # new env
    e_t = new_env(e_t_1,v_t)

    # abs distance
    dis = abs_dist(e_t)
    
    # assemble output
    EHT_t = (e_t,h_t,theta,sel)
    R_dis = (R_t,dis)
    
    return (EHT_t,R_dis)

def tot_reward(e0,h0,theta,sel,eps,epoch): #sel
    EHT_0 = (e0,h0,theta,sel)
    IT = theta["ENV"]["IT"]
    EHT_,R_dis = jax.lax.scan(single_step,EHT_0,eps,length=IT)
    R_t,dis = R_dis
    if (epoch==0)|(epoch%50==0)|(epoch==(theta["ENV"]["EPOCHS"]-1)):
        csv_write(dis,'csv_test_5.csv') ###
    return jnp.sum(R_t)

# main routine
def main():
    # ENV parameters
    SIGMA_A = jnp.float32(0.9)
    SIGMA_R = jnp.float32(0.3) # 0.3
    SIGMA_N = jnp.float32(2) # 1.6
    STEP = jnp.float32(0.02) # play around with! 0.005
    APERTURE = jnp.pi/3
    COLORS = jnp.float32([[255,100,50],[50,255,100],[100,50,255],[100,100,100],[200,200,200]]) # [[255,0,0],[0,200,200],[100,100,100]]
    N_DOTS = COLORS.shape[0]
    NEURONS = 11
    
    # GRU parameters
    N = NEURONS**2 #
    G = 50 # size of mGRU
    KEY_INIT = rnd.PRNGKey(0) # 0
    INIT = jnp.float32(0.15) # 0.1
    
    # main() params
    EPOCHS = 1000
    IT = 25
    VMAPS = 200
    UPDATE = jnp.float32(0.001) # 0.001
    R_arr = jnp.empty(EPOCHS)*jnp.nan
    std_arr = jnp.empty(EPOCHS)*jnp.nan
    optimizer = optax.adam(learning_rate=UPDATE)
    
    # generate initial values
    ki = rnd.split(KEY_INIT,num=20)
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
    DOTS = create_dots(N_DOTS,ki[8],VMAPS,EPOCHS)
    EPS = rnd.normal(ki[9],shape=[EPOCHS,IT,2,VMAPS],dtype="float32") ### NEED TO CHANGE TO ALSO HAVE EPOCH DIMENSION; CHANGE DOTS TO BE PRE-GENERATED TOO
    SELECT = jnp.eye(N_DOTS)[rnd.choice(ki[10],N_DOTS,(EPOCHS,VMAPS))] # [EPOCHS,VMAPS,N_DOTS]
    
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
            "COLORS"       : COLORS,
            "SIGMA_N"      : SIGMA_N,
            "SIGMA_A"      : SIGMA_A,
            "SIGMA_R"      : SIGMA_R,
            "STEP"         : STEP,
            "NEURONS"      : NEURONS,
            "N_DOTS"       : N_DOTS,
            "SELECT"       : SELECT,
            "IT"           : IT,
            "EPOCHS"       : EPOCHS
        }
             }
    jax.lax.stop_gradient(theta["ENV"])
    opt_state = optimizer.init(theta["GRU"])

    #figure
    plt.figure()
    def fig_():
        plt.errorbar(jnp.arange(EPOCHS),R_arr,yerr=std_arr/2,ecolor="black",elinewidth=0.5,capsize=1.5) # jnp.arange(EPOCHS),
        plt.show(block=False)
        # title_ = "epochs = " + str(EPOCHS) + ", it = " + str(IT) + ", vmaps = " + str(VMAPS) + ", update = " + str(UPDATE) + ", SIGMA_A = " + str(SIGMA_A) + ", SIGMA_R = " + str(SIGMA_R) + "\n" + "colors = " + jnp.array_str(N_COLORS[0][:]) + "," + jnp.array_str(N_COLORS[1][:]) + "," + jnp.array_str(N_COLORS[2][:]) + "," + jnp.array_str(N_COLORS[3][:]) + "," + jnp.array_str(N_COLORS[4][:]) # N_COLORS.tolist()) #title_ = "NO UPDATE (control) " '''
        title__ = f'epochs={EPOCHS}, it={IT}, vmaps={VMAPS}, update={UPDATE:.3f}, SIGMA_A={SIGMA_A:.1f}, SIGMA_R={SIGMA_R:.1f}, SIGMA_N={SIGMA_N:.1f} \n colors={jnp.array_str(COLORS[0][:]) + jnp.array_str(COLORS[1][:]) + jnp.array_str(COLORS[2][:]) + jnp.array_str(COLORS[3][:]) + jnp.array_str(COLORS[4][:])}'
        plt.title(title__,fontsize=8)
        plt.xlabel('Iteration')
        plt.ylabel('Reward')

    #save important values to file
    ### save_params(theta,'pkl_theta_.pkl') ###
    save_params(COLORS,'pkl_colors_5.pkl')
    save_params(theta["ENV"]["SELECT"],'pkl_select_5.pkl')

    #main loop
    for e in range(EPOCHS):

        # get dots, select, eps
        e0 = DOTS[e,:,:,:] 
        sel = SELECT[e,:,:]
        eps = EPS[e,:,:,:]
        
        # vmap tot_reward over dots (e0), eps (EPS) and sel (SELECT)); find avg r_tot, grad
        val_grad_vmap = jax.vmap(jax.value_and_grad(tot_reward,argnums=2,allow_int=True),in_axes=(2,None,None,0,2,None),out_axes=(0,0))
        R_tot,grads = val_grad_vmap(e0,h0,theta,sel,eps,e)
        grads_ = jax.tree_util.tree_map(lambda g: jnp.mean(g,axis=0), grads["GRU"])
        R_arr = R_arr.at[e].set(jnp.mean(R_tot))
        std_arr = std_arr.at[e].set(jnp.std(R_tot))
        # print(grads_["C"])
        
        # update theta
        update, opt_state = optimizer.update(grads_, opt_state, theta["GRU"])
        theta["GRU"] = optax.apply_updates(theta["GRU"], update)
        print(f'epoch: {e}, R_tot: {R_arr[e]}')

        drawnow(fig_)

    plt.show() 

if __name__ == "__main__":
    main() 