# -*- coding: utf-8 -*-
"""
Created on Fri Jan  29 2023

@author: lukej
"""
import jax
import jax.numpy as jnp
from jax import jit
import jax.random as rnd
# from jax.experimental.host_callback import id_print
# from jax.experimental.host_callback import call
import optax
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
import csv
import pickle
from pathlib import Path
from datetime import datetime
import re
import os
# from os.path import dirname, abspath
# jax.config.update('jax_platform_name', 'cpu')

# fnc definitions
def csv_write(data,rav): # 1 (dis values in tot_reward), 0 (R scalars in R_arr)
    if rav==1:
        data = data.ravel()
    else:
        pass
    path_ = str(Path(__file__).resolve().parents[1]) + '\\csv_plotter\\'
    dt = datetime.now().strftime("%d_%m-%H%M")
    # file_ = os.path.basename(__file__).split('.')[0]
    with open(path_+'dis_csv'+dt,'a',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
# csv_write=jit(csv_write,static_argnums=(1))

def save_params(param,str_):  # can't jit (can't pickle jax tracers)
    path_ = str(Path(__file__).resolve().parents[1]) + '\\pkl\\'
    dt = datetime.now().strftime("%d_%m-%H%M")
    # file_ = os.path.basename(__file__).split('.')[0]
    with open(path_+str_+dt+'.pkl','wb') as file:
        pickle.dump(param,file,pickle.HIGHEST_PROTOCOL)

def eval_(jaxpr_in): ### to do
    reg_ = r'(?<=DeviceArray\()(.*)(?=dtype)' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
    jaxpr_str = repr(jaxpr_in)
    jaxpr_ = re.findall(reg_,jaxpr_str,re.DOTALL)
    jaxpr_ = ''.join(jaxpr_)
    jaxpr_ = re.sub(r'\s+','',jaxpr_)
    jaxpr_ = jaxpr_.rstrip(r',')
    jaxpr_ = re.sub(r'(\[|\]|\.)','',jaxpr_)
    jaxpr_ = jaxpr_.split(",")

@jit
def sigmoid(x):
  return 1 / (1 + jnp.exp(-x))

def gen_neurons(NEURONS,APERTURE):
    return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32")
gen_neurons = jit(gen_neurons,static_argnums=(0,1))  

def create_dots(N_DOTS,KEY_DOT,VMAPS,EPOCHS):
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
    return act_C

@jit
def obj(e_t_1,sel,SIGMA_R):
    obj = -jnp.exp(-jnp.sum((e_t_1)**2,axis=1)/SIGMA_R**2)
    return jnp.dot(obj,sel)

@jit
def new_env(e_t_1,v_t): # e_t
    return e_t_1 - v_t

@jit
def abs_dist(e_t):
    e_t_ = (e_t + jnp.pi)%(2*jnp.pi)-jnp.pi
    return jnp.sqrt(e_t_[:,0]**2+e_t_[:,1]**2)

@jit
def single_step(EHT_t_1,eps):
    # unpack values
    e_t_1,h_t_1,theta,sel = EHT_t_1
    
    # extract data from theta
    Wr_f = theta["GRU"]["Wr_f"]
    Wg_f = theta["GRU"]["Wg_f"]
    Wb_f = theta["GRU"]["Wb_f"]
    U_f = theta["GRU"]["U_f"]
    b_f = theta["GRU"]["b_f"]
    Wr_h = theta["GRU"]["Wr_h"]
    Wg_h = theta["GRU"]["Wg_h"]
    Wb_h = theta["GRU"]["Wb_h"]
    W_s = theta["GRU"]["W_s"]
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
    (act_r,act_g,act_b) = neuron_act(e_t_1,THETA_J,THETA_I,SIGMA_A,COLORS)
    
    # reward from neurons
    R_t = obj(e_t_1,sel,SIGMA_R)
    
    # minimal GRU equations
    f_t = sigmoid(jnp.append( jnp.matmul(Wr_f,act_r) + jnp.matmul(Wg_f,act_g) + jnp.matmul(Wb_f,act_b), jnp.matmul(W_s,sel) ) + jnp.matmul(U_f,h_t_1) + b_f)
    hhat_t = jnp.tanh(jnp.append(jnp.matmul(Wr_h,act_r)  + jnp.matmul(Wg_h,act_g) + jnp.matmul(Wb_h,act_b), jnp.matmul(W_s,sel) ) + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t) # dot?
    
    # v_t = C*h_t + eps
    v_t = STEP*(jnp.matmul(C,h_t) + SIGMA_N*eps) # 'motor noise'
    
    # new env
    e_t = new_env(e_t_1,v_t)

    # abs distance
    dis = abs_dist(e_t) # finds dis with current e_t...
    
    # assemble output
    EHT_t = (e_t,h_t,theta,sel)
    R_dis = (R_t,dis)
    
    return (EHT_t,R_dis)

@jit
def tot_reward(e0,h0,theta,sel,eps,epoch):
    EHT_0 = (e0,h0,theta,sel)
    EHT_,R_dis = jax.lax.scan(single_step,EHT_0,eps) # ,length=theta["ENV"]["IT"].astype(jnp.int32)) # DOESNT WORK WITH TRACER IT.astype(jnp.int32)) # IT.astype(int))
    R_t,dis = R_dis
    # if (epoch==0)|(epoch%50==0)|(epoch==(theta["ENV"]["EPOCHS"]-1)):
    #     csv_write(dis,1) ### dt?
    return jnp.sum(R_t)

@jit
def body_fnc(e,UTORR): # returns theta
    # unpack
    (UPDATE,theta,opt_state,R_arr,std_arr) = UTORR #opt_state
    # h0 = theta["GRU"]["h0"]
    optimizer = optax.adam(learning_rate=UPDATE) # theta["ENV"]["OPT"] # put in GRU?

    # use e
    # e0 = theta["ENV"]["DOTS"][e,:,:,:] 
    # sel = theta["ENV"]["SELECT"][e,:,:]
    # eps = theta["ENV"]["EPS"][e,:,:,:]

    # each iteration effects next LTRR (L{R_arr,std_arr},T{GRU}) # vmap tot_reward over dots (e0), eps (EPS) and sel (SELECT)); find avg r_tot, grad
    val_grad_vmap = jax.vmap(jax.value_and_grad(tot_reward,argnums=2,allow_int=True),in_axes=(2,None,None,0,2,None),out_axes=(0,0))
    R_tot,grads = val_grad_vmap(theta["ENV"]["DOTS"][e,:,:,:],theta["GRU"]["h0"],theta,theta["ENV"]["SELECT"][e,:,:],theta["ENV"]["EPS"][e,:,:,:],e)
    grads_ = jax.tree_util.tree_map(lambda g: jnp.mean(g,axis=0), grads["GRU"])
    R_arr = R_arr.at[e].set(jnp.mean(R_tot))
    std_arr = std_arr.at[e].set(jnp.std(R_tot))
    
    # update theta
    opt_update, opt_state = optimizer.update(grads_, opt_state, theta["GRU"])
    theta["GRU"] = optax.apply_updates(theta["GRU"], opt_update)
    UTORR = (UPDATE,theta,opt_state,R_arr,std_arr)

    return UTORR # becomes new input

@jit
def full_loop(loop_params,theta): # main routine: R_arr, std_arr = full_loop(params) 
    # jax.lax.stop_gradient(theta["ENV"])
    optimizer = optax.adam(learning_rate=loop_params['UPDATE'])
    opt_state = optimizer.init(theta["GRU"])
    UTORR_0 = (loop_params['UPDATE'],theta,opt_state,loop_params['R_arr'],loop_params['std_arr'])
    UPDATE_,theta_,opt_state_,R_arr,std_arr = jax.lax.fori_loop(0, EPOCHS, body_fnc, UTORR_0)
    return (R_arr,std_arr)

startTime = datetime.now()
# ENV parameters
SIGMA_A = jnp.float32(1) # 0.9
SIGMA_R = jnp.float32(0.5) # 0.3
SIGMA_N = jnp.float32(1.8) # 1.6
STEP = jnp.float32(0.008) # play around with! 0.005
APERTURE = jnp.pi/3
COLORS = jnp.float32([[255,100,50],[50,255,100],[100,50,255]]) # ,[100,100,100],[200,200,200]]) # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[50,0,200]]) # [[255,0,0],[0,200,200],[100,100,100]]
N_DOTS = COLORS.shape[0]
NEURONS = 11

# GRU parameters
N = NEURONS**2
G = 50 # size of mGRU (total size = G+N_DOTS)
KEY_INIT = rnd.PRNGKey(0) # 0
INIT = jnp.float32(0.1) # 0.1

# loop params
EPOCHS = 100
IT = 25
VMAPS = 200
UPDATE = jnp.float32(0.001) # 0.001
R_arr = jnp.empty(EPOCHS)*jnp.nan
std_arr = jnp.empty(EPOCHS)*jnp.nan
optimizer = optax.adam(learning_rate=UPDATE) # (can't pass in to jit'd function)

# assemble loop_params pytree
loop_params = {
        'UPDATE':   jnp.float32(UPDATE),
        'R_arr':    jnp.zeros(EPOCHS)*jnp.nan,
        'std_arr':  jnp.zeros(EPOCHS)*jnp.nan
    } # vmaps,it,epochs

# generate initial values
ki = rnd.split(KEY_INIT,num=20)
h0 = rnd.normal(ki[0],(G+N_DOTS,),dtype="float32")
Wr_f0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype="float32")
Wg_f0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype="float32")
Wb_f0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype="float32")
U_f0 = (INIT/G*G)*rnd.normal(ki[2],(G+N_DOTS,G+N_DOTS),dtype="float32")
b_f0 = (INIT/G)*rnd.normal(ki[3],(G+N_DOTS,),dtype="float32")
Wr_h0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype="float32")
Wg_h0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype="float32")
Wb_h0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype="float32")
W_s = (INIT)*rnd.normal(ki[5],(N_DOTS,N_DOTS),dtype="float32")
U_h0 = (INIT/G*G)*rnd.normal(ki[6],(G+N_DOTS,G+N_DOTS),dtype="float32")
b_h0 = (INIT/G)*rnd.normal(ki[7],(G+N_DOTS,),dtype="float32")
C0 = (INIT/2*G)*rnd.normal(ki[8],(2,G+N_DOTS),dtype="float32")
THETA_I = gen_neurons(NEURONS,APERTURE)
THETA_J = gen_neurons(NEURONS,APERTURE)
DOTS = create_dots(N_DOTS,ki[9],VMAPS,EPOCHS)
EPS = rnd.normal(ki[10],shape=[EPOCHS,IT,2,VMAPS],dtype="float32")
SELECT = jnp.eye(N_DOTS)[rnd.choice(ki[11],N_DOTS,(EPOCHS,VMAPS))]

# assemble theta pytree
theta = { "GRU" : {
        "h0"   : h0, # ?
        "Wr_f" : Wr_f0,
        "Wg_f" : Wg_f0,
        "Wb_f" : Wb_f0,            
        "U_f"  : U_f0,
        "b_f"  : b_f0,
        "Wr_h" : Wr_h0,
        "Wg_h" : Wg_h0,
        "Wb_h" : Wb_h0,
        "W_s"  : W_s,
        "U_h"  : U_h0,
        "b_h"  : b_h0,
        "C"    : C0
    },
            "ENV" : {
        "THETA_I"      : THETA_I,
        "THETA_J"      : THETA_J,
        "COLORS"       : COLORS,
        "SIGMA_N"      : SIGMA_N,
        "SIGMA_A"      : SIGMA_A,
        "SIGMA_R"      : SIGMA_R,
        "STEP"         : STEP,
        "DOTS"         : DOTS,
        "EPS"          : EPS,
        "SELECT"       : SELECT,
        # removed aperture,neurons,n_dots,it,epochs

    }
            }
###
R_arr,std_arr = full_loop(loop_params,theta)
print(f'R_arr: {R_arr}','\n',f'std_arr: {std_arr}')
time_elapsed = datetime.now() - startTime
print(f'Completed in: {time_elapsed}, {time_elapsed/EPOCHS} s/epoch')
###
    #figure
plt.figure()
plt.errorbar(jnp.arange(EPOCHS),R_arr,yerr=std_arr/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.show(block=False)
title__ = f'epochs={EPOCHS}, it={IT}, vmaps={VMAPS}, update={UPDATE:.3f}, SIGMA_A={SIGMA_A:.1f}, SIGMA_R={SIGMA_R:.1f}, SIGMA_N={SIGMA_N:.1f} \n colors={jnp.array_str(COLORS[0][:]) + jnp.array_str(COLORS[1][:]) + jnp.array_str(COLORS[2][:])}' #  + jnp.array_str(COLORS[3][:]) + jnp.array_str(COLORS[4][:])}'
plt.title(title__,fontsize=8)
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.show()

# path_ = str(Path(__file__).resolve().parents[1]) + '\\figs\\task6_multi\\'
# plt.savefig(path_ + 'fig_' + dt + '.png')
# plt.show()
# csv_write(R_arr,'R_ARR_TEST.csv',0)