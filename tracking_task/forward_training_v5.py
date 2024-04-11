# -*- coding: utf-8 -*-
# scan; const dot/v0
"""
Created on Wed May 10 2023

@author: lukej
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import jax.random as rnd
import optax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from drawnow import drawnow
import numpy as np
import csv
import pickle
from pathlib import Path
from datetime import datetime
import re
import os
import sys
import scipy
import gc

@jit # @partial(jax.jit,static_argnums=())
def neuron_act(params,dots,pos): #e_t_1,th_j,th_i,SIGMA_A,COLORS
    COLORS = params["COLORS"]
    th_x = params["THETA_X"]
    th_y = params["THETA_Y"]
    SIGMA_A = params["SIGMA_A"]
    D_ = COLORS.shape[0]
    N_ = th_x.size
    G_0 = jnp.vstack((jnp.tile(th_x,N_),jnp.tile(th_y,N_)))
    G = jnp.tile(G_0.reshape(2,N_**2,1),(1,1,D_))
    C = (COLORS/255).transpose((1,0))
    E = G.transpose((1,0,2)) - (dots-pos).T
    act = jnp.exp((jnp.cos(E[:,0,:]) + jnp.cos(E[:,1,:]) - 2)/SIGMA_A**2).reshape((D_,N_**2))
    act_r,act_g,act_b = jnp.matmul(C,act)
    act_rgb = jnp.concatenate((act_r,act_g,act_b))
    return act_rgb

# @jit # @partial(jax.jit,static_argnums=())
# def gen_hp(key_,params):
#     H = params["H"]
#     INIT = params["INIT"]
#     hp_0 = (INIT/H)*rnd.normal(key_,shape=(H,))
#     return hp_0

# @jit # @partial(jax.jit,static_argnums=())
# def gen_v(dots,key,params): #
#     key_ = rnd.PRNGKey(key)
#     pos_0 = rnd.uniform(key=key_,shape=(2,),minval=-params["APERTURE"],maxval=params["APERTURE"])
#     v_0 = neuron_act(params,dots,pos_0)
#     return v_0

# @partial(jax.jit,static_argnums=(1,2,3))
# def gen_dots(key,EPOCHS,VMAPS,N_DOTS,APERTURE):
#     DOTS = rnd.uniform(key,shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE)
#     return DOTS

@partial(jax.jit,static_argnums=(0,))
def gen_vectors(MODULES,APERTURE):
    m = MODULES
    M = m**2
    A = APERTURE
    x = jnp.linspace(-(A-A/m),(A-A/m),m) # CHECK
    x_ = jnp.tile(x,(m,1))
    x_ = x_.reshape((M,)) # .transpose((1,0))
    y = jnp.linspace(-(A-A/m),(A-A/m),m) # CHECK
    y_ = jnp.tile(y.reshape(m,1),(1,m))
    y_ = y_.reshape((M,))
    v = jnp.vstack([x_,y_])
    return v

# @jit
def new_params(params,e):
    params_ = params
    EPOCHS = params["EPOCHS"]
    VMAPS = params["VMAPS"]
    N_DOTS = params["N_DOTS"]
    APERTURE = params["APERTURE"]
    N = params["N"]
    H = params["H"]
    ki = rnd.split(rnd.PRNGKey(e),num=10)
    params_["DOTS"] = rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE)
    params_["V0"] = (INIT/(N))*rnd.normal(ki[1],(EPOCHS,VMAPS,N))
    params_["HP0"] = (INIT/(H))*rnd.normal(ki[2],(EPOCHS,VMAPS,H))
    return params_

# @jit
# def RNN_value(hv_t_1,v_t,r_t_1,r_weights): # WRITE (GRU computation of hv_t_1,v_t,r_t_1->hv_t,r_t)
#     hv_t <= hv_t_1
#     r_t <= r_t_1
#     return hv_t,r_t

# @partial(jax.jit,static_argnums=())
def value(SC,h_t_1,v_t,r_t_1,r_weights): # self,hv_t_1,v_t,r_t_1,weights,params
    h_t,r_t = RNN_value(h_t_1,v_t,r_t_1,r_weights)
    return h_t,r_t

@jit
def RNN_forward(vals,x): # THINK... GRU computation of hp_t_1,v_t_1->hp_t,v_t
    # gc.collect()
    h1vec,v_t_1,h_t_1,vh_t_1,p_weights = vals
    W_h1 = p_weights["W_h1"]
    W_v = p_weights["W_v"]
    U_vh = p_weights["U_vh"]
    b_vh = p_weights["b_vh"]
    vh_t = jax.nn.tanh(jnp.matmul(W_h1,h1vec) + jnp.matmul(W_v,v_t_1) + jnp.matmul(U_vh,vh_t_1) + b_vh) # + jnp.matmul(W_v_z,v_t_1)
    return (h1vec,v_t_1,h_t_1,vh_t,p_weights),None #v_t,hp_t

# @jit # @partial(jax.jit,static_argnums=(5))
def plan(h1vec,v_0,h_0,p_weights,params): # THINK... self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    vh_t_1 = jnp.concatenate((v_0,h_0))
    vals_0 = (h1vec,v_0,h_0,vh_t_1,p_weights)
    (h1vec_,v_t_,h_t_,vh_t_,p_weights_),_ = jax.lax.scan(RNN_forward,vals_0,xs=None,length=params["PLAN_ITS"])
    split_ = jnp.split(vh_t_,jnp.array([v_0.shape[0]]))
    v_t = split_[0] # hp_t_1 = split_[1]
    return v_t

@jit # @partial(jax.jit,static_argnums=())
def move(vec,dots,params): # self,pos_t_1,v_t_1,r_t_1,weights,params
        v_t = neuron_act(params,dots,vec) # pos_t = vector_ # pos_t_1 + self.vector
        return v_t

# @jit # @partial(jax.jit,static_argnums=())
def body_fnc(SC,p_weights,params,dots,s,v_0,h_0):
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    PLAN_ITS = params["PLAN_ITS"]
    h1vec = H1VEC_ARR[:,s]
    vec = VEC_ARR[:,s]
    v_pred = plan(h1vec,v_0,h_0,p_weights,params)
    v_t = move(vec,dots,params)
    loss = jnp.sum((v_pred-v_t)**2)
    return loss

def train_loop(SC,p_weights,params,opt_state,l):
        loss_arr_,loss_std_ = (jnp.empty((params["EPOCHS"])),jnp.empty((params["EPOCHS"])))
        optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
        for e in range(params["EPOCHS"]):
            dots = params["DOTS"][e,0,:,:]
            s = params["SAMPLE"][e,:]
            v_0 = params["V0"][e,0,:]
            h_0 = params["HP0"][e,:,:]
            val_grad = jax.value_and_grad(body_fnc,argnums=1,allow_int=True)
            val_grad_vmap = jax.vmap(val_grad,in_axes=(None,None,None,None,0,None,0),out_axes=(0,0))
            loss,grads = val_grad_vmap(SC,p_weights,params,dots,s,v_0,h_0) # ,v_0
            jax.debug.print("e={}",l*(params["EPOCHS"])+e)
            jax.debug.print("mean loss={}",jnp.mean(loss))
            loss_arr_ = loss_arr_.at[e].set(jnp.mean(loss))
            loss_std_ = loss_std_.at[e].set(jnp.std(loss)/jnp.sqrt(params["VMAPS"]))
            grads_ = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
            opt_update,opt_state = optimizer.update(grads_,opt_state,p_weights)
            p_weights = optax.apply_updates(p_weights,opt_update)
        return p_weights,loss_arr_,loss_std_,opt_state

# @partial(jax.jit,static_argnums=())
def forward_model_loop(SC,weights,params):
    p_weights = weights["p_weights"]
    E = params["EPOCHS"]
    loss_arr,loss_std = (jnp.empty(params["TOT_EPOCHS"]) for _ in range(2))
    optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    opt_state = optimizer.init(p_weights)
    for l in range(params["LOOPS"]):
        p_weights,loss_arr_,loss_std_,opt_state = train_loop(SC,p_weights,params,opt_state,l)
        loss_arr = loss_arr.at[l*E:(l+1)*E].set(loss_arr_)
        loss_std = loss_std.at[l*E:(l+1)*E].set(loss_std_)
        params = new_params(params,l)
    p_weights_trained = p_weights
    return loss_arr,loss_std,p_weights_trained

# hyperparams
TOT_EPOCHS = 1000 # 1000
EPOCHS = 1
LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 500 # 800,500
PLAN_ITS = 8 # 10,8,5
# STEPS = 10
LR = 0.000001 # 0.0001
WD = 0.0001 # 0.0001
H = 1000 # 500,300
INIT = 0.1 # 0.5,0.1

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
MODULES = 5 # 8
M = MODULES**2
NEURONS = 10
N = 3*(NEURONS**2)
N_DOTS = 3
SIGMA_A = 1 # 0.5,0.3,1,0.5,1,0.1
APERTURE = jnp.pi
THETA_X = jnp.linspace(-APERTURE,APERTURE,NEURONS)
THETA_Y = jnp.linspace(-APERTURE,APERTURE,NEURONS)
COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]])
DOTS = rnd.uniform(ke[0],shape=(EPOCHS,1,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
# KEYS = rnd.split(ke[1],(EPOCHS*VMAPS*3)/2).reshape(EPOCHS,VMAPS,3) # rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
SAMPLE = rnd.choice(ke[2],M,(EPOCHS,VMAPS)) # [E,V] rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
V0 = (INIT/(N))*rnd.normal(ke[3],(EPOCHS,1,N)) # [E,V,N]
HP0 = (INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) # [E,V,H]
ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True) # CHANGE TO PERMUATION
VEC_ARR = gen_vectors(MODULES,APERTURE)
H1VEC_ARR = jnp.diag(jnp.ones(M))[:,ID_ARR]
SC = (ID_ARR,VEC_ARR,H1VEC_ARR)

# INITIALIZATION
ki = rnd.split(rnd.PRNGKey(1),num=50)
W_h10 = (INIT/((H+N)*M))*rnd.normal(ki[0],(H+N,M))
W_v0 = (INIT/((H+N)*N))*rnd.normal(ki[1],(H+N,N))
# Wv_f0 = (INIT/((H+N)*N))*rnd.normal(ki[1],(H+N,N))
# Wv_h0 = (INIT/((H+N)*N))*rnd.normal(ki[2],(H+N,N))
U_vh0 = (INIT/((H+N)**2))*rnd.normal(ki[2],(H+N,H+N))
# U_f0 = (INIT/((H+N)*N))*rnd.normal(ki[4],(H+N,N))
# U_h0 = (INIT/((H+N)*N))*rnd.normal(ki[5],(H+N,N))
b_vh0 = (INIT/(H+N))*rnd.normal(ki[3],(H+N,))
# b_f0 = (INIT/(H+N))*rnd.normal(ki[7],(H+N,1))
# b_h0 = (INIT/(H+N))*rnd.normal(ki[8],(H+N,1))

params = {
    "TOT_EPOCHS" : TOT_EPOCHS,
    "EPOCHS" : EPOCHS,
    "LOOPS" : LOOPS,
    # "STEPS" : STEPS,
    "MODULES" : MODULES,
    "APERTURE" : APERTURE,
    "PLAN_ITS" : PLAN_ITS,
    # "SIGMA_N" : SIGMA_N,
    "LR" : LR,
    "WD" : WD,
    "H" : H,
    "N" : N,
    "INIT" : INIT,
    "VMAPS" : VMAPS,
    # "KEYS" : KEYS,
    "N_DOTS" : N_DOTS,
    "DOTS" : DOTS,
    "THETA_X" : THETA_X,
    "THETA_Y" : THETA_Y,
    "SIGMA_A" : SIGMA_A,
    "COLORS" : COLORS,
    "SAMPLE" : SAMPLE,
    "V0" : V0,
    "HP0" : HP0
}

weights = {
    's_weights' : {

    },
    'p_weights' : {
        "W_h1" : W_h10,
        "W_v" : W_v0,
        "U_vh" : U_vh0,
        "b_vh" : b_vh0
    },
    'r_weights' : {

    }
}

# (planning call wrapper)
        # hp_0 = gen_hm(params)
        # H = params["HORIZON"]
        # m = params["MODULES"]
        # pos_arr = jnp.empty(H,2)
        # v_arr = jnp.empty(H,m**2)
        # r_arr = jnp.empty(H)
        # for i in range(H):
        #     hm_t_1,pos_t_1,v_t_1 = forward_model(self,hp_t_1,pos_t_1,v_t_1,weights["m_weights"])
        #     hv_t_1,r_t_1 = value_model(hv_t_1,v_t_1,r_t_1,weights["v_weights"])
        #     pos_arr.at[i].set[pos_t_1]
        #     v_arr.at[i].set[v_t_1]
        #     r_arr.at[i].set[r_t_1]
        # return pos_arr,v_arr,r_arr

###
startTime = datetime.now()
loss_arr,loss_std,p_weights_trained = forward_model_loop(SC,weights,params)

# plot training loss
print("loss_arr=",loss_arr)
print("loss_std=",loss_std)
print("Training time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
plt.figure(figsize=(12,6))
title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, init={INIT:.2f}, update={LR:.6f}, WD={WD:.5f}, \n SIGMA_A={SIGMA_A:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H={H}'
plt.suptitle('forward_training_v5, '+title__,fontsize=14)
plt.errorbar(jnp.arange(TOT_EPOCHS),loss_arr,yerr=loss_std,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
plt.ylabel(r'Loss',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M")
plt.savefig(path_ + 'forward_train_' + dt + '.png')
# def gen_sc(params):
#     m = params["MODULES"]
#     A = params["APERTURE"]
#     M=m**2
#     sc = [] # jnp.empty(M)6
#     x = np.linspace(-(A-A/m),(A-A/m),m)
#     x_ = np.tile(x,(m,1))
#     x_ = x_.reshape((M,)) # .transpose((1,0))
#     y = np.linspace(-(A-A/m),(A-A/m),m)
#     y_ = np.tile(y.reshape(m,1),(1,m))
#     y_ = y_.reshape((M,))
#     for i in range(M):
#         sc.append(Module(i,[x_[i],y_[i]],M))
#     return sc

    # jax.debug.print('*********************',sample)
    # hp_0 = gen_hp(sample[0],params)
    # v_0 = gen_v(dots,key[0],params) # ,pos_0
    # M = params["MODULES"]**2
    # S = params["STEPS"]
    # v_pred = jnp.empty(S,M)
    # v_arr = jnp.empty(S,M)

    # vals_0 = (h1vec,v_t_1,hp_t_1,p_weights)
    # (h1vec_,v_t_,hp_t_,p_weights_),_ = jax.lax.scan(RNN_loop,vals_0,xs=None,length=params["PLAN_ITS"])