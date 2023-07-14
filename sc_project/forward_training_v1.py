# -*- coding: utf-8 -*-
"""
Created on Wed May 10 2023

@author: lukej
"""
import jax
import jax.numpy as jnp
from jax import jit
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

class Module:
    def __init__(self,id,vector,M):
        self.id = id
        self.vector = vector
        self.h1vec = np.zeros(M)
        self.h1vec[id] = 1

    def move(self,dots,params): # self,pos_t_1,v_t_1,r_t_1,weights,params
        pos_t = self.vector # pos_t_1 + self.vector
        v_t = neuron_act(params,dots,pos_t)
        return v_t
    
    def plan(self,v_t_1,weights,params): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
        hp_t_1 = gen_hp(jnp.int32(jnp.floor(jnp.sum(v_t_1))),params)
        vh_t_1 = jnp.concatenate((v_t_1,hp_t_1))
        for i in range(params["PLAN_ITS"]):
            vh_t_1 = RNN_forward(vh_t_1,weights["p_weights"])
        v_t,hp_t = jnp.split(vh_t_1,2)
        return v_t

    def value(self,hv_t_1,v_t,r_t_1,weights): # self,hv_t_1,v_t,r_t_1,weights,params
        hv_t,r_t = RNN_value(hv_t_1,v_t,r_t_1,weights["r_weights"])
        return hv_t,r_t

def forward_model_loop(sc,weights,params):
    p_weights = weights["p_weights"]
    loss_arr,loss_std = (jnp.empty(params["EPOCHS"]) for _ in range(2))
    optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    # pos_0 = jnp.array([0,0])
    for e in range(params["EPOCHS"]):
        sample = params["SAMPLES"][e,:,:]
        dots = params["DOTS"][e,:,:,:]
        val_grad = jax.value_and_grad(body_fnc,argnums=1,allow_int=True)
        val_grad_vmap = jax.vmap(val_grad,in_axes=(None,None,None,0,0,None),out_axes=(0,0))
        loss,grads = val_grad_vmap(sc,p_weights,params,dots,sample,e) # ,v_0
        loss_arr.at[e].set[jnp.mean(loss)]
        loss_std.at[e].set[jnp.std(loss)]
        grads_ = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
        opt_update,opt_state = optimizer.update(grads_,opt_state,p_weights)
        p_weights = optax.apply_updates(p_weights,opt_update)
    p_weights_trained = p_weights
    return loss_arr,loss_std,p_weights_trained

def body_fnc(sc,p_weights,params,dots,sample,e):
    # jax.debug.print('*********************',sample)
    hp_0 = gen_hp(sample[0],params)
    v_0 = gen_v(dots,sample[1],params) # ,pos_0
    # M = params["MODULES"]**2
    # S = params["STEPS"]
    # v_pred = jnp.empty(S,M)
    # v_arr = jnp.empty(S,M)
    module = sample_module(sc,sample[2],params,e)
    v_pred = module.plan(hp_0,v_0,p_weights)
    v_t = module.move(dots,params)
    loss = (v_pred-v_t)**2
    return loss

def RNN_forward(vh_t_1,p_weights): # GRU computation of hp_t_1,v_t_1->hp_t,v_t
    Wv_z = p_weights["Wv_z"]
    U_z = p_weights["U_z"]
    b_z = p_weights["b_z"]
    vh_t = jax.nn.tanh(jnp.matmul(Wv_z,vh_t_1) + jnp.matmul(U_z,vh_t_1) + b_z) # + jnp.matmul(W_v_z,v_t_1)
    return vh_t

def RNN_value(hv_t_1,v_t,r_t_1,r_weights): # GRU computation of hv_t_1,v_t,r_t_1->hv_t,r_t
    hv_t = hv_t_1
    r_t = r_t_1
    return hv_t,r_t

def sample_module(sc,sample,params,epoch):
    m = params["MODULES"]
    M = m**2
    key_ = rnd.PRNGKey(sample) #params["SAMPLES"][epoch,sample]
    s = rnd.choice(key_,M,shape=())
    jax.debug.print('*********************={}',s)
    module = sc[s]
    return #module

def gen_sc(params):
    m = params["MODULES"]
    A = params["APERTURE"]
    M=m**2
    sc = [] # jnp.empty(M)6
    x = np.linspace(-(A-A/m),(A-A/m),m)
    x_ = np.tile(x,(m,1))
    x_ = x_.reshape((M,)) # .transpose((1,0))
    y = np.linspace(-(A-A/m),(A-A/m),m)
    y_ = np.tile(y.reshape(m,1),(1,m))
    y_ = y_.reshape((M,))
    for i in range(M):
        sc.append(Module(i,[x_[i],y_[i]],M))
    return sc

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

def gen_hp(key,params):
    key_ = rnd.PRNGKey(key)
    H = params["H"]
    INIT = params["INIT"]
    hp_0 = (INIT/H)*rnd.normal(key=key_,shape=(H,))
    return hp_0

def gen_v(dots,key,params): #
    key_ = rnd.PRNGKey(key)
    pos_0 = rnd.uniform(key=key_,shape=(2,),minval=-params["APERTURE"],maxval=params["APERTURE"])
    v_0 = neuron_act(params,dots,pos_0)
    return v_0

def gen_dots(key,EPOCHS,VMAPS,N_DOTS,APERTURE):
    DOTS = rnd.uniform(key,shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE)
    return DOTS

# hyperparams
EPOCHS = 2
VMAPS = 100
PLAN_ITS = 5
# STEPS = 10
LR = 0.001
WD = 0.001
H = 400
INIT = 0.5

# ENV/sc params
MODULES = 8
NEURONS = 10
APERTURE = jnp.pi
SAMPLES = rnd.split(rnd.PRNGKey(0),(EPOCHS*VMAPS*3)/2).reshape(EPOCHS,VMAPS,3) # rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
N_DOTS = 3
DOTS = gen_dots(rnd.PRNGKey(1),EPOCHS,VMAPS,N_DOTS,APERTURE)
THETA_X = jnp.linspace(-APERTURE,APERTURE,NEURONS)
THETA_Y = jnp.linspace(-APERTURE,APERTURE,NEURONS)
SIGMA_A = 0.1
COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]])

# INITIALIZATION
M = MODULES
N = 3*(NEURONS**2)
ki = rnd.split(rnd.PRNGKey(1),num=50)
W_v0 = (INIT/((H+N)*N))*rnd.normal(ki[0],(H+N,N))
# Wv_f0 = (INIT/((H+N)*N))*rnd.normal(ki[1],(H+N,N))
# Wv_h0 = (INIT/((H+N)*N))*rnd.normal(ki[2],(H+N,N))
U_v0 = (INIT/((H+N)*N))*rnd.normal(ki[3],(H+N,N))
# U_f0 = (INIT/((H+N)*N))*rnd.normal(ki[4],(H+N,N))
# U_h0 = (INIT/((H+N)*N))*rnd.normal(ki[5],(H+N,N))
b_v0 = (INIT/(H+N))*rnd.normal(ki[6],(H+N,1))
# b_f0 = (INIT/(H+N))*rnd.normal(ki[7],(H+N,1))
# b_h0 = (INIT/(H+N))*rnd.normal(ki[8],(H+N,1))

params = {
    "EPOCHS" : EPOCHS,
    # "STEPS" : STEPS,
    "MODULES" : MODULES,
    "APERTURE" : APERTURE,
    "PLAN_ITS" : PLAN_ITS,
    # "SIGMA_N" : SIGMA_N,
    "LR" : LR,
    "WD" : WD,
    "H" : H,
    "INIT" : INIT,
    "VMAPS" : VMAPS,
    "SAMPLES" : SAMPLES,
    "N_DOTS" : N_DOTS,
    "DOTS" : DOTS,
    "THETA_X" : THETA_X,
    "THETA_Y" : THETA_Y,
    "SIGMA_A" : SIGMA_A,
    "COLORS" : COLORS
}

weights = {
    's_weights' : {

    },
    'p_weights' : {
        "W_v" : W_v0,
        "U_v" : U_v0,
        "b_v" : b_v0,
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
sc = gen_sc(params)
print(SAMPLES.shape)
# print(type(sc),type(sc[0]),sc[0])
loss_arr,loss_std,p_weights_trained = forward_model_loop(sc,weights,params)
startTime = datetime.now()

# plot training loss
plt.figure()
# plt.errorbar(jnp.arange(EPOCHS),loss_arr,yerr=loss_std)
