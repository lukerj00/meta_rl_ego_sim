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

# class Module:
#     def __init__(self,id,vector,M):
#         self.id = id
#         self.vector = vector
#         self.h1vec = np.zeros(M)
#         self.h1vec[id] = 1

def sample_module(SC,key_,params):
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    M = params["MODULES"]**2
    jax.debug.print("M={}",M)
    jax.debug.print("key_.shape___={}",key_.shape)
    s = rnd.choice(key_,jnp.arange(M))
    vec = VEC_ARR[:,s]
    h1vec = H1VEC_ARR[:,s]
    return h1vec,vec

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

def gen_hp(key_,params):
    H = params["H"]
    INIT = params["INIT"]
    hp_0 = (INIT/H)*rnd.normal(key_,shape=(H,))
    return hp_0

def gen_v(dots,key,params): #
    key_ = rnd.PRNGKey(key)
    pos_0 = rnd.uniform(key=key_,shape=(2,),minval=-params["APERTURE"],maxval=params["APERTURE"])
    v_0 = neuron_act(params,dots,pos_0)
    return v_0

def gen_dots(key,EPOCHS,VMAPS,N_DOTS,APERTURE):
    DOTS = rnd.uniform(key,shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE)
    return DOTS

def gen_vectors(MODULES,APERTURE):
    m = MODULES
    M = m**2
    A = APERTURE
    x = np.linspace(-(A-A/m),(A-A/m),m) # CHECK
    x_ = np.tile(x,(m,1))
    x_ = x_.reshape((M,)) # .transpose((1,0))
    y = np.linspace(-(A-A/m),(A-A/m),m) # CHECK
    y_ = np.tile(y.reshape(m,1),(1,m))
    y_ = y_.reshape((M,))
    v = jnp.vstack([x_,y_])
    return v

def RNN_forward(h1vec,v_t_1,hp_t_1,p_weights): # GRU computation of hp_t_1,v_t_1->hp_t,v_t
    W_h1 = p_weights["W_h1"]
    W_v = p_weights["W_v"]
    U_vh = p_weights["U_vh"]
    b_vh = p_weights["b_vh"]
    vh_t_1 = jnp.concatenate((v_t_1,hp_t_1))
    # jax.debug.print("***hp_t_1.shape={}",hp_t_1.shape)
    # jax.debug.print("***hp_t_1.shape[0]={}",v_t_1.shape[0])
    # jax.debug.print("W_h1.shape={}",W_h1.shape)
    # jax.debug.print("h1vec.shape={}",h1vec.shape)
    # jax.debug.print("W_vh.shape={}",W_v.shape)
    # jax.debug.print("v_t_1.shape={}",v_t_1.shape)
    # jax.debug.print("v_t_1.shape[0]={}",v_t_1.shape[0])
    # jax.debug.print("U_vh.shape={}",U_vh.shape)
    # jax.debug.print("vh_t_1.shape={}",vh_t_1.shape)
    # jax.debug.print("b_vh.shape={}",b_vh.shape)
    vh_t = jax.nn.tanh(jnp.matmul(W_h1,h1vec) + jnp.matmul(W_v,v_t_1) + jnp.matmul(U_vh,vh_t_1) + b_vh) # + jnp.matmul(W_v_z,v_t_1)
    split_ = jnp.split(vh_t,jnp.array([v_t_1.shape[0]]))
    v_t = split_[0]
    hp_t = split_[1]
    # jax.debug.print("v_t.shape={}",v_t.shape)
    # jax.debug.print("hp_t.shape={}",hp_t.shape)
    return v_t,hp_t

def RNN_value(hv_t_1,v_t,r_t_1,r_weights): # GRU computation of hv_t_1,v_t,r_t_1->hv_t,r_t
    hv_t = hv_t_1
    r_t = r_t_1
    return hv_t,r_t

def move(vec,dots,params): # self,pos_t_1,v_t_1,r_t_1,weights,params
        v_t = neuron_act(params,dots,vec) # pos_t = vector_ # pos_t_1 + self.vector
        return v_t
    
def plan(h1vec,v_0,p_weights,params,hp_0): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    # hp_0 = gen_hp(key_,params) # PREGEN , jnp.int32(jnp.floor(1000*jnp.sum(v_0)))
    v_t_1 = v_0
    hp_t_1 = hp_0
    for i in range(params["PLAN_ITS"]):
        v_t_1,hp_t_1 = RNN_forward(h1vec,v_t_1,hp_t_1,p_weights)
    return v_t_1

def value(SC,hv_t_1,v_t,r_t_1,r_weights): # self,hv_t_1,v_t,r_t_1,weights,params
    hv_t,r_t = RNN_value(hv_t_1,v_t,r_t_1,r_weights)
    return hv_t,r_t

def body_fnc(SC,p_weights,params,dots,s,v_0,hp_0):
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    # jax.debug.print("key={}",key)
    # s = rnd.choice(key[0:2],jnp.arange(M))
    h1vec = H1VEC_ARR[:,s]
    vec = VEC_ARR[:,s]
    # h1vec,vec = sample_module(SC,key[1],params) # just sample random number to choose module
    # v_0 = gen_v(dots,key[1],params)
    v_pred = plan(h1vec,v_0,p_weights,params,hp_0)
    # jax.debug.print("v_pred={}",v_pred)
    v_t = move(vec,dots,params)
    # jax.debug.print("v_t={}",v_t)
    loss = jnp.sum((v_pred-v_t)**2)
    # jax.debug.print("loss={}",loss)
    return loss

def forward_model_loop(SC,weights,params):
    p_weights = weights["p_weights"]
    loss_arr,loss_std = (jnp.empty(params["EPOCHS"]) for _ in range(2))
    # jax.debug.print("loss_arr_0={}",loss_arr)
    optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    opt_state = optimizer.init(p_weights)
    for e in range(params["EPOCHS"]):
        dots = params["DOTS"][e,:,:,:]
        s = params["SAMPLE"][e,:]
        v_0 = params["V0"][e,:,:]
        hp_0 = params["HP0"][e,:,:]
        val_grad = jax.value_and_grad(body_fnc,argnums=1,allow_int=True)
        val_grad_vmap = jax.vmap(val_grad,in_axes=(None,None,None,0,0,0,0),out_axes=(0,0))
        loss,grads = val_grad_vmap(SC,p_weights,params,dots,s,v_0,hp_0) # ,v_0
        jax.debug.print("epoch={}",e)
        jax.debug.print("mean loss={}",jnp.mean(loss))
        # jax.debug.print("grads={}",grads)
        loss_arr = loss_arr.at[e].set(jnp.mean(loss))
        loss_std = loss_std.at[e].set(jnp.std(loss))
        # jax.debug.print("loss_arr={}",loss_arr)
        grads_ = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
        # jax.debug.print("grads_={}",grads_)
        opt_update,opt_state = optimizer.update(grads_,opt_state,p_weights)
        p_weights = optax.apply_updates(p_weights,opt_update)
    p_weights_trained = p_weights
    return loss_arr,loss_std,p_weights_trained

# hyperparams
EPOCHS = 50
VMAPS = 1000 # 500
PLAN_ITS = 8 # 5
# STEPS = 10
LR = 0.0001 # 0.0001
WD = 0.0001
H = 500 # 300
INIT = 0.5 # 0.1

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
MODULES = 5 # 8
M = MODULES**2
NEURONS = 10
N = 3*(NEURONS**2)
APERTURE = jnp.pi
N_DOTS = 3
THETA_X = jnp.linspace(-APERTURE,APERTURE,NEURONS)
THETA_Y = jnp.linspace(-APERTURE,APERTURE,NEURONS)
SIGMA_A = 0.5 # 0.5,1,0.1
COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]])
DOTS = gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
KEYS = rnd.split(ke[1],(EPOCHS*VMAPS*3)/2).reshape(EPOCHS,VMAPS,3) # rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
SAMPLE = rnd.choice(ke[2],M,(EPOCHS,VMAPS)) # [E,V] rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
V0 = (INIT/(N))*rnd.normal(ke[3],(EPOCHS,VMAPS,N)) # [E,V,N]
HP0 = (INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) # [E,V,H]
ID_ARR = rnd.shuffle(ke[5],jnp.arange(0,M))
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
    "KEYS" : KEYS,
    "N_DOTS" : N_DOTS,
    "DOTS" : DOTS,
    "THETA_X" : THETA_X,
    "THETA_Y" : THETA_Y,
    "SIGMA_A" : SIGMA_A,
    "COLORS" : COLORS,
    "SAMPLE" : SAMPLE,
    "V0" : V0,
    "HP0" : HP0,
}

weights = {
    's_weights' : {

    },
    'p_weights' : {
        "W_h1" : W_h10,
        "W_v" : W_v0,
        "U_vh" : U_vh0,
        "b_vh" : b_vh0,
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
startTime = datetime.now()

# plot training loss
print("loss_arr=",loss_arr)
print("loss_std=",loss_std)
print("Training time: ",datetime.now()-startTime)
plt.figure()
title__ = f'epochs={EPOCHS}, vmaps={VMAPS}, init={INIT:.2f}, update={LR:.5f}, SIGMA_A={SIGMA_A:.1f}, \n WD={WD:.5f}, NEURONS={NEURONS**2}, MODULES={M},'#, p=R_norm/{5}' # \n colors={jnp.array_str(COLORS[0][:]) + jnp.array_str(COLORS[1][:]) + jnp.array_str(COLORS[2][:])}' #  + jnp.array_str(COLORS[3][:]) + jnp.array_str(COLORS[4][:])}'
plt.errorbar(jnp.arange(EPOCHS),loss_arr,yerr=loss_std)
plt.ylabel(r'Loss',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M")
plt.savefig(path_ + 's_train_' + dt + '.png')
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