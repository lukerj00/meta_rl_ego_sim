# -*- coding: utf-8 -*-
# readout
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
    MODULES = params["MODULES"]
    STEPS = params["STEPS"]
    N = params["N"]
    M = params["M"]
    H = params["H"]
    ki = rnd.split(rnd.PRNGKey(0),num=10) #e
    params_["DOTS"] = rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #jnp.tile(jnp.array([[2,3]]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,1)) #
    params_["POS_0"] = rnd.choice(ke[3],jnp.linspace(-APERTURE,APERTURE,MODULES),(EPOCHS,VMAPS,2))
    params_["SAMPLES"] = rnd.choice(ke[2],M,(EPOCHS,VMAPS,STEPS))
    params_["HP_0"] = (INIT/(H))*rnd.normal(ki[2],(EPOCHS,VMAPS,STEPS,H))
    # params_["V0"] = (INIT/(N))*rnd.normal(ki[1],(EPOCHS,VMAPS,N))
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

# @jit
# def RNN_forward_scan(vals,x): # THINK... GRU computation of hp_t_1,v_t_1->hp_t,v_t
#     # gc.collect()
#     h1vec,v_t_1,h_t_1,p_weights = vals
#     W_h1 = p_weights["W_h1"]
#     W_v = p_weights["W_v"]
#     U_vh = p_weights["U_vh"]
#     b_vh = p_weights["b_vh"]
#     h_t = jax.nn.tanh(jnp.matmul(W_h1,h1vec) + jnp.matmul(W_v,v_t_1) + jnp.matmul(U_vh,h_t_1) + b_vh) # + jnp.matmul(W_v_z,v_t_1)
#     return (h1vec,v_t_1,h_t,p_weights),None #v_t,hp_t

@jit
def RNN_forward(h1vec,v_t_1,h_t_1,p_weights): # THINK... GRU computation of hp_t_1,v_t_1->hp_t,v_t
    W_h1 = p_weights["W_h1"]
    W_v = p_weights["W_v"]
    U_vh = p_weights["U_vh"]
    b_vh = p_weights["b_vh"]
    h_t = jax.nn.sigmoid(jnp.matmul(W_h1,h1vec) + jnp.matmul(W_v,v_t_1) + jnp.matmul(U_vh,h_t_1) + b_vh) # + jnp.matmul(W_v_z,v_t_1)
    return h_t #v_t,hp_t

# @jit # @partial(jax.jit,static_argnums=())
def plan(h1vec,v_0,h_0,p_weights,params): # THINK... self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    W_r = p_weights["W_r"]
    # vh_t_1 = jnp.concatenate((v_0,h_0))
    vals_0 = (h1vec,v_0,h_0,p_weights)
    h_t = jax.lax.scan(RNN_forward,vals_0,xs=None,length=params["PLAN_ITS"])
    v_t = jnp.matmul(W_r,h_t)
    # split_ = jnp.split(vh_t_,jnp.array([v_0.shape[0]]))
    # v_t = split_[0] # hp_t_1 = split_[1]
    return v_t

@jit # @partial(jax.jit,static_argnums=())
def move(pos_t,dots,params): # self,pos_t_1,v_t_1,r_t_1,weights,params
        # pos_t = pos_0 + vec
        v_t = neuron_act(params,dots,pos_t) # pos_t = vector_ # pos_t_1 + self.vector
        return v_t

# @jit # @partial(jax.jit,static_argnums=())
def body_fnc(SC,p_weights,params,dots,samples,pos_0,h_0):
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    loss_tot = 0
    pos_t_1 = pos_0 # VEC_ARR[:,0] #
    jax.debug.print("pos_t_1={}",pos_t_1)
    for s in samples: # range(params["STEPS"]):
        v_t_1 = neuron_act(params,dots,pos_t_1)
        h1vec = H1VEC_ARR[:,s] #,s
        v_pred = plan(h1vec,v_t_1,h_0,p_weights,params) # h_0[s,:]
        v_pred_norm = v_pred/jnp.sum(v_pred)
        pos_t = VEC_ARR[:,s] #,s
        # jax.debug.print("pos_t={}",pos_t)
        v_t = move(pos_t,dots,params)
        # loss_e = params["LAMBDA_E"]*(jnp.sum(jnp.multiply(v_pred_norm,jnp.log(v_pred_norm)))) # penalize low entropy
        # jax.debug.print("loss_e={}",loss_e)
        loss = jnp.sum((v_pred-v_t)**2)
        # jax.debug.print("loss={}",loss)
        loss_tot += loss #+ loss_e
        pos_t_1 = pos_t
    # jax.debug.print("loss={}",loss)
    # jax.debug.print("loss_e={}",params["LAMBDA_E"]*(jnp.sum(jnp.multiply(v_pred,jnp.log(v_pred)))))
    avg_loss = loss_tot/len(samples)
    return avg_loss,(v_pred,v_t,pos_t)

def train_loop(SC,p_weights,params,l): # ,opt_state
        loss_arr_,loss_std_ = (jnp.empty((params["EPOCHS"])),jnp.empty((params["EPOCHS"])))
        # optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
        # opt_state = optimizer.init(p_weights)
        v_pred_arr,v_t_arr = (jnp.empty((params["EPOCHS"],params["VMAPS"],params["N"])) for _ in range(2))
        pos_arr = jnp.empty((params["EPOCHS"],params["VMAPS"],2))
        for e in range(params["EPOCHS"]):
            dots = params["DOTS"][0,0,:,:] # same dots across vmaps
            jax.debug.print("dots={}",dots)
            samples = params["SAMPLES"][0,0,:]
            pos_0 = params["POS_0"][0,0,:] # same pos_0 across vmaps
            jax.debug.print("pos_0={}",pos_0)
            h_0 = params["HP_0"][0,0,0,:]
            val_grad = jax.value_and_grad(body_fnc,argnums=1,allow_int=True,has_aux=True)
            # val_grad_vmap = jax.vmap(val_grad,in_axes=(None,None,None,None,0,None,None),out_axes=(0,0))
            (loss,aux),grads = val_grad(SC,p_weights,params,dots,samples,pos_0,h_0) # val_grad_vmap ,,v_0
            v_pred,v_t,pos_t = aux
            grads_ = grads #jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
            p_weights = jax.tree_util.tree_map(lambda x,y: x - params["LR"]*y,p_weights,grads_)
            jax.debug.print("grads={}",grads_["W_h1"].shape)
            # grads_ = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
            # opt_update,opt_state = optimizer.update(grads_,opt_state,p_weights)
            # p_weights = optax.apply_updates(p_weights,opt_update)
            v_pred_arr = v_pred_arr.at[e,:,:].set(v_pred)
            v_t_arr = v_t_arr.at[e,:,:].set(v_t)
            pos_arr = pos_arr.at[e,:,:].set(pos_t)
            # v_aux = (aux[0][:,:],aux[1][:,:]) # v_pred[0],v_t[0] [N] (0th vmap,E-1'th sample)
            loss_arr_ = loss_arr_.at[e].set(jnp.mean(loss))
            loss_std_ = loss_std_.at[e].set(jnp.std(loss)/jnp.sqrt(params["VMAPS"]))
            jax.debug.print("e={}",l*(params["EPOCHS"])+e)
            jax.debug.print("mean loss={}",jnp.mean(loss))
        v_aux = (v_pred_arr,v_t_arr,pos_arr,params["DOTS"])
        return p_weights,loss_arr_,loss_std_,v_aux # ,opt_state

# @partial(jax.jit,static_argnums=())
def forward_model_loop(SC,weights,params):
    p_weights = weights["p_weights"]
    E = params["EPOCHS"]
    L = params["LOOPS"]
    loss_arr,loss_std = (jnp.empty(params["TOT_EPOCHS"]) for _ in range(2))
    v_pred_arr,v_t_arr = (jnp.empty((2,params["N"])) for _ in range(2))
    optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    opt_state = optimizer.init(p_weights)
    for l in range(L):
        p_weights,loss_arr_,loss_std_,opt_state,aux = train_loop(SC,p_weights,params,opt_state,l)
        loss_arr = loss_arr.at[l*E:(l+1)*E].set(loss_arr_)
        loss_std = loss_std.at[l*E:(l+1)*E].set(loss_std_)
        params = new_params(params,l)
        if(l==0)or(l==(L-1)):
            v_pred_arr = v_pred_arr.at[(l//(L-1)),:].set(aux[0])
            v_t_arr = v_t_arr.at[(l//(L-1)),:].set(aux[1])
    p_weights_trained = p_weights
    return loss_arr,loss_std,p_weights_trained,v_pred_arr,v_t_arr

# hyperparams
TOT_EPOCHS = 2000
EPOCHS = 1
LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 500 # 800,500
PLAN_ITS = 10 # 8,5
# STEPS = 10
LR = 0.00001 # 0.0001
WD = 0.0001 # 0.0001
H = 1000 # 500,300
INIT = 0.5 # 0.1

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
MODULES = 5 # 8
M = MODULES**2
NEURONS = 10
N = 3*(NEURONS**2)
N_DOTS = 3
SIGMA_A = 0.5 # 1,0.3,1,0.5,1,0.1
APERTURE = jnp.pi
THETA_X = jnp.linspace(-APERTURE,APERTURE,NEURONS)
THETA_Y = jnp.linspace(-APERTURE,APERTURE,NEURONS)
COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]])
DOTS = rnd.uniform(ke[0],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
# KEYS = rnd.split(ke[1],(EPOCHS*VMAPS*3)/2).reshape(EPOCHS,VMAPS,3) # rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
SAMPLE = rnd.choice(ke[2],M,(EPOCHS,VMAPS)) # [E,V] rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
POS_0 = rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
# V0 = None ### CHANGE TO ACTUAL ACTIVATIONS CORRESPONDING TO POS_0(INIT/(N))*rnd.normal(ke[3],(EPOCHS,VMAPS,N)) # [E,V,N]
HP_0 = (INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) # [E,V,H]
ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True) # CHANGE TO PERMUATION
VEC_ARR = gen_vectors(MODULES,APERTURE)
H1VEC_ARR = jnp.diag(jnp.ones(M))[:,ID_ARR]
SC = (ID_ARR,VEC_ARR,H1VEC_ARR)

# INITIALIZATION
ki = rnd.split(rnd.PRNGKey(1),num=50)
W_h10 = (INIT/((H)*M))*rnd.normal(ki[0],(H,M))
W_v0 = (INIT/((H)*N))*rnd.normal(ki[1],(H,N))
W_r0 = (INIT/(H*N))*rnd.normal(ki[2],(N,H))
# Wv_f0 = (INIT/((H+N)*N))*rnd.normal(ki[1],(H+N,N))
# Wv_h0 = (INIT/((H+N)*N))*rnd.normal(ki[2],(H+N,N))
U_vh0 = (INIT/((H)**2))*rnd.normal(ki[2],(H,H))
# U_f0 = (INIT/((H+N)*N))*rnd.normal(ki[4],(H+N,N))
# U_h0 = (INIT/((H+N)*N))*rnd.normal(ki[5],(H+N,N))
b_vh0 = (INIT/(H))*rnd.normal(ki[3],(H,))
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
    "POS_0" : POS_0,
    # "V_0" : V_0,
    "HP_0" : HP_0
}

weights = {
    's_weights' : {

    },
    'p_weights' : {
        "W_h1" : W_h10,
        "W_v" : W_v0,
        "W_r" : W_r0,
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
plt.suptitle('forward_training_v4, '+title__,fontsize=14)
plt.errorbar(jnp.arange(TOT_EPOCHS),loss_arr,yerr=loss_std,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
plt.ylabel(r'Loss',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M")
plt.savefig(path_ + 'forward_train_' + dt + '.png')

# plot before and after heatmaps using v_pred_arr and v_t_arr:
print("v_pred_arr=",v_pred_arr)
print("v_t_arr=",v_t_arr)
# separate each v in to r/g/b channels
# 
# reshape each channel to neuron^2
#
# plot (dots+pos_0) [low alpha] and (dots+pos_t) as scatter 
# plot each channel as heatmap

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