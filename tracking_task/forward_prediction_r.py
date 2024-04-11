# -*- coding: utf-8 -*-
# proof of concept for activation prediction
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

# @jit # @partial(jax.jit,static_argnums=())
# def neuron_act_old(params,dots,pos): #e_t_1,th_j,th_i,SIGMA_A,COLORS
#     COLORS = params["COLORS"]
#     th_x = params["THETA_X"]
#     th_y = params["THETA_Y"]
#     SIGMA_A = params["SIGMA_A"]
#     D_ = COLORS.shape[0]
#     N_ = th_x.size
#     G_0 = jnp.vstack((jnp.tile(th_x,N_),jnp.tile(th_y,N_)))
#     G = jnp.tile(G_0.reshape(2,N_**2,1),(1,1,D_))
#     C = (COLORS/255).transpose((1,0))
#     E = G.transpose((1,0,2)) - (dots-pos).T
#     act = jnp.exp((jnp.cos(E[:,0,:]) + jnp.cos(E[:,1,:]) - 2)/SIGMA_A**2).reshape((D_,N_**2))
#     act_r,act_g,act_b = jnp.matmul(C,act)
#     act_rgb = jnp.concatenate((act_r,act_g,act_b))
#     return act_rgb

@jit # @partial(jax.jit,static_argnums=())
def neuron_act(params,dots,pos): #e_t_1,th_j,th_i,SIGMA_A,COLORS
    COLORS = params["COLORS"]
    th_x = params["THETA_X"]
    th_y = params["THETA_Y"]
    SIGMA_A = params["SIGMA_A"]
    D_ = COLORS.shape[0]
    N_ = th_x.size
    th_x = jnp.tile(th_x,(N_,1)).reshape((N_**2,))
    th_y = jnp.tile(jnp.flip(th_y).reshape(N_,1),(1,N_)).reshape((N_**2,))
    G_0 = jnp.vstack([th_x,th_y])
    # G_0 = jnp.vstack((jnp.tile(th_x,N_),jnp.tile(th_y,N_)))
    G = jnp.tile(G_0.reshape(2,N_**2,1),(1,1,D_))
    # jax.debug.print("G={}",G)
    # jax.debug.print("dots={}",dots)
    C = (COLORS/255).transpose((1,0))
    E = G.transpose((1,0,2)) - ((dots-pos).T) #.reshape((2,1))
    # jax.debug.print("E={}",E)
    act = jnp.exp((jnp.cos(E[:,0,:]) + jnp.cos(E[:,1,:]) - 2)/SIGMA_A**2).T #.reshape((D_,N_**2))
    act_r,act_g,act_b = jnp.matmul(C,act) #.reshape((3*N_**2,))
    act_rgb = jnp.concatenate((act_r,act_g,act_b))
    return act_rgb

def mod_(val):
    return (val+jnp.pi)%(2*jnp.pi)-jnp.pi

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
def gen_dots(key,EPOCHS,VMAPS,N_DOTS,APERTURE):
    keys = rnd.split(key,N_DOTS)
    dots_0 = rnd.uniform(keys[0],shape=(EPOCHS,VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([APERTURE/4,APERTURE/4]),maxval=jnp.array([3*APERTURE/4,3*APERTURE/4]))
    dots_1 = rnd.uniform(keys[1],shape=(EPOCHS,VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([APERTURE/4,-APERTURE/4]),maxval=jnp.array([3*APERTURE/4,-3*APERTURE/4]))
    dots_2 = rnd.uniform(keys[2],shape=(EPOCHS,VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([-3*APERTURE/4,-APERTURE]),maxval=jnp.array([-APERTURE/4,APERTURE]))
    dots_tot = jnp.concatenate((dots_0,dots_1,dots_2),axis=2)
    # DOTS = rnd.uniform(key,shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE)
    return dots_tot[:,:,:N_DOTS,:]

@partial(jax.jit,static_argnums=(0,))
def gen_vectors(MODULES,APERTURE):
    m = MODULES
    M = m**2
    A = APERTURE
    x = jnp.linspace(-(A-A/m),(A-A/m),m) # x = jnp.linspace(-(A-A/m),(A-A/m),m) # CHECK
    x_ = jnp.tile(x,(m,1))
    x_ = x_.reshape((M,)) # .transpose((1,0))
    y = jnp.linspace(-(A-A/m),(A-A/m),m) # y = jnp.linspace(-(A-A/m),(A-A/m),m) # CHECK
    y_ = jnp.tile(jnp.flip(y).reshape(m,1),(1,m))
    y_ = y_.reshape((M,))
    v = jnp.vstack([x_,y_]) # [2,M]
    return v

def new_params(params,l):
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
    ki = rnd.split(rnd.PRNGKey(0),num=10) #l/0
    params_["DOTS"] = gen_dots(ki[0],EPOCHS,VMAPS,N_DOTS,APERTURE) #rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #jnp.tile(jnp.array([[2,3]]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,1)) #
    # params_["POS_0"] = rnd.choice(ki[1],jnp.linspace(-(APERTURE-APERTURE/MODULES),(APERTURE-APERTURE/MODULES),MODULES),(EPOCHS,VMAPS,2))
    params_["SAMPLES"] = rnd.choice(ki[2],M,(EPOCHS,VMAPS,STEPS))
    params_["HP_0"] = (INIT/(H))*rnd.normal(ki[3],(EPOCHS,VMAPS,H))
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
def RNN_forward(h1vec,v_0,h_t_1,p_weights): # GRU computation of hp_t_1,v_t_1->hp_t,v_t
    # gc.collect()
    W_h1 = p_weights["W_h1"]
    W_v = p_weights["W_v"]
    U_vh = p_weights["U_vh"]
    b_vh = p_weights["b_vh"]
    h_t = jax.nn.sigmoid(jnp.matmul(W_h1,h1vec) + jnp.matmul(W_v,v_0) + jnp.matmul(U_vh,h_t_1) + b_vh) # + jnp.matmul(W_v_z,v_t_1)
    return h_t #v_t,hp_t

# @partial(jax.jit,static_argnums=())
def plan(h1vec,v_0,h_0,p_weights,params): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    W_r = p_weights["W_r"]
    h_t_1 = h_0
    for i in range(params["PLAN_ITS"]):
        h_t_1 = RNN_forward(h1vec,v_0,h_t_1,p_weights) # vh_t_1
    h_t = h_t_1
    v_t = jnp.matmul(W_r,h_t)#####
    # split_ = jnp.split(vh_t_1,jnp.array([v_0.shape[0]]))
    # v_t = split_[0] # hp_t_1 = split_[1]
    return v_t,h_t

# @partial(jax.jit,static_argnums=())
def move(pos_t,dots,params): # self,pos_t_1,v_t_1,r_t_1,weights,params
        v_t = neuron_act(params,dots,pos_t) # pos_t = pos_0 + vec
        return v_t

# @partial(jax.jit,static_argnums=())
def body_fnc(SC,p_weights,params,dots,samples,h_0): # SC
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    loss_tot = 0
    pos_t_1 = VEC_ARR[:,0]
    h_t_1 = h_0
    # jax.debug.print("pos_t_1={}",pos_t_1)
    for s in range(1,params["STEPS"]):
        v_t_1 = neuron_act(params,dots,pos_t_1)
        h1vec = H1VEC_ARR[:,s]
        # h_t_1 = h_0[s,:]
        v_pred,h_t = plan(h1vec,v_t_1,h_t_1,p_weights,params) # h_0[s,:]
        # v_pred_norm = v_pred/jnp.sum(v_pred)
        pos_t = VEC_ARR[:,s]
        # jax.debug.print("dots={}",dots)
        # jax.debug.print("pos_t={}",pos_t)
        v_t = move(pos_t,dots,params) # pos_t
        # loss_e = params["LAMBDA_E"]*(jnp.sum(jnp.multiply(v_pred_norm,jnp.log(v_pred_norm)))) # penalize low entropy
        # loss_ex = jnp.sum(v_pred[0:10])
        # jax.debug.print("loss_e={}",loss_e)
        loss = jnp.sum((v_pred-v_t)**2)
        # jax.debug.print("loss={}",loss)
        loss_tot += loss #+ loss_e + 10000*loss_ex
        pos_t_1 = pos_t
        h_t_1 = h_t ###
        # jax.debug.print("loss={}",loss)
        # jax.debug.print("loss_e={}",params["LAMBDA_E"]*(jnp.sum(jnp.multiply(v_pred,jnp.log(v_pred)))))
    avg_loss = loss_tot/len(samples)
    return avg_loss,(v_pred,v_t,pos_t) # avg_loss,,pos_t

def train_loop(SC,p_weights,params,l,opt_state): # ,opt_state SC
        loss_arr_,loss_std_ = (jnp.empty((params["EPOCHS"])),jnp.empty((params["EPOCHS"])))
        # optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
        v_pred_arr,v_t_arr = (jnp.empty((params["EPOCHS"],params["VMAPS"],params["N"])) for _ in range(2))
        pos_arr = jnp.empty((params["EPOCHS"],params["VMAPS"],2))
        for e in range(params["EPOCHS"]):
            dots = params["DOTS"][0,0,:,:] # same dots across vmaps
            # jax.debug.print("dots={}",dots)
            samples = params["SAMPLES"][0,0,:]
            # pos_0 = params["POS_0"][0,0,:] # same pos_0 across vmaps
            # jax.debug.print("pos_0={}",pos_0)
            h_0 = params["HP_0"][0,0,:] # [e,v,h],
            val_grad = jax.value_and_grad(body_fnc,argnums=1,allow_int=True,has_aux=True)
            val_grad_vmap = jax.vmap(val_grad,in_axes=(None,None,None,0,0,0),out_axes=(0,0))
            (loss,aux),grads = val_grad(SC,p_weights,params,dots,samples,h_0) # val_grad_vmap ,,v_0
            v_pred,v_t,pos_t = aux
            grads_ = grads #jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
            p_weights = jax.tree_util.tree_map(lambda x,y: x - params["LR"]*y,p_weights,grads_)
            # grads_ = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads) # grads
            # opt_update,opt_state = optimizer.update(grads_,opt_state,p_weights)
            # p_weights = optax.apply_updates(p_weights,opt_update)
            v_pred_arr = v_pred_arr.at[e,:,:].set(v_pred)
            v_t_arr = v_t_arr.at[e,:,:].set(v_t)
            pos_arr = pos_arr.at[e,:,:].set(pos_t) # [e,v,2]
            loss_arr_ = loss_arr_.at[e].set(jnp.mean(loss))
            loss_std_ = loss_std_.at[e].set(jnp.std(loss)/jnp.sqrt(params["VMAPS"]))
            jax.debug.print("e={}",l*(params["EPOCHS"])+e)
            jax.debug.print("mean loss={}",jnp.mean(loss))
        v_aux = (v_pred_arr,v_t_arr,pos_arr,params["DOTS"]) # [e,v,n,2]
        return p_weights,loss_arr_,loss_std_,v_aux,opt_state

# @partial(jax.jit,static_argnums=())
def forward_model_loop(SC,weights,params):
    p_weights = weights["p_weights"]
    E = params["EPOCHS"]
    L = params["LOOPS"]
    loss_arr,loss_std = (jnp.empty(params["TOT_EPOCHS"]) for _ in range(2))
    v_pred_arr,v_t_arr = (jnp.empty((params["PLOTS"],params["N"])) for _ in range(2))
    optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    opt_state = optimizer.init(p_weights)
    for l in range(L):
        params = new_params(params,l)
        p_weights,loss_arr_,loss_std_,(v_pred_arr,v_t_arr,pos_arr,dots_arr),opt_state = train_loop(SC,p_weights,params,l,opt_state) #,opt_state,opt_state
        loss_arr = loss_arr.at[l*E:(l+1)*E].set(loss_arr_)
        loss_std = loss_std.at[l*E:(l+1)*E].set(loss_std_)
        # if(l==(L-1)): # (l==0)or
            # v_pred_arr = v_pred_arr.at[0,:].set(v_aux[0]) # (l//(L-1))
            # v_t_arr = v_t_arr.at[0,:].set(v_aux[1]) # (l//(L-1))
            # dots_arr = dots_arr.at[0,:,:].set(params["DOTS"][-1,0,:,:]) # final epoch, 0th vmap
    p_weights_trained = p_weights
    return loss_arr,loss_std,p_weights_trained,v_pred_arr,v_t_arr,pos_arr,dots_arr

# hyperparams
TOT_EPOCHS = 150
EPOCHS = 50
PLOTS = 1
LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 100 # 800,500
PLAN_ITS = 10 # 8,5
STEPS = 20 # 10
LR = 0.0001 # 0.0001
WD = 0.0001 # 0.0001
H = 500 # 500,300
INIT = 0.1 # 0.5,0.1
LAMBDA_E = 10 # 0.1

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
MODULES = 2 # 8
M = MODULES**2
NEURONS = 20
N = 3*(NEURONS**2)
SIGMA_A = 0.5 # 0.5,1,0.3,1,0.5,1,0.1
APERTURE = jnp.pi
THETA_X = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
THETA_Y = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]]) # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = COLORS.shape[0]
DOTS = gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
# KEYS = rnd.split(ke[1],(EPOCHS*VMAPS*3)/2).reshape(EPOCHS,VMAPS,3) # rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
SAMPLES = rnd.choice(ke[2],M,(EPOCHS,VMAPS,STEPS)) # [E,V] rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
POS_0 = rnd.choice(ke[3],jnp.linspace(-(APERTURE-APERTURE/MODULES),(APERTURE-APERTURE/MODULES),MODULES),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
# V0 = None ### CHANGE TO ACTUAL ACTIVATIONS CORRESPONDING TO POS_0(INIT/(N))*rnd.normal(ke[3],(EPOCHS,VMAPS,N)) # [E,V,N]
HP_0 = (INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) #rnd.normal(ke[4],(EPOCHS,VMAPS,STEPS,H)) # [E,V,H]
ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True) # CHANGE TO PERMUATION
VEC_ARR = gen_vectors(MODULES,APERTURE)
H1VEC_ARR = jnp.diag(jnp.ones(M))[:,ID_ARR]
SC = (ID_ARR,VEC_ARR,H1VEC_ARR)

# INITIALIZATION
ki = rnd.split(rnd.PRNGKey(1),num=50)
W_h10 = (INIT/((H)*M))*rnd.normal(ki[0],(H,M))
W_v0 = (INIT/((H)*N))*rnd.normal(ki[1],(H,N))
W_r0 = (INIT/((H)*N))*rnd.normal(ki[2],(N,H))
U_vh0 = (INIT/((H)*H))*rnd.normal(ki[2],(H,H))
b_vh0 = (INIT/(H))*rnd.normal(ki[3],(H,))

params = {
    "TOT_EPOCHS" : TOT_EPOCHS,
    "EPOCHS" : EPOCHS,
    "LOOPS" : LOOPS,
    "STEPS" : STEPS,
    "MODULES" : MODULES,
    "APERTURE" : APERTURE,
    "PLAN_ITS" : PLAN_ITS,
    # "SIGMA_N" : SIGMA_N,
    "LR" : LR,
    "WD" : WD,
    "H" : H,
    "N" : N,
    "M" : M,
    "INIT" : INIT,
    "VMAPS" : VMAPS,
    "PLOTS" : PLOTS,
    # "KEYS" : KEYS,
    "N_DOTS" : N_DOTS,
    "DOTS" : DOTS,
    "THETA_X" : THETA_X,
    "THETA_Y" : THETA_Y,
    "SIGMA_A" : SIGMA_A,
    "COLORS" : COLORS,
    "SAMPLES" : SAMPLES,
    "STEPS" : STEPS,
    # "POS_0" : POS_0,
    # "V_0" : V_0,
    "HP_0" : HP_0,
    "LAMBDA_E" : LAMBDA_E,
}

weights = {
    's_weights' : {

    },
    'p_weights' : {
        "W_h1" : W_h10,
        "W_r" : W_r0,
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
loss_arr,loss_std,p_weights_trained,v_pred_arr,v_t_arr,pos_arr,dots_arr = forward_model_loop(SC,weights,params)

# plot training loss
print("Training time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
plt.figure(figsize=(12,6))
title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, STEPS = {STEPS}, init={INIT:.2f}, update={LR:.6f}, WD={WD:.5f}, \n SIGMA_A={SIGMA_A:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H={H}'
plt.suptitle('forward_prediction, '+title__,fontsize=14)
plt.errorbar(jnp.arange(TOT_EPOCHS),loss_arr,yerr=loss_std,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
plt.ylabel(r'Loss',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_ + 'forward_train_' + dt + '.png')

# plot before and after heatmaps using v_pred_arr and v_t_arr:
# print("v_pred_arr=",v_pred_arr.shape) # (e,v,3,neurons**2)
# print("v_t_arr=",v_t_arr.shape) # (e,v,3,neurons**2)
# print("pos_arr=",pos_arr) # (e,v,2)
# print("dots_arr=",dots_arr,dots_arr.shape) # (e,v,3,2)
# print('dots_arr_0',dots_arr[0,0,:,:])
# print('dots_arr_1',dots_arr[-1,0,:,:])
colors_ = np.float32([[255,0,0],[0,255,0],[0,0,255]])/255 # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])/255

# separate each v in to r/g/b channels
print(v_pred_arr.shape)
v_pred_rgb = jnp.abs(v_pred_arr[-1,0,:].reshape((3,NEURONS**2))) # ((-1 if e))
print(v_pred_rgb.shape)
# print("v_pred_rgb=",v_pred_rgb[0,:],v_pred_rgb[1,:],v_pred_rgb[2,:])
print(v_t_arr.shape)
v_t_rgb = v_t_arr[-1,0,:].reshape((3,NEURONS**2))
print(v_t_rgb.shape)
# print("v_t_rgb=",v_t_rgb[0,:],v_t_rgb[1,:],v_t_rgb[2,:])
print(pos_arr[0,0,:],pos_arr[1,0,:],pos_arr[-2,0,:],pos_arr[-1,0,:],pos_arr.shape)
pos_arr = pos_arr[0,0,:] # (0th vmap of '0th' epoch of final gen)
print(pos_arr.shape)
print(dots_arr[0,0,:,:],dots_arr[1,0,:,:],dots_arr[-2,0,:,:],dots_arr[-1,0,:,:],dots_arr.shape)
dots_arr = dots_arr[0,0,:,:] # (0th vmap of '0th' epoch of final gen)
print(dots_arr.shape)
# plot rgb values at each neuron location
neuron_locs = gen_vectors(NEURONS,APERTURE)
# plot true dot locations
fig,axis = plt.subplots(PLOTS,2,figsize=(9,4*PLOTS)) # plt.figure(figsize=(12,6))
title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, STEPS = {STEPS}, init={INIT:.2f}, update={LR:.6f}, WD={WD:.5f}, \n SIGMA_A={SIGMA_A:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H={H}'
plt.suptitle('figs_forward_prediction, '+title__,fontsize=10)
for i in range(params["PLOTS"]):
    # print("i=",i)
    # predicted
    ax0 = plt.subplot2grid((2*PLOTS,2),(2*i,0),colspan=1,rowspan=1)
    ax0.set_title('v_pred_arr')
    ax0.scatter(-pos_arr[0],-pos_arr[1],c='k',marker='+',s=40)
    for ind,n in enumerate(neuron_locs.T):
        print('0:ind=',ind,'n=',n,'v_pred_rgb[:,ind]=',v_pred_rgb[:,ind],v_pred_rgb[:,ind].shape)
        ax0.scatter(n[0],n[1],s=np.sum(25*v_pred_rgb[:,ind]),marker='o',color=np.float32(jnp.sqrt((v_pred_rgb[:,ind])))) #,sum,np.float32((v_pred_rgb[:,ind]))
    # true
    ax1 = plt.subplot2grid((2*PLOTS,2),(2*i,1),colspan=1,rowspan=1)
    ax1.set_title('v_t_arr')
    ax1.scatter(-pos_arr[0],-pos_arr[1],c='k',marker='+',s=40)
    for d in range(N_DOTS):
        ax0.scatter(mod_(dots_arr[d,0]-pos_arr[0]),mod_(dots_arr[d,1]-pos_arr[1]),color=(colors_[d,:]),s=60,marker='x') # [d,0],[d,1].. dots_arr[i,d,0]
        ax1.scatter(mod_(dots_arr[d,0]-pos_arr[0]),mod_(dots_arr[d,1]-pos_arr[1]),color=(colors_[d,:]),s=60,marker='x') # dots_arr[i,d,0]
        # print('dashape',dots_arr,'pos_arr=',pos_arr,'mod_=',mod_(dots_arr[0]-pos_arr[0]),mod_(dots_arr[1]-pos_arr[1]))
    # ax1.scatter(dots_arr[i][:,0],dots_arr[i][:,1],color='black',s=100)
    for ind,n in enumerate(neuron_locs.T):
        print('1:ind=',ind,'n=',n,'v_t_rgb[:,ind]=',v_t_rgb[:,ind],v_t_rgb[:,ind].shape)
        ax1.scatter(n[0],n[1],s=np.sum(20*v_t_rgb[:,ind]),marker='o',color=np.float32((v_t_rgb[:,ind]))) #,,sum,np.float32((v_t_rgb[:,ind]))
ax0.set_xticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
ax0.set_xticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=12)
ax0.set_yticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
ax0.set_yticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=12)
ax0.set_xlim(-jnp.pi,jnp.pi)
ax0.set_ylim(-jnp.pi,jnp.pi)
ax0.set_aspect('equal')
ax1.set_xticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
ax1.set_xticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=12)
ax1.set_yticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
ax1.set_yticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=12)
ax1.set_xlim(-jnp.pi,jnp.pi)
ax1.set_ylim(-jnp.pi,jnp.pi)
ax1.set_aspect('equal')
plt.subplots_adjust(top=0.8) 

dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_ + 'figs_forward_train_' + dt + '.png') # + dt + 

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
