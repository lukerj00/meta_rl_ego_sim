# -*- coding: utf-8 -*-
# multiple steps
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

def save_pkl(param,str_):  # can't jit (can't pickle jax tracers)
	path_ = str(Path(__file__).resolve().parents[1]) + '/pkl/' # '/scratch/lrj34/'
	dt = datetime.now().strftime("%d_%m-%H%M")
	with open(path_+str_+'_'+dt+'.pkl','wb') as file:
		pickle.dump(param,file,pickle.HIGHEST_PROTOCOL)

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

@jit
def loss_obj(dots,sel,pos,SIGMA_R): # R_t
    dots = dots - pos
    obj = jnp.exp((jnp.cos(dots[:,0]) + jnp.cos(dots[:,1]) - 2)/SIGMA_R**2) ### (positive), standard loss (-) (1/(sigma_e*jnp.sqrt(2*jnp.pi)))*
    R_obj = jnp.dot(obj,sel)
    return R_obj #sigma_e

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

def gen_vr_arr(SC,dots,sel,samples,pos_0,params):
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    M = params["M"]
    MODULES = params["MODULES"]
    pos_arr = jnp.zeros((2,params["TRIAL_LENGTH"]))
    pos_arr = pos_arr.at[:,0].set(pos_0) #VEC_ARR[:,rnd.choice(samples[0],jnp.arange(params["M"]),shape=())])
    choice_ = jnp.array([M//2-MODULES,M//2-1,M//2+1,M//2+MODULES])
    # jax.debug.print("choice_={}",choice_)
    for i,s in enumerate(samples[1:]):
        pos_arr = pos_arr.at[:,i+1].set(pos_arr[:,i] + VEC_ARR[:,rnd.choice(rnd.PRNGKey(s),choice_,shape=())]) # jnp.arange(params["M"])
    v_arr = jax.vmap(neuron_act,in_axes=(None,None,1),out_axes=(0))(params,dots,pos_arr) # TL,N
    r_arr = jax.vmap(loss_obj,in_axes=(None,None,1,None),out_axes=(0))(dots,sel,pos_arr,params["SIGMA_R"]) #TL,
    return v_arr,r_arr,pos_arr.T

def new_params(params,l):
    params_ = params
    EPOCHS = params["EPOCHS"]
    VMAPS = params["VMAPS"]
    N_DOTS = params["N_DOTS"]
    APERTURE = params["APERTURE"]
    MODULES = params["MODULES"]
    N = params["N"]
    M = params["M"]
    H = params["H"]
    T = params["TRIAL_LENGTH"]
    key_ = rnd.PRNGKey(0)
    ki = rnd.split(rnd.PRNGKey(l),num=10) #l/0
    params_["DOTS"] = gen_dots(ki[0],EPOCHS,VMAPS,N_DOTS,APERTURE) #key_,rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #jnp.tile(jnp.array([[2,3]]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,1)) #
    params_["SELECT"] = jnp.eye(N_DOTS)[rnd.choice(ki[1],N_DOTS,(EPOCHS,VMAPS))] #rnd.choice(ki[1],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2))
    params_["SAMPLES"] = rnd.choice(ki[2],M,(EPOCHS,VMAPS,T))
    params_["POS_0"] = rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2))
    params_["HV_0"] = (INIT/(H))*rnd.normal(ki[4],(EPOCHS,VMAPS,H))
    return params_

# @jit
# def RNN_forward(h1vec,v_0,h_t_1,p_weights): # THINK... GRU computation of hp_t_1,v_t_1->hp_t,v_t
#     W_h1 = p_weights["W_h1"]
#     W_v = p_weights["W_v"]
#     U_vh = p_weights["U_vh"]
#     b_vh = p_weights["b_vh"]
#     h_t = jax.nn.sigmoid(jnp.matmul(W_h1,h1vec) + jnp.matmul(W_v,v_0) + jnp.matmul(U_vh,h_t_1) + b_vh) # + jnp.matmul(W_v_z,v_t_1)
#     return h_t #v_t,hp_t

# @partial(jax.jit,static_argnums=())
# def plan(h1vec,v_0,h_0,p_weights,params): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
#     W_r = p_weights["W_r"]
#     h_t_1 = h_0
#     for i in range(params["PLAN_ITS"]):
#         h_t_1 = RNN_forward(h1vec,v_0,h_t_1,p_weights)
#     h_t = h_t_1
#     v_t = jnp.matmul(W_r,h_t)
#     return v_t,h_t

# @partial(jax.jit,static_argnums=())
# def move(pos_t,dots,params): # self,pos_t_1,v_t_1,r_t_1,weights,params
#         v_t = neuron_act(params,dots,pos_t) # pos_t = pos_0 + vec
#         return v_t

# W_vt0 = (INIT/((H)*M))*rnd.normal(ki[0],(H,N))
# W_vt_10 = (INIT/((H)*N))*rnd.normal(ki[1],(H,N))
# W_rt_10 = (INIT/((H)*N))*rnd.normal(ki[2],(H,))
# U_h0 = (INIT/((H)*H))*rnd.normal(ki[3],(H,H))
# b_h0 = (INIT/(H))*rnd.normal(ki[4],(H,))
# W_read0 = (INIT/(H))*rnd.normal(ki[5],(H,))

@jit
def value(v_weights,v_t,v_t_1,r_t_1,h_t_1):
    W_vt = v_weights["W_vt"]
    W_vt_1 = v_weights["W_vt_1"]
    W_rt_1 = v_weights["W_rt_1"]
    U_h = v_weights["U_h"]
    b_h = v_weights["b_h"]
    W_read = v_weights["W_read"]
    W_sel = v_weights["W_sel"]
    h_t = jax.nn.sigmoid(jnp.matmul(W_vt,v_t) + jnp.matmul(W_vt_1,v_t_1) + W_rt_1*r_t_1 + jnp.matmul(U_h,h_t_1) + b_h)
    # jax.debug.print('v_t={}',v_t[0:5])
    # jax.debug.print('v_t_1={}',v_t_1[0:5])
    # jax.debug.print('r_t_1={}',r_t_1[0:5])
    # jax.debug.print('h_t={}',h_t[0:5])
    r_hat_t = jnp.matmul(W_read,h_t)
    sel_hat_t = jnp.matmul(W_sel,h_t)
    return r_hat_t,h_t,sel_hat_t

# @partial(jax.jit,static_argnums=())
def body_fnc(SC,v_weights,params,dots,sel,samples,pos_0,h_0): # vmap over dots,sel,samples,h_0
    v_arr,r_arr,pos_arr = gen_vr_arr(SC,dots,sel,samples,pos_0,params)
    # jax.debug.print('v_arr={}',v_arr)
    r_hat_arr = jnp.zeros(params["TRIAL_LENGTH"])
    h_t = h_0
    loss = 0
    loss_s = 0
    loss_tot = 0
    for i in range(1,params["TRIAL_LENGTH"]):
        if i < params["TRAIN_LENGTH"]:
            v_t_1 = v_arr[i-1,:]
            v_t = v_arr[i,:]
            r_t_1 = r_arr[i-1]
            r_t = r_arr[i]
            r_hat_t,h_t,_ = value(v_weights,v_t,v_t_1,r_t_1,h_t)
            r_hat_arr = r_hat_arr.at[i].set(r_hat_t)
        # if i == params["TRAIN_LENGTH"]:
            # v_t_1 = v_arr[i-1,:]
            # v_t = v_arr[i,:]
            # r_t_1 = r_arr[i-1] #r_hat_t
            # r_t = r_arr[i] # no more r_t used
            # r_hat_t,h_t = value(v_weights,v_t,v_t_1,r_t_1,h_t)
            # r_hat_arr = r_hat_arr.at[i].set(r_hat_t)
            # loss += (r_hat_t - r_t)**2
            # r_t_1 = r_hat_t
        if i >= params["TRAIN_LENGTH"]: ###CHANGE
            v_t_1 = v_arr[i-1,:]
            v_t = v_arr[i,:]
            # r_t_1 = r_arr[i-1] #r_hat_t
            r_t = r_arr[i]
            r_hat_t,h_t,sel_hat_t = value(v_weights,v_t,v_t_1,r_hat_t,h_t) # r_t_1
            r_hat_arr = r_hat_arr.at[i].set(r_hat_t)
            loss += (r_hat_t - r_t)**2
            loss_s += optax.softmax_cross_entropy(logits=sel_hat_t,labels=sel)
            loss_tot += loss + params["LAMBDA_S"]*loss_s
            # r_t_1 = r_hat_t
            # jax.debug.print("sum(v_t)={}",jnp.sum(v_t)) # (correct)
            # jax.debug.print("r_hat_t={}",r_hat_t)
            # jax.debug.print("r_t={}",r_t)
            # jax.debug.print("loss={}",loss)
            # jax.debug.print("L_S*loss_s={}",params["LAMBDA_S"]*loss_s)
        # avg_loss = loss/(params["TEST_LENGTH"])
        # jax.debug.print("loss={}",loss)
    return loss_tot,(v_arr,r_arr,r_hat_arr,pos_arr,loss_s)

def train_loop(SC,v_weights,params,l,opt_state): # 
        loss_arr,loss_std,s_arr,s_std = (jnp.empty((params["EPOCHS"])) for _ in range(4))
        v_arr = jnp.empty((params["EPOCHS"],params["VMAPS"],params["TRIAL_LENGTH"],params["N"]))
        pos_arr = jnp.empty((params["EPOCHS"],params["VMAPS"],params["TRIAL_LENGTH"],2))
        r_arr,r_hat_arr = (jnp.empty((params["EPOCHS"],params["VMAPS"],params["TRIAL_LENGTH"])) for _ in range(2))
        optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
        for e in range(params["EPOCHS"]):
            dots = params["DOTS"][e,:,:,:] # same dots across vmaps
            sel = params["SELECT"][e,:,:]
            samples = params["SAMPLES"][e,:,:]
            pos_0 = params["POS_0"][e,:,:]
            h_0 = params["HV_0"][e,:,:]
            val_grad = jax.value_and_grad(body_fnc,argnums=1,allow_int=True,has_aux=True)
            val_grad_vmap = jax.vmap(val_grad,in_axes=(None,None,None,0,0,0,0,0),out_axes=(0,0))
            (loss,aux),grads = val_grad_vmap(SC,v_weights,params,dots,sel,samples,pos_0,h_0) # val_grad_vmap ,,v_0
            v_arr_,r_arr_,r_hat_arr_,pos_arr_,loss_s = aux
            grads_ = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
            # p_weights = jax.tree_util.tree_map(lambda x,y: x - params["LR"]*y,p_weights,grads_)
            opt_update,opt_state = optimizer.update(grads_,opt_state,v_weights)
            v_weights = optax.apply_updates(v_weights,opt_update)
            v_arr = v_arr.at[e,:,:,:].set(v_arr_) #
            r_arr = r_arr.at[e,:,:].set(r_arr_) ##
            r_hat_arr = r_hat_arr.at[e,:,:].set(r_hat_arr_) ##
            pos_arr = pos_arr.at[e,:,:,:].set(pos_arr_) ##
            loss_arr = loss_arr.at[e].set(jnp.mean(loss))
            loss_std = loss_std.at[e].set(jnp.std(loss)/jnp.sqrt(params["VMAPS"]))
            s_arr = s_arr.at[e].set(jnp.mean(loss_s))
            s_std = s_std.at[e].set(jnp.std(loss_s)/jnp.sqrt(params["VMAPS"]))
            # jax.debug.print("std={}",jnp.std(loss)/jnp.sqrt(params["VMAPS"]))
            jax.debug.print("e={}",l*(params["EPOCHS"])+e)
            jax.debug.print("mean loss={}",jnp.mean(loss))
        v_aux = (v_arr,r_arr,r_hat_arr,pos_arr,s_arr,s_std) #,params["DOTS"])
        return v_weights,loss_arr,loss_std,v_aux,opt_state

# @partial(jax.jit,static_argnums=())
def value_model_loop(SC,weights,params):
    v_weights = weights["v_weights"]
    E = params["EPOCHS"]
    L = params["LOOPS"]
    loss_arr,loss_std,s_arr,s_std = (jnp.empty(params["TOT_EPOCHS"]) for _ in range(4))
    v_arr = jnp.empty((params["PLOTS"],params["VMAPS"],params["TRIAL_LENGTH"],params["N"]))
    r_arr,r_hat_arr = (jnp.empty((params["PLOTS"],params["VMAPS"],params["TRIAL_LENGTH"])) for _ in range(2))
    pos_arr = jnp.empty((params["PLOTS"],params["VMAPS"],params["TRIAL_LENGTH"],2))
    optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    opt_state = optimizer.init(v_weights)
    for l in range(L):
        params = new_params(params,l)
        p_weights,loss_arr_,loss_std_,(v_arr_,r_arr_,r_hat_arr_,pos_arr_,s_arr_,s_std_),opt_state = train_loop(SC,v_weights,params,l,opt_state)
        loss_arr = loss_arr.at[l*E:(l+1)*E].set(loss_arr_)
        loss_std = loss_std.at[l*E:(l+1)*E].set(loss_std_)
        s_arr = s_arr.at[l*E:(l+1)*E].set(s_arr_)
        s_std = s_std.at[l*E:(l+1)*E].set(s_std_)
        if (E>=PLOTS)and(l//(L-1)==0):
            v_arr = v_arr.at[-PLOTS:,:,:,:].set(v_arr_[-PLOTS:,:,:,:]) # l*E:(l+1)*E,(l//(L-1))
            r_arr = r_arr.at[-PLOTS:,:,:].set(r_arr_[-PLOTS:,:,:]) # (l//(L-1))
            r_hat_arr = r_hat_arr.at[-PLOTS:,:,:].set(r_hat_arr_[-PLOTS:,:,:])
            pos_arr = pos_arr.at[-PLOTS:,:,:,:].set(pos_arr_[-PLOTS:,:,:,:])
        elif (E<PLOTS)and(l>=(L-PLOTS)):
            v_arr = v_arr.at[l-L,:,:].set(v_arr_[0,:,:,:])
            r_arr = r_arr.at[l-L,:,:].set(r_arr_[0,:,:])
            r_hat_arr = r_hat_arr.at[l-L,:,:].set(r_hat_arr_[0,:,:])
            pos_arr = pos_arr.at[l-L,:,:,:].set(pos_arr_[0,:,:,:])
    v_weights_trained = v_weights
    return loss_arr,loss_std,v_weights_trained,v_arr,r_arr,r_hat_arr,pos_arr,s_arr,s_std

# hyperparams
TOT_EPOCHS = 100
EPOCHS = 1
PLOTS = 3
LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 600 # 1100,1000,800,500
LR = 0.0001 # 0.000001,0.0001
WD = 0.001 # 0.0001
LAMBDA_S = 0.1
INIT = 25
H = 800 # 500,300 
TRIAL_LENGTH = 100
TRAIN_LENGTH = 70
TEST_LENGTH = TRIAL_LENGTH-TRAIN_LENGTH
# (H=800->INIT=25;1500*)

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
MODULES = 11 # 8
M = MODULES**2
NEURONS = 10
N = 3*(NEURONS**2)
SIGMA_R = 1.2 # 1.5,1
SIGMA_A = 0.5 # 0.5
APERTURE = jnp.pi
THETA_X = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
THETA_Y = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]]) # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = COLORS.shape[0]
DOTS = gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
# KEYS = rnd.split(ke[1],(EPOCHS*VMAPS*3)/2).reshape(EPOCHS,VMAPS,3) # rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
SELECT = jnp.eye(N_DOTS)[rnd.choice(ke[1],N_DOTS,(EPOCHS,VMAPS))]
SAMPLES = rnd.choice(ke[2],M,(EPOCHS,VMAPS,TRIAL_LENGTH)) # [E,V] rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
POS_0 = rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
HP_0 = (INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) # [E,V,H]
ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True) # CHANGE TO PERMUATION
VEC_ARR = gen_vectors(MODULES,APERTURE)
H1VEC_ARR = jnp.diag(jnp.ones(M))[:,ID_ARR]
SC = (ID_ARR,VEC_ARR,H1VEC_ARR)

# INITIALIZATION
ki = rnd.split(rnd.PRNGKey(1),num=10)
W_vt0 = 2200*(INIT/(H*N))*rnd.normal(ki[0],(H,N)) # (INIT/(H*N))*
W_vt_10 = 2200*(INIT/(H*N))*rnd.normal(ki[1],(H,N)) # (INIT/(H*N))*
W_rt_10 = (INIT/(H))*rnd.normal(ki[2],(H,))
U_h0 = (INIT/(H*H))*rnd.normal(ki[3],(H,H))
b_h0 = (INIT/(H))*rnd.normal(ki[4],(H,))
W_read0 = (INIT/(H))*rnd.normal(ki[5],(H,))
W_sel0 = (INIT/(3*H))*rnd.normal(ki[6],(3,H))

params = {
    "TOT_EPOCHS" : TOT_EPOCHS,
    "EPOCHS" : EPOCHS,
    "LOOPS" : LOOPS,
    "MODULES" : MODULES,
    "NEURONS" : NEURONS,
    "APERTURE" : APERTURE,
    "LR" : LR,
    "WD" : WD,
    "H" : H,
    "N" : N,
    "M" : M,
    "INIT" : INIT,
    "VMAPS" : VMAPS,
    "PLOTS" : PLOTS,
    "N_DOTS" : N_DOTS,
    "DOTS" : DOTS,
    "THETA_X" : THETA_X,
    "THETA_Y" : THETA_Y,
    "COLORS" : COLORS,
    "SAMPLES" : SAMPLES,
    "POS_0" : POS_0,
    "HP_0" : HP_0,
    "TRIAL_LENGTH" : TRIAL_LENGTH,
    "TRAIN_LENGTH" : TRAIN_LENGTH,
    "TEST_LENGTH" : TEST_LENGTH,
    "SIGMA_R" : SIGMA_R,
    "SIGMA_A" : SIGMA_A,
    "LAMBDA_S" : LAMBDA_S,
    }

weights = {
    's_weights' : {

    },
    'p_weights' : {
        # "W_h1" : W_h10,
        # "W_r" : W_r0,
        # "W_v" : W_v0,
        # "U_vh" : U_vh0,
        # "b_vh" : b_vh0
    },
    'v_weights' : {
        "W_vt"      : W_vt0,
        "W_vt_1"    : W_vt_10,
        "W_rt_1"    : W_rt_10,
        "U_h"       : U_h0,
        "b_h"       : b_h0,
        "W_read"    : W_read0,
        "W_sel"     : W_sel0
    }
}

###
startTime = datetime.now()
(loss_arr,loss_std,p_weights_trained,v_arr,r_arr,r_hat_arr,pos_arr,s_arr,s_std) = value_model_loop(SC,weights,params)

# plot training loss
print("Training time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
fig,ax = plt.subplots(1,2,figsize=(18,6))
# plt.figure(figsize=(12,6))
title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, TRIAL_LENGTH={TRIAL_LENGTH}, TRAIN_LENGTH={TRAIN_LENGTH}, init={INIT:.2f}, update={LR:.6f}, WD={WD:.5f}, \n SIGMA_R={SIGMA_R:.1f}, SIGMA_A={SIGMA_A:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H={H}'
plt.suptitle('value_training_v1, '+title__,fontsize=14)
plt.subplot(1,2,1)
plt.errorbar(jnp.arange(TOT_EPOCHS),loss_arr,yerr=loss_std,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
# plt.xscale('log')
plt.ylabel(r'loss_tot',fontsize=15)
plt.xlabel(r'iteration',fontsize=15)
plt.subplot(1,2,2)
plt.errorbar(jnp.arange(TOT_EPOCHS),s_arr,yerr=s_std,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
plt.ylabel(r'loss_sel',fontsize=15)
plt.xlabel(r'iteration',fontsize=15)
# plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_ + 'value_training_v1_' + dt + '.png')

# colors_ = np.float32([[255,0,0],[0,255,0],[0,0,255]])/255 # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])/255

# separate each v in to r/g/b channels
v_arr = jnp.linalg.norm(v_arr[-PLOTS:,0,:,:],axis=2) # norm? (PLOTS,T),((-1 if e))
r_arr = r_arr[-PLOTS:,0,:] # (3,0,T),((-1 if e))
r_hat_arr = r_hat_arr[-PLOTS:,0,:] # (3,0,T),((-1 if e))
pos_arr = jnp.linalg.norm(mod_(pos_arr[-PLOTS:,0,:,:]),axis=2) # (3,0,T),((-1 if e))
# plot true dot locations
fig,axis = plt.subplots(PLOTS,2,figsize=(10,6*PLOTS)) #9,4x plt.figure(figsize=(12,6))
# title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, init={INIT:.2f}, update={LR:.6f}, WD={WD:.5f}, \n SIGMA_R={SIGMA_R:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H={H}'
plt.suptitle('value_training_v1, '+title__,fontsize=10)
for i in range(params["PLOTS"]):
    ax0 = plt.subplot2grid((2*PLOTS,2),(2*i,0),colspan=2,rowspan=1)
    ax0.set_title('pos_arr') # v_arr
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.set_xticks([])
    ax0.ticklabel_format(useOffset=False)
    ax1 = plt.subplot2grid((2*PLOTS,2),(2*i+1,0),colspan=2,rowspan=1)
    ax1.set_title('r_arr vs r_hat_arr')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # v_arr
    ax0.plot(pos_arr[i,:],linewidth=2,color='b') # v_arr
    ax0.axvline(x=TRAIN_LENGTH,color='k',linestyle='--',linewidth=1)
    # r_arr
    ax1.plot(r_arr[i,:],linewidth=2,color='k',label='r')
    ax1.plot(jnp.arange(1,TRIAL_LENGTH),r_hat_arr[i,1:],linewidth=2,color='r',label='r_hat')
    ax1.axvline(x=TRAIN_LENGTH,color='k',linestyle='--',linewidth=1)
    ax1.legend()
    if i != params["PLOTS"]-1:
        # ax0.set_xticks([])
        ax1.set_xticks([])
plt.axis('tight')
plt.subplots_adjust(top=0.9)

dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_ + 'figs_value_training_v1_' + dt + '.png')

save_pkl((loss_arr,loss_std,p_weights_trained,v_arr,r_arr,r_hat_arr),'value_v1_')