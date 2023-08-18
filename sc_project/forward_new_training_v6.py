# -*- coding: utf-8 -*-
# 'nice' hp's; const dot speed; penalise similarity
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

def load_(str_):
    path_ = str(Path(__file__).resolve().parents[1]) # .../meta_rl_ego_sim/
    with open(path_+str_,'rb') as file_:
        # param = pickle.load(file_)
        param_ = jnp.load(file_,allow_pickle=True)
    return param_

def save_pkl(param,str_):  # can't jit (can't pickle jax tracers)
	path_ = str(Path(__file__).resolve().parents[0]) + '/test_data/' # '/scratch/lrj34/'
	dt = datetime.now().strftime("%d_%m-%H%M%S")
	with open(path_+str_+'_'+dt+'.pkl','wb') as file:
		pickle.dump(param,file,pickle.HIGHEST_PROTOCOL)

@jit
def neuron_act_noise(val,THETA,SIGMA_A,SIGMA_N,dot,pos):
    key = rnd.PRNGKey(val)
    N_ = THETA.size
    dot = dot.reshape((1, 2))
    xv,yv = jnp.meshgrid(THETA,THETA[::-1])
    G_0 = jnp.vstack([xv.flatten(),yv.flatten()])
    E = G_0.T - (dot - pos)
    act = jnp.exp((jnp.cos(E[:, 0]) + jnp.cos(E[:, 1]) - 2) / SIGMA_A**2)
    noise = (SIGMA_N*act)*rnd.normal(key,shape=(N_**2,))
    return act + noise

# @jit
# def loss_dot(dot_hat,dot_t,pos_t):
#     rel_vec = dot_t - pos_t
#     rel_x_hat = jnp.arctan2(dot_hat[1],dot_hat[0])
#     rel_y_hat = jnp.arctan2(dot_hat[3],dot_hat[2])
#     rel_vec_hat = jnp.array([rel_x_hat,rel_y_hat])
#     loss_d = -jnp.sum(jnp.exp(jnp.cos(rel_vec_hat[0] - rel_vec[0]) + jnp.cos(rel_vec_hat[1] - rel_vec[1]) - 2)/params["SIGMA_D"]**2)
#     return loss_d,rel_vec_hat

def mod_(x):
    return (x+jnp.pi)%(2*jnp.pi)-jnp.pi

def gen_sc(keys,MODULES,ACTION_SPACE,PLAN_SPACE):
    index_range = jnp.arange(MODULES**2)
    x = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)
    y = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)[::-1]
    xv,yv = jnp.meshgrid(x,y)
    A_full = jnp.vstack([xv.flatten(),yv.flatten()])

    inner_mask = (jnp.abs(xv) <= ACTION_SPACE) & (jnp.abs(yv) <= ACTION_SPACE)
    A_inner_ind = index_range[inner_mask.flatten()]
    A_outer_ind = index_range[~inner_mask.flatten()]
    A_inner_perm = rnd.permutation(keys[0],A_inner_ind)
    A_outer_perm = rnd.permutation(keys[1],A_outer_ind)
    ID_ARR = jnp.concatenate((A_inner_perm,A_outer_perm),axis=0)

    VEC_ARR = A_full[:,ID_ARR]
    H1VEC_ARR = jnp.eye(MODULES**2) # [:,ID_ARR]
    SC = (ID_ARR,VEC_ARR,H1VEC_ARR)
    return SC

# @partial(jax.jit,static_argnums=(1,2,3))
def gen_dot(key,VMAPS,N_dot,APERTURE):
    keys = rnd.split(key,N_dot)
    dot_0 = rnd.uniform(keys[0],shape=(VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([APERTURE/4,APERTURE/4]),maxval=jnp.array([3*APERTURE/4,3*APERTURE/4]))
    # dot_1 = rnd.uniform(keys[1],shape=(VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([APERTURE/4,-APERTURE/4]),maxval=jnp.array([3*APERTURE/4,-3*APERTURE/4]))
    # dot_2 = rnd.uniform(keys[2],shape=(VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([-3*APERTURE/4,-APERTURE]),maxval=jnp.array([-APERTURE/4,APERTURE]))
    # dot_tot = jnp.concatenate((dot_0,dot_1,dot_2),axis=1)
    # dot = rnd.uniform(key,shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE)
    return dot_0 #dot_tot[:,:N_dot,:]

# @partial(jax.jit,static_argnums=(0,))
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

def gen_samples(key,MODULES,ACTION_SPACE,PLANNING_SPACE,INIT_STEPS,TOT_STEPS):###
    M_P = (MODULES-1)//2 # (1/ACTION_FRAC)
    M_A = jnp.int32(M_P*(ACTION_SPACE/PLANNING_SPACE)) ### FIX
    init_vals = jnp.arange((2*M_A+1)**2)
    main_vals = jnp.arange((2*M_A+1)**2,MODULES**2)
    keys = rnd.split(key,2)
    init_samples = rnd.choice(keys[0],init_vals,shape=(INIT_STEPS,)) # INIT_STEPS,
    # main_samples = rnd.choice(keys[1],main_vals,shape=(TOT_STEPS-INIT_STEPS-1,))
    return init_samples #jnp.concatenate([init_samples,main_samples],axis=0) # init_samples

def new_params(params,e):
    EPOCHS = params["EPOCHS"]
    VMAPS = params["VMAPS"]
    N_DOTS = params["N_DOTS"]
    APERTURE = params["APERTURE"]
    MAX_DOT_SPEED = params["MAX_DOT_SPEED"]
    MODULES = params["MODULES"]
    ACTION_SPACE = params["ACTION_SPACE"]
    PLAN_SPACE = params["PLAN_SPACE"]
    INIT_STEPS = params["INIT_STEPS"]
    TOT_STEPS = params["TOT_STEPS"]
    # N = params["N"]
    M = params["M"]
    H = params["H"]
    ki = rnd.split(rnd.PRNGKey(e),num=10) #l/0
    POS_0 = rnd.uniform(ki[0],shape=(VMAPS,2),minval=-jnp.pi,maxval=jnp.pi)
    params["POS_0"] = POS_0
    params["DOT_0"] = POS_0 + rnd.uniform(ki[1],shape=(VMAPS,2),minval=-APERTURE,maxval=APERTURE) #gen_dot(ki[0],VMAPS,N_DOTS,APERTURE) #key_,rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #jnp.tile(jnp.array([[2,3]]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,1)) #
    params["DOT_VEC"] = gen_dot_vecs(ki[2],VMAPS,MAX_DOT_SPEED) 
    kv = rnd.split(ki[3],num=VMAPS)
    params["SAMPLES"] = jax.vmap(gen_samples,in_axes=(0,None,None,None,None,None),out_axes=0)(kv,MODULES,ACTION_SPACE,PLAN_SPACE,INIT_STEPS,TOT_STEPS) #rnd.choice(ki[3],M,shape=(VMAPS,(TOT_STEPS-1)))
    params["HP_0"] = jnp.sqrt(INIT/(H))*rnd.normal(ki[4],(VMAPS,H))

def gen_dot_vecs(key,VMAPS,MAX_DOT_SPEED):
    dot_vecs = rnd.uniform(key,shape=(VMAPS,2),minval=-MAX_DOT_SPEED/2,maxval=MAX_DOT_SPEED/2) ###
    # mask = jnp.all((-APERTURE/2 <= dot_vecs)&(dot_vecs <= APERTURE/2),axis=1)
    # while mask.any():
    #     key = rnd.split(key)[0]
    #     new_vecs = rnd.uniform(key,shape=(jnp.sum(mask),2),minval=-APERTURE,maxval=APERTURE)
    #     dot_vecs = dot_vecs.at[mask].set(new_vecs)
    #     mask = jnp.all((-APERTURE/2 <= dot_vecs)&(dot_vecs <= APERTURE/2),axis=1)
    return dot_vecs

@jit
def gen_timeseries(SC,pos_0,dot_0,dot_vec,samples,step_array):
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    h1vec_arr = H1VEC_ARR[:,samples]
    vec_arr = VEC_ARR[:,samples]
    cum_vecs = jnp.cumsum(vec_arr,axis=1)
    pos_1_end = (pos_0+cum_vecs.T).T
    dot_1_end = (dot_0+jnp.outer(dot_vec,step_array).T).T
    pos_arr = jnp.concatenate((pos_0.reshape(2,1),pos_1_end),axis=1)
    dot_arr = jnp.concatenate((dot_0.reshape(2,1),dot_1_end),axis=1)
    return pos_arr,dot_arr,h1vec_arr,vec_arr

# @jit
def get_inner_activation_indices(N, K):
    inner_N = jnp.int32(K) # Length of one side of the central region
    start_idx = (N - inner_N) // 2 # Starting index
    end_idx = start_idx + inner_N # Ending index
    row_indices, col_indices = jnp.meshgrid(jnp.arange(start_idx, end_idx), jnp.arange(start_idx, end_idx))
    flat_indices = jnp.ravel_multi_index((row_indices.flatten(), col_indices.flatten()), (N, N))
    return flat_indices

# def get_inner_activations(activations, NEURONS_AP, NEURONS_FULL):
#     x = NEURONS_AP / NEURONS_FULL # Fraction of neurons in the central region
#     N = int(jnp.sqrt(activations.shape[0])) # Length of one side of the square array
#     indices = get_inner_activation_indices(N, x)
#     # Use jnp.take to gather the values at the given indices
#     inner_activations = jnp.take(activations, indices)
#     return inner_activations

@jit
def cosine_similarity(x,y):
    dot_product = jnp.dot(x,y)
    x_norm = jnp.linalg.norm(x)
    y_norm = jnp.linalg.norm(y)
    return dot_product/(x_norm*y_norm)

@partial(jax.jit,static_argnums=(4,))
def plan(h1vec,v_0,h_0,p_weights,PLAN_ITS): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    def loop_body(carry,i):
        return RNN_forward(carry)
    def RNN_forward(carry):
        p_weights,h1vec,v_0,h_t_1 = carry
        W_h1 = p_weights["W_h1"]
        W_v = p_weights["W_v"]
        U_vh = p_weights["U_vh"]
        b_vh = p_weights["b_vh"]
        h_t = jax.nn.sigmoid(jnp.matmul(W_h1,h1vec) + jnp.matmul(W_v,v_0) + jnp.matmul(U_vh,h_t_1) + b_vh)
        return (p_weights,h1vec,v_0,h_t),None
    W_r_f = p_weights["W_r_f"]
    W_r_a = p_weights["W_r_a"]
    W_r_pl = p_weights["W_r_pl"]
    carry_0 = (p_weights,h1vec,v_0,h_0)
    (*_,h_t),_ = jax.lax.scan(loop_body,carry_0,jnp.zeros(PLAN_ITS))
    v_pred_pl = jnp.matmul(W_r_pl,h_t)

    v_pred_ap = jnp.take(v_pred_pl, params["INDICES"])
    # v_pred_ap = jnp.matmul(W_r_a,h_t)
    return v_pred_ap,v_pred_pl,h_t # dot_hat_t,

def body_fnc(SC,p_weights,params,pos_0,dot_0,dot_vec,h_0,samples,e):###
    loss_v_arr,loss_c_arr = (jnp.zeros(params["TOT_STEPS"]) for _ in range(2))
    v_pred_arr,v_t_arr = (jnp.zeros((params["TOT_STEPS"],params["N_P"])) for _ in range(2))
    rel_vec_hat_arr = jnp.zeros((params["TOT_STEPS"],2))
    pos_arr,dot_arr,h1vec_arr,vec_arr = gen_timeseries(SC,pos_0,dot_0,dot_vec,samples,params["STEP_ARRAY"])
    tot_loss_v,tot_loss_c = 0,0
    h_t_1 = h_0
    v_t_1_ap = neuron_act_noise(samples[0],params["THETA_AP"],params["SIGMA_A"],params["SIGMA_N"],dot_arr[:,0],pos_arr[:,0])
    v_t_arr = v_t_arr.at[0,:params["N_A"]].set(v_t_1_ap)
    # v_t_1_pl = neuron_act_noise(samples[0],params["THETA_PLAN"],params["SIGMA_A"],params["SIGMA_N"],dot_arr[:,0],pos_arr[:,0])
    for t in range(1,params["TOT_STEPS"]):
        v_pred_ap,v_pred_pl,h_t = plan(h1vec_arr[:,t-1],v_t_1_ap,h_t_1,p_weights,params["PLAN_ITS"]) # ,dot_hat_t
        v_t_ap = neuron_act_noise(samples[t-1],params["THETA_AP"],params["SIGMA_A"],params["SIGMA_N"],dot_arr[:,t],pos_arr[:,t])
        # v_t_pl = neuron_act_noise(samples[t-1],params["THETA_PLAN"],params["SIGMA_A"],params["SIGMA_N"],dot_arr[:,t],pos_arr[:,t])
        # loss_v_pl = jnp.sum((v_pred_pl-v_t_pl)**2)
        loss_v_ap = jnp.sum((v_pred_ap-v_t_1_ap)**2) # v_t_1 not v_t
        loss_cos_ap = cosine_similarity(v_pred_ap,v_t_ap)
        loss_prev_mse_ap = jnp.sum((v_pred_ap-v_t_1_ap)**2)
        v_t_arr = v_t_arr.at[t,:params["N_A"]].set(v_t_ap)
        h_t_1,v_t_1_ap = h_t,v_t_ap
        if t >= 0: # params["INIT_STEPS"]:
            # if e < params["INIT_TRAIN_EPOCHS"]:
            tot_loss_v += loss_v_ap + params["LAMBDA_C"]*loss_cos_ap #- params["LAMBDA_P"]*loss_prev_mse_ap
            loss_v_arr = loss_v_arr.at[t].set(loss_v_ap)
            loss_c_arr = loss_c_arr.at[t].set(loss_cos_ap)
            v_pred_arr = v_pred_arr.at[t,:params["N_A"]].set(v_pred_ap)
            # else:
            #     tot_loss_v += loss_v_pl
            #     loss_v_arr = loss_v_arr.at[t].set(loss_v_pl)
            #     v_pred_arr = v_pred_arr.at[t,:].set(v_pred_pl)
    # loss_tot = tot_loss_v # + params["LAMBDA_D"]*tot_loss_d
    avg_loss = tot_loss_v/(params["TOT_STEPS"]-1) # params["PRED_STEPS"]
    return avg_loss,(v_pred_arr,v_t_arr,pos_arr,dot_arr,rel_vec_hat_arr,loss_v_arr,loss_c_arr)

# @partial(jax.jit,static_argnums=())
def forward_model_loop(SC,weights,params):
    p_weights = weights["p_weights"]
    T = params["TOT_EPOCHS"]
    loss_arr,loss_sem = (jnp.empty(params["TOT_EPOCHS"]) for _ in range(2))
    loss_v_arr,v_std_arr,loss_c_arr,c_std_arr = (jnp.empty((params["TOT_EPOCHS"],params["TOT_STEPS"])) for _ in range(4))
    optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    opt_state = optimizer.init(p_weights)
    print("Starting training")
    for e in range(T):
        new_params(params,e) # 'params = new_params(params,e)'
        dot_0 = params["DOT_0"] #[e,:,:,:] # same dot across vmaps
        dot_vec = params["DOT_VEC"] #[e,:,:,:]
        samples = params["SAMPLES"] #[e,:,:]
        pos_0 = params["POS_0"] #[e,:,:] ## (change) same pos_0 across vmaps
        h_0 = params["HP_0"] #[e,:,:]
        val_grad = jax.value_and_grad(body_fnc,argnums=1,allow_int=True,has_aux=True)
        val_grad_vmap = jax.vmap(val_grad,in_axes=(None,None,None,0,0,0,0,0,None),out_axes=(0,0))
        (loss,aux),grads = val_grad_vmap(SC,p_weights,params,pos_0,dot_0,dot_vec,h_0,samples,e) # val_grad_vmap ,,v_0
        v_pred_arr,v_t_arr,pos_arr,dot_arr,rel_vec_hat_arr,loss_v_arr_,loss_c_arr_ = aux # [VMAPS,STEPS,N]x2,[VMAPS,STEPS,2]x3,[VMAPS,STEPS]x2 (final timestep)
        grads_ = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
        opt_update,opt_state = optimizer.update(grads_,opt_state,p_weights)
        p_weights = optax.apply_updates(p_weights,opt_update)
        loss_arr = loss_arr.at[e].set(jnp.mean(loss))
        loss_sem = loss_sem.at[e].set(jnp.std(loss)/jnp.sqrt(params["VMAPS"]))
        loss_v_arr = loss_v_arr.at[e,:].set(jnp.mean(loss_v_arr_,axis=0))
        loss_c_arr = loss_c_arr.at[e,:].set(jnp.mean(loss_c_arr_,axis=0))
        v_std_arr = v_std_arr.at[e,:].set(jnp.std(loss_v_arr_,axis=0)/jnp.sqrt(params["VMAPS"]))
        c_std_arr = c_std_arr.at[e,:].set(jnp.std(loss_c_arr_,axis=0)/jnp.sqrt(params["VMAPS"]))
        print("e={}",e)
        print("loss_avg={}",jnp.mean(loss))
        print("loss_sem={}",jnp.std(loss)/jnp.sqrt(params["VMAPS"]))
        print("loss_v={}",jnp.mean(loss_v_arr_,axis=0))
        print("loss_c={}",jnp.mean(loss_c_arr_,axis=0))
        # print("std_v={}",jnp.std(loss_v_arr)/jnp.sqrt(params["VMAPS"]))
        # print("loss_d={}",jnp.mean(loss_d_arr_,axis=0))
        # print("std_d={}",jnp.std(loss_d_arr)/jnp.sqrt(params["VMAPS"]))
        # if e == T-1:
        #     print("v_pred_arr.shape=",v_pred_arr.shape)
        #     print("v_t_arr.shape=",v_t_arr.shape)
        #     print("pos_arr.shape=",pos_arr.shape)
        #     print("dot_arr.shape=",dot_arr.shape)
        #     print("rel_vec_hat_arr.shape=",rel_vec_hat_arr.shape)
        #     print("loss_v_arr.shape=",loss_v_arr.shape)
        #     print("loss_d_arr.shape=",loss_d_arr.shape)
        arrs = (loss_arr,loss_sem,loss_v_arr,v_std_arr,loss_c_arr,c_std_arr)
        aux = (v_pred_arr,v_t_arr,loss_v_arr_,loss_c_arr_,pos_arr,dot_arr,rel_vec_hat_arr,opt_state,p_weights)
    return arrs,aux # [VMAPS,STEPS,N]x2,[VMAPS,STEPS,2]x3,[VMAPS,STEPS]x2,..

# hyperparams
TOT_EPOCHS = 10000 #10000 # 1000 #250000
EPOCHS = 1
INIT_TRAIN_EPOCHS = 50000 ### epochs until phase 2
PLOTS = 3
# LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 200 # 800,500
PLAN_ITS = 10 # 8,5
INIT_STEPS = 30 # (taking loss over all steps so doesnt matter)
TOT_STEPS = 30
PRED_STEPS = TOT_STEPS-INIT_STEPS
LR = 0.0001 # 0.003,,0.0001
WD = 0.0001 # 0.0001
H = 1000 # 500,300
INIT = 2 # 0.5,0.1
LAMBDA_D = 1 # 1,0.1
LAMBDA_C = 10

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
MODULES = 7 # 17 # (3*N+1)
M = MODULES**2
APERTURE = (jnp.sqrt(2)/2)*jnp.pi ###
ACTION_FRAC = 1/4 # 
ACTION_SPACE = ACTION_FRAC*APERTURE # 'AGENT_SPEED'
PLAN_FRAC_REL = 3/2
PLAN_SPACE = PLAN_FRAC_REL*ACTION_SPACE
MAX_DOT_SPEED_REL_FRAC = 5/4
MAX_DOT_SPEED = MAX_DOT_SPEED_REL_FRAC*ACTION_SPACE
ALPHA = 1
NEURONS_FULL = 20 # 15 # 12 # jnp.int32(NEURONS_AP*(jnp.pi//APERTURE))
N_F = (NEURONS_FULL**2)
NEURONS_AP = jnp.int32(jnp.floor(NEURONS_FULL*(APERTURE/jnp.pi))) # 6 # 10
N_A = (NEURONS_AP**2)
NEURONS_PLAN = NEURONS_AP + 2*ALPHA
N_P = (NEURONS_PLAN**2)
THETA_FULL = jnp.linspace(-(jnp.pi-jnp.pi/NEURONS_FULL),(jnp.pi-jnp.pi/NEURONS_FULL),NEURONS_FULL)
THETA_AP = THETA_FULL[NEURONS_FULL//2 - NEURONS_AP//2 : NEURONS_FULL//2 + NEURONS_AP//2]
THETA_PLAN = THETA_FULL[NEURONS_FULL//2 - NEURONS_PLAN//2 : NEURONS_FULL//2 + NEURONS_PLAN//2]
SIGMA_A = 0.3 # 0.3,0.5,1,0.3,1,0.5,1,0.1
# SIGMA_D = 0.5
SIGMA_N = 0.05 # 0.05
COLORS = jnp.array([[255]]) # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = 1 #COLORS.shape[0]
STEP_ARRAY = jnp.arange(1,TOT_STEPS)
DOT_0 = None # gen_dot(ke[0],EPOCHS,VMAPS,N_dot,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #gen_dot(ke[0],EPOCHS,VMAPS,N_dot,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #gen_dot(ke[0],EPOCHS,VMAPS,N_dot,APERTURE)
DOT_VEC = None
SAMPLES = None # rnd.choice(ke[2],M,(EPOCHS,VMAPS,STEPS)) # [E,V] rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
POS_0 = None # rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
HP_0 = None # jnp.sqrt(INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) # [E,V,H]
# ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True) # CHANGE TO PERMUATION
# VEC_ARR = gen_vectors(MODULES,ACTION_SPACE)
# H1VEC_ARR = jnp.diag(jnp.ones(M))[:,ID_ARR]
# SC = (ID_ARR,VEC_ARR,H1VEC_ARR)
SC = gen_sc(ke,MODULES,ACTION_SPACE,PLAN_SPACE)

# x = NEURONS_AP / NEURONS_PLAN
# N = jnp.int32(jnp.sqrt(v_pred_full.shape[0]))
INDICES = get_inner_activation_indices(NEURONS_PLAN, NEURONS_AP)

# INITIALIZATION ### FIX
ki = rnd.split(rnd.PRNGKey(1),num=50)
W_h10 = jnp.sqrt(INIT/(H+M))*rnd.normal(ki[0],(H,M))
W_v0 = jnp.sqrt(INIT/(H+N_A))*rnd.normal(ki[1],(H,N_A))
W_r_f0 = jnp.sqrt(INIT/(N_F+H))*rnd.normal(ki[2],(N_F,H))
W_r_a0 = jnp.sqrt(INIT/(N_A+H))*rnd.normal(ki[3],(N_A,H))
W_r_pl0 = jnp.sqrt(INIT/(N_P+H))*rnd.normal(ki[4],(N_P,H))
# W_dot0 = jnp.sqrt(INIT/(H+4))*rnd.normal(ki[3],(4,H))
W_v0 = jnp.sqrt(INIT/(H+N_A))*rnd.normal(ki[1],(H,N_A))
W_r0 = jnp.sqrt(INIT/(N_F+H))*rnd.normal(ki[2],(N_F,H))
# W_dot0 = jnp.sqrt(INIT/(H+4))*rnd.normal(ki[3],(4,H))
U_vh0,_ = jnp.linalg.qr(jnp.sqrt(INIT/(H+H))*rnd.normal(ki[2],(H,H)))
b_vh0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[3],(H,))

params = {
    "TOT_EPOCHS" : TOT_EPOCHS,
    "EPOCHS" : EPOCHS,
    "INIT_TRAIN_EPOCHS" : INIT_TRAIN_EPOCHS,
    # "LOOPS" : LOOPS,
    "TOT_STEPS" : TOT_STEPS,
    "INIT_STEPS" : INIT_STEPS,
    "PRED_STEPS" : PRED_STEPS,
    "MODULES" : MODULES,
    "APERTURE" : APERTURE,
    "PLAN_ITS" : PLAN_ITS,
    # "SIGMA_N" : SIGMA_N,
    "LR" : LR,
    "WD" : WD,
    "H" : H,
    "NEURONS_FULL" : NEURONS_FULL,
    "NEURONS_AP" : NEURONS_AP,
    "NEURONS_PLAN" : NEURONS_PLAN,
    "N_F" : N_F,
    "N_A" : N_A,
    "N_P" : N_P,
    "THETA_FULL": THETA_FULL,
    "THETA_AP" : THETA_AP,
    "THETA_PLAN" : THETA_PLAN,
    "M" : M,
    "INIT" : INIT,
    "VMAPS" : VMAPS,
    "PLOTS" : PLOTS,
    # "KEYS" : KEYS,
    "N_DOTS" : N_DOTS,
    "DOT_0" : DOT_0,
    "DOT_VEC" : DOT_VEC,
    "SIGMA_A" : SIGMA_A,
    # "SIGMA_D" : SIGMA_D,
    "SIGMA_N" : SIGMA_N,
    "COLORS" : COLORS,
    "SAMPLES" : SAMPLES,
    "STEP_ARRAY" : STEP_ARRAY,
    "POS_0" : POS_0,
    # "V_0" : V_0,
    "HP_0" : HP_0,
    "LAMBDA_D" : LAMBDA_D,
    "LAMBDA_C" : LAMBDA_C,
    "ACTION_FRAC" : ACTION_FRAC,
    "ACTION_SPACE" : ACTION_SPACE,
    "PLAN_SPACE" : PLAN_SPACE,
    "MAX_DOT_SPEED" : MAX_DOT_SPEED,
    "INDICES" : INDICES,
    "ALPHA" : ALPHA,
}

weights = {
    's_weights' : {

    },
    'p_weights' : {
        "W_h1" : W_h10,
        "W_r_f" : W_r_f0,
        "W_r_a" : W_r_a0,
        "W_r_pl" : W_r_pl0,
        "W_v" : W_v0,
        # "W_dot" : W_dot0,
        "U_vh" : U_vh0,
        "b_vh" : b_vh0
    },
    'r_weights' : {

    }
}

# *_,p_weights = load_('/sc_project/pkl/forward_v9_225_16_06-0023.pkl') # ...14-06...,opt_state'/sc_project/pkl/forward_v9_225_13_06-0014.pkl','/pkl/forward_v8M_08_06-1857.pkl'
# weights["p_weights"] = p_weights
###
startTime = datetime.now()
arrs,aux = forward_model_loop(SC,weights,params)
(loss_arr,loss_sem,loss_v_arr,v_std_arr,loss_c_arr,c_std_arr) = arrs
(v_pred_arr,v_t_arr,loss_v_arr_,loss_c_arr_,pos_arr,dot_arr,rel_vec_hat_arr,opt_state,p_weights) = aux

# plot training loss
print("Training time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
print("Time finished:",datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
plt.figure(figsize=(12,6))
title__ = f'EPOCHS={TOT_EPOCHS}, INIT_EPOCHS={INIT_TRAIN_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, init={INIT:.2f}, update={LR:.6f}, WD={WD:.5f}, TOT_STEPS={TOT_STEPS}, PRED_STEPS={PRED_STEPS} \n ALPHA={ALPHA}, APERTURE={APERTURE:.2f}, ACTION_SPACE={ACTION_SPACE:.2f}, SIGMA_A={SIGMA_A:.1f}, NEURONS_FULL={NEURONS_FULL**2}, MODULES={M}, H={H}'
fig,ax = plt.subplots(1,3,figsize=(13,6))
plt.suptitle('forward_new_training_v6, '+title__,fontsize=10)
plt.subplot(1,3,1)
plt.errorbar(jnp.arange(TOT_EPOCHS),loss_arr,yerr=loss_sem,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
# plt.xscale('log')
# plt.yscale('log')
plt.ylabel(r'Total loss',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.subplot(1,3,2)
plt.errorbar(jnp.arange(TOT_EPOCHS),jnp.mean(loss_v_arr,axis=1),yerr=jnp.mean(v_std_arr,axis=1),color='black',ecolor='lightgray',elinewidth=2,capsize=0)
# plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Visual model loss',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.subplot(1,3,3)
plt.errorbar(jnp.arange(TOT_EPOCHS),jnp.mean(loss_c_arr,axis=1),yerr=jnp.mean(c_std_arr,axis=1),color='black',ecolor='lightgray',elinewidth=2,capsize=0)
# plt.xscale('log')
# plt.yscale('log')
plt.ylabel(r'cosine similarity',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.tight_layout()
plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_ + 'forward_new_training_v6_' + dt + '.png')

# plot before and after heatmaps using v_pred_arr and v_t_arr:
# print("v_pred_arr=",v_pred_arr.shape) # (v,n)
# print("v_t_arr=",v_t_arr.shape) # (v,n)
# print("pos_arr=",pos_arr) # (v,2)
# print("dot_arr=",dot_arr,dot_arr.shape) # (v,3,2)
# print('dot_arr_0',dot_arr[0,0,:,:])
# print('dot_arr_1',dot_arr[-1,0,:,:])
# colors_ = np.float32([[255,0,0],[0,255,0],[0,0,255]])/255 # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])/255
# separate each v in to r/g/b channels
# v_pred_rgb = np.clip(jnp.abs(v_pred_arr[-PLOTS:,:].reshape((PLOTS,STEPS,3,NEURONS**2))),0,1) # ((-1 if e))
# v_t_rgb = v_t_arr[-PLOTS:,:].reshape((PLOTS,3,NEURONS**2))
# pos_arr = pos_arr[-PLOTS:,:] #0
# dot_arr = dot_arr[-PLOTS:,:,:] #0
# plot rgb values at each neuron location
# neuron_locs = gen_vectors(NEURONS,APERTURE)
# # plot true dot locations
# fig,axis = plt.subplots(PLOTS,2,figsize=(12,5*PLOTS)) #9,4x plt.figure(figsize=(12,6))
# title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, STEPS = {STEPS}, PRED_STEPS={PRED_STEPS}, init={INIT:.2f}, update={LR:.6f}, WD={WD:.5f}, \n SIGMA_A={SIGMA_A:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H={H}'
# plt.suptitle('forward_new_training_v6, '+title__,fontsize=10)
# for i in range(params["PLOTS"]):
#     ax0 = plt.subplot2grid((2*PLOTS,2),(2*i,0),colspan=1,rowspan=1)
#     ax0.set_title('v_pred_arr')
#     ax0.set_aspect('equal')
#     ax1 = plt.subplot2grid((2*PLOTS,2),(2*i,1),colspan=1,rowspan=1)
#     ax1.set_title('v_t_arr')
#     ax1.set_aspect('equal')
#     # predicted
#     for ind,n in enumerate(neuron_locs.T):
#         ax0.scatter(n[0],n[1],color=np.float32(np.sqrt(v_pred_rgb[i,:,ind])),s=np.sum(17*v_pred_rgb[i,:,ind]),marker='o') ### sum,np.float32((v_pred_rgb[:,ind]))
#     # both
#     for d in range(N_DOTS):
#         ax0.scatter(mod_(dot_arr[i,d,0]-pos_arr[i,0]),mod_(dot_arr[i,d,1]-pos_arr[i,1]),color=(colors_[d,:]),s=52,marker='x') # dot_arr[i,d,0]
#         ax1.scatter(mod_(dot_arr[i,d,0]-pos_arr[i,0]),mod_(dot_arr[i,d,1]-pos_arr[i,1]),color=(colors_[d,:]),s=52,marker='x') # dot_arr[i,d,0]
#     # true
#     for ind,n in enumerate(neuron_locs.T):
#         ax1.scatter(n[0],n[1],color=np.float32(jnp.sqrt(v_t_rgb[i,:,ind])),s=np.sum(17*v_t_rgb[i,:,ind]),marker='o') ### sum,np.float32((v_t_rgb[:,ind]))
# ax0.set_xticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
# ax0.set_xticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=12)
# ax0.set_yticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
# ax0.set_yticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=12)
# ax0.set_xlim(-jnp.pi,jnp.pi)
# ax0.set_ylim(-jnp.pi,jnp.pi)
# ax1.set_xticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
# ax1.set_xticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=12)
# ax1.set_yticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
# ax1.set_yticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=12)
# ax1.set_xlim(-jnp.pi,jnp.pi)
# ax1.set_ylim(-jnp.pi,jnp.pi)
# plt.subplots_adjust(top=0.85)

# dt = datetime.now().strftime("%d_%m-%H%M%S")
# plt.savefig(path_ + 'figs_forward_new_training_v6_' + dt + '.png')

save_pkl((arrs,aux),'forward_new_v6_'+str(M)+'M_'+str(N_F)+'N') # (no rel_vec / loss_v / loss_d)