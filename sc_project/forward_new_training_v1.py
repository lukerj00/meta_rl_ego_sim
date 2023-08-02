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

def load_(str_):
    path_ = str(Path(__file__).resolve().parents[1]) # .../meta_rl_ego_sim/
    with open(path_+str_,'rb') as file_:
        # param = pickle.load(file_)
        param_ = jnp.load(file_,allow_pickle=True)
    return param_

def save_pkl(param,str_):  # can't jit (can't pickle jax tracers)
	path_ = str(Path(__file__).resolve().parents[0]) + '/test_data/' # '/scratch/lrj34/'
	dt = datetime.now().strftime("%d_%m-%H%M")
	with open(path_+str_+'_'+dt+'.pkl','wb') as file:
		pickle.dump(param,file,pickle.HIGHEST_PROTOCOL)

@jit # @partial(jax.jit,static_argnums=(0,1,2))
def neuron_act(COLORS,THETA,SIGMA_A,dot,pos): #e_t_1,th_j,th_i,SIGMA_A,COLORS # COLORS,THETA,SIGMA_A
    D_ = COLORS.shape[0]
    N_ = THETA.size
    dot = dot.reshape((1,2))
    th_x = jnp.tile(THETA,(N_,1)).reshape((N_**2,))
    th_y = jnp.tile(jnp.flip(THETA).reshape(N_,1),(1,N_)).reshape((N_**2,))
    G_0 = jnp.vstack([th_x,th_y])
    G = jnp.tile(G_0.reshape(2,N_**2,1),(1,1,D_))
    C = (COLORS/255).T # transpose((1,0))
    E = G.transpose((1,0,2)) - ((dot-pos).T) #.reshape((2,1))
    act = jnp.exp((jnp.cos(E[:,0,:]) + jnp.cos(E[:,1,:]) - 2)/SIGMA_A**2).T #.reshape((D_,N_**2))
    act_r,act_g,act_b = jnp.matmul(C,act) #.reshape((3*N_**2,))
    act_rgb = jnp.concatenate((act_r,act_g,act_b))
    return act_rgb

def mod_(x):
    return (x+jnp.pi)%(2*jnp.pi)-jnp.pi

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

def new_params(params,e):
    EPOCHS = params["EPOCHS"]
    VMAPS = params["VMAPS"]
    N_DOTS = params["N_DOTS"]
    APERTURE = params["APERTURE"]
    DOT_SPEED = params["DOT_SPEED"]
    MODULES = params["MODULES"]
    TOT_STEPS = params["TOT_STEPS"]
    N = params["N"]
    M = params["M"]
    H = params["H"]
    ki = rnd.split(rnd.PRNGKey(e),num=10) #l/0
    POS_0 = rnd.uniform(ki[0],shape=(VMAPS,2),minval=-jnp.pi,maxval=jnp.pi)
    params["POS_0"] = POS_0
    params["DOT_0"] = POS_0 + rnd.uniform(ki[1],shape=(VMAPS,2),minval=-APERTURE,maxval=APERTURE) #gen_dot(ki[0],VMAPS,N_DOTS,APERTURE) #key_,rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #jnp.tile(jnp.array([[2,3]]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,1)) #
    params["DOT_VEC"] = rnd.uniform(ki[2],shape=(VMAPS,2),minval=-(DOT_SPEED*APERTURE),maxval=(DOT_SPEED*APERTURE)) #gen_dot(ki[0],VMAPS,N_DOTS,APERTURE) #key_,rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #jnp.tile(jnp.array([[2,3]]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,1)) #
    params["SAMPLES"] = rnd.choice(ki[3],M,shape=(VMAPS,TOT_STEPS))
    params["HP_0"] = jnp.sqrt(INIT/(H))*rnd.normal(ki[4],(VMAPS,H))

# @jit
# def RNN_value(hv_t_1,v_t,r_t_1,r_weights): # WRITE (GRU computation of hv_t_1,v_t,r_t_1->hv_t,r_t)
#     hv_t <= hv_t_1
#     r_t <= r_t_1
#     return hv_t,r_t

# @partial(jax.jit,static_argnums=())
# def value(SC,h_t_1,v_t,r_t_1,r_weights): # self,hv_t_1,v_t,r_t_1,weights,params
#     h_t,r_t = RNN_value(h_t_1,v_t,r_t_1,r_weights)
#     return h_t,r_t

def conditional_gen_timeseries(SC,pos_0,dot_0,dot_vec,samples,params,step_array):
    # def check_aperture(args):
    #     (pos_arr,dot_arr,*_),_ = args
    #     pos_1_3 = pos_arr[:,:3]
    #     dot_1_3 = dot_arr[:,:3]
    #     rel_vec = pos_1_3 - dot_1_3
    #     cond = jnp.any(jnp.abs(rel_vec) > params["APERTURE"],axis=None)
    #     return cond
    # def loop_body(args):
    #     carry,inputs = args
    #     (*_,samples) = carry
    #     (SC,pos_0,dot_0,dot_vec,params,step_array) = inputs
    #     pos_arr,dot_arr,h1vec_arr,vec_arr = gen_timeseries(SC,pos_0,dot_0,dot_vec,samples,params,step_array)
    #     samples = (samples + 1) % params["M"]
    #     return (pos_arr,dot_arr,h1vec_arr,vec_arr,samples),inputs
    def gen_timeseries(SC,pos_0,dot_0,dot_vec,samples,params,step_array):
        ID_ARR,VEC_ARR,H1VEC_ARR = SC
        h1vec_arr = H1VEC_ARR[:,samples]
        vec_arr = VEC_ARR[:,samples]
        cum_vecs = jnp.cumsum(vec_arr,axis=1)
        pos_1_end = (pos_0+cum_vecs.T).T
        dot_1_end = (dot_0+jnp.outer(dot_vec,step_array).T).T # jnp.arange(1,params["TOT_STEPS"]+1)
        pos_arr = jnp.concatenate((pos_0.reshape(2,1),pos_1_end),axis=1)
        dot_arr = jnp.concatenate((dot_0.reshape(2,1),dot_1_end),axis=1)
        return pos_arr,dot_arr,h1vec_arr,vec_arr
    
    pos_arr,dot_arr,h1vec_arr,vec_arr = gen_timeseries(SC,pos_0,dot_0,dot_vec,samples,params,step_array)
    # carry = (pos_arr,dot_arr,h1vec_arr,vec_arr,samples)
    # inputs = (SC,pos_0,dot_0,dot_vec,params,step_array)
    # (pos_arr,dot_arr,h1vec_arr,vec_arr,_),inputs = jax.lax.while_loop(check_aperture,loop_body,(carry,inputs))
    return pos_arr,dot_arr,h1vec_arr,vec_arr

# @jit
# def RNN_forward(h1vec,v_0,h_t_1,p_weights): # THINK... GRU computation of hp_t_1,v_t_1->hp_t,v_t
#     W_h1 = p_weights["W_h1"]
#     W_v = p_weights["W_v"]
#     U_vh = p_weights["U_vh"]
#     b_vh = p_weights["b_vh"]
#     h_t = jax.nn.sigmoid(jnp.matmul(W_h1,h1vec) + jnp.matmul(W_v,v_0) + jnp.matmul(U_vh,h_t_1) + b_vh) # + jnp.matmul(W_v_z,v_t_1)
#     return h_t #v_t,hp_t

# # @partial(jax.jit,static_argnums=())
# def plan(h1vec,v_0,h_0,p_weights,params): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
#     W_r = p_weights["W_r"]
#     h_t_1 = h_0
#     for i in range(params["PLAN_ITS"]):
#         h_t_1 = RNN_forward(h1vec,v_0,h_t_1,p_weights)
#     h_t = h_t_1
#     v_t = jnp.matmul(W_r,h_t)
#     return v_t,h_t

# @partial(jax.jit,static_argnums=(4,))
def plan(h1vec,v_0,h_0,p_weights,params): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
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
    W_r = p_weights["W_r"]
    W_dot = p_weights["W_dot"]
    carry_0 = (p_weights,h1vec,v_0,h_0)
    (*_,h_t),_ = jax.lax.scan(loop_body,carry_0,jnp.zeros(params["PLAN_ITS"]))
    v_pred = jnp.matmul(W_r,h_t)
    dot_hat_t = jnp.matmul(W_dot,h_t)
    return v_pred,dot_hat_t,h_t

def loss_dot(dot_hat,dot_t,pos_t):
    rel_vec = dot_t - pos_t
    rel_x_hat = jnp.arctan2(dot_hat[1],dot_hat[0])
    rel_y_hat = jnp.arctan2(dot_hat[3],dot_hat[2])
    rel_vec_hat = jnp.array([rel_x_hat,rel_y_hat])
    loss_d = -jnp.sum(jnp.exp(jnp.cos(rel_vec_hat - rel_vec)-1))
    return loss_d,rel_vec_hat

# @partial(jax.jit,static_argnums=())
def body_fnc(SC,p_weights,params,pos_0,dot_0,dot_vec,h_0,samples):
    # ID_ARR,VEC_ARR,H1VEC_ARR = SC
    loss_v_arr,loss_d_arr = (jnp.zeros(params["TOT_STEPS"]) for _ in range(2))
    v_pred_arr,v_t_arr = (jnp.zeros((params["TOT_STEPS"]+1,params["N"])) for _ in range(2))
    rel_vec_hat_arr = jnp.zeros((params["TOT_STEPS"],2))
    pos_arr,dot_arr,h1vec_arr,vec_arr = conditional_gen_timeseries(SC,pos_0,dot_0,dot_vec,samples,params,params["STEP_ARRAY"])
    loss_v,loss_d = 0,0
    h_t_1 = h_0
    v_t_1 = neuron_act(params["COLORS"],params["THETA"],params["SIGMA_A"],dot_arr[:,0],pos_arr[:,0])
    v_t_arr = v_t_arr.at[0,:].set(v_t_1)
    # think: -v_t_1 freewheel correct ((-movement correctly 'updated'/returned from move)) -correct indexing
    for t in range(params["TOT_STEPS"]):
        v_pred,dot_hat_t,h_t = plan(h1vec_arr[:,t],v_t_1,h_t_1,p_weights,params) # h_0[s,:]
        loss_d,rel_vec_hat = loss_dot(dot_hat_t,dot_arr[:,t+1],pos_arr[:,t+1])
        v_t = neuron_act(params["COLORS"],params["THETA"],params["SIGMA_A"],dot_arr[:,t+1],pos_arr[:,t+1]) # move(pos_arr[:,t+1],dot_arr[:,t+1],params)
        loss_v = jnp.sum((v_pred-v_t)**2)
        v_pred_arr = v_pred_arr.at[t+1,:].set(v_pred)
        rel_vec_hat_arr = rel_vec_hat_arr.at[t,:].set(rel_vec_hat)
        v_t_arr = v_t_arr.at[t+1,:].set(v_t)
        loss_v_arr = loss_v_arr.at[t].set(loss_v)
        loss_d_arr = loss_d_arr.at[t].set(loss_d)
        h_t_1,v_t_1 = h_t,v_t # continuous hidden state
        if t >= params["INIT_STEPS"]:
            loss_v += loss_v
            loss_d += loss_d
    loss_tot = loss_v + params["LAMBDA_D"]*loss_d
    avg_loss = loss_tot/(params["PRED_STEPS"])
    return avg_loss,(v_pred_arr,v_t_arr,pos_arr,dot_arr,rel_vec_hat_arr,loss_v_arr,loss_d_arr) # final v_pred,v_t,pos_t

# @partial(jax.jit,static_argnums=())
def forward_model_loop(SC,weights,params):
    p_weights = weights["p_weights"]
    T = params["TOT_EPOCHS"]
    loss_arr,loss_sem = (jnp.empty(params["TOT_EPOCHS"]) for _ in range(2))
    loss_v_arr,v_std_arr,loss_r_arr,r_std_arr = (jnp.empty((params["TOT_EPOCHS"],params["TOT_STEPS"])) for _ in range(4))
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
        val_grad_vmap = jax.vmap(val_grad,in_axes=(None,None,None,0,0,0,0,0),out_axes=(0,0))
        (loss,aux),grads = val_grad_vmap(SC,p_weights,params,pos_0,dot_0,dot_vec,h_0,samples) # val_grad_vmap ,,v_0
        v_pred_arr,v_t_arr,pos_arr,dot_arr,rel_vec_hat_arr,loss_v_arr_,loss_d_arr_ = aux # [VMAPS,STEPS,N]x2,[VMAPS,STEPS,2]x3,[VMAPS,STEPS]x2 (final timestep)
        grads_ = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
        opt_update,opt_state = optimizer.update(grads_,opt_state,p_weights)
        p_weights = optax.apply_updates(p_weights,opt_update)
        loss_arr = loss_arr.at[e].set(jnp.mean(loss))
        loss_sem = loss_sem.at[e].set(jnp.std(loss)/jnp.sqrt(params["VMAPS"]))
        loss_v_arr = loss_v_arr.at[e,:].set(jnp.mean(loss_v_arr_,axis=0))
        loss_r_arr = loss_r_arr.at[e,:].set(jnp.mean(loss_d_arr_,axis=0))
        v_std_arr = v_std_arr.at[e,:].set(jnp.std(loss_v_arr_,axis=0)/jnp.sqrt(params["VMAPS"]))
        r_std_arr = r_std_arr.at[e,:].set(jnp.std(loss_d_arr_,axis=0)/jnp.sqrt(params["VMAPS"]))
        print("e={}",e)
        print("loss_tot={}",jnp.mean(loss))
        print("loss_sem={}",jnp.std(loss)/jnp.sqrt(params["VMAPS"]))
        print("loss_v={}",jnp.mean(loss_v_arr_,axis=0))
        # print("std_v={}",jnp.std(loss_v_arr)/jnp.sqrt(params["VMAPS"]))
        print("loss_d={}",jnp.mean(loss_d_arr_,axis=0))
        # print("std_d={}",jnp.std(loss_d_arr)/jnp.sqrt(params["VMAPS"]))
        # if e == T-1:
        #     print("v_pred_arr.shape=",v_pred_arr.shape)
        #     print("v_t_arr.shape=",v_t_arr.shape)
        #     print("pos_arr.shape=",pos_arr.shape)
        #     print("dot_arr.shape=",dot_arr.shape)
        #     print("rel_vec_hat_arr.shape=",rel_vec_hat_arr.shape)
        #     print("loss_v_arr.shape=",loss_v_arr.shape)
        #     print("loss_d_arr.shape=",loss_d_arr.shape)
        arrs = (loss_arr,loss_sem,loss_v_arr,v_std_arr,loss_r_arr,r_std_arr)
        aux = (v_pred_arr,v_t_arr,loss_v_arr_,loss_d_arr_,pos_arr,dot_arr,rel_vec_hat_arr,opt_state,p_weights)
    return arrs,aux # [VMAPS,STEPS,N]x2,[VMAPS,STEPS,2]x3,[VMAPS,STEPS]x2,..

# hyperparams
TOT_EPOCHS = 20000 #250000
EPOCHS = 1
PLOTS = 3
# LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 550 # 800,500
PLAN_ITS = 10 # 8,5
INIT_STEPS = 3 
PRED_STEPS = 3 # 1
TOT_STEPS = INIT_STEPS + PRED_STEPS
LR = 0.00001 # 0.003,,0.0001
WD = 0.0001 # 0.0001
H = 500 # 500,300
INIT = 2 # 0.5,0.1
LAMBDA_D = 1 # 0.1

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
MODULES = 15 # 10,8
M = MODULES**2
NEURONS = 10
N = 3*(NEURONS**2)
SIGMA_A = 0.5 # 0.5,1,0.3,1,0.5,1,0.1
APERTURE = jnp.pi/2 ###
ACTION_FRAC = 2/5
ACTION_SPACE = ACTION_FRAC*APERTURE # 'AGENT_SPEED'
DOT_SPEED = 0.5
THETA = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
COLORS = jnp.array([[255,255,255]]) # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = 1 #COLORS.shape[0]
STEP_ARRAY = jnp.arange(1,TOT_STEPS+1)
DOT_0 = None # gen_dot(ke[0],EPOCHS,VMAPS,N_dot,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #gen_dot(ke[0],EPOCHS,VMAPS,N_dot,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #gen_dot(ke[0],EPOCHS,VMAPS,N_dot,APERTURE)
DOT_VEC = None
SAMPLES = None # rnd.choice(ke[2],M,(EPOCHS,VMAPS,STEPS)) # [E,V] rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
POS_0 = None # rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
HP_0 = None # jnp.sqrt(INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) # [E,V,H]
ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True) # CHANGE TO PERMUATION
VEC_ARR = gen_vectors(MODULES,ACTION_SPACE)
H1VEC_ARR = jnp.diag(jnp.ones(M))[:,ID_ARR]
SC = (ID_ARR,VEC_ARR,H1VEC_ARR)

# INITIALIZATION ### FIX
ki = rnd.split(rnd.PRNGKey(1),num=50)
W_h10 = jnp.sqrt(INIT/(H+M))*rnd.normal(ki[0],(H,M))
W_v0 = jnp.sqrt(INIT/(H+N))*rnd.normal(ki[1],(H,N))
W_r0 = jnp.sqrt(INIT/(N+H))*rnd.normal(ki[2],(N,H))
W_dot0 = jnp.sqrt(INIT/(H+4))*rnd.normal(ki[3],(4,H))
U_vh0,_ = jnp.linalg.qr(jnp.sqrt(INIT/(H+H))*rnd.normal(ki[2],(H,H)))
b_vh0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[3],(H,))

params = {
    "TOT_EPOCHS" : TOT_EPOCHS,
    "EPOCHS" : EPOCHS,
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
    "N" : N,
    "M" : M,
    "INIT" : INIT,
    "VMAPS" : VMAPS,
    "PLOTS" : PLOTS,
    # "KEYS" : KEYS,
    "N_DOTS" : N_DOTS,
    "DOT_0" : DOT_0,
    "DOT_VEC" : DOT_VEC,
    "THETA" : THETA,
    "SIGMA_A" : SIGMA_A,
    "COLORS" : COLORS,
    "SAMPLES" : SAMPLES,
    "STEP_ARRAY" : STEP_ARRAY,
    "POS_0" : POS_0,
    # "V_0" : V_0,
    "HP_0" : HP_0,
    "LAMBDA_D" : LAMBDA_D,
    "ACTION_SPACE" : ACTION_SPACE,
    "DOT_SPEED" : DOT_SPEED,
}

weights = {
    's_weights' : {

    },
    'p_weights' : {
        "W_h1" : W_h10,
        "W_r" : W_r0,
        "W_v" : W_v0,
        "W_dot" : W_dot0,
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
(loss_arr,loss_sem,loss_v_arr,v_std_arr,loss_r_arr,r_std_arr) = arrs
(v_pred_arr,v_t_arr,loss_v_arr_,loss_d_arr_,pos_arr,dot_arr,rel_vec_hat_arr,opt_state,p_weights) = aux

# plot training loss
print("Training time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
plt.figure(figsize=(12,6))
title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, init={INIT:.2f}, update={LR:.6f}, WD={WD:.5f}, \n SIGMA_A={SIGMA_A:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H={H}'
fig,ax = plt.subplots(1,3,figsize=(12,6))
plt.suptitle('forward_new_training_v1, '+title__,fontsize=14)
plt.subplot(1,3,1)
plt.errorbar(jnp.arange(TOT_EPOCHS),loss_arr,yerr=loss_sem,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
# plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Total loss',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.subplot(1,3,2)
plt.errorbar(jnp.arange(TOT_EPOCHS),jnp.mean(loss_v_arr,axis=1),yerr=jnp.mean(v_std_arr,axis=1),color='black',ecolor='lightgray',elinewidth=2,capsize=0)
# plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Loss_v',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.subplot(1,3,3)
plt.errorbar(jnp.arange(TOT_EPOCHS),jnp.mean(loss_r_arr,axis=1),yerr=jnp.mean(r_std_arr,axis=1),color='black',ecolor='lightgray',elinewidth=2,capsize=0)
# plt.xscale('log')
# plt.yscale('log')
plt.ylabel(r'Loss_r',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.tight_layout()
plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_ + 'forward_new_training_v1_' + dt + '.png')

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
# plt.suptitle('forward_new_training_v1, '+title__,fontsize=10)
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
# plt.savefig(path_ + 'figs_forward_new_training_v1_' + dt + '.png')

save_pkl((arrs,aux),'forward_new_v1_'+str(M)) # (no rel_vec / loss_v / loss_d)