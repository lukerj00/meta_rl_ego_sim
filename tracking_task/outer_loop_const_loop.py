# -*- coding: utf-8 -*-
# simple test using constant policy
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

# @jit
def sample_action_old(r,ind,params):
    key = rnd.PRNGKey(ind) ##
    p_ = 1/(1+jnp.exp(-(r-params["SIGMOID_MEAN"])/params["SIGMA_S"])) # (shifted sigmoid)
    return jnp.int32(rnd.bernoulli(key,p=p_)) # 1=action, 0=plan

def sample_action_new_(r,ind,params,alpha):
    key = rnd.PRNGKey(ind) ##
    p_ = 1/(1+jnp.exp(-(r-alpha)/params["SIGMA_S"])) # (shifted sigmoid)
    return jnp.int32(rnd.bernoulli(key,p=p_)) # 1=action, 0=plan

def sample_action_new(r,ind,params,alpha):
    key = rnd.PRNGKey(ind) ##
    p_ = jnp.power(r,alpha)
    return jnp.int32(rnd.bernoulli(key,p=p_)) # 1=action, 0=plan

# @jit
def sample_vec(SC,ind):
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    h1vec_t = H1VEC_ARR[:,ind] # rnd.choice(key,params["M"])]
    vec_t = VEC_ARR[:,ind] # rnd.choice(key,params["M"])]
    return h1vec_t,vec_t

# @partial(jax.jit,static_argnums=(1,2,3))
def gen_dots(key,VMAPS,N_DOTS,APERTURE):
    keys = rnd.split(key,N_DOTS)
    dots_0 = rnd.uniform(keys[0],shape=(VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([APERTURE/4,APERTURE/4]),maxval=jnp.array([3*APERTURE/4,3*APERTURE/4]))
    dots_1 = rnd.uniform(keys[1],shape=(VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([APERTURE/4,-APERTURE/4]),maxval=jnp.array([3*APERTURE/4,-3*APERTURE/4]))
    dots_2 = rnd.uniform(keys[2],shape=(VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([-3*APERTURE/4,-APERTURE]),maxval=jnp.array([-APERTURE/4,APERTURE]))
    dots_tot = jnp.concatenate((dots_0,dots_1,dots_2),axis=1)
    # DOTS = rnd.uniform(key,shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE)
    return dots_tot[:,:N_DOTS,:]

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

def new_params(params,e):
    params_ = params
    VMAPS = params["VMAPS"]
    N_DOTS = params["N_DOTS"]
    APERTURE = params["APERTURE"]
    MODULES = params["MODULES"]
    N = params["N"]
    M = params["M"]
    H_P = params["H_P"]
    H_R = params["H_R"]
    T = params["TRIAL_LENGTH"]
    # key_ = rnd.PRNGKey(0)
    ki = rnd.split(rnd.PRNGKey(e),num=10) #l/0
    params_["HR_0"] = jnp.zeros((VMAPS,H_R)) #jnp.sqrt(INIT_R/(H_R))*rnd.normal(ki[0],(VMAPS,H_R))
    params_["HP_0"] = jnp.sqrt(INIT_P/(H_P))*rnd.normal(ki[1],(VMAPS,H_P)) ### CHECK/CHANGE
    params_["POS_0"] = rnd.choice(ke[2],jnp.arange(-APERTURE,APERTURE,0.01),(VMAPS,2))
    params_["DOTS"] = gen_dots(ki[3],VMAPS,N_DOTS,APERTURE) #key_,rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #jnp.tile(jnp.array([[2,3]]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,1)) #
    params_["SELECT"] = jnp.eye(N_DOTS)[rnd.choice(ki[4],N_DOTS,(VMAPS,))] #rnd.choice(ki[1],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2))
    params_["IND"] = rnd.randint(ki[5],(VMAPS,T),minval=0,maxval=M,dtype=jnp.int32)
    return params_

@jit
def r_predict(v_t,v_t_1,r_t_1,hr_t_1,r_weights):
    W_vt_z = r_weights["W_vt_z"]
    W_vt_1z = r_weights["W_vt_1z"]
    W_rt_1z = r_weights["W_rt_1z"]
    U_z = r_weights["U_z"]
    b_z = r_weights["b_z"]
    W_vt_f = r_weights["W_vt_f"]
    W_vt_1f = r_weights["W_vt_1f"]
    W_rt_1f = r_weights["W_rt_1f"]
    U_f = r_weights["U_f"]
    b_f = r_weights["b_f"]
    W_vt_h = r_weights["W_vt_h"]
    W_vt_1h = r_weights["W_vt_1h"]
    W_rt_1h = r_weights["W_rt_1h"]
    U_h = r_weights["U_h"]
    b_h = r_weights["b_h"]
    W_read = r_weights["W_read"]
    W_sel = r_weights["W_sel"]
    z_t = jax.nn.sigmoid(jnp.matmul(W_vt_z,v_t) + jnp.matmul(W_vt_1z,v_t_1) + W_rt_1z*r_t_1 + jnp.matmul(U_z,hr_t_1) + b_z)
    f_t = jax.nn.sigmoid(jnp.matmul(W_vt_f,v_t) + jnp.matmul(W_vt_1f,v_t_1) + W_rt_1f*r_t_1 + jnp.matmul(U_f,hr_t_1) + b_f)
    hhat_t = jax.nn.tanh(jnp.matmul(W_vt_h,v_t) + jnp.matmul(W_vt_1h,v_t_1) + W_rt_1h*r_t_1 + jnp.matmul(U_h,jnp.multiply(f_t,hr_t_1)) + b_h)
    hr_t = jnp.multiply(z_t,hr_t_1) + jnp.multiply(1-z_t,hhat_t)
    r_hat_t = jnp.matmul(W_read,hr_t)
    return r_hat_t,hr_t

@partial(jax.jit,static_argnums=(5,))
def v_predict(h1vec_t,v_t_1,hv_t_1,p_weights,params,PLAN_ITS): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    W_h1 = p_weights["W_h1"]
    W_v = p_weights["W_v"]
    U_vh = p_weights["U_vh"]
    b_vh = p_weights["b_vh"]
    W_r = p_weights["W_r"]
    for i in range(PLAN_ITS):
        hv_t_1 = jax.nn.sigmoid(jnp.matmul(W_h1,h1vec_t) + jnp.matmul(W_v,v_t_1) + jnp.matmul(U_vh,hv_t_1) + b_vh)
    v_t = jnp.matmul(W_r,hv_t_1)
    return v_t,hv_t_1 # v_t,hv_t

def plan(h1vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,weights,params,pos_t_1,dots,vec_t,sel):#(vec_t,r_t_1,hr_t_1,v_t_1,hv_t_1,weights,params):
    v_t_1_true = neuron_act(params,dots,pos_t_1)###
    r_t_1_true = loss_obj(dots,sel,pos_t_1,params["SIGMA_R"])
    pos_t = pos_t_1 + vec_t###
    v_t_true = neuron_act(params,dots,pos_t)###
    v_t,hv_t = v_predict(h1vec_t,v_t_1,hv_t_1,weights["p"],params,params["PLAN_ITS"])#(hr_t_1,v_t_1,pos_t_1,weights["v"],params)
    r_t,hr_t = r_predict(v_t,v_t_1,r_t_1,hr_t_1,weights["r"])
    v_tp,_ = v_predict(h1vec_t,v_t_1_true,hv_t_1,weights["p"],params,params["PLAN_ITS"])#(hr_t_1,v_t_1,pos_t_1,weights["v"],params)
    r_tp,_ = r_predict(v_t_true,v_t_1_true,r_t_1_true,hr_t_1,weights["r"])
    return r_t,hr_t,v_t,r_tp,v_tp #,hv_t # predictions

def move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,params): # shouldnt be random; should take action dictated by plan...
    pos_t = pos_t_1 + vec_t
    v_t = neuron_act(params,dots,pos_t)
    r_t = loss_obj(dots,sel,pos_t,params["SIGMA_R"])
    vhat_t_,_ = v_predict(h1vec_t,v_t_1,hv_t_1,weights["p"],params,params["PLAN_ITS"])
    rhat_t_,hr_t = r_predict(vhat_t_,v_t_1,r_t_1,hr_t_1,weights["r"])
    # r_loss = (r_t-rhat_t_)**2
    # v_loss = jnp.sum((v_t-vhat_t_)**2)
    # print('v_loss=',v_loss,'r_loss=',r_loss)
    return r_t,hr_t,v_t,pos_t,rhat_t_,vhat_t_ # hv_t, true

def body_fnc(SC,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,weights,params,c,alpha):
    sample_arr,r_arr,pos_arr,r_true_arr,r_tp_arr = ([] for _ in range(5))
    r_loss_arr,v_loss_arr = (jnp.nan*jnp.zeros((params["TRIAL_LENGTH"],)) for _ in range(2))
    s_ = 0
    t = 0
    R = 0
    cost = 0
    v_t_1 = _v_t_1 = neuron_act(params,dots,pos_t_1)
    r_t_1 = _r_t_1 = loss_obj(dots,sel,pos_t_1,params["SIGMA_R"])
    _hr_t_1 = hr_t_1
    while t < params["TRIAL_LENGTH"]:
        if t < params["INIT_STEPS"]:
            h1vec_t,vec_t = sample_vec(SC,ind[t])
            r_t,hr_t,v_t,pos_t,*_ = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,params)
            r_arr.append(r_t)
            r_true_arr.append(r_t)
            pos_arr.append(pos_t)
            sample_arr.append(1)
            r_t_1,hr_t_1,v_t_1,pos_t_1 = r_t,hr_t,v_t,pos_t
            t += 1
            # print('t=',t,'s=',s_,'r=',r_t,'pos=',pos_t,'vec=',vec_t,'h1vec=',jnp.argmax(h1vec_t),'LEN=',len(r_arr))
        else:
            if s_ == 0:
                h1vec_t,vec_t = sample_vec(SC,ind[t])
                r_t_,_,v_t_,*_ = plan(h1vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,weights,params,pos_t_1,dots,vec_t,sel)
                r_arr.append(r_t_)
                pos_arr.append(pos_t)
                sample_arr.append(s_)
                s_ = sample_action_new_(r_t_,ind[t],params,alpha)
                _r_t,_,_v_t,*_ = move(h1vec_t,vec_t,_v_t_1,hv_t_1,_r_t_1,_hr_t_1,pos_t_1,dots,sel,weights,params)
                r_true_arr.append(_r_t)
                r_loss_arr = r_loss_arr.at[t].set(jnp.log((_r_t-r_t_)**2))
                v_loss_arr = v_loss_arr.at[t].set(jnp.log(jnp.sum((_v_t-v_t_)**2)))
                cost += params["C"]*c
                t += 1
                # print('t=',t-1,'s=',s_,'r=',r_t_,'pos=',pos_t,'h1vec=',jnp.argmax(h1vec_t),'v_loss=',v_loss_arr[t-1],'r_loss=',r_loss_arr[t-1])
            if s_ == 1:
                r_t,hr_t,v_t,pos_t,_,_ = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,params)
                r_arr.append(r_t)
                pos_arr.append(pos_t)
                sample_arr.append(s_)
                s_ = 0
                _r_t,_,_v_t,*_ = move(h1vec_t,vec_t,_v_t_1,hv_t_1,_r_t_1,_hr_t_1,pos_t_1,dots,sel,weights,params)
                r_true_arr.append(_r_t)
                r_loss_arr = r_loss_arr.at[t].set(jnp.log((_r_t-r_t)**2))
                v_loss_arr = v_loss_arr.at[t].set(jnp.log(jnp.sum((_v_t-v_t)**2)))
                r_t_1,hr_t_1,v_t_1,pos_t_1 = r_t,hr_t,v_t,pos_t
                cost += params["C"]*1
                R += params["R"]*r_t
                t += 1
                # print('t=',t,'s=',s_,'r=',r_t,'pos=',pos_t,'h1vec=',jnp.argmax(h1vec_t),'LEN=',len(r_arr))
        R_trial = R - cost
        # r_arr,pos_arr,sample_arr,v_loss_arr,r_loss_arr,r_true_arr = (lambda:)
    return jnp.array(r_arr[:params["TRIAL_LENGTH"]]),jnp.array(pos_arr[:params["TRIAL_LENGTH"]]),jnp.array(sample_arr[:params["TRIAL_LENGTH"]]),jnp.array(v_loss_arr[:params["TRIAL_LENGTH"]]),jnp.array(r_loss_arr[:params["TRIAL_LENGTH"]]),jnp.array(r_true_arr[:params["TRIAL_LENGTH"]]),R_trial

def full_loop(SC,weights,params,c,alpha):
    r_arr,sample_arr = (jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"])) for _ in range(2)) # ,params["VMAPS"]
    pos_arr = jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"],2)) # ,params["VMAPS"]
    dots_arr = jnp.zeros((params["TOT_EPOCHS"],3,2))
    v_loss_arr = jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"]))
    r_loss_arr = jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"]))
    r_true_arr = jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"]))
    r_tp_arr = jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"]))
    v_tp_arr = jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"]))
    R_trial_tot = []
    E = params["TOT_EPOCHS"]
    for e in range(E):
        params = new_params(params,e)
        hp_0 = params["HP_0"][0,:]
        hr_0 = params["HR_0"][0,:]
        pos_0 = params["POS_0"][0,:]
        dots = params["DOTS"][0,:,:]
        sel = params["SELECT"][0,:]
        ind = params["IND"][0,:]
        # body_fnc_vmap = jax.vmap(body_fnc,in_axes=(None,0,0,0,0,0,0,None,None,None),out_axes=(0,0))
        r_arr_,pos_arr_,sample_arr_,v_loss_arr_,r_loss_arr_,r_true_arr_,R_trial_ = body_fnc(SC,hr_0,hp_0,pos_0,dots,sel,ind,weights,params,c,alpha)
        r_arr = r_arr.at[e,:].set(r_arr_)
        pos_arr = pos_arr.at[e,:,:].set(pos_arr_)
        sample_arr = sample_arr.at[e,:].set(sample_arr_)
        dots_arr = dots_arr.at[e,:,:].set(dots)
        v_loss_arr = v_loss_arr.at[e,:].set(v_loss_arr_)
        r_loss_arr = r_loss_arr.at[e,:].set(r_loss_arr_)
        r_true_arr = r_true_arr.at[e,:].set(r_true_arr_)
        R_trial_tot.append(R_trial_)
        # r_tp_arr = r_tp_arr.at[e,:].set(r_tp_arr_)
        # r_tp_arr = r_tp_arr.at[e,:].set(r_tp_arr_)
        # v_tp_arr = v_tp_arr.at[e,:].set(v_tp_arr_)
        # r_arr_,r_std_,pos_arr_,sample_arr_ = train_loop(SC,weights,params,l)
        # r_arr.append(r_arr_)
        # r_std.append(r_std_)
        # pos_arr.append(pos_arr_)
        # sample_arr.append(sample_arr_)
        print("e={}",e)
        if e == E-1:
            pass
    R_trial_avg = jnp.mean(jnp.array(R_trial_tot))
    # cat_squeeze = lambda arr: jnp.concatenate(jnp.array(arr),axis=0)
    # r_arr,r_std,pos_arr,sample_arr = (cat_squeeze(arr) for arr in [r_arr,r_std,pos_arr,sample_arr])
    return r_arr,pos_arr,sample_arr,dots_arr,v_loss_arr,r_loss_arr,r_true_arr,R_trial_avg

# hyperparams
TOT_EPOCHS = 20
PLOTS = 5
VMAPS = 1 # 1100,1000,800,500
INIT_P = 2 # 0.5
INIT_R = 3 # 5,2
H_P = 500 # 500,300
H_R = 100
PLAN_ITS = 10
INIT_STEPS = 30
TRIAL_LENGTH = 100 # 100

# ENV/sc params
R = 4
C = 1.5
c_ = [0,0.25,0.5,0.75,1]
ALPHA = jnp.linspace(0,1,20) # jnp.logspace(-2,2,20)
ke = rnd.split(rnd.PRNGKey(0),10)
MODULES = 15 # 10,8
M = MODULES**2
NEURONS = 10
N = 3*(NEURONS**2)
SIGMA_R = 1 # 1.2,1.5,1
SIGMA_A = 0.5 # 0.5
SIGMA_S = 0.05 #0.01
SIGMOID_MEAN = 0.5
APERTURE = jnp.pi
THETA_X = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
THETA_Y = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]]) # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = COLORS.shape[0]
HP_0 = None #jnp.sqrt(INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) # [E,V,H]
HR_0 = None #jnp.sqrt(INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) # [E,V,H]
POS_0 = None #rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
DOTS = None #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
SELECT = None #jnp.eye(N_DOTS)[rnd.choice(ke[1],N_DOTS,(EPOCHS,VMAPS))]
IND = None
ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True) # CHANGE TO PERMUATION
VEC_ARR = gen_vectors(MODULES,APERTURE)
H1VEC_ARR = jnp.diag(jnp.ones(M))[:,ID_ARR]
SC = (ID_ARR,VEC_ARR,H1VEC_ARR)

# INITIALIZATION
pass

params = {
    "TOT_EPOCHS" : TOT_EPOCHS,
    "MODULES" : MODULES,
    "NEURONS" : NEURONS,
    "APERTURE" : APERTURE,
    "H_P" : H_P,
    "H_R" : H_R,
    "N" : N,
    "M" : M,
    "INIT_P" : INIT_P,
    "INIT_R" : INIT_R,
    "VMAPS" : VMAPS,
    "PLOTS" : PLOTS,
    "N_DOTS" : N_DOTS,
    "DOTS" : DOTS,
    "THETA_X" : THETA_X,
    "THETA_Y" : THETA_Y,
    "COLORS" : COLORS,
    "POS_0" : POS_0,
    "HP_0" : HP_0,
    "HR_0" : HR_0,
    "IND" : IND,
    "INIT_STEPS" : INIT_STEPS,
    "TRIAL_LENGTH" : TRIAL_LENGTH,
    "PLAN_ITS" : PLAN_ITS,
    "SIGMOID_MEAN" : SIGMOID_MEAN,
    "SIGMA_R" : SIGMA_R,
    "SIGMA_A" : SIGMA_A,
    "SIGMA_S" : SIGMA_S,
    "ALPHA" : ALPHA,
    "c_" : c_,
    "R" : R,
    "C" : C,
    }

weights = {
    "s" : { # policy weights

    },
    "p" : { # p_weights
    # LOAD IN
    },
    "r" : { # r_weights
    # LOAD IN
    }
}

###
*_,p_weights = load_('/sc_project/pkl/forward_v9_225_17_06-2358.pkl') # /pkl/forward_v8M_08_06-1857.pkl
*_,r_weights = load_('/sc_project/pkl/value_v5_225_13_06-1849.pkl') # sc_project/pkl/value_v5_225_10_06-0722.pkl /homes/lrj34/projects/meta_rl_ego_sim/sc_project/pkl/value_v5_225_09_06-1830.pkl
weights['p'] = p_weights
weights['r'] = r_weights
# print("Loaded weights=******************** \n ",weights['p'].keys(),"\n",weights['r'].keys())
###
startTime = datetime.now()

R_trial_arr = np.zeros((len(c_),len(ALPHA)))
for i,c in enumerate(c_):
    for j,alpha in enumerate(ALPHA):
        *_,R_trial_avg = full_loop(SC,weights,params,c,alpha) # r_arr,pos_arr,sample_arr,dots_arr,v_loss_arr,r_loss_arr,r_true_arr
        R_trial_arr[i,j] = R_trial_avg
R_trial_norm = scipy.stats.zscore(R_trial_arr,axis=None)
# plot training loss
print("Sim time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
print('R_trial_norm=',R_trial_norm)

# create blue colormap:
cmap = plt.cm.Blues

# use matplotlib to plot each row of R_trial_arr with a different color:
fig,ax = plt.subplots()
title_ = f'R_trial_avg (p_act = sigmoid with centre alpha), R={R}, C={C})' # sigmoid with centre alpha , (p_act = x^alpha)
plt.suptitle(title_)
for i in range(len(c_)):
    ax.plot(ALPHA,R_trial_norm[i,:],label='c='+str(c_[i]),color=cmap((i+1)/(len(c_)+1)))
ax.set_xlabel('alpha',fontsize=14)
ax.set_ylabel('mean reward (zscore)',fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.xscale('log')
plt.legend()

# r_arr = r_arr[-PLOTS:,:] # (3,0,T)
# pos_arr = pos_arr[-PLOTS:,:,:] # (3,0,T),((-1 if e))
# dots_arr = dots_arr[-PLOTS:,:,:] # (3,0,T,N,2)
# sample_arr = sample_arr[-PLOTS:,:]
# v_loss_arr = v_loss_arr[-PLOTS:,:]
# r_loss_arr = r_loss_arr[-PLOTS:,:]
# r_true_arr = r_true_arr[-PLOTS:,:]

# Manually define legend
# legend_handles = [
#     plt.Line2D([], [], color='r', marker='x', label='r_t'),
#     plt.Line2D([], [], color='k', marker='o', label='r_hat_t'),
#     plt.Line2D([], [], color='b', label='v_loss'), # marker='.', 
#     plt.Line2D([], [], color='purple', label='r_loss'), # marker='.', 
#     plt.Line2D([], [], color='g', marker='o', label='r_true'), # marker='.',
    # plt.Line2D([], [], color='orange', marker='o', label='r_tp'), # marker='.',
# ]

# colors_ = np.float32([[255,0,0],[0,255,0],[0,0,255]])/255 # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])/255
# T_ = [15,35,90] # ,[5,15,25]
# neuron_locs = gen_vectors(NEURONS,APERTURE)

# plot timeseries of true/planned reward
# fig,axis = plt.subplots(PLOTS,2,figsize=(10,6*PLOTS)) #9,4x plt.figure(figsize=(12,6))
# title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, INIT_STEPS={INIT_STEPS}, TRIAL_LENGTH={TRIAL_LENGTH} \n SIGMA_R={SIGMA_R:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H_P={H_P}, H_R={H_R}'
# plt.suptitle('outer_loop_const, '+title__,fontsize=10)
# for i in range(params["PLOTS"]):
#     ax0 = plt.subplot2grid((2*PLOTS,2),(2*i,0),colspan=2,rowspan=1)
#     ax0.set_title('r_t/r_hat_t') # v_arr
#     ax0.spines['right'].set_visible(False)
#     ax0.spines['top'].set_visible(False)
#     ax0.set_ylabel('r_t',fontsize=15)
#     ax0.set_xlabel('t',fontsize=15)
#     ax0.ticklabel_format(useOffset=False)
#     for t in range(TRIAL_LENGTH):
#         ax0.scatter(x=t,y=r_true_arr[i,t],color='g',s=20,marker='+',label='r_true')
#         # ax0.scatter(x=t,y=r_tp_arr[i,t],color='orange',s=15,marker='+',label='r_tp')
#         if sample_arr[i,t]==0:
#             ax0.scatter(x=t,y=r_arr[i,t],color='k',s=5,marker='o',label='rhat_t')
#         else:
#             ax0.scatter(x=t,y=r_arr[i,t],color='r',s=5,marker='x',label='r_t')
#             #
#     # ax0.axhline(y=SIGMOID_MEAN,color='k',linestyle='--',linewidth=1)
#     ax0.axvline(x=INIT_STEPS,color='k',linestyle='--',linewidth=1)
#     ax0.legend(handles=legend_handles)
#     ax01 = ax0.twinx()
#     ax01.plot(v_loss_arr[i,:],color='b',linewidth=1)
#     ax01.plot(r_loss_arr[i,:],color='purple',linewidth=1)
#     ax01.spines['top'].set_visible(False)
#     ax01.set_ylabel('r / v_loss (log)',fontsize=15)
#     if i!=params["PLOTS"]-1:
#         ax0.set_xlabel('')
#         ax0.set_xticks([])
#         ax0.set_xticklabels([])
#     # ax1 = plt.subplot2grid((3*PLOTS,3),(3*i+1,0),colspan=1,rowspan=1)
#     # ax1.set_title('v_pred')#,t='+str(T_[0])+'\n r_t='+str(r_arr[i,T_[0]])+',r_pred='+str(r_pred_arr[i,T_[0]]),fontsize=8) # v_arr
#     # ax1.set_aspect('equal')
#     # for ind,n in enumerate(neuron_locs.T):
#     #     ax1.scatter(n[0],n[1],color=np.float32(jnp.sqrt(v_pred_rgb[i,T_[0],:,ind])),s=np.sum(17*v_pred_rgb[i,T_[0],:,ind]),marker='o') ### sum,np.float32((v_pred_rgb[:,ind]))
#     # for d in range(N_DOTS):
#     #     ax1.scatter(mod_(dots_arr[i,d,0]-pos_arr[i,T_[0],0]),mod_(dots_arr[i,d,1]-pos_arr[i,T_[0],1]),color=(colors_[d,:]),s=52,marker='x') # v_arr[i,T_[0],:], r_arr[i,T_[0]]
#     # ax2 = plt.subplot2grid((3*PLOTS,3),(3*i+1,1),colspan=1,rowspan=1)
#     # ax2.set_title('v_pred')#,t='+str(T_[1])+'\n r_t='+str(r_arr[i,T_[1]])+',r_pred='+str(r_pred_arr[i,T_[1]]),fontsize=8) # v_arr
#     # ax2.set_aspect('equal')
#     # for ind,n in enumerate(neuron_locs.T):
#     #     ax2.scatter(n[0],n[1],color=np.float32(jnp.sqrt(v_pred_rgb[i,T_[1],:,ind])),s=np.sum(17*v_pred_rgb[i,T_[1],:,ind]),marker='o') ### sum,np.float32((v_pred_rgb[:,ind]))
#     # for d in range(N_DOTS):
#     #     ax2.scatter(mod_(dots_arr[i,d,0]-pos_arr[i,T_[1],0]),mod_(dots_arr[i,d,1]-pos_arr[i,T_[1],1]),color=(colors_[d,:]),s=52,marker='x') # v_arr[i,T_[0],:], r_arr[i,T_[0]]
#     # ax3 = plt.subplot2grid((3*PLOTS,3),(3*i+1,2),colspan=1,rowspan=1)
#     # ax3.set_title('v_pred')#,t='+str(T_[2])+'\n r_t='+str(r_arr[i,T_[2]])+',r_pred='+str(r_pred_arr[i,T_[2]]),fontsize=8) # v_arr
#     # ax3.set_aspect('equal')
#     # for ind,n in enumerate(neuron_locs.T):
#     #     ax3.scatter(n[0],n[1],color=np.float32(jnp.sqrt(v_pred_rgb[i,T_[2],:,ind])),s=np.sum(17*v_pred_rgb[i,T_[2],:,ind]),marker='o') ### sum,np.float32((v_pred_rgb[:,ind]))
#     # for d in range(N_DOTS):
#     #     ax3.scatter(mod_(dots_arr[i,d,0]-pos_arr[i,T_[2],0]),mod_(dots_arr[i,d,1]-pos_arr[i,T_[2],1]),color=(colors_[d,:]),s=52,marker='x') # v_arr[i,T_[0],:], r_arr[i,T_[0]]
#     # ax4 = plt.subplot2grid((3*PLOTS,3),(3*i+2,0),colspan=1,rowspan=1)
#     # ax4.set_title('v_t')#,t='+str(T_[0])+'\n r_t='+str(r_arr[i,T_[0]])+',r_pred='+str(r_pred_arr[i,T_[0]]),fontsize=8) # v_arr
#     # ax4.set_aspect('equal')
#     # for ind,n in enumerate(neuron_locs.T):
#     #     # print('v_t_rgb[0]=',v_t_rgb[i,T_[0],:,ind])
#     #     ax4.scatter(n[0],n[1],color=np.float32(jnp.sqrt(v_t_rgb[i,T_[0],:,ind])),s=np.sum(17*v_t_rgb[i,T_[0],:,ind]),marker='o') ### sum,np.float32((v_pred_rgb[:,ind]))
#     # for d in range(N_DOTS):
#     #     ax4.scatter(mod_(dots_arr[i,d,0]-pos_arr[i,T_[0],0]),mod_(dots_arr[i,d,1]-pos_arr[i,T_[0],1]),color=(colors_[d,:]),s=52,marker='x') # v_arr[i,T_[0],:], r_arr[i,T_[0]]
#     # ax5 = plt.subplot2grid((3*PLOTS,3),(3*i+2,1),colspan=1,rowspan=1)
#     # ax5.set_title('v_t')#,t='+str(T_[1])+'\n r_t='+str(r_arr[i,T_[1]])+',r_pred='+str(r_pred_arr[i,T_[1]]),fontsize=8) # v_arr
#     # ax5.set_aspect('equal')
#     # for ind,n in enumerate(neuron_locs.T):
#     #     ax5.scatter(n[0],n[1],color=np.float32(jnp.sqrt(v_t_rgb[i,T_[1],:,ind])),s=np.sum(17*v_t_rgb[i,T_[1],:,ind]),marker='o') ### sum,np.float32((v_pred_rgb[:,ind]))
#     # for d in range(N_DOTS):
#     #     ax5.scatter(mod_(dots_arr[i,d,0]-pos_arr[i,T_[1],0]),mod_(dots_arr[i,d,1]-pos_arr[i,T_[1],1]),color=(colors_[d,:]),s=52,marker='x') # v_arr[i,T_[0],:], r_arr[i,T_[0]]
#     # ax6 = plt.subplot2grid((3*PLOTS,3),(3*i+2,2),colspan=1,rowspan=1)
#     # ax6.set_title('v_t')#,t='+str(T_[2])+'\n r_t='+str(r_arr[i,T_[2]])+',r_pred='+str(r_pred_arr[i,T_[2]]),fontsize=8) # v_arr
#     # ax6.set_aspect('equal')
#     # for ind,n in enumerate(neuron_locs.T):
#     #     ax6.scatter(n[0],n[1],color=np.float32(jnp.sqrt(v_t_rgb[i,T_[2],:,ind])),s=np.sum(17*v_t_rgb[i,T_[2],:,ind]),marker='o') ### sum,np.float32((v_pred_rgb[:,ind]))
#     # for d in range(N_DOTS):
#     #     ax6.scatter(mod_(dots_arr[i,d,0]-pos_arr[i,T_[2],0]),mod_(dots_arr[i,d,1]-pos_arr[i,T_[2],1]),color=(colors_[d,:]),s=52,marker='x') # v_arr[i,T_[0],:], r_arr[i,T_[0]]

plt.axis('tight')
plt.subplots_adjust(top=0.95)

dt = datetime.now().strftime("%d_%m-%H%M%S")
path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
plt.savefig(path_ + 'figs_outer_loop_const_loop_' + dt + '.png') # ctrl_v7_(v,r,h normal, r_tp changed)

# save_pkl((loss_arr,loss_std,p_weights_trained,v_arr,r_arr,r_hat_arr),'value_v3_')
