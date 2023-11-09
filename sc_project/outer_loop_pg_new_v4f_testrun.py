# -*- coding: utf-8 -*-
# adjust dot_vec=, ratio=10, dot sigma=.5, timesteps=120, M_D_S=1.2, add act_kl in properly
# add in hv (FF) and r_hat, as in v6
"""
Created on Wed May 10 2023

@author: lukej
"""
import jax
import jax.numpy as jnp
from jax import jit
import jax.profiler
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
import jax.scipy
from jax.scipy.optimize import minimize
import gc
import copy

import faulthandler; faulthandler.enable()

def load_(str_):
    path_ = str(Path(__file__).resolve().parents[1]) # .../meta_rl_ego_sim/
    with open(path_+str_,'rb') as file_:
        # param = pickle.load(file_)
        param_ = jnp.load(file_,allow_pickle=True)
    return param_

def save_pkl_sc(param,str_):  # can't jit (can't pickle jax tracers)
	path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/pkl_sc/' # '/scratch/lrj34/'
	dt = datetime.now().strftime("%d_%m-%H%M%S")
	with open(path_+str_+'_'+dt+'.pkl','wb') as file:
		pickle.dump(param,file,pickle.HIGHEST_PROTOCOL)

def save_large_outputs(param,str_):  # can't jit (can't pickle jax tracers)
	path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/large_outputs/' # '/sc_project/test_data/' # '/scratch/lrj34/'
	dt = datetime.now().strftime("%d_%m-%H%M%S")
	with open(path_+str_+'_'+dt+'.pkl','wb') as file:
		pickle.dump(param,file,pickle.HIGHEST_PROTOCOL)

@partial(jax.jit,static_argnums=(0,1))
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
    vec_norm = jnp.linalg.norm(VEC_ARR,axis=0)
    prior_vec = jnp.exp(-vec_norm)/jnp.sum(jnp.exp(-vec_norm))
    H1VEC_ARR = jnp.eye(MODULES**2) # [:,ID_ARR]
    zero_vec_index = jnp.where(jnp.all(VEC_ARR == jnp.array([0,0])[:, None], axis=0))[0][0]
    SC = (ID_ARR,VEC_ARR,H1VEC_ARR)
    return SC,prior_vec,zero_vec_index


def gen_dot(key,VMAPS,N_DOTS,ACTION_SPACE):
    keys = rnd.split(key,N_DOTS)
    dot_0 = rnd.uniform(keys[0],shape=(VMAPS,2),minval=-(2*jnp.pi)/4,maxval=(2*jnp.pi)/4)#minval=jnp.array([APERTURE/4,APERTURE/4]),maxval=jnp.array([3*APERTURE/4,3*APERTURE/4]))
    # dot_0 = rnd.uniform(keys[0],shape=(VMAPS,2),minval=-ACTION_SPACE,maxval=ACTION_SPACE)#minval=jnp.array([APERTURE/4,APERTURE/4]),maxval=jnp.array([3*APERTURE/4,3*APERTURE/4]))
    
    # dot_1 = rnd.uniform(keys[1],shape=(VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([APERTURE/4,-APERTURE/4]),maxval=jnp.array([3*APERTURE/4,-3*APERTURE/4]))
    # dot_2 = rnd.uniform(keys[2],shape=(VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi)#minval=jnp.array([-3*APERTURE/4,-APERTURE]),maxval=jnp.array([-APERTURE/4,APERTURE]))
    # dot_tot = jnp.concatenate((dot_0,dot_1,dot_2),axis=1)
    # dot = rnd.uniform(key,shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE)
    return dot_0 #dot_tot[:,:N_dot,:]

def gen_dot_vecs(key,VMAPS,MAX_DOT_SPEED,ACTION_SPACE): # rejection sampling
    dot_vecs = rnd.uniform(key,shape=(VMAPS,2),minval=-ACTION_SPACE,maxval=ACTION_SPACE) # / 2
    # mask = jnp.all((-APERTURE/2 <= dot_vecs)&(dot_vecs <= APERTURE/2),axis=1)
    # while mask.any():
    #     key = rnd.split(key)[0]
    #     new_vecs = rnd.uniform(key,shape=(jnp.sum(mask),2),minval=-APERTURE,maxval=APERTURE)
    #     dot_vecs = dot_vecs.at[mask].set(new_vecs)
    #     mask = jnp.all((-APERTURE/2 <= dot_vecs)&(dot_vecs <= APERTURE/2),axis=1)
    return dot_vecs

def gen_samples(key,MODULES,ACTION_SPACE,PLANNING_SPACE,INIT_STEPS,TOT_STEPS):## (not used)
    M_P = (MODULES-1)//2 # (1/ACTION_FRAC)
    M_A = jnp.int32(M_P*(ACTION_SPACE/PLANNING_SPACE)) ### FIX (AS/PS=2/3)
    init_vals = jnp.arange((2*M_A+1)**2)
    main_vals = jnp.arange((2*M_A+1)**2,MODULES**2)
    keys = rnd.split(key,2)
    init_samples = rnd.choice(keys[0],init_vals,shape=(INIT_STEPS,)) # INIT_STEPS,
    main_samples = rnd.choice(keys[1],main_vals,shape=(TOT_STEPS-INIT_STEPS-1,))
    return jnp.concatenate([init_samples,main_samples],axis=0) # init_samples

def get_inner_activation_indices(N_A, K):
    inner_N = jnp.int32(K) # Length of one side of the central region
    start_idx = (N_A - inner_N) // 2 # Starting index
    end_idx = start_idx + inner_N # Ending index
    row_indices, col_indices = jnp.meshgrid(jnp.arange(start_idx, end_idx), jnp.arange(start_idx, end_idx))
    flat_indices = jnp.ravel_multi_index((row_indices.flatten(), col_indices.flatten()), (N_A, N_A))
    return flat_indices

def get_inner_activation_coords(N_A, K):
    inner_N = jnp.int32(K)  # Length of one side of the central region
    start_idx = (N_A - inner_N) // 2  # Starting index
    end_idx = start_idx + inner_N  # Ending index
    x_coordinates, y_coordinates = jnp.meshgrid(jnp.arange(start_idx, end_idx), jnp.arange(start_idx, end_idx))
    return jnp.array([x_coordinates.flatten(), y_coordinates.flatten()])

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

def mod_(val):
    return (val+jnp.pi)%(2*jnp.pi)-jnp.pi

@jit
def sigma_fnc(VMAPS,SIGMA_RINF,SIGMA_R0,TAU,e):
    k = e #*(VMAPS//BATCH)
    sigma_k = SIGMA_RINF*(1-jnp.exp(-k/TAU))+SIGMA_R0*jnp.exp(-k/TAU) # exp decay to 1/e mag in 1/e time
    return sigma_k

@jit
def loss_obj(dot,pos,params,e): # R_t
    sigma_k = sigma_fnc(params["VMAPS"],params["SIGMA_RINF"],params["SIGMA_R0"],params["TAU"],e)
    dis = dot - pos
    obj = (params["SIGMA_R0"]/sigma_k)*jnp.exp((jnp.cos(dis[0]) + jnp.cos(dis[1]) - 2)/sigma_k**2) ### (positive), standard loss (-) (1/(sigma_e*jnp.sqrt(2*jnp.pi)))*
    return obj #sigma_e

# @jit
# def sample_action(r,ind,params):
#     key = rnd.PRNGKey(ind) ##
#     p_ = 1/(1+jnp.exp(-(r-params["SIGMOID_MEAN"])/params["SIGMA_S"])) # (shifted sigmoid)
#     return jnp.int32(rnd.bernoulli(key,p=p_)) # 1=action, 0=plan

@jit
def sample_vec_rand(SC,ind):
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    h1vec_t = H1VEC_ARR[:,ind] # rnd.choice(key,params["M"])]
    vec_t = VEC_ARR[:,ind] # rnd.choice(key,params["M"])]
    return h1vec_t,vec_t

# def gen_vectors(m, A): # modules/neurons, aperture
#     x = jnp.linspace(-(A-A/m), (A-A/m), m)
#     x_ = jnp.tile(x, (m, ))
#     y_ = jnp.repeat(jnp.flip(x), m)
#     return jnp.vstack([x_, y_])

@jit
def get_plan_rate(data,mask):
    masked_data = data * mask
    sum_masked_data = jnp.sum(masked_data)
    valid_entries = jnp.sum(mask)
    mean = sum_masked_data / valid_entries
    variance = jnp.sum((masked_data - mean) ** 2 * mask) / valid_entries
    sem = jnp.sqrt(variance)/jnp.sqrt(valid_entries)
    return mean, sem

@jit
def curve_obj(params,x,weights):
    mu,log_sigma = params
    sigma = jnp.exp(log_sigma)
    gaussian = jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return jnp.sum((gaussian - weights) ** 2)

# @jit
def curve_fit(v_t,MODULES,APERTURE,NEURON_GRID_AP):
    x_data = NEURON_GRID_AP[0,:]
    y_data = NEURON_GRID_AP[1,:]
    init_params_x = jnp.array([jnp.mean(x_data), jnp.log(jnp.std(x_data)) + 1e-8])
    init_params_y = jnp.array([jnp.mean(y_data), jnp.log(jnp.std(y_data)) + 1e-8])
    result_x = jax.scipy.optimize.minimize(curve_obj, init_params_x, args=(x_data, v_t), method='BFGS', options={'gtol': 1e-3})
    result_y = jax.scipy.optimize.minimize(curve_obj, init_params_y, args=(y_data, v_t), method='BFGS', options={'gtol': 1e-3})
    mu_x, log_sigma_x = result_x.x
    sigma_x = jnp.exp(log_sigma_x)
    mu_y, log_sigma_y = result_y.x
    sigma_y = jnp.exp(log_sigma_y)
    mean_reward = 1/(2*jnp.pi*sigma_x*sigma_y)
    return jnp.array([mu_x,mu_y]),jnp.array([sigma_x,sigma_y]),mean_reward

def circular_mean(v_t,NEURON_GRID_AP):
    x_data = NEURON_GRID_AP[0,:]
    y_data = NEURON_GRID_AP[1,:]
    theta = jnp.arctan2(y_data, x_data)
    mean_sin = jnp.sum(jnp.sin(theta) * v_t) / jnp.sum(v_t)
    mean_cos = jnp.sum(jnp.cos(theta) * v_t) / jnp.sum(v_t)
    mean_x = jnp.arctan2(mean_sin, mean_cos) # cartesian
    mean_y = jnp.sqrt(mean_sin**2 + mean_cos**2)
    return jnp.array([mean_x,mean_y])

###############

def new_params(params, e): # Modify in place
    VMAPS = params["VMAPS"]
    INIT_S = params["INIT_S"]
    H_S = params["H_S"]
    # H_R = params["H_R"]
    H_P = params["H_P"]
    APERTURE = params["APERTURE"]
    MODULES = params["MODULES"]
    N_A = params["N_A"]
    M = params["M"]
    T = params["TRIAL_LENGTH"]
    N_DOTS = params["N_DOTS"]
    ACTION_SPACE = params["ACTION_SPACE"]
    ki = rnd.split(rnd.PRNGKey(e), num=10)
    params["HS_0"] = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[0],(VMAPS,H_S))
    # params["HR_0"] = jnp.zeros((VMAPS,H_R))
    params["HP_0"] = jnp.sqrt(INIT_P/(H_P))*rnd.normal(ki[1],(VMAPS,H_P))
    params["POS_0"] = rnd.uniform(ki[0],shape=(VMAPS,2),minval=-jnp.pi,maxval=jnp.pi) # rnd.choice(ke[2],jnp.arange(-APERTURE,APERTURE,0.01),(VMAPS,2))
    DOT_0_VEC = gen_dot(ki[3],VMAPS,N_DOTS,ACTION_SPACE)
    params["DOT_0"] = params["POS_0"] + DOT_0_VEC
    params["DOT_VEC"] = (-(2/2)*DOT_0_VEC + gen_dot_vecs(ki[4],VMAPS,MAX_DOT_SPEED,ACTION_SPACE))*(MAX_DOT_SPEED/(2*PLAN_RATIO)) # (pi/2 + pi/2) * DS/(2*P)
    # params["SELECT"] = jnp.eye(N_DOTS)[rnd.choice(ki[4],N_DOTS,(VMAPS,))]
    params["IND"] = rnd.randint(ki[5],(VMAPS,T),minval=0,maxval=M,dtype=jnp.int32)

# @partial(jax.jit,static_argnums=(1,))
def kl_loss(policy,params):
    # *_,PRIOR_STAT,PRIOR_PLAN,_,_ = consts
    len_ = len(policy[0])
    val_ = (1-params["PRIOR_STAT"])/(len_-1)
    vec_prior = jnp.full(len_,1/len_)
    # vec_prior = jnp.full(len_,val_).at[len_//2].set(PRIOR_STAT)
    vec_prior = vec_prior/jnp.sum(vec_prior)
    vec_kl = optax.kl_divergence(jnp.log(policy[0]),vec_prior)
    act_prior = jnp.array([params["PRIOR_PLAN"],1-params["PRIOR_PLAN"]])
    act_kl = optax.kl_divergence(jnp.log(policy[1]),act_prior)
    return vec_kl,act_kl

@jit
def get_policy(args_t,weights_s,params): # hs_t_1,v_t,r_t,rp_t_1,rm_t_1,weights_s,params
    # (hs_t_1,*_,rp_t_1,rm_t_1,v_t,r_t,r_tot,move_flag) = args_t
    (hs_t_1,hv_t_1,*_,act_t,v_t,r_t,rp_t,move_counter,e) = args_t
    r_arr = jnp.array([r_t,rp_t])

    Ws_vt_z = weights_s["Ws_vt_z"]
    Ws_vt_f = weights_s["Ws_vt_f"]
    Ws_vt_h = weights_s["Ws_vt_h"]
    Ws_rt_z = weights_s["Ws_rt_z"]
    Ws_rt_f = weights_s["Ws_rt_f"]
    Ws_rt_h = weights_s["Ws_rt_h"]
    Ws_at_1z = weights_s["Ws_at_1z"]
    Ws_at_1f = weights_s["Ws_at_1f"]
    Ws_at_1h = weights_s["Ws_at_1h"]
    Ws_ht_1z = weights_s["Ws_ht_1z"]
    Ws_ht_1f = weights_s["Ws_ht_1f"]
    Ws_ht_1h = weights_s["Ws_ht_1h"]
    Us_z = weights_s["Us_z"]
    Us_f = weights_s["Us_f"]
    Us_h = weights_s["Us_h"]
    bs_z = weights_s["bs_z"]
    bs_f = weights_s["bs_f"]
    bs_h = weights_s["bs_h"]
    Ws_vec = weights_s["Ws_vec"]
    Ws_act = weights_s["Ws_act"]
    Ws_val = weights_s["Ws_val"]
    z_t = jax.nn.sigmoid(jnp.matmul(Ws_vt_z,v_t) + jnp.matmul(Ws_rt_z,r_arr) + jnp.matmul(Ws_at_1z,act_t) + jnp.matmul(Ws_ht_1z,hv_t_1) + jnp.matmul(Us_z,hs_t_1) + bs_z)
    f_t = jax.nn.sigmoid(jnp.matmul(Ws_vt_f,v_t) + jnp.matmul(Ws_rt_f,r_arr) + jnp.matmul(Ws_at_1f,act_t) + jnp.matmul(Ws_ht_1f,hv_t_1) + jnp.matmul(Us_f,hs_t_1) + bs_f)
    hhat_t = jax.nn.tanh(jnp.matmul(Ws_vt_h,v_t) + jnp.matmul(Ws_rt_h,r_arr) + jnp.matmul(Ws_at_1h,act_t) + jnp.matmul(Ws_ht_1h,hv_t_1) + jnp.matmul(Us_h,jnp.multiply(f_t,hs_t_1)) + bs_h)
    hs_t = jnp.multiply(1-z_t,hs_t_1) + jnp.multiply(z_t,hhat_t) #
    vec_logits = jnp.matmul(Ws_vec,hs_t)
    act_logits = jnp.matmul(Ws_act,hs_t)
    val_t = jnp.squeeze(jnp.matmul(Ws_val,hs_t))
    # jax.debug.print('vec_t={}',vec_t)
    # jax.debug.print('act_t={}',act_t)
    vectors_t = jax.nn.softmax((vec_logits/params["TEMP_VS"]) - jnp.max(vec_logits/params["TEMP_VS"]) + 1e-8) # stable
    actions_t = jax.nn.softmax((act_logits/params["TEMP_AS"]) - jnp.max(act_logits/params["TEMP_AS"]) + 1e-8) # stable
    return (vectors_t,actions_t),val_t,hs_t

def sample_policy(policy,SC,ind):
    vectors,actions = policy
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    keys = rnd.split(rnd.PRNGKey(ind),num=2)
    vec_ind = rnd.choice(key=keys[0],a=jnp.arange(len(vectors)),p=vectors)
    h1vec = H1VEC_ARR[:,vec_ind]
    vec = VEC_ARR[:,vec_ind]
    act_ind = rnd.choice(key=keys[1],a=jnp.arange(len(actions)),p=actions) #plan=0,move1
    act = jnp.eye(len(actions))[act_ind] #plan=[1,0],move=[0,1]
    logit_t = jnp.log(vectors[vec_ind] + 1e-8) + jnp.log(actions[act_ind] + 1e-8)
    return h1vec,vec,vec_ind,act_ind,act,logit_t

def get_vectors(policy,SC,vec_ind,act_ind):
    vectors,actions = policy
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    h1vec = H1VEC_ARR[:,vec_ind]
    vec = VEC_ARR[:,vec_ind]
    logit_t = jnp.log(vectors[vec_ind] + 1e-8) + jnp.log(actions[act_ind] + 1e-8)
    act = jnp.eye(len(actions))[act_ind]
    # rp_t = jnp.int32(act_ind == 0)
    # rm_t = jnp.int32(act_ind == 1)
    return h1vec,vec,act,logit_t

# @jit
# def r_predict(v_t,v_t_1,r_t_1,hr_t_1,r_weights):
#     W_vt_z = r_weights["W_vt_z"]
#     W_vt_1z = r_weights["W_vt_1z"]
#     W_rt_1z = r_weights["W_rt_1z"]
#     U_z = r_weights["U_z"]
#     b_z = r_weights["b_z"]
#     W_vt_f = r_weights["W_vt_f"]
#     W_vt_1f = r_weights["W_vt_1f"]
#     W_rt_1f = r_weights["W_rt_1f"]
#     U_f = r_weights["U_f"]
#     b_f = r_weights["b_f"]
#     W_vt_h = r_weights["W_vt_h"]
#     W_vt_1h = r_weights["W_vt_1h"]
#     W_rt_1h = r_weights["W_rt_1h"]
#     U_h = r_weights["U_h"]
#     b_h = r_weights["b_h"]
#     W_read = r_weights["W_read"]
#     W_sel = r_weights["W_sel"]
#     z_t = jax.nn.sigmoid(jnp.matmul(W_vt_z,v_t) + jnp.matmul(W_vt_1z,v_t_1) + W_rt_1z*r_t_1 + jnp.matmul(U_z,hr_t_1) + b_z)
#     f_t = jax.nn.sigmoid(jnp.matmul(W_vt_f,v_t) + jnp.matmul(W_vt_1f,v_t_1) + W_rt_1f*r_t_1 + jnp.matmul(U_f,hr_t_1) + b_f)
#     hhat_t = jax.nn.tanh(jnp.matmul(W_vt_h,v_t) + jnp.matmul(W_vt_1h,v_t_1) + W_rt_1h*r_t_1 + jnp.matmul(U_h,jnp.multiply(f_t,hr_t_1)) + b_h)
#     hr_t = jnp.multiply(z_t,hr_t_1) + jnp.multiply(1-z_t,hhat_t)
#     r_hat_t = jnp.matmul(W_read,hr_t)
#     return r_hat_t,hr_t

@jit
def r_predict(v_t,dot_t,MODULES,APERTURE,NEURON_GRID_AP,e): # v_t,v_t_1,r_t_1,hr_t_1,r_weights): #PLACEHOLDER
    mean = circular_mean(v_t,NEURON_GRID_AP) # curve_fit(v_t,MODULES,APERTURE,NEURON_GRID_AP)
    r_pred_t = loss_obj(mean,dot_t,params,e) # (unused)
    return r_pred_t #,r_fit_t

@jit # @partial(jax.jit,static_argnums=(6,))
def v_predict(h1vec,v_0,hv_t_1,act_t,weights_v,NONE_PLAN): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    def loop_body(carry,i):
        return RNN_forward(carry)
    def RNN_forward(carry):
        weights_v,h1vec,v_0,act_t,hv_t_1 = carry
        plan_t = act_t[0]
        move_t = act_t[1]
        W_h1_f = weights_v["W_h1_f"]
        W_h1_hh = weights_v["W_h1_hh"]
        W_p_f = weights_v["W_p_f"]
        W_p_hh = weights_v["W_p_hh"]
        W_m_f = weights_v["W_m_f"]
        W_m_hh = weights_v["W_m_hh"]
        W_v_f = weights_v["W_v_f"]
        W_v_hh = weights_v["W_v_hh"]
        U_f = weights_v["U_f"]
        U_hh = weights_v["U_hh"]
        b_f = weights_v["b_f"]
        b_hh = weights_v["b_hh"]
        f_t = jax.nn.sigmoid(W_p_f*plan_t + W_m_f*move_t + jnp.matmul(W_h1_f,h1vec) + jnp.matmul(W_v_f,v_0) + jnp.matmul(U_f,hv_t_1) + b_f)
        hhat_t = jax.nn.tanh(W_p_hh*plan_t + W_m_hh*move_t + jnp.matmul(W_h1_hh,h1vec) + jnp.matmul(W_v_hh,v_0) + jnp.matmul(U_hh,jnp.multiply(f_t,hv_t_1)) + b_hh)
        hv_t = jnp.multiply((1-f_t),hv_t_1) + jnp.multiply(f_t,hhat_t)
        return (weights_v,h1vec,v_0,act_t,hv_t),None
    W_full = weights_v["W_full"]
    W_ap = weights_v["W_ap"]
    carry_0 = (weights_v,h1vec,v_0,act_t,hv_t_1)
    (*_,hv_t),_ = jax.lax.scan(loop_body,carry_0,jnp.zeros(PLAN_ITS))
    v_pred_full = jnp.matmul(W_full,hv_t)
    v_pred_ap = jnp.take(v_pred_full, params["INDICES"])
    return v_pred_ap,v_pred_full,hv_t # dot_hat_t,

def plan(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,act_t,val,weights_v,params,e):#,pos_t_1,dots,vec_t,sel(vec_t,r_t_1,hr_t_1,v_t_1,hv_t_1,weights,params):
    
    dot_t = dot_t_1 + dot_vec ### _dont_ freeze dot
    pos_plan_t = pos_plan_t_1 + vec_t
    pos_t = pos_t_1 ## + vec_t

    v_pred_t,v_full_t,hv_t = v_predict(h1vec_t,v_t_1,hv_t_1,act_t,weights_v,params["NONE_PLAN"])### #(hr_t_1,v_t_1,pos_t_1,weights["v"],params)
    rp_t = r_predict(v_full_t,dot_t,params["MODULES"],params["APERTURE"],params["NEURON_GRID_AP"],e)# (using true pos)
    return rp_t,v_pred_t,hv_t,pos_plan_t,pos_t,dot_t #dot_t# ,r_tp,v_tp #,hv_t # predictions

def move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,act_t,val,weights_v,params,e): # sel, shouldnt be random; should take action dictated by plan...
    
    dot_t = dot_t_1 + dot_vec
    pos_plan_t = pos_t_1 + vec_t
    pos_t = pos_t_1 + vec_t

    v_t_,_,hv_t = v_predict(h1vec_t,v_t_1,hv_t_1,act_t,weights_v,params["NONE_PLAN"])###
    v_t = neuron_act_noise(val,params["THETA_AP"],params["SIGMA_A"],params["SIGMA_N"],dot_t,pos_t)
    r_t = loss_obj(dot_t,pos_t,params,e)###
    return r_t,v_t,hv_t,pos_plan_t,pos_t,dot_t #,rhat_t_,vhat_t_ # hv_t, true

def continue_fnc(carry_args):
    (t,args_t,logit_t,vec_kl,act_kl,theta,h1vec_t,vec_t,act_t,policy_t) = carry_args
    return jax.lax.cond(act_t[0] == 1,plan_fnc,move_fnc,(carry_args))

def env_fnc(carry_args):
    (t,args_t,logit_t,vec_kl,act_kl,theta,h1vec_t,vec_t,act_t,policy_t) = carry_args
    (hs_t,hv_t_1,val_t,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,ind,act_t,v_t_1,r_t_1,rp_t_1,move_counter,e) = args_t
    (SC,weights,weights_s,params) = theta
    (ID_ARR,VEC_ARR,H1VEC_ARR) = SC

    act_t = jnp.array([0,1],dtype=jnp.float32) # refac period of 'move'
    h1vec_t = H1VEC_ARR[:,params["ZERO_VEC_IND"]]
    vec_t = VEC_ARR[:,params["ZERO_VEC_IND"]]
    _r_t_,v_t,hv_t,pos_plan_t,pos_t,dot_t = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,act_t,ind[t],weights_v,params,e) # sel,hr update v,r;dont update h
    dot_t = dot_t_1 + dot_vec
    t += 1 # params["T_MOVE"]
    move_counter += 1

    # r_t = jnp.float32(0) # no grads through no-op
    MDV = (jnp.pi*params["MAX_DOT_SPEED"])/(jnp.sqrt(2)*params["PLAN_RATIO"])
    r_t = _r_t_*(1+0.5*(jnp.linalg.norm(dot_vec)/MDV))
    rp_t = jnp.float32(0)
    lp_t = jnp.float32(0) # no grads

    args_t = (hs_t,hv_t,val_t,pos_plan_t,pos_t,dot_t,dot_vec,ind,act_t,v_t,r_t,rp_t,move_counter,e) #(hs_t,hv_t_1,pos_plan_t_1,pos_t_1, dot_t ,dot_vec,ind,rp_t,rm_t,v_t_1, r_t,r_tot, move_counter) # sel,hr update v,r;dont update h
    return (t,args_t,(rp_t,r_t,pos_plan_t,pos_t,dot_t,logit_t,val_t,jnp.float32(0),jnp.float32(0),vec_kl,act_kl,policy_t,hs_t,hv_t)) #jnp.array([lp_t]),jnp.array([rp_t]),jnp.array([r_t]), val_t ,jnp.array([0]),jnp.array([vec_kl]),jnp.array([act_kl]),pos_plan_t_1,pos_t_1,dot_t]))

def plan_fnc(carry_args):
    (t,args_t,logit_t,vec_kl,act_kl,theta,h1vec_t,vec_t,act_t,policy_t) = carry_args # (add rp,rm)
    (hs_t,hv_t_1,val_t,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,ind,act_t,v_t_1,r_t_1,rp_t_1,move_counter,e) = args_t # sel,hr
    (SC,weights_v,weights_s,params) = theta

    rp_t,v_t,hv_t,pos_plan_t,pos_t,dot_t = plan(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,act_t,ind[t],weights_v,params,e)
    MDV = (jnp.pi*params["MAX_DOT_SPEED"])/(jnp.sqrt(2)*params["PLAN_RATIO"])
    rp_t = rp_t*(1+0.5*(jnp.linalg.norm(dot_vec)/MDV))
    r_t = jnp.float32(0) # rp_t = rp_t ### # r_t
    t += 1 # params["T_PLAN"]
    move_counter = 0
    # hs_old_t = hs_t

    args_t = (hs_t,hv_t,val_t,pos_plan_t_1,pos_t_1,dot_t,dot_vec,ind,act_t,v_t,r_t,rp_t,move_counter,e) # sel,hr update v,r;dont update h
    return (t,args_t,(rp_t,r_t,pos_plan_t,pos_t,dot_t,logit_t,val_t,jnp.float32(1),jnp.float32(1),vec_kl,act_kl,policy_t,hs_t,hv_t))

def move_fnc(carry_args):
    (t,args_t,logit_t,vec_kl,act_kl,theta,h1vec_t,vec_t,act_t,policy_t) = carry_args
    (hs_t,hv_t_1,val_t,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,ind,act_t,v_t_1,r_t_1,rp_t_1,move_counter,e) = args_t # ,sel,hr
    (SC,weights_v,weights_s,params) = theta

    r_t,v_t,hv_t,pos_plan_t,pos_t,dot_t = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,act_t,ind[t],weights_v,params,e) # ,sel,hr
    MDV = (jnp.pi*params["MAX_DOT_SPEED"])/(jnp.sqrt(2)*params["PLAN_RATIO"])
    r_t = r_t*(1+0.5*(jnp.linalg.norm(dot_vec)/MDV))
    rp_t = jnp.float32(0) ###
    t += 1 # params["T_MOVE"]
    move_counter = 1
    # hs_old_t = hs_t

    args_t = (hs_t,hv_t,val_t,pos_plan_t,pos_t,dot_t,dot_vec,ind,act_t,v_t,r_t,rp_t,move_counter,e) # ,sel,hr update v,r,hr,pos
    return (t,args_t,(rp_t,r_t,pos_plan_t,pos_t,dot_t,logit_t,val_t,jnp.float32(0),jnp.float32(1),vec_kl,act_kl,policy_t,hs_t,hv_t))

@jit
def scan_body(carry_t_1,x):
    t,args_t_1,theta = carry_t_1
    (hs_t_1,hv_t_1,val_t_1,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,ind,act_t_1,v_t_1,r_t_1,rp_t_1,move_counter,e) = args_t_1 # sel,hr
    (SC,weights_v,weights_s,params) = theta    

    policy_t,val_t,hs_t = get_policy(args_t_1,weights_s,params) # hs_t_1,v_t_1,r_t_1,rp_t_1,rm_t_1,weights_s,params
    vec_kl,act_kl = kl_loss(policy_t,params) #-jnp.dot(policy_t[1],jnp.log(policy_t[1]))
    h1vec_t,vec_t,vec_ind,act_ind,act_t,logit_t = sample_policy(policy_t,SC,ind[t])

    args_t = (hs_t,hv_t_1,val_t,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,ind,act_t,v_t_1,r_t_1,rp_t_1,move_counter,e) # sel,hr update rp/rm
    carry_args = (t,args_t,logit_t,vec_kl,act_kl,theta,h1vec_t,vec_t,act_t,policy_t) # (lp_arr,r_arr,sample_arr) assemble carry with sampled vecs and updated args
    t,args_t,arrs_t = jax.lax.cond((move_counter > 0)&(move_counter < params["PLAN_RATIO"]),env_fnc,continue_fnc,(carry_args))
    return (t,args_t,theta),(vec_ind,act_ind,arrs_t)

# @jit
# def scan_body_old(carry_t_1,x):
#     t,args_t_1,theta = carry_t_1
#     (hs_t_1,hv_t_1,val_t_1,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,ind,act_t_1,v_t_1,r_t_1,rp_t_1,move_counter,e) = args_t_1 # sel,hr
#     (SC,weights_v,weights_s,params) = theta

#     policy_t,val_t,hs_t = get_policy(args_t_1,weights_s,params) # hs_t_1,v_t_1,r_t_1,rp_t_1,rm_t_1,weights_s,params
#     vec_kl,act_kl = kl_loss(policy_t,params) #-jnp.dot(policy_t[1],jnp.log(policy_t[1]))
#     h1vec_t,vec_t,vec_ind,act_ind,act_t,logit_t = sample_policy(policy_t,SC,ind[t])

#     args_t = (hs_t,hv_t_1,val_t,pos_plan_t_1,pos_t_1,dot_t_1,dot_vec,ind,act_t,v_t_1,r_t_1,rp_t_1,move_counter,e) # sel,hr update rp/rm
#     carry_args = (t,args_t,logit_t,vec_kl,act_kl,theta,h1vec_t,vec_t,act_t) # (lp_arr,r_arr,sample_arr) assemble carry with sampled vecs and updated args
#     t,args_t,arrs_t = jax.lax.cond((move_counter > 0)&(move_counter < params["PLAN_RATIO"]),env_fnc,continue_fnc,(carry_args))
#     # t,args_t,arrs_t = jax.lax.cond(rp_t == 1,plan_fnc,move_fnc,(carry_args))###DEBUG;CHANGE
#     return (t,args_t,theta),(vec_ind,act_ind,arrs_t)

def body_fnc(SC,hs_0,hv_0,pos_0,dot_0,dot_vec,ind,weights_v,weights_s,params,e): # # ,sel,hr PLAN_ITS
    v_0 = neuron_act_noise(ind[-1],params["THETA_AP"],params["SIGMA_A"],params["SIGMA_N"],dot_0,pos_0)
    r_0 = loss_obj(dot_0,pos_0,params,e)### ,sel

    val_0 = jnp.float32(0)
    t = 0
    act_0 = jnp.array([0,1],dtype=jnp.float32) # refac period of 'move'
    rp_0 = 0
    # rm_0 = 1
    # r_tot_0 = 0
    pos_plan_0 = pos_0
    move_counter = 0
    # consts = params["INIT_LENGTH"],params["TEST_LENGTH"],params["NONE_PLAN"],params["MODULES"],params["APERTURE"],params["SIGMA_R"],params["SIGMA_A"],params["SIGMA_N"],params["COLORS"],params["THETA_AP"],params["NEURON_GRID_AP"],params["PRIOR_STAT"],params["PRIOR_PLAN"],params["C_MOVE"],params["C_PLAN"]
    # args_0 = (hs_0,hv_0,pos_plan_0,pos_0,dot_0,dot_vec,ind,rp_0,rm_0,v_0,r_0,r_tot_0,move_flag) # sel,hr r_tot_1 = 0
    args_0 = (hs_0,hv_0,val_0,pos_plan_0,pos_0,dot_0,dot_vec,ind,act_0,v_0,r_0,rp_0,move_counter,e)
    theta = (SC,weights_v,weights_s,params)
    # _,args_init,rpos_init_arr = init_scan((t,args_0,theta))
    # r_init_arr,pos_init_arr = rpos_init_arr[:,0],rpos_init_arr[:,1:]
    (_,args_final,_),arrs_stack = jax.lax.scan(scan_body,(t,args_0,theta),None,params["TEST_LENGTH"])# dynamic_scan((t,args_0,theta)) # arrs_0
    vec_ind_arr,act_ind_arr,(rp_arr,r_arr,pos_plan_arr,pos_arr,dot_arr,lp_arr,val_arr,sample_arr,mask_arr,vec_kl_arr,act_kl_arr,policy_arr,hs_arr,hv_arr) = arrs_stack
    # lp_arr,r_arr,rt_arr,val_arr,sample_arr,vec_kl_arr,act_kl_arr = (arrs_stack[:,i] for i in range(7)) #,t_arr
    # pos_plan_arr = arrs_stack[:,7:9]
    # pos_arr = arrs_stack[:,9:11]
    # dot_arr = arrs_stack[:,11:13]
    pos_plan_arr,pos_arr = jnp.concatenate([jnp.array([pos_0]),pos_plan_arr]),jnp.concatenate([jnp.array([pos_0]),pos_arr])##
    dot_arr = jnp.concatenate([jnp.array([dot_0]),dot_arr])##
    # sample_arr = jnp.concatenate([jnp.array([0]),sample_arr])
    return lp_arr,r_arr,rp_arr,val_arr,sample_arr,vec_ind_arr,vec_kl_arr,act_ind_arr,act_kl_arr,pos_plan_arr,pos_arr,dot_arr,mask_arr,policy_arr,hs_arr,hv_arr #,t_arr #


def pg_obj(SC,hs_0,hp_0,pos_0,dot_0,dot_vec,ind,weights,weights_s,params,e): # ,sel
    body_fnc_vmap = jax.vmap(body_fnc,in_axes=(None,0,0,0,0,0,0,None,None,None,None),out_axes=(1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0))
    lp_arr,r_arr,rp_arr,val_arr,sample_arr,vec_ind_arr,vec_kl_arr,act_ind_arr,act_kl_arr,pos_plan_arr,pos_arr,dot_arr,mask_arr,policy_arr,hs_arr,hv_arr = body_fnc_vmap(SC,hs_0,hp_0,pos_0,dot_0,dot_vec,ind,weights["v"],weights_s,params,e) # ,sel ,params["PLAN_ITS"] [TEST_LENGTH,VMAPS] arrs_(lp,r),aux
    r_to_go = jnp.cumsum(r_arr.T[:,::-1],axis=1)[:,::-1] # reward_to_go(r_arr.T)
    adv_arr = r_to_go - val_arr.T
    adv_norm = (adv_arr-jnp.mean(adv_arr,axis=None))/jnp.std(adv_arr,axis=None) ###
    adv_norm_masked = jnp.multiply(adv_norm,mask_arr.T)
    actor_loss_arr = -(jnp.multiply(lp_arr.T,adv_norm_masked)) # negative for adam
    actor_loss = jnp.mean(actor_loss_arr)# jnp.mean(jnp.sum(actor_loss_arr,axis=1))
    sem_actor = jnp.std(actor_loss_arr)/params["VMAPS"]**0.5
    vec_kl_masked = jnp.multiply(vec_kl_arr,mask_arr)
    vec_kl_loss = jnp.mean(vec_kl_masked,axis=None) # jnp.mean(vec_kl_arr,axis=None)
    sem_vec_kl = jnp.std(vec_kl_masked,axis=None)/params["VMAPS"]**0.5
    act_kl_masked = jnp.multiply(act_kl_arr,mask_arr)
    act_kl_loss = jnp.mean(act_kl_masked,axis=None) # jnp.mean(act_kl_arr,axis=None)
    sem_act_kl = jnp.std(act_kl_masked,axis=None)/params["VMAPS"]**0.5

    actor_losses = actor_loss + params["LAMBDA_VEC_KL"]*vec_kl_loss + params["LAMBDA_ACT_KL"]*act_kl_loss
    std_actor_loss = (params["VMAPS"]*sem_actor**2+(params["VMAPS"]*params["LAMBDA_VEC_KL"]**2)*(sem_vec_kl**2))**0.5
    sem_loss = std_actor_loss/params["VMAPS"]**0.5

    critic_loss = jnp.mean(jnp.square(adv_arr),axis=None)
    # critic_loss_masked = jnp.multiply(jnp.square(adv_arr),mask_arr.T)
    # critic_loss = jnp.mean(critic_loss_masked) # params["LAMBDA_CRITIC"]*
    sem_critic = jnp.std(jnp.square(adv_arr),axis=None)/params["VMAPS"]**0.5
    # std_critic = jnp.std(critic_loss_masked)

    tot_loss = actor_losses + critic_loss
    
    r_tot = jnp.mean(r_to_go[:,0])
    sem_r = jnp.std(r_to_go[:,0])/params["VMAPS"]**0.5
    plan_rate,sem_plan_rate = get_plan_rate(sample_arr,mask_arr)

    losses = (tot_loss,actor_loss,critic_loss,vec_kl_loss,act_kl_loss,r_tot,plan_rate)
    stds = (sem_loss,sem_actor,sem_critic,sem_act_kl,sem_vec_kl,sem_r,sem_plan_rate)
    other = (r_arr.T,rp_arr.T,sample_arr.T,mask_arr.T,pos_plan_arr.transpose([1,0,2]),pos_arr.transpose([1,0,2]),dot_arr.transpose([1,0,2]),policy_arr,hs_arr,hv_arr,vec_ind_arr.T,act_ind_arr.T) # ,t_arr.T
    return (actor_losses,critic_loss),(losses,stds,other)
    # tot_loss,(actor_loss,std_actor,critic_loss,std_critic,avg_vec_kl,std_vec_kl,avg_act_kl,std_act_kl,r_std,l_sem,plan_rate,avg_tot_r,kl_loss,r_init_arr.T,r_arr.T,rt_arr.T,sample_arr.T,pos_init_arr.transpose([1,0,2]),pos_arr.transpose([1,0,2])) # ,t_arr.T

def global_norm(grad):
    squared_norms = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), grad)
    total_norm = jnp.sqrt(jax.tree_util.tree_reduce(jnp.add, squared_norms))
    return total_norm

def get_entropy(policy): # shape [1000,61,81]
    mean_entropy = jnp.mean(jnp.sum(-jnp.multiply(policy,jnp.log(policy)),axis=2),axis=None)
    sem_entropy = jnp.std(jnp.sum(-jnp.multiply(policy,jnp.log(policy)),axis=2),axis=None)/params["VMAPS"]**0.5
    return mean_entropy,sem_entropy

# def full_loop(SC,weights,params):
def full_loop(SC,weights,params,actor_opt_state,critic_opt_state,weights_s):
    loss_arr,sem_loss_arr,actor_loss_arr,sem_actor_arr,critic_loss_arr,sem_critic_arr,vec_kl_arr,sem_vec_kl_arr,act_kl_arr,sem_act_kl_arr,r_tot_arr,sem_r_arr,plan_rate_arr,sem_plan_rate_arr,policy_entropy_arr,sem_policy_entropy_arr = (jnp.zeros((params["TOT_EPOCHS"],)) for _ in range(16)) #jnp.zeros((params["TOT_EPOCHS"]))
    E = params["TOT_EPOCHS"]
    weights_s = weights["s"]

    actor_optimizer = optax.chain(optax.clip_by_global_norm(params["ACTOR_GC"]), optax.adam(learning_rate=params["ACTOR_LR"],eps=1e-7))# optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"]),
    critic_optimizer = optax.chain(
    optax.clip_by_global_norm(params["CRITIC_GC"]),optax.adam(learning_rate=params["CRITIC_LR"],eps=1e-7))# 
    # optax.clip_by_global_norm(params["CRITIC_GC"]),optax.adamw(learning_rate=params["CRITIC_LR"],weight_decay=params["CRITIC_WD"],eps=1e-7))
    
    # actor_opt_state = actor_optimizer.init(weights_s) ##
    # critic_opt_state = critic_optimizer.init(weights_s) ##
    
    # other_buffer = []
    # r_tot_buffer = []
    max_r_tot = jnp.float32(0)

    max_rx_plan_rate = jnp.float32(0)
    rx_plan_rate = jnp.float32(0)
    plan_info = {}
    for e in range(E):
        new_params(params,e)
        hs_0 = params["HS_0"]
        hp_0 = params["HP_0"]
        pos_0 = params["POS_0"]
        dot_0 = params["DOT_0"]
        dot_vec = params["DOT_VEC"]
        ind = params["IND"]
        # print('hs_0[:1]=',hs_0[:1],'ind[:1]=',ind[:1])

        for _ in range(params["CRITIC_UPDATES"]):
            # Recompute losses using updated weights
            isolated_pg_obj = lambda weights_s: pg_obj(SC, hs_0, hp_0, pos_0, dot_0, dot_vec, ind, weights, weights_s, params, e)
            (actor_losses, critic_loss), pg_obj_vjp, (losses,stds,other) = jax.vjp(isolated_pg_obj, weights_s, has_aux=True)
            critic_grad, = pg_obj_vjp((0.0, 1.0))  # Pull back gradient for critic loss

            critic_global_norm = global_norm(critic_grad)
            # print(f"Critic Gradient Norm: {critic_global_norm}")

            critic_update, critic_opt_state = critic_optimizer.update(critic_grad, critic_opt_state, weights_s)
            weights_s = optax.apply_updates(weights_s, critic_update)

        # compute actor gradient
        actor_grad, = pg_obj_vjp((1.0, 0.0))
        (tot_loss,actor_loss,critic_loss,vec_kl_loss,act_kl_loss,r_tot,plan_rate) = losses
        print('actor_loss=',actor_loss,'vec_kl_loss=',vec_kl_loss,'act_kl_loss=',act_kl_loss)

        actor_global_norm = global_norm(actor_grad)
        # print(f"Actor Gradient Norm: {actor_global_norm}")

        actor_update, actor_opt_state = actor_optimizer.update(actor_grad, actor_opt_state)
        weights_s = optax.apply_updates(weights_s, actor_update)

        # leaves,_ = jax.tree_util.tree_flatten(grads)
        # jax.debug.print("all_grads={}",grads)
        # jax.debug.print("max_grad={}",jnp.max(jnp.concatenate(jnp.array(leaves)),axis=None))
        (tot_loss,actor_loss,critic_loss,vec_kl_loss,act_kl_loss,r_tot,plan_rate) = losses
        (sem_loss,sem_actor,sem_critic,sem_act_kl,sem_vec_kl,sem_r,sem_plan_rate) = stds
        (r_arr,rp_arr,sample_arr,mask_arr,pos_plan_arr,pos_arr,dot_arr,policy_arr,hs_arr,hv_arr,vec_ind_arr,act_ind_arr) = other
        # print('vec_ind_arr=',vec_ind_arr[0],'act_ind_arr=',act_ind_arr[0])

        if r_tot > max_r_tot:
            max_r_tot = r_tot
            selected_other = (other,copy.deepcopy(params))

        if (r_tot*plan_rate) > max_rx_plan_rate:
            max_rx_plan_rate = r_tot*plan_rate
            rx_plan_rate = plan_rate
            plan_info = {
                'other_': (other,copy.deepcopy(params)),
                'actor_opt_state': actor_opt_state,
                'critic_opt_state': critic_opt_state,
                'weights_s': weights_s,
                'rx_plan_rate': plan_rate,
                'max_r_tot' : max_r_tot,
            }

        mean_policy_entropy,sem_policy_entropy = get_entropy(policy_arr[0])
        policy_entropy_arr = policy_entropy_arr.at[e].set(mean_policy_entropy)
        std_policy_entropy_arr = sem_policy_entropy_arr.at[e].set(sem_policy_entropy)
        loss_arr = loss_arr.at[e].set(tot_loss)
        sem_loss_arr = sem_loss_arr.at[e].set(sem_loss)
        actor_loss_arr = actor_loss_arr.at[e].set(actor_loss)
        sem_actor_arr = sem_actor_arr.at[e].set(sem_actor)
        critic_loss_arr = critic_loss_arr.at[e].set(critic_loss)
        sem_critic_arr = sem_critic_arr.at[e].set(sem_critic)
        vec_kl_arr = vec_kl_arr.at[e].set(vec_kl_loss)
        sem_vec_kl_arr = sem_vec_kl_arr.at[e].set(sem_vec_kl)
        act_kl_arr = act_kl_arr.at[e].set(act_kl_loss)
        sem_act_kl_arr = sem_act_kl_arr.at[e].set(sem_act_kl)
        r_tot_arr = r_tot_arr.at[e].set(r_tot)
        sem_r_arr = sem_r_arr.at[e].set(sem_r)
        plan_rate_arr = plan_rate_arr.at[e].set(plan_rate)
        sem_plan_rate_arr = sem_plan_rate_arr.at[e].set(sem_plan_rate)
        print("e=",e,"r=",r_tot,"r_sem=",sem_r) #,"loss=",tot_loss,"sem_loss=",sem_loss)
        print("actor_loss=",actor_loss,"sem_actor=",sem_actor,"critic_loss=",critic_loss,"sem_critic=",sem_critic)
        print("vec_kl=",vec_kl_loss,"sem_vec=",sem_vec_kl,"act_kl=",act_kl_loss,"sem_act=",sem_act_kl,"plan=",plan_rate,'entropy=',mean_policy_entropy,'sem_entropy=',sem_policy_entropy,'\n')
        print('rx_plan_rate=',rx_plan_rate,'max_r_tot=',max_r_tot)
        if e == E-1:
            pass
        ### loss.block_until_ready()
        ### jax.profiler.save_device_memory_profile(f"memory{e}.prof")
        if e>0 and (e % 50) == 0:
            print("*clearing cache*")
            jax.clear_caches()

    # max_index = jnp.argmax(jnp.array(r_tot_buffer))
    # selected_other = other_buffer[max_index]

    loss_arrs = (loss_arr,actor_loss_arr,critic_loss_arr,vec_kl_arr,act_kl_arr,r_tot_arr,plan_rate_arr,policy_entropy_arr)
    sem_arrs = (sem_loss_arr,sem_actor_arr,sem_critic_arr,sem_act_kl_arr,sem_vec_kl_arr,sem_r_arr,sem_plan_rate_arr,sem_policy_entropy_arr)
    return loss_arrs,sem_arrs,selected_other,actor_opt_state,critic_opt_state,weights_s,plan_info #r_arr,pos_arr,sample_arr,dots_arr

# hyperparams ###
TOT_EPOCHS = 3000 ## 1000
CRITIC_UPDATES = 5 # 8 5
# LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 1000 ## 2000,500,1100,1000,800,500
ACTOR_LR = 0.0003 # 0.0002 # 0.001 # 0.0005
CRITIC_LR = 0.0008 #  # 0.0005 # 0.0001 # 0.0005   0.001,0.0008,0.0005,0.001,0.000001,0.0001
CRITIC_WD = 0 # 0.00001 # 0.0001
PLAN_RATIO = 5 ## 5 2 10
ACTOR_GC = 0.4 ##0.6 0.3, 0.5 1.0
CRITIC_GC = 0.9 ## 0.7
INIT_S = 2
INIT_P = 2 # 0.5
# INIT_R = 3 # 5,2
H_S = 100
H_P = 300 # 400,500,300
# H_R = 100
PLAN_ITS = 10
NONE_PLAN = jnp.zeros((PLAN_ITS,)) #[None] * PLAN_ITS
INIT_LENGTH = 0
TRIAL_LENGTH = 60 ## 50 90 120 100
TEST_LENGTH = TRIAL_LENGTH - INIT_LENGTH
LAMBDA_VEC_KL = 0.01 # 0.5, 0.05, 0.1, 0.5
LAMBDA_ACT_KL = 0.01 # 0.01,0.1,1
TEMP_VS = 1 # 0.1,0.05
TEMP_AS = 0.1 # 1 # 0.1,0.05

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
C_MOVE = 0.0 #0.30 #0.28 0.30
C_PLAN = 0.0 # 0.05 # 0.0
PRIOR_STAT = 0.2 # 1 # 0.2
PRIOR_PLAN_FRAC = 0.2
PRIOR_PLAN = PRIOR_PLAN_FRAC #(PRIOR_PLAN_FRAC*PLAN_RATIO)/(1+PRIOR_PLAN_FRAC*PLAN_RATIO) # 0.2
MODULES = 9 # 10,8
M = MODULES**2
APERTURE = (1/2)*jnp.pi # (3/5)*jnp.pi # (jnp.sqrt(2)/2)*jnp.pi ### unconstrained
ACTION_FRAC = 1 # CHANGED FROM 1/2... unconstrained
ACTION_SPACE = ACTION_FRAC*APERTURE # 'AGENT_SPEED'
PLAN_FRAC_REL = 1 # 3/2
PLAN_SPACE = PLAN_FRAC_REL*ACTION_SPACE
MAX_DOT_SPEED_REL_FRAC = 1.5 ## 1.2, 3/2 # 5/4
MAX_DOT_SPEED = ACTION_SPACE*MAX_DOT_SPEED_REL_FRAC
NEURONS_FULL = 12
N_F = (NEURONS_FULL**2)
NEURONS_AP = jnp.int32(jnp.floor(NEURONS_FULL*(APERTURE/jnp.pi))) # 6 # 10
N_A = (NEURONS_AP**2)
THETA_FULL = jnp.linspace(-(APERTURE-APERTURE/NEURONS_FULL),(APERTURE-APERTURE/NEURONS_FULL),NEURONS_FULL)
THETA_AP = THETA_FULL[NEURONS_FULL//2 - NEURONS_AP//2 : NEURONS_FULL//2 + NEURONS_AP//2]

SIGMA_A = 0.3 # CHANGED FROM 0.5
SIGMA_R0 = 0.3 # 0.5,0.3
SIGMA_RINF = SIGMA_R0 # SIGMA_A # 1.2,1.5,1
TAU = 1000 ## heuristic
SIGMA_S = 0.1
SIGMA_N = 0.2 # 0.1,0.05
SIGMOID_MEAN = 0.5
COLORS = jnp.array([255]) #,[0,255,0],[0,0,255]]) # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = COLORS.shape[0]
HS_0 = None
HP_0 = None #jnp.sqrt(INIT/(H_S))*rnd.normal(ke[4],(EPOCHS,VMAPS,H_S)) # [E,V,H_S]
# HR_0 = None #jnp.sqrt(INIT/(H_S))*rnd.normal(ke[4],(EPOCHS,VMAPS,H_S)) # [E,V,H_S]
POS_0 = None #rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
DOTS = None #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
# SELECT = None #jnp.eye(N_DOTS)[rnd.choice(ke[1],N_DOTS,(EPOCHS,VMAPS))]
IND = None
# ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True)
# VEC_ARR = gen_vectors(MODULES,APERTURE)
# H1VEC_ARR = jnp.diag(jnp.ones(M))[:,ID_ARR]
SC,PRIOR_VEC,ZERO_VEC_IND = gen_sc(ke,MODULES,ACTION_SPACE,PLAN_SPACE) # (ID_ARR,VEC_ARR,H1VEC_ARR)
INDICES = get_inner_activation_indices(NEURONS_FULL,NEURONS_AP)
NEURON_GRID_AP = get_inner_activation_coords(NEURONS_AP,APERTURE)

# INITIALIZATION
ki = rnd.split(rnd.PRNGKey(1),num=20)
Ws_vt_z0 = jnp.sqrt(INIT_S/(H_S+N_A))*rnd.normal(ki[0],(H_S,N_A))
Ws_vt_f0 = jnp.sqrt(INIT_S/(H_S+N_A))*rnd.normal(ki[1],(H_S,N_A))
Ws_vt_h0 = jnp.sqrt(INIT_S/(H_S+N_A))*rnd.normal(ki[2],(H_S,N_A))
Ws_rt_z0 = jnp.sqrt(INIT_S/(H_S+2))*rnd.normal(ki[3],(H_S,2)) # (H_S,)
Ws_rt_f0 = jnp.sqrt(INIT_S/(H_S+2))*rnd.normal(ki[4],(H_S,2))
Ws_rt_h0 = jnp.sqrt(INIT_S/(H_S+2))*rnd.normal(ki[5],(H_S,2))
Ws_at_1z0 = jnp.sqrt(INIT_S/(H_S+2))*rnd.normal(ki[6],(H_S,2))
Ws_at_1f0 = jnp.sqrt(INIT_S/(H_S+2))*rnd.normal(ki[7],(H_S,2))
Ws_at_1h0 = jnp.sqrt(INIT_S/(H_S+2))*rnd.normal(ki[8],(H_S,2))
Ws_ht_1z0 = jnp.sqrt(INIT_S/(H_S+2))*rnd.normal(ki[9],(H_S,H_P))
Ws_ht_1f0 = jnp.sqrt(INIT_S/(H_S+2))*rnd.normal(ki[10],(H_S,H_P))
Ws_ht_1h0 = jnp.sqrt(INIT_S/(H_S+2))*rnd.normal(ki[11],(H_S,H_P))
Us_z0,_ = jnp.linalg.qr(jnp.sqrt(INIT_S/(H_S+H_S))*rnd.normal(ki[12],(H_S,H_S)))
Us_f0,_ = jnp.linalg.qr(jnp.sqrt(INIT_S/(H_S+H_S))*rnd.normal(ki[13],(H_S,H_S)))
Us_h0,_ = jnp.linalg.qr(jnp.sqrt(INIT_S/(H_S+H_S))*rnd.normal(ki[14],(H_S,H_S)))
bs_z0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[15],(H_S,))
bs_f0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[16],(H_S,))
bs_h0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[17],(H_S,))
Ws_vec0 = jnp.sqrt(INIT_S/(M+H_S))*rnd.normal(ki[18],(M,H_S))
Ws_act0 = jnp.sqrt(INIT_S/(2+H_S))*rnd.normal(ki[19],(2,H_S))
Ws_val0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[19],(1,H_S))

params = {
    "TOT_EPOCHS" : TOT_EPOCHS,
    "MODULES" : MODULES,
    "NEURONS_AP" : NEURONS_AP,
    "NEURONS_FULL" : NEURONS_FULL,
    "APERTURE" : APERTURE,
    "ACTOR_LR" : ACTOR_LR,
    "CRITIC_LR" : CRITIC_LR,
    "CRITIC_WD" : CRITIC_WD,
    "ACTOR_GC" : ACTOR_GC,
    "CRITIC_GC" : CRITIC_GC,
    "H_S" : H_S,
    "H_P" : H_P,
    # "H_R" : H_R,
    "N_A" : N_A,
    "N_F" : N_F,
    "M" : M,
    "INIT_P" : INIT_P,
    "INIT_S" : INIT_S,
    # "INIT_R" : INIT_R,
    "VMAPS" : VMAPS,
    "N_DOTS" : N_DOTS,
    "DOTS" : DOTS,
    "THETA_AP" : THETA_AP,
    "THETA_FULL" : THETA_FULL,
    "COLORS" : COLORS,
    # "SAMPLES" : SAMPLES,
    "POS_0" : POS_0,
    "HP_0" : HP_0,
    # "HR_0" : HR_0,
    "IND" : IND,
    "INIT_LENGTH" : INIT_LENGTH,
    "TRIAL_LENGTH" : TRIAL_LENGTH,
    "TEST_LENGTH" : TEST_LENGTH,
    "PLAN_ITS" : PLAN_ITS,
    "NONE_PLAN" : NONE_PLAN,
    "SIGMOID_MEAN" : SIGMOID_MEAN,
    "SIGMA_R0" : SIGMA_R0,
    "SIGMA_RINF" : SIGMA_RINF,
    "TAU" : TAU,
    "SIGMA_A" : SIGMA_A,
    "SIGMA_S" : SIGMA_S,
    "SIGMA_N" : SIGMA_N,
    "TEMP_VS" : TEMP_VS,
    "TEMP_AS" : TEMP_AS,
    "LAMBDA_VEC_KL" : LAMBDA_VEC_KL,
    "LAMBDA_ACT_KL" : LAMBDA_ACT_KL,
    "C_MOVE" : C_MOVE,
    "C_PLAN" : C_PLAN,
    "PRIOR_STAT" : PRIOR_STAT,
    "PRIOR_PLAN" : PRIOR_PLAN,
    "INDICES" : INDICES,
    "NEURON_GRID_AP" : NEURON_GRID_AP,
    "ACTION_SPACE" : ACTION_SPACE,
    "CRITIC_UPDATES" : CRITIC_UPDATES,
    "PLAN_RATIO" : PLAN_RATIO,
    "PRIOR_VEC" : PRIOR_VEC,
    "ZERO_VEC_IND" : ZERO_VEC_IND,
    "MAX_DOT_SPEED" : MAX_DOT_SPEED,
    }

weights = {
    "s" : { # policy weights
    "Ws_vt_z" : Ws_vt_z0,
    "Ws_vt_f" : Ws_vt_f0,
    "Ws_vt_h" : Ws_vt_h0,
    "Ws_rt_z" : Ws_rt_z0,
    "Ws_rt_f" : Ws_rt_f0,
    "Ws_rt_h" : Ws_rt_h0,
    "Ws_at_1z" : Ws_at_1z0,
    "Ws_at_1f" : Ws_at_1f0,
    "Ws_at_1h" : Ws_at_1h0,
    "Ws_ht_1z" : Ws_ht_1z0,
    "Ws_ht_1f" : Ws_ht_1f0,
    "Ws_ht_1h" : Ws_ht_1h0,
    "Us_z" : Us_z0,
    "Us_f" : Us_f0,
    "Us_h" : Us_h0,
    "bs_z" : bs_z0,
    "bs_f" : bs_f0,
    "bs_h" : bs_h0,
    "Ws_vec" : Ws_vec0,
    "Ws_act" : Ws_act0,
    "Ws_val" : Ws_val0,
    },
    "v" : { # weights_v
    # LOAD IN
    },
    # "r" : { # r_weights
    # # LOAD IN
    # }
}

###
# MDS=1.5 (rnd dot gen): 2 = x, 5 = 2210 021511, 10 = 2210 021849
# sc_project/test_data/forward_new_v10_81M_144N_22_10-021511.pkl, sc_project/test_data/forward_new_v10_81M_144N_22_10-021849.pkl
# MDS=1.5: 2 = 1410 1628, 5 = 1410 1630, 10 = 1410 1632
# sc_project/test_data/forward_new_v10_81M_144N_14_10-162845.pkl, sc_project/test_data/forward_new_v10_81M_144N_14_10-163001.pkl, sc_project/test_data/forward_new_v10_81M_144N_14_10-163228.pkl
# MDS=1.2: 2 = 1110 2032, 5 = 1210 1324, 10 = 1210 0233
(_),(*_,weights_v) = load_('/sc_project/test_data/forward_new_v10_81M_144N_14_10-163001.pkl') #/sc_project/test_data/forward_new_v10_81M_144N_22_10-021511.pkl') # '/sc_project/test_data/forward_new_v10_81M_144N_22_10-021849.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_12_10-023341.pkl') #/sc_project/test_data/forward_new_v10_81M_144N_12_10-132458.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_11_10-203208.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_12_10-023341.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_11_10-234644.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_11_10-234704.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_11_10-234644.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_11_10-203208.pkl') #'/sc_project/test_data/forward_new_v11_81M_144N_11_10-210349.pkl') #'/sc_project/test_data/forward_new_v8_81M_144N_06_09-211442.pkl') #
weights['v'] = weights_v
(actor_opt_state,critic_opt_state,weights_s) = load_('/sc_project/pkl_sc/outer_loop_pg_new_v4f_04_11-181832.pkl') #'/sc_project/pkl_sc/outer_loop_pg_new_v4f_01_11-124405.pkl') #/sc_project/pkl_sc/outer_loop_pg_new_v4f_27_10-173751.pkl') #'/sc_project/pkl_sc/outer_loop_pg_new_v4f_15_10-112328.pkl') #'/sc_project/pkl_sc/outer_loop_pg_new_v4f__13_10-084115.pkl') #/sc_project/pkl_sc/outer_loop_pg_new_v4f__12_10-175620.pkl') #'/sc_project/test_data/outer_loop_pg_new_v4f_12_10-173828.pkl') #'/sc_project/pkl_sc/outer_loop_pg_new_v6__21_09-125738.pkl') #'/sc_project/pkl_sc/outer_loop_pg_new_v1_ppo__13_09-235540.pkl') #'/sc_project/pkl_sc/outer_loop_pg_new_v3_c__13_09-174514.pkl', '/sc_project/test_data/outer_loop_pg_new_v3_c__13_09-174514.pkl')
weights['s'] = weights_s
### run from best planning rate:
selected_other,plan_info = load_('/sc_project/large_outputs/outer_loop_pg_new_v4f_04_11-181832.pkl') #'/sc_project/large_outputs/outer_loop_pg_new_v4f_28_10-210746.pkl')
(r_arr,*_,vec_ind_arr,act_ind_arr),_ = plan_info["other_"] # dont load params
weights['s'] = plan_info["weights_s"]
actor_opt_state = plan_info["actor_opt_state"]
critic_opt_state = plan_info["critic_opt_state"]
# params["MAX_DOT_SPEED"] = MAX_DOT_SPEED
# *_,r_weights = load_('') #
# weights['r'] = r_weights
###
# load weights, opt_state init; weights['s'] = weights_s
startTime = datetime.now()
# losses,stds,other,actor_opt_state,critic_opt_state,weights_s = full_loop(SC,weights,params) # full_loop(SC,weights,params) (loss_arr,actor_loss_arr,critic_loss_arr,kl_loss_arr,vec_kl_arr,act_kl_arr,r_std_arr,l_sem_arr,plan_rate_arr,avg_tot_r_arr,avg_pol_kl_arr,r_init_arr,r_arr,rt_arr,sample_arr,pos_init_arr,pos_arr,dots,sel)
loss_arrs,sem_arrs,selected_other,actor_opt_state,critic_opt_state,weights_s,plan_info = full_loop(SC,weights,params,actor_opt_state,critic_opt_state,weights_s) # full_loop(SC,weights,params) (loss_arr,actor_loss_arr,critic_loss_arr,kl_loss_arr,vec_kl_arr,act_kl_arr,r_std_arr,l_sem_arr,plan_rate_arr,avg_tot_r_arr,avg_pol_kl_arr,r_init_arr,r_arr,rt_arr,sample_arr,pos_init_arr,pos_arr,dots,sel)
print("Sim time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
(loss_arr,actor_loss_arr,critic_loss_arr,vec_kl_arr,act_kl_arr,r_tot_arr,plan_rate_arr,policy_entropy_arr) = loss_arrs
(sem_loss_arr,sem_actor_arr,sem_critic_arr,sem_act_kl_arr,sem_vec_kl_arr,sem_r_arr,sem_plan_rate_arr,sem_policy_entropy_arr) = sem_arrs
((r_arr,rp_arr,sample_arr,mask_arr,pos_plan_arr,pos_arr,dot_arr,policy_arr,hs_arr,hv_arr,vec_ind_arr,act_ind_arr),params) = selected_other # ,sel
print('policy_arr.shape=',policy_arr[0].shape,policy_arr[1].shape)
print('vec_ind_arr.shape=',vec_ind_arr.shape)
print('act_ind_arr.shape=',act_ind_arr.shape)

# plot loss_arr:
#####
legend_handles = [
    plt.Line2D([], [], color='k', label='loss'), # 
    plt.Line2D([], [], color='b', label='plan rate'), # 
]

fig,axes = plt.subplots(2,4,figsize=(16,9)) # fig,axes = plt.subplots(2,3,figsize=(12,9))
title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, TEST_LENGTH={TEST_LENGTH}, CTC_UP={CRITIC_UPDATES}, actor_lr={ACTOR_LR:.6f}, critic_lr={CRITIC_LR:.6f}, A/C GRAD_CLIP={ACTOR_GC:.2f},{CRITIC_GC:.2f}, PLAN_RATIO={PLAN_RATIO} \n CRITIC_WD={CRITIC_WD}, TEMP_VS,AS={TEMP_VS},{TEMP_AS}, L_VEC_KL={LAMBDA_VEC_KL}, L_ACT_KL={LAMBDA_ACT_KL}, SIGMA_R={SIGMA_R0}, PRIOR_PLAN={PRIOR_PLAN}, MAX_DOT_SPEED={MAX_DOT_SPEED:.2f}, ACTION_SPACE={ACTION_SPACE:.2f}, PLAN_SPACE={PLAN_SPACE:.2f}'
plt.suptitle('outer_loop_pg_new_v4f, '+title__,fontsize=10)
axes[0,0].errorbar(np.arange(TOT_EPOCHS),r_tot_arr,yerr=1.96*sem_r_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[0,0].set_xlabel('iteration')
axes[0,0].set_ylabel('total reward')
axes[0,1].errorbar(np.arange(TOT_EPOCHS),loss_arr,yerr=1.96*sem_loss_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[0,1].set_xlabel('iteration')
axes[0,1].set_ylabel('total loss')
axes[0,2].errorbar(np.arange(TOT_EPOCHS),actor_loss_arr,yerr=1.96*sem_actor_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[0,2].set_xlabel('iteration')
axes[0,2].set_ylabel('actor loss')
axes[0,3].errorbar(np.arange(TOT_EPOCHS),critic_loss_arr,yerr=1.96*sem_critic_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[0,3].set_xlabel('iteration')
axes[0,3].set_ylabel('critic loss')

axes[1,0].errorbar(np.arange(TOT_EPOCHS),plan_rate_arr,yerr=1.96*sem_plan_rate_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[1,0].set_xlabel('iteration')
axes[1,0].set_ylabel('plan rate')
axes[1,1].errorbar(np.arange(TOT_EPOCHS),policy_entropy_arr,yerr=1.96*sem_policy_entropy_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[1,1].set_xlabel('iteration')
axes[1,1].set_ylabel('policy entropy')
axes[1,2].errorbar(np.arange(TOT_EPOCHS),vec_kl_arr,yerr=1.96*sem_vec_kl_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[1,2].set_xlabel('iteration')
axes[1,2].set_ylabel('vector kl')
axes[1,3].errorbar(np.arange(TOT_EPOCHS),act_kl_arr,yerr=1.96*sem_act_kl_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[1,3].set_xlabel('iteration')
axes[1,3].set_ylabel('action kl')

# line1, _, barlinecols1 = axes[1,2].errorbar(np.arange(TOT_EPOCHS), vec_kl_arr, yerr=1.96*sem_vec_kl_arr, color='blue', ecolor='lightgray', elinewidth=2, capsize=0, label='vector kl')
# for barlinecol in barlinecols1:
#     barlinecol.set_alpha(0.1)
# axes[1,2].set_ylabel('vector kl')
# ax2 = axes[1,2].twinx()
# line2, _, barlinecols2 = ax2.errorbar(np.arange(TOT_EPOCHS), act_kl_arr, yerr=1.96*sem_act_kl_arr, color='red', ecolor='gray', elinewidth=2, capsize=0, label='action kl')
# for barlinecol in barlinecols2:
#     barlinecol.set_alpha(0.1)
# ax2.set_ylabel('action kl')
# ax2.tick_params(axis='y') #, colors='red')
# axes[1,2].set_xlabel('iteration')
# plt.legend([line1,line2],['vector kl','action kl'])

for ax in axes.flatten():
    ax.ticklabel_format(style='plain', useOffset=False)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_+'outer_loop_pg_new_v4f_tr_'+dt+'.png')

# PLOT TESTING DATA
# r_init_arr = r_init_arr[-PLOTS:,:] # [PLOTS,TEST_LENGTH]
# rt_arr = rt_arr[-PLOTS:,:] # [PLOTS,TEST_LENGTH]
# sample_arr = sample_arr[-PLOTS:,:] # [PLOTS,TEST_LENGTH]
# # pos_init_arr = pos_init_arr[-PLOTS:,:,:] # [PLOTS,TEST_LENGTH,2]
# pos_arr = pos_arr[-PLOTS:,:,:] # [PLOTS,TEST_LENGTH,2]
# print('new pos_arr=',pos_arr,'new pos_arr.shape=',pos_arr.shape)
# dots = dots[-PLOTS:,:,:] # [PLOTS,3,2]
# sel = sel[-PLOTS:,:] # [PLOTS,3]

# # Manually define legend
# legend_handles = [
#     plt.Line2D([], [], color='r', marker='x', label='r_t'),
#     plt.Line2D([], [], color='k', marker='o', label='r_hat_t'),
#     plt.Line2D([], [], color='b', label='v_loss'), # marker='.', 
#     plt.Line2D([], [], color='purple', label='r_loss'), # marker='.', 
#     plt.Line2D([], [], color='g', marker='o', label='r_true'), # marker='.',
#     # plt.Line2D([], [], color='orange', marker='o', label='r_tp'), # marker='.',
# ]

# colors_ = np.float32([[255,0,0],[0,255,0],[0,0,255]])/255 # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])/255

# # plot timeseries of true/planned reward
# fig,axis = plt.subplots(2*PLOTS,4,figsize=(15,6*PLOTS)) #9,4x plt.figure(figsize=(12,6))
# # title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, INIT_STEPS={INIT_STEPS}, TRIAL_LENGTH={TRIAL_LENGTH} \n SIGMA_R={SIGMA_R:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H_P={H_P}, H_R={H_R}'
# plt.suptitle('outer_loop_pg_new_v4_, '+title__,fontsize=10)
# for i in range(params["PLOTS"]):
#     ax0 = plt.subplot2grid((2*PLOTS,4),(2*i,0),colspan=2,rowspan=1)
#     # ax0.set_title('r_t') # v_arr
#     ax0.spines['right'].set_visible(False)
#     ax0.spines['top'].set_visible(False)
#     ax0.set_ylabel('r_t',fontsize=15)
#     ax0.set_xlabel('t',fontsize=15)
#     ax0.ticklabel_format(useOffset=False)
#     for t in range(INIT_LENGTH):
#         ax0.scatter(x=t,y=r_init_arr[i,t],color='k',s=5,marker='x')
#     for t in range(TEST_LENGTH):
#         if sample_arr[i,t] == 1:
#             ax0.scatter(x=t+INIT_LENGTH,y=rt_arr[i,t],color='red',s=5,marker='x')
#         else:
#             ax0.scatter(x=t+INIT_LENGTH,y=rt_arr[i,t],color='k',s=5,marker='x')
#     ax0.axvline(x=INIT_LENGTH,color='k',linestyle='--',linewidth=1)
#     ax0.set_ylim([-(C_MOVE+0.1),1.1])
    
#     ax1 = plt.subplot2grid((2*PLOTS,4),(2*i,2),colspan=2,rowspan=2)
#     # for i in range(M):
#     #     ax1.scatter(x=VEC_ARR[i,0],y=VEC_ARR[i,1],color='light grey',s=20,marker='o',label='modules')
#     # for t in range(INIT_LENGTH):
#     #     ax1.scatter(x=pos_init_arr[i,t,0],y=pos_init_arr[i,t,1],color='black',alpha=0.5,s=60,marker='+')
#     for t in range(TEST_LENGTH):
#         if sample_arr[i,t] == 1:
#             ax1.scatter(x=mod_(pos_arr[i,t,0]),y=mod_(pos_arr[i,t,1]),color='red',alpha=0.4,s=60,marker='o')
#         else:
#             ax1.scatter(x=mod_(pos_arr[i,t,0]),y=mod_(pos_arr[i,t,1]),color='black',alpha=0.2,s=60,marker='o')
#     for d in range(N_DOTS):
#         ax1.scatter(x=dots[i,d,0],y=dots[i,d,1],color=colors_[d,:],s=120,marker='x')
#     ax1.set_xlim(-jnp.pi,jnp.pi)
#     ax1.set_ylim(-jnp.pi,jnp.pi)
#     ax1.set_xticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
#     ax1.set_xticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=14)
#     ax1.set_yticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
#     ax1.set_yticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=14)
#     ax1.set_aspect('equal')
#     ax1.set_title(f'vector heatmap, sel={sel[i,:]}',fontsize=14)

#     ax2 = plt.subplot2grid((2*PLOTS,4),(2*i+1,0),colspan=2,rowspan=1)
#     ax2.axis('off')
# plt.tight_layout()
# plt.subplots_adjust(top=0.94)

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
#     # plt = plt.subplot2grid((3*PLOTS,3),(3*i+1,0),colspan=1,rowspan=1)
#     # plt.set_title('v_pred')#,t='+str(T_[0])+'\n r_t='+str(r_arr[i,T_[0]])+',r_pred='+str(r_pred_arr[i,T_[0]]),fontsize=8) # v_arr
#     # plt.set_aspect('equal')
#     # for ind,n in enumerate(neuron_locs.T):
#     #     plt.scatter(n[0],n[1],color=np.float32(jnp.sqrt(v_pred_rgb[i,T_[0],:,ind])),s=np.sum(17*v_pred_rgb[i,T_[0],:,ind]),marker='o') ### sum,np.float32((v_pred_rgb[:,ind]))
#     # for d in range(N_DOTS):
#     #     plt.scatter(mod_(dots_arr[i,d,0]-pos_arr[i,T_[0],0]),mod_(dots_arr[i,d,1]-pos_arr[i,T_[0],1]),color=(colors_[d,:]),s=52,marker='x') # v_arr[i,T_[0],:], r_arr[i,T_[0]]
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

# plt.axis('tight')
# plt.subplots_adjust(top=0.95)

# dt = datetime.now().strftime("%d_%m-%H%M%S")
# path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
# plt.savefig(path_ + 'figs_outer_loop_pg_new_v4_' + dt + '.png') # ctrl_v7_(v,r,h normal, r_tp changed)

save_pkl_sc((actor_opt_state,critic_opt_state,weights_s),'outer_loop_pg_new_v4f') # no _
save_large_outputs((selected_other,plan_info),'outer_loop_pg_new_v4f')