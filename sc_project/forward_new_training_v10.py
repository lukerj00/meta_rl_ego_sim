# -*- coding: utf-8 -*-
# mGRU, with trained v8 loaded in
# change to match v6/v7: plan ratio=10, change timeseries, dot speed to account for this
# KL term over movement size (instead of small initial movements), and use scheduler to change dot init loc over time
# remove conditional statement in loop
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

# import faulthandler; faulthandler.enable()

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

# def gen_sc(keys,MODULES,ACTION_SPACE,PLAN_SPACE):
#     index_range = jnp.arange(MODULES**2)
#     x = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)
#     y = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)[::-1]
#     xv,yv = jnp.meshgrid(x,y)
#     A_full = jnp.vstack([xv.flatten(),yv.flatten()])

#     # inner_mask = (jnp.abs(xv) <= ACTION_SPACE) & (jnp.abs(yv) <= ACTION_SPACE)
#     # A_inner_ind = index_range[inner_mask.flatten()]
#     # A_outer_ind = index_range[~inner_mask.flatten()]
#     # A_inner_perm = rnd.permutation(keys[0],A_inner_ind)
#     # A_outer_perm = rnd.permutation(keys[1],A_outer_ind)
#     # ID_ARR = jnp.concatenate((A_inner_perm,A_outer_perm),axis=0)
#     ID_ARR = rnd.permutation(keys[0],index_range) ## (different permutation)

#     VEC_ARR = A_full[:,ID_ARR]
#     vec_norm = jnp.linalg.norm(VEC_ARR,axis=0)
#     prior_vec = jnp.exp(-vec_norm)/jnp.sum(jnp.exp(-vec_norm))

#     H1VEC_ARR = jnp.eye(MODULES**2) # [:,ID_ARR]
#     zero_vec_index = jnp.where(jnp.all(VEC_ARR == jnp.array([0,0])[:, None], axis=0))[0][0]

#     SC = (ID_ARR,VEC_ARR,H1VEC_ARR)
#     return SC,prior_vec,zero_vec_index

def gen_sc(keys,MODULES,ACTION_SPACE,PLAN_SPACE): # (old to match with v8)
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
    zero_vec_index = jnp.where(jnp.all(VEC_ARR == jnp.array([0,0])[:, None], axis=0))[0][0]
    vec_norm = jnp.linalg.norm(VEC_ARR,axis=0)
    prior_vec = jnp.exp(-vec_norm)/jnp.sum(jnp.exp(-vec_norm))
    H1VEC_ARR = jnp.eye(MODULES**2) # [:,ID_ARR]
    SC = (ID_ARR,VEC_ARR,H1VEC_ARR)
    return SC,prior_vec,zero_vec_index

# @partial(jax.jit,static_argnums=(1,2,3))
def gen_dot_scheduled(key,N_A,N_dot,APERTURE,e,TOT_EPOCHS):
    keys = rnd.split(key,N_dot)
    lim_0 = APERTURE
    lim_inf = jnp.pi
    ###
    lim_e = lim_inf # lim_0 + (lim_inf-lim_0)*(e/TOT_EPOCHS)
    ###
    dot_0 = rnd.uniform(keys[0],shape=(N_A,2),minval=-lim_e,maxval=lim_e)#lim_e,minval=jnp.array([APERTURE/4,APERTURE/4]),maxval=jnp.array([3*APERTURE/4,3*APERTURE/4]))
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

def gen_samples(key,MODULES,ACTION_SPACE,PLANNING_SPACE,INIT_STEPS,TOT_STEPS,PRIOR_VEC):###
    # M_P = (MODULES-1)//2 # (1/ACTION_FRAC)
    # M_A = jnp.int32(M_P*(ACTION_SPACE/PLANNING_SPACE)) ### FIX (AS/PS=2/3)
    # init_vals = jnp.arange((2*M_A+1)**2)
    # main_vals = jnp.arange((2*M_A+1)**2,MODULES**2)
    keys = rnd.split(key,2)

    # init_samples = rnd.choice(keys[0],init_vals,shape=(INIT_STEPS,)) # INIT_STEPS,
    # main_samples = rnd.choice(keys[1],main_vals,shape=(TOT_STEPS-INIT_STEPS-1,))
    # return jnp.concatenate([init_samples,main_samples],axis=0) # init_samples
    return rnd.choice(keys[0],MODULES**2,shape=(TOT_STEPS,))#,p=PRIOR_VEC)

def new_params(params,key):
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
    PRIOR_VEC = params["PRIOR_VEC"]
    # N = params["N"]
    M = params["M"]
    H = params["H"]
    key, subkey = rnd.split(key) #l/0

    # POS_0 = rnd.uniform(ki[0],shape=(VMAPS,2),minval=-jnp.pi,maxval=jnp.pi)
    # DOT_0 = gen_dot_scheduled(ki[1],VMAPS,N_DOTS,APERTURE,e)
    # params["POS_0"] = POS_0
    # params["DOT_0"] = POS_0 + DOT_0 # rnd.uniform(ki[1],shape=(VMAPS,2),minval=-APERTURE,maxval=APERTURE) #gen_dot(ki[0],VMAPS,N_DOTS,APERTURE) #key_,rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #jnp.tile(jnp.array([[2,3]]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,1)) #
    # params["DOT_VEC"] = gen_dot_vecs(ki[2],VMAPS,MAX_DOT_SPEED)
    # kv = rnd.split(ki[3],num=VMAPS)
    # params["SAMPLES"] = jax.vmap(gen_samples,in_axes=(0,None,None,None,None,None,None),out_axes=0)(kv,MODULES,ACTION_SPACE,PLAN_SPACE,INIT_STEPS,TOT_STEPS,PRIOR_VEC) #rnd.choice(ki[3],M,shape=(VMAPS,(TOT_STEPS-1)))
    params["HP_0"] = jnp.sqrt(INIT/(H))*rnd.normal(subkey,(VMAPS,H))
    # params["KEY"] = rnd.split(ki[5],num=VMAPS)

def gen_dot_vecs(key,VMAPS,MAX_DOT_SPEED):
    dot_vecs = rnd.uniform(key,shape=(VMAPS,2),minval=-MAX_DOT_SPEED,maxval=MAX_DOT_SPEED) # / 2
    # mask = jnp.all((-APERTURE/2 <= dot_vecs)&(dot_vecs <= APERTURE/2),axis=1)
    # while mask.any():
    #     key = rnd.split(key)[0]
    #     new_vecs = rnd.uniform(key,shape=(jnp.sum(mask),2),minval=-APERTURE,maxval=APERTURE)
    #     dot_vecs = dot_vecs.at[mask].set(new_vecs)
    #     mask = jnp.all((-APERTURE/2 <= dot_vecs)&(dot_vecs <= APERTURE/2),axis=1)
    return dot_vecs

# def gen_binary_timeseries(keys, N, switch_prob, max_plan_length):
#     ts_init = jnp.zeros(N)  # start with 0s
#     if max_plan_length == 0:
#         return ts_init
#     def body_fn(i, carry):
#         ts, count_ones, keys = carry
#         true_fn = lambda _: (jnp.int32(rnd.bernoulli(keys[i], p=switch_prob)), 0)
#         def false_fn(_):
#             return jax.lax.cond(count_ones >= max_plan_length - 1,
#                                 lambda _: (0, 0), # ts, count_ones
#                                 lambda _: (jnp.int32(rnd.bernoulli(keys[i], p=1-switch_prob)), count_ones + 1),
#                                 None)
#         next_val, count_ones_next = jax.lax.cond(ts[i] == 0, true_fn, false_fn, None)
#         ts = ts.at[i+1].set(next_val)
#         i += 1
#         return (ts, count_ones_next, keys)
#     result, _, _ = jax.lax.fori_loop(1, N, body_fn, (ts_init, 0, keys))
#     return result

# def gen_binary_timeseries(keys, N, switch_prob, K):
#     ts_list = [0]  # Start with a zero
#     current_val = 0  # Initial value
#     i = 0  # Initialize index
    
#     while i < N - 1:  # Continue until the end of the array
#         if rnd.uniform(keys[i]) < switch_prob:  # Decide whether to switch the value
#             current_val = 1 - current_val  # Switch the value
        
#         if current_val == 0:  # If current value is 0 and there is enough space
#             ts_list.extend([0] * (K))  # Insert K zeros
#             i += K   # Move index K steps forward
#         else:  # If there is space for one more element
#             ts_list.append(current_val)  # Insert current value
#             i += 1  # Move index 1 step forward

#     # Convert list to a numpy array, ensuring the length is exactly N
#     ts_array = jnp.array(ts_list[:N], dtype=jnp.int32)
#     return ts_array

def gen_binary_action_array(N_A, T, switch_prob, P):
    binary_array = np.zeros((N_A, T), dtype=np.int32)
    action_array = np.ones((N_A, T), dtype=np.int32)  # initialize the second array as all 1s (plan)
    refac_sequence = [0] + [2] * (P - 1)  # 0 followed by P - 1 2's
    
    for idx in range(N_A):
        ts_list = [0]  # Start with a zero
        action_list = [0]  #
        current_val = 0  # Initial value
        i = 0  # Initialize index
        
        while i < T:  # Continue until the end of the array
            if np.random.uniform() < switch_prob:  # Decide whether to switch the value
                current_val = 1 - current_val  # Switch the value
            
            if current_val == 0:  # If current value is 0 and there is enough space
                ts_list.extend([0] * P)  # Insert P zeros in large_array
                action_list.extend(refac_sequence)  # Insert refac_sequence in action_array
                i += P  # Move index P steps forward
            else:  # If there is space for one more element
                ts_list.append(current_val)  # Insert current value in large_array
                action_list.append(1)  # Insert plan in action_array
                i += 1  # Move index 1 step forward
                
        binary_array[idx, :] = ts_list[:T]
        action_array[idx, :] = action_list[:T]  # action_list might be shorter than T

    return jnp.array(binary_array), jnp.array(action_array)

# def gen_timeseries(SC, pos_0, dot_0, dot_vec, samples, switch_prob, T, PLAN_RATIO, N_A, binary_series, action_series):
#     ID_ARR, VEC_ARR, H1VEC_ARR = SC  # Number of instances to generate
#     M = H1VEC_ARR.shape[0]  # Assuming H1VEC_ARR has shape (M, ...)
    
#     # Initialize large arrays
#     binary_large_array = np.zeros((N_A, 2, T), dtype=np.int32)
#     pos_plan_large_array = np.zeros((N_A, 2, T), dtype=np.float32)
#     pos_large_array = np.zeros((N_A, 2, T), dtype=np.float32)
#     dot_large_array = np.zeros((N_A, 2, T), dtype=np.float32)
#     h1vec_large_array = np.zeros((N_A, M, T), dtype=np.float32)
#     vec_large_array = np.zeros((N_A, 2, T), dtype=np.float32)
    
#     for idx in range(N_A):
#         binary_array = np.vstack([binary_series[idx,:], 1 - binary_series[idx,:]])
#         h1vec_arr = H1VEC_ARR[:, samples[idx,:]]  # Assuming samples[idx] is valid
#         vec_arr = VEC_ARR[:, samples[idx,:]]  # Assuming samples[idx] is valid

#         pos_plan_arr = np.zeros((2, T), dtype=np.float32)
#         pos_arr = np.zeros((2, T), dtype=np.float32)
#         dot_arr = np.zeros((2, T), dtype=np.float32)
        
#         # Initialize the first time step
#         pos_plan_arr[:, 0] = pos_0[idx,:]
#         pos_arr[:, 0] = pos_0[idx,:]
#         dot_arr[:, 0] = dot_0[idx,:]
#         dot_vec_i = dot_vec[idx,:]
        
#         for i in range(1, T):
#             if action_series[idx, i] == 0:  # Move
#                 pos_plan_arr[:, i] = pos_arr[:, i-1] + vec_arr[:, i]
#                 pos_arr[:, i] = pos_arr[:, i-1] + vec_arr[:, i]
#             elif action_series[idx, i] == 1:  # Plan
#                 pos_plan_arr[:, i] = pos_plan_arr[:, i-1] + vec_arr[:, i]
#                 pos_arr[:, i] = pos_arr[:, i-1]
#             else:  # Refac period (action_series[idx, i] == 2)
#                 pos_plan_arr[:, i] = pos_plan_arr[:, i-1]
#                 pos_arr[:, i] = pos_arr[:, i-1]
#             dot_arr[:, i] = dot_arr[:, i-1] + dot_vec_i

#         # Assign to large arrays
#         binary_large_array[idx, :, :] = binary_array
#         pos_plan_large_array[idx, :, :] = pos_plan_arr
#         pos_large_array[idx, :, :] = pos_arr
#         dot_large_array[idx, :, :] = dot_arr
#         h1vec_large_array[idx, :, :] = h1vec_arr
#         vec_large_array[idx, :, :] = vec_arr

#     return binary_large_array, pos_plan_large_array, pos_large_array, dot_large_array, h1vec_large_array, vec_large_array

# def gen_timeseries_row(SC, pos_0, dot_0, dot_vec, samples, params, binary_series, action_series):
#     ID_ARR, VEC_ARR, H1VEC_ARR = SC
#     M = H1VEC_ARR.shape[0]  # Assuming H1VEC_ARR has shape (M, ...)
#     switch_prob = params["SWITCH_PROB"]
#     T = params["TOT_STEPS"]
#     PLAN_RATIO = params["PLAN_RATIO"]
#     ZERO_VEC_IND = params["ZERO_VEC_IND"]

#     binary_array = jnp.vstack([binary_series, 1 - binary_series])
#     h1vec_arr = H1VEC_ARR[:, samples]  # Assuming samples is valid
#     vec_arr = VEC_ARR[:, samples]  # Assuming samples is valid

#     pos_plan_arr = jnp.zeros((2, T), dtype=jnp.float32)
#     pos_arr = jnp.zeros((2, T), dtype=jnp.float32)
#     dot_arr = jnp.zeros((2, T), dtype=jnp.float32)
#     h1vec_arr = jnp.zeros((M, T), dtype=jnp.float32)
#     vec_arr = jnp.zeros((2, T), dtype=jnp.float32)

#     # Initialize the first time step
#     pos_plan_arr = pos_plan_arr.at[:, 0].set(pos_0)
#     pos_arr = pos_arr.at[:, 0].set(pos_0)
#     dot_arr = dot_arr.at[:, 0].set(dot_0)
#     h1vec_arr = h1vec_arr.at[:, 0].set(H1VEC_ARR[:, samples[0]])
#     vec_arr = vec_arr.at[:, 0].set(VEC_ARR[:, samples[0]])

#     dot_vec_i = dot_vec
    
#     def loop_body(i, arrays):
#         pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = arrays

#         def move_branch(arrays):
#             pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = arrays
#             pos_plan_arr = pos_plan_arr.at[:, i].set(pos_arr[:, i-1] + vec_arr[:, i])
#             pos_arr = pos_arr.at[:, i].set(pos_arr[:, i-1] + vec_arr[:, i])
#             h1vec_arr = h1vec_arr.at[:, i].set(H1VEC_ARR[:, samples[i]])
#             vec_arr = vec_arr.at[:, i].set(VEC_ARR[:, samples[i]])
#             return pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr

#         def plan_branch(arrays):
#             pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = arrays
#             pos_plan_arr = pos_plan_arr.at[:, i].set(pos_plan_arr[:, i-1] + vec_arr[:, i])
#             pos_arr = pos_arr.at[:, i].set(pos_arr[:, i-1])
#             h1vec_arr = h1vec_arr.at[:, i].set(H1VEC_ARR[:, samples[i]])
#             vec_arr = vec_arr.at[:, i].set(VEC_ARR[:, samples[i]])
#             return pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr

#         def refac_branch(arrays):
#             pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = arrays
#             pos_plan_arr = pos_plan_arr.at[:, i].set(pos_plan_arr[:, i-1] + vec_arr[:, ZERO_VEC_IND]) #
#             pos_arr = pos_arr.at[:, i].set(pos_arr[:, i-1] + vec_arr[:, ZERO_VEC_IND]) #
#             h1vec_arr = h1vec_arr.at[:, i].set(H1VEC_ARR[:, ZERO_VEC_IND]) #
#             vec_arr = vec_arr.at[:, i].set(VEC_ARR[:, ZERO_VEC_IND]) #
#             return pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr

#         pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = jax.lax.cond(
#             action_series[i] == 0,
#             lambda arrays: move_branch(arrays),
#             lambda arrays: jax.lax.cond(
#                 action_series[i] == 1,
#                 lambda arrays: plan_branch(arrays),
#                 lambda arrays: refac_branch(arrays),
#                 arrays
#             ),
#             (pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr)
#         )

#         dot_arr = dot_arr.at[:, i].set(dot_arr[:, i-1] + dot_vec_i)
        
#         return pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr
    
#     pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = jax.lax.fori_loop(1, T, loop_body, (pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr))
    
#     return binary_array, pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr

def gen_timeseries_row(SC, pos_0, dot_0, dot_vec, samples, params, binary_series, action_series):
    ID_ARR, VEC_ARR, H1VEC_ARR = SC
    M = H1VEC_ARR.shape[0]  # Assuming H1VEC_ARR has shape (M, ...)
    switch_prob = params["SWITCH_PROB"]
    T = params["TOT_STEPS"]
    PLAN_RATIO = params["PLAN_RATIO"]
    ZERO_VEC_IND = params["ZERO_VEC_IND"]

    binary_array = jnp.vstack([binary_series, 1 - binary_series])
    h1vecs = H1VEC_ARR[:, samples]  # Assuming samples is valid
    vecs = VEC_ARR[:, samples]  # Assuming samples is valid
    zero_vec = VEC_ARR[:, ZERO_VEC_IND]
    zero_h1vec = H1VEC_ARR[:, ZERO_VEC_IND]

    pos_plan_arr = jnp.zeros((2, T), dtype=jnp.float32)
    pos_arr = jnp.zeros((2, T), dtype=jnp.float32)
    dot_arr = jnp.zeros((2, T), dtype=jnp.float32)
    h1vec_arr = jnp.zeros((M, T), dtype=jnp.float32)
    vec_arr = jnp.zeros((2, T), dtype=jnp.float32)

    # Initialize the first time step
    pos_plan_arr = pos_plan_arr.at[:, 0].set(pos_0)
    pos_arr = pos_arr.at[:, 0].set(pos_0)
    dot_arr = dot_arr.at[:, 0].set(dot_0)
    h1vec_arr = h1vec_arr.at[:, 0].set(h1vecs[:, 0])
    vec_arr = vec_arr.at[:, 0].set(vecs[:, 0])

    dot_vec_i = dot_vec
    
    def loop_body(i, arrays):
        pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = arrays

        def move_branch(arrays):
            pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = arrays
            pos_plan_arr = pos_plan_arr.at[:, i].set(pos_arr[:, i-1] + vecs[:, i])
            pos_arr = pos_arr.at[:, i].set(pos_arr[:, i-1] + vecs[:, i])
            h1vec_arr = h1vec_arr.at[:, i].set(h1vecs[:, i])
            vec_arr = vec_arr.at[:, i].set(vecs[:, i])
            return pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr

        def plan_branch(arrays):
            pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = arrays
            pos_plan_arr = pos_plan_arr.at[:, i].set(pos_plan_arr[:, i-1] + vecs[:, i])
            pos_arr = pos_arr.at[:, i].set(pos_arr[:, i-1])
            h1vec_arr = h1vec_arr.at[:, i].set(h1vecs[:, i])
            vec_arr = vec_arr.at[:, i].set(vecs[:, i])
            return pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr

        def refac_branch(arrays):
            pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = arrays
            pos_plan_arr = pos_plan_arr.at[:, i].set(pos_plan_arr[:, i-1] + zero_vec) #
            pos_arr = pos_arr.at[:, i].set(pos_arr[:, i-1] + zero_vec) #
            h1vec_arr = h1vec_arr.at[:, i].set(zero_h1vec) #
            vec_arr = vec_arr.at[:, i].set(zero_vec) #
            return pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr

        pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = jax.lax.cond(
            action_series[i] == 0,
            lambda arrays: move_branch(arrays),
            lambda arrays: jax.lax.cond(
                action_series[i] == 1,
                lambda arrays: plan_branch(arrays),
                lambda arrays: refac_branch(arrays),
                arrays
            ),
            (pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr)
        )

        dot_arr = dot_arr.at[:, i].set(dot_arr[:, i-1] + dot_vec_i)
        
        return pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr
    
    pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = jax.lax.fori_loop(1, T, loop_body, (pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr))
    
    return binary_array, pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr

gen_timeseries = jax.vmap(gen_timeseries_row, in_axes=(None, 0, 0, 0, 0, None, 0, 0), out_axes=(0, 0, 0, 0, 0, 0))

# def gen_timeseries(key, SC, pos_0, dot_0, dot_vec, samples, switch_prob, N, max_plan_length, PLAN_RATIO):
#     ID_ARR, VEC_ARR, H1VEC_ARR = SC
#     keys = rnd.split(key, N)
#     binary_series = gen_binary_timeseries(keys, N, switch_prob, max_plan_length)
#     binary_array = jnp.vstack([binary_series, 1 - binary_series])
#     h1vec_arr = H1VEC_ARR[:, samples]
#     vec_arr = VEC_ARR[:, samples]
#     def time_step(carry, i):
#         pos_plan, pos, dot = carry
#         # dot_next = dot + dot_vec  # Assuming dot_vec is a globally available vector
#         def true_fn(_):
#             return pos + vec_arr[:, i-1], pos + vec_arr[:, i-1], dot + dot_vec
#         def false_fn(_):
#             return pos_plan + vec_arr[:, i-1], pos, dot + dot_vec
#         pos_plan_next, pos_next, dot_next = jax.lax.cond(binary_series[i] == 0, true_fn, false_fn, None)
#         return (pos_plan_next, pos_next, dot_next), (pos_plan_next, pos_next, dot_next)
#     init_carry = (jnp.array(pos_0), jnp.array(pos_0), jnp.array(dot_0))
#     _, (pos_plan_stacked, pos_stacked, dot_stacked) = jax.lax.scan(time_step, init_carry, jnp.arange(1,N))
#     return binary_array.T,jnp.concatenate([jnp.reshape(pos_0,(1,2)),pos_plan_stacked]).T,jnp.concatenate([jnp.reshape(pos_0,(1,2)),pos_stacked]).T,jnp.concatenate([jnp.reshape(dot_0,(1,2)),dot_stacked]).T,h1vec_arr,vec_arr # should be 2,N

# def gen_timeseries(key, SC, pos_0, dot_0, dot_vec, samples, switch_prob, N, PLAN_RATIO, binary_series, action_series):
#     ID_ARR, VEC_ARR, H1VEC_ARR = SC
#     keys = rnd.split(key, N)
    
#     binary_array = jnp.vstack([binary_series, 1 - binary_series])
#     h1vec_arr = H1VEC_ARR[:, samples]
#     vec_arr = VEC_ARR[:, samples]

#     def time_step(carry, i):
#         pos_plan, pos, dot = carry

#         # action_series[i] == 0: Move
#         def move_fn(_):
#             return pos + vec_arr[:, i], pos + vec_arr[:, i], dot + dot_vec
        
#         # action_series[i] == 1: Plan
#         def plan_fn(_):
#             return pos_plan + vec_arr[:, i], pos, dot + dot_vec
        
#         # action_series[i] == 2: Refac period
#         def refac_fn(_):
#             return pos_plan, pos, dot + dot_vec

#         pos_plan_next, pos_next, dot_next = jax.lax.cond(
#             action_series[i] == 0, move_fn,  # Move condition
#             lambda _: jax.lax.cond(
#                 action_series[i] == 1, plan_fn,  # Plan condition
#                 refac_fn,  # Refac condition (default if action_series[i] is not 0 or 1)
#                 None
#             ),
#             None
#         )
        
#         return (pos_plan_next, pos_next, dot_next), (pos_plan_next, pos_next, dot_next)

#     init_carry = (jnp.array(pos_0), jnp.array(pos_0), jnp.array(dot_0))
#     _, (pos_plan_stacked, pos_stacked, dot_stacked) = jax.lax.scan(time_step, init_carry, jnp.arange(1,N))

#     cat_ = lambda init,stacked: jnp.concatenate([jnp.reshape(init,(-1,2)),stacked]).T
#     pos_plan_arr = cat_(pos_0,pos_plan_stacked)
#     pos_arr = cat_(pos_0,pos_stacked)
#     dot_arr = cat_(dot_0,dot_stacked)

#     return binary_array,pos_plan_arr,pos_arr,dot_arr,h1vec_arr,vec_arr # 2,N
    # jnp.concatenate([jnp.reshape(pos_0,(-1,2)),pos_plan_stacked]).T,jnp.concatenate([jnp.reshape(pos_0,(-1,2)),pos_stacked]).T,jnp.concatenate([jnp.reshape(dot_0,(-1,2)),dot_stacked]).T,h1vec_arr,vec_arr

# def gen_timeseries(key, SC, pos_0, dot_0, dot_vec, samples, switch_prob, N, PLAN_RATIO, binary_series, action_series):
#     ID_ARR, VEC_ARR, H1VEC_ARR = SC
#     keys = rnd.split(key, N)
#     # binary_series = gen_binary_timeseries(keys[0], N, switch_prob, PLAN_RATIO)
#     # print('binary_series=',binary_series,binary_series.shape)
#     binary_array = jnp.vstack([binary_series, 1 - binary_series])
#     h1vec_arr = H1VEC_ARR[:, samples]
#     vec_arr = VEC_ARR[:, samples]
#     def time_step(carry, i):
#         pos_plan, pos, dot = carry
#         # dot_next = dot + dot_vec  # Assuming dot_vec is a globally available vector
#         def true_fn(_):
#             return pos + vec_arr[:, i], pos + vec_arr[:, i], dot + dot_vec
#         def false_fn(_):
#             return pos_plan + vec_arr[:, i], pos, dot + dot_vec
#         pos_plan_next, pos_next, dot_next = jax.lax.cond(binary_series[i] == 0, true_fn, false_fn, None)
#         return (pos_plan_next, pos_next, dot_next), (pos_plan_next, pos_next, dot_next)
#     init_carry = (jnp.array(pos_0), jnp.array(pos_0), jnp.array(dot_0))
#     _, (pos_plan_stacked, pos_stacked, dot_stacked) = jax.lax.scan(time_step, init_carry, jnp.arange(N-1))
#     return binary_array,jnp.concatenate([jnp.reshape(pos_0,(-1,2)),pos_plan_stacked]).T,jnp.concatenate([jnp.reshape(pos_0,(-1,2)),pos_stacked]).T,jnp.concatenate([jnp.reshape(dot_0,(-1,2)),dot_stacked]).T,h1vec_arr,vec_arr # should be 2,N

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

@partial(jax.jit,static_argnums=(5,))
def plan(h1vec,v_0,h_0,pm_t,p_weights,PLAN_ITS): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    def loop_body(carry,i):
        return RNN_forward(carry)
    def RNN_forward(carry):
        p_weights,h1vec,v_0,pm_t,h_t_1 = carry
        W_h1_f = p_weights["W_h1_f"]
        W_h1_hh = p_weights["W_h1_hh"]
        W_p_f = p_weights["W_p_f"]
        W_p_hh = p_weights["W_p_hh"]
        W_m_f = p_weights["W_m_f"]
        W_m_hh = p_weights["W_m_hh"]
        W_v_f = p_weights["W_v_f"]
        W_v_hh = p_weights["W_v_hh"]
        U_f = p_weights["U_f"]
        U_hh = p_weights["U_hh"]
        b_f = p_weights["b_f"]
        b_hh = p_weights["b_hh"]
        f_t = jax.nn.sigmoid(W_p_f*pm_t[0] + W_m_f*pm_t[1] + jnp.matmul(W_h1_f,h1vec) + jnp.matmul(W_v_f,v_0) + jnp.matmul(U_f,h_t_1) + b_f)
        hhat_t = jax.nn.tanh(W_p_hh*pm_t[0] + W_m_hh*pm_t[1] + jnp.matmul(W_h1_hh,h1vec) + jnp.matmul(W_v_hh,v_0) + jnp.matmul(U_hh,jnp.multiply(f_t,h_t_1)) + b_hh)
        h_t = jnp.multiply((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t)
        return (p_weights,h1vec,v_0,pm_t,h_t),None
    W_full = p_weights["W_full"]
    W_ap = p_weights["W_ap"]
    carry_0 = (p_weights,h1vec,v_0,pm_t,h_0)
    (*_,h_t),_ = jax.lax.scan(loop_body,carry_0,jnp.zeros(PLAN_ITS))
    v_pred_full = jnp.matmul(W_full,h_t)
    v_pred_ap = jnp.take(v_pred_full, params["INDICES"])
    return v_pred_ap,v_pred_full,h_t # dot_hat_t,

# def body_fnc(SC,p_weights,params,key,pos_0,dot_0,dot_vec,h_0,samples,e):###
#     loss_v_arr,loss_c_arr = (jnp.zeros(params["TOT_STEPS"]) for _ in range(2))
#     v_pred_arr,v_t_arr = (jnp.zeros((params["TOT_STEPS"],params["N_F"])) for _ in range(2))
#     rel_vec_hat_arr = jnp.zeros((params["TOT_STEPS"],2))
#     pm_arr,pos_plan_arr,pos_arr,dot_arr,h1vec_arr,vec_arr = gen_timeseries(key,SC,pos_0,dot_0,dot_vec,samples,params["SWITCH_PROB"],params["TOT_STEPS"],params["MAX_PLAN_LENGTH"])
#     tot_loss_v,tot_loss_c = 0,0
#     h_t_1 = h_0
#     v_t_1_ap = neuron_act_noise(samples[0],params["THETA_AP"],params["SIGMA_A"],params["SIGMA_N"],dot_arr[:,0],pos_plan_arr[:,0]) # pos_arr
#     v_t_1_full = neuron_act_noise(samples[0],params["THETA_FULL"],params["SIGMA_A"],params["SIGMA_N"],dot_arr[:,0],pos_plan_arr[:,0]) # pos_arr
#     v_t_arr = v_t_arr.at[0,:].set(v_t_1_full)
#     for t in range(1,params["TOT_STEPS"]):
#         v_pred_ap,v_pred_full,h_t = plan(h1vec_arr[:,t-1],v_t_1_ap,h_t_1,pm_arr[t,:],p_weights,params["PLAN_ITS"]) # ,dot_hat_t
#         v_t_ap = neuron_act_noise(samples[t],params["THETA_AP"],params["SIGMA_A"],params["SIGMA_N"],dot_arr[:,t],pos_plan_arr[:,t]) # pos_arr
#         v_t_full = neuron_act_noise(samples[t],params["THETA_FULL"],params["SIGMA_A"],params["SIGMA_N"],dot_arr[:,t],pos_plan_arr[:,t]) # pos_arr
        
#         loss_v_full = jnp.sum((v_pred_full-v_t_full)**2)
#         # loss_v_ap = jnp.sum((v_pred_ap-v_t_1_ap)**2) # loss between new/prev predictions (v_t_1 not v_t)
#         # loss_cos_full = cosine_similarity(v_pred_full,v_t_full)
        
#         v_t_arr = v_t_arr.at[t,:].set(v_t_full)
#         v_pred_arr = v_pred_arr.at[t,:].set(v_pred_full)    
#         loss_v_arr = loss_v_arr.at[t].set(loss_v_full)
#         loss_c_arr = loss_c_arr.at[t].set(loss_cos_full)
#         h_t_1 = h_t
#         v_t_1_ap = jax.lax.cond(pm_arr[t,0] == 1, lambda _: v_pred_ap, lambda _: v_t_ap, None)
#         # if t >= 0: # params["INIT_STEPS"]:
#             # if e < params["INIT_TRAIN_EPOCHS"]:
#         tot_loss_v += loss_v_full #+ params["LAMBDA_C"]*loss_cos_ap #- params["LAMBDA_P"]*loss_prev_mse_ap
#             # else:
#             #     tot_loss_v += loss_v_pl
#             #     loss_v_arr = loss_v_arr.at[t].set(loss_v_pl)
#             #     v_pred_arr = v_pred_arr.at[t,:].set(v_pred_pl)
#     # loss_tot = tot_loss_v # + params["LAMBDA_D"]*tot_loss_d
#     avg_loss = tot_loss_v/(params["TOT_STEPS"]) # params["PRED_STEPS"]
#     return avg_loss,(v_pred_arr,v_t_arr,pos_plan_arr,pos_arr,dot_arr,pm_arr,loss_v_arr,loss_c_arr)

def loop_body(carry, current_input):
    h_t_1, v_t_1_ap, tot_loss_v, p_weights = carry
    sample, h1vec_t, pm_t, dot_t, pos_plan_t = current_input
    v_pred_ap, v_pred_full, h_t = plan(h1vec_t, v_t_1_ap, h_t_1, pm_t, p_weights, params["PLAN_ITS"])
    v_t_ap = neuron_act_noise(sample, params["THETA_AP"], params["SIGMA_A"], params["SIGMA_N"], dot_t, pos_plan_t)
    v_t_full = neuron_act_noise(sample, params["THETA_FULL"], params["SIGMA_A"], params["SIGMA_N"], dot_t, pos_plan_t)
    loss_v = jnp.sum((v_pred_full - v_t_full) ** 2)
    loss_c = cosine_similarity(v_pred_full, v_t_full)
    tot_loss_v = tot_loss_v + loss_v
    v_stacked = jnp.vstack([v_pred_ap, v_t_ap])
    v_t_1_ap = jnp.dot(pm_t, v_stacked)
    # v_t_1_ap = jax.lax.cond(pm_t[0] == 1, lambda _: v_pred_ap, lambda _: v_t_ap, None)
    return (h_t, v_t_1_ap, tot_loss_v, p_weights), (v_pred_full, v_t_full, loss_v, loss_c)

def body_fnc(SC, p_weights, params, pos_plan_arr, pos_arr, dot_arr, h1vec_arr, dot_vec, h_0, samples, binary_arr, action_series, e):
    # pm_arr, pos_plan_arr, pos_arr, dot_arr, h1vec_arr, vec_arr = gen_timeseries(key, SC, pos_0, dot_0, dot_vec, samples, params["SWITCH_PROB"], params["TOT_STEPS"], params["PLAN_RATIO"], binary_series, action_series)
    # loss_c_stacked = jnp.zeros(params["TOT_STEPS"])
    v_0_ap = neuron_act_noise(samples[0], params["THETA_AP"], params["SIGMA_A"], params["SIGMA_N"], dot_arr[:, 0], pos_plan_arr[:, 0])
    tot_loss_v = 0
    v_t_1_full = neuron_act_noise(samples[0], params["THETA_FULL"], params["SIGMA_A"], params["SIGMA_N"], dot_arr[:, 0], pos_plan_arr[:, 0])
    initial_carry = (h_0, v_0_ap, tot_loss_v, p_weights)
    sequence_inputs = (samples[1:], h1vec_arr[:, 1:].T, binary_arr[:,1:].T, dot_arr[:, 1:].T, pos_plan_arr[:, 1:].T) # jnp.arange(params["TOT_STEPS"] - 1), 
    
    final_carry, outputs = jax.lax.scan(loop_body, initial_carry, sequence_inputs)
    h_t, v_t_1_ap, tot_loss_v, p_weights = final_carry
    v_pred_full_stacked, v_t_full_stacked, loss_v_stacked, loss_c_stacked = outputs 
    # jax.debug.print('**** tot_loss_v={}', tot_loss_v)
    # jax.debug.print('**** sum(loss_v_stacked)={}', jnp.sum(loss_v_stacked))
    v_pred_full_stacked = jnp.concatenate([jnp.expand_dims(jnp.zeros(params["N_F"]),axis=0), v_pred_full_stacked])
    v_t_full_stacked = jnp.concatenate([jnp.expand_dims(v_t_1_full,axis=0), v_t_full_stacked])
    loss_v_stacked = jnp.concatenate([jnp.zeros(1), loss_v_stacked])
    loss_c_stacked = jnp.concatenate([jnp.zeros(1), loss_c_stacked])

    avg_loss = jnp.sum(tot_loss_v) / params["TOT_STEPS"] ## loss_v_stacked
    return avg_loss, (v_pred_full_stacked, v_t_full_stacked, pos_plan_arr, pos_arr, dot_arr, binary_arr, action_series, loss_v_stacked, loss_c_stacked)

# @partial(jax.jit,static_argnums=())
# def forward_model_loop(SC,weights,params,INIT_0):
def forward_model_loop(SC,weights,params,INIT_0,opt_state):
    p_weights = weights["p_weights"]
    T = params["TOT_EPOCHS"]
    loss_arr,loss_std = (jnp.empty(params["TOT_EPOCHS"]) for _ in range(2))
    loss_v_arr,v_std_arr,loss_c_arr,c_std_arr = (jnp.empty((params["TOT_EPOCHS"])) for _ in range(4)) #,params["TOT_STEPS"])) for _ in range(4))
    optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    ###
    # opt_state = optimizer.init(p_weights)#
    ###
    POS_0,DOT_0,DOT_VEC,SAMPLES = INIT_0
    print("Starting binary/action pregen")
    binary_series,action_series = gen_binary_action_array(params["N_ARRAY"], params["TOT_STEPS"], params["SWITCH_PROB"], params["PLAN_RATIO"])
    # print('binary_series=',binary_series,binary_series.shape)
    # print('action_series=',action_series,action_series.shape)
    print("Starting ts pregen")
    binary_array,pos_plan_array,pos_array,dot_array,h1vec_array,vec_array = gen_timeseries(SC,POS_0,DOT_0,DOT_VEC,SAMPLES,params,binary_series,action_series)
    # print('binary_array=',binary_array[0,:,:],binary_array.shape)
    # print('pos_plan_array=',pos_plan_array[0,:,:],pos_plan_array.shape)
    # print('pos_array=',pos_array[0,:,:],pos_array.shape)
    # print('dot_array=',dot_array[0,:,:],dot_array.shape)
    jax.clear_caches()
    key = jax.random.PRNGKey(0)
    print("Starting training")
    for e in range(T):
        key,subkey = jax.random.split(key)
        new_params(params,subkey) # 'params = new_params(params,e)'
        indexes = jax.random.choice(subkey, binary_array.shape[0], shape=(params["VMAPS"],), replace=False)
        binary_arr_vmap = binary_array[indexes,:,:] # N_A,2,T
        action_series_vmap = action_series[indexes,:]
        pos_plan_vmap = pos_plan_array[indexes,:,:]
        pos_vmap = pos_array[indexes,:,:]
        dot_vmap = dot_array[indexes,:,:]
        h1vec_vmap = h1vec_array[indexes,:,:]
        vec_vmap = vec_array[indexes,:,:]
        dot_vec_vmap = DOT_VEC[indexes,:]
        samples_vmap = SAMPLES[indexes,:]
        # arrays = (binary_arr_vmap,pos_plan_vmap,pos_vmap,dot_vmap,h1vec_vmap,vec_vmap,dot_vec_vmap)

        # dot_0 = params["DOT_0"][indexes,:]
        # dot_vec = params["DOT_VEC"][indexes,:]
        # samples = params["SAMPLES"][indexes,:]
        # pos_0 = params["POS_0"][indexes,:]
        h_0 = params["HP_0"] #[e,:,:]
        # key = params["KEY"]

        val_grad = jax.value_and_grad(body_fnc,argnums=1,allow_int=True,has_aux=True)
        val_grad_vmap = jax.vmap(val_grad,in_axes=(None,None,None,0,0,0,0,0,0,0,0,0,None),out_axes=(0,0))
        (loss,aux),grads = val_grad_vmap(SC,p_weights,params,pos_plan_vmap,pos_vmap,dot_vmap,h1vec_vmap,dot_vec_vmap,h_0,samples_vmap,binary_arr_vmap,action_series_vmap,e) # val_grad_vmap ,,v_0
        v_pred_arr,v_t_arr,pos_plan_arr,pos_arr,dot_arr,binary_arr,action_ser,loss_v_epoch,loss_c_epoch = aux # [VMAPS,STEPS,N]x2,[VMAPS,STEPS,2]x3,[VMAPS,STEPS]x2 (final timestep)
        # print('pos_plan_arr=',pos_plan_arr[0,:,:])
        # print('pos_plan_arr_vmap=',pos_plan_vmap[0,:,:])
        # print('pos_arr=',pos_arr[0,:,:])
        # print('pos_arr_vmap=',pos_vmap[0,:,:])
        # print('binary_arr=',binary_arr[0,:,:])
        # print('action_ser=',action_ser[0,:])
        grads_ = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0),grads)
        opt_update,opt_state = optimizer.update(grads_,opt_state,p_weights)
        p_weights = optax.apply_updates(p_weights,opt_update)
        loss_arr = loss_arr.at[e].set(jnp.mean(loss))
        loss_std = loss_std.at[e].set(jnp.std(loss)) #/jnp.sqrt(params["VMAPS"]))
        loss_v_arr = loss_v_arr.at[e].set(jnp.mean(loss_v_epoch,axis=None))
        loss_c_arr = loss_c_arr.at[e].set(jnp.mean(loss_c_epoch,axis=None))
        v_std_arr = v_std_arr.at[e].set(jnp.std(loss_v_epoch,axis=None)/jnp.sqrt(params["VMAPS"]))
        c_std_arr = c_std_arr.at[e].set(jnp.std(loss_c_epoch,axis=None)/jnp.sqrt(params["VMAPS"]))
        print("e={}",e)
        print("loss_avg={}",jnp.mean(loss))
        print("loss_std={}",jnp.std(loss)) #/jnp.sqrt(params["VMAPS"]))
        # print("loss_v={}",jnp.mean(loss_v_epoch,axis=None))
        print("loss_c_avg={}",jnp.mean(jnp.mean(loss_c_epoch,axis=None)))

        # print("std_v={}",jnp.std(loss_v_arr)/jnp.sqrt(params["VMAPS"]))
        # print("loss_d={}",jnp.mean(loss_d_arr_,axis=0))
        # print("std_d={}",jnp.std(loss_d_arr)/jnp.sqrt(params["VMAPS"]))

        # if e>0 and (e % 100) == 0:
        #     print("*clearing cache*")
        #     jax.clear_caches()

        #     print("v_pred_arr.shape=",v_pred_arr.shape)
        #     print("v_t_arr.shape=",v_t_arr.shape)
        #     print("pos_arr.shape=",pos_arr.shape)
        #     print("dot_arr.shape=",dot_arr.shape)
        #     print("rel_vec_hat_arr.shape=",rel_vec_hat_arr.shape)
        #     print("loss_v_arr.shape=",loss_v_arr.shape)
        #     print("loss_d_arr.shape=",loss_d_arr.shape)
        arrs = (loss_arr,loss_std,loss_v_arr,v_std_arr,loss_c_arr,c_std_arr)
        aux = (v_pred_arr,v_t_arr,loss_v_epoch,loss_c_epoch,pos_plan_arr,pos_arr,dot_arr,binary_arr,action_ser,opt_state,p_weights) # _vmap
        # print('pos_plan_arr=',pos_plan_arr[0,0,:],pos_plan_arr.shape)
        # print('pos_arr=',pos_arr[0,0,:],pos_arr.shape)
        # print('binary_arr_vmap=',binary_arr_vmap,binary_arr_vmap.shape)
        # print('action_series_vmap=',action_series_vmap,action_series_vmap.shape)
    return arrs,aux # [VMAPS,STEPS,N]x2,[VMAPS,STEPS,2]x3,[VMAPS,STEPS]x2,..

# hyperparams
TOT_EPOCHS = 1000 #10000 # 1000 #250000
EPOCHS = 1
INIT_TRAIN_EPOCHS = 500000 ### epochs until 'phase 2'
PLOTS = 3
# LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 200 # 800,500
PLAN_ITS = 10 # 10,8,5
INIT_STEPS = 0 # (taking loss over all steps so doesnt matter)
TOT_STEPS = 100 # steps in rnn-time inc refac period
PRED_STEPS = TOT_STEPS-INIT_STEPS
MAX_PLAN_LENGTH = TOT_STEPS # 1,3,5
PLAN_RATIO = 10 # 5,2
LR = 0.0001 # 0.0001, 0.003,,0.0001
WD = 0.0001 # 0.0001
H = 300 # 500,300
INIT = 2 # 0.5,0.1
LAMBDA_D = 1 # 1,0.1
LAMBDA_C = 10

# ENV/sc params
ks = rnd.split(rnd.PRNGKey(0),10) #DONT CHANGE#
ke = rnd.split(rnd.PRNGKey(2),10) #change#
MODULES = 9 # 17 # (3*N+1)
M = MODULES**2
APERTURE = (1/2)*jnp.pi # (3/5)*jnp.pi # (jnp.sqrt(2)/2)*jnp.pi ### unconstrained
ACTION_FRAC = 1 # CHANGED FROM 1/2... unconstrained
ACTION_SPACE = ACTION_FRAC*APERTURE # 'AGENT_SPEED'
PLAN_FRAC_REL = 1 #... 3/2
PLAN_SPACE = PLAN_FRAC_REL*ACTION_SPACE
MAX_DOT_SPEED_REL_FRAC = 1.5 # 1.2 # 3/2 # 5/4
MAX_DOT_SPEED = (MAX_DOT_SPEED_REL_FRAC*ACTION_SPACE)/PLAN_RATIO
ALPHA = 1
NEURONS_FULL = 12 # 15 # 12 # jnp.int32(NEURONS_AP*(jnp.pi//APERTURE))
N_F = (NEURONS_FULL**2)
NEURONS_AP = jnp.int32(jnp.floor(NEURONS_FULL*(APERTURE/jnp.pi))) # 6 # 10
N_A = (NEURONS_AP**2)
NEURONS_PLAN = NEURONS_FULL #NEURONS_AP + 2*ALPHA
N_P = (NEURONS_PLAN**2)
THETA_FULL = jnp.linspace(-(jnp.pi-jnp.pi/NEURONS_FULL),(jnp.pi-jnp.pi/NEURONS_FULL),NEURONS_FULL)
THETA_AP = THETA_FULL[NEURONS_FULL//2 - NEURONS_AP//2 : NEURONS_FULL//2 + NEURONS_AP//2]
THETA_PLAN = THETA_FULL #[NEURONS_FULL//2 - NEURONS_PLAN//2 : NEURONS_FULL//2 + NEURONS_PLAN//2]
SIGMA_A = 0.3 # CHANGED FROM 0.5 0.3,0.5,1,0.3,1,0.5,1,0.1
# SIGMA_D = 0.5
SIGMA_N = 0.2 # 0.1,0.05
SWITCH_PROB = 0.2 # 0.25
N_ARRAY = 2000
# 
SC,PRIOR_VEC,ZERO_VEC_IND = gen_sc(ks,MODULES,ACTION_SPACE,PLAN_SPACE) # (ID_ARR,VEC_ARR,H1VEC_ARR)
INDICES = get_inner_activation_indices(NEURONS_PLAN, NEURONS_AP)
COLORS = jnp.array([[255]]) # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = 1 #COLORS.shape[0]
# STEP_ARRAY = jnp.arange(1,TOT_STEPS)
POS_0 = rnd.uniform(ke[0],shape=(N_ARRAY,2),minval=-jnp.pi,maxval=jnp.pi) # POS_0 = None # rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
DOT_0 = POS_0 + gen_dot_scheduled(ke[1],N_ARRAY,N_DOTS,APERTURE,0,TOT_EPOCHS) # DOT_0 = None # gen_dot(ke[0],EPOCHS,VMAPS,N_dot,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #gen_dot(ke[0],EPOCHS,VMAPS,N_dot,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_dot,2),minval=-APERTURE,maxval=APERTURE) #gen_dot(ke[0],EPOCHS,VMAPS,N_dot,APERTURE)
DOT_VEC = gen_dot_vecs(ke[2],N_ARRAY,MAX_DOT_SPEED) # DOT_VEC = None
kv = rnd.split(ke[3],num=N_ARRAY)
SAMPLES = jax.vmap(gen_samples,in_axes=(0,None,None,None,None,None,None),out_axes=0)(kv,MODULES,ACTION_SPACE,PLAN_SPACE,INIT_STEPS,TOT_STEPS,PRIOR_VEC) # SAMPLES = None # rnd.choice(ke[2],M,(EPOCHS,VMAPS,STEPS)) # [E,V] rnd.randint(rnd.PRNGKey(0),0,EPOCHS*STEPS,(EPOCHS,STEPS))
HP_0 = None # jnp.sqrt(INIT/(H))*rnd.normal(ke[4],(EPOCHS,VMAPS,H)) # [E,V,H]
INIT_0 = (POS_0,DOT_0,DOT_VEC,SAMPLES)

# INITIALIZATION ### FIX
ki = rnd.split(rnd.PRNGKey(1),num=50)
# W_h10 = jnp.sqrt(INIT/(H+M))*rnd.normal(ki[0],(H,M))
# W_v0 = jnp.sqrt(INIT/(H+N_A))*rnd.normal(ki[1],(H,N_A))
# W_r_f0 = jnp.sqrt(INIT/(N_F+H))*rnd.normal(ki[2],(N_F,H))
# W_r_a0 = jnp.sqrt(INIT/(N_A+H))*rnd.normal(ki[3],(N_A,H))
# W_r_full0 = jnp.sqrt(INIT/(N_P+H))*rnd.normal(ki[4],(N_P,H))
# # W_dot0 = jnp.sqrt(INIT/(H+4))*rnd.normal(ki[3],(4,H))
# W_v0 = jnp.sqrt(INIT/(H+N_A))*rnd.normal(ki[5],(H,N_A))
# W_r0 = jnp.sqrt(INIT/(N_F+H))*rnd.normal(ki[6],(N_F,H))
# # W_dot0 = jnp.sqrt(INIT/(H+4))*rnd.normal(ki[3],(4,H))
# W_p0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[7],(H,))
# W_m0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[8],(H,))
# U_vh0,_ = jnp.linalg.qr(jnp.sqrt(INIT/(H+H))*rnd.normal(ki[9],(H,H)))
# b_vh0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[10],(H,))
W_h1_f0 = jnp.sqrt(INIT/(H+M))*rnd.normal(ki[0],(H,M))
W_h1_hh0 = jnp.sqrt(INIT/(H+M))*rnd.normal(ki[1],(H,M))
W_p_f0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[2],(H,))
W_p_hh0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[3],(H,))
W_m_f0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[4],(H,))
W_m_hh0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[5],(H,))
W_v_f0 = jnp.sqrt(INIT/(H+N_A))*rnd.normal(ki[6],(H,N_A))
W_v_hh0 = jnp.sqrt(INIT/(H+N_A))*rnd.normal(ki[7],(H,N_A))
U_f0,_ = jnp.linalg.qr(jnp.sqrt(INIT/(H+H))*rnd.normal(ki[8],(H,H)))
U_hh0,_ = jnp.linalg.qr(jnp.sqrt(INIT/(H+H))*rnd.normal(ki[9],(H,H)))
b_f0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[10],(H,))
b_hh0 = jnp.sqrt(INIT/(H))*rnd.normal(ki[11],(H,))
W_ap0 = jnp.sqrt(INIT/(N_A+H))*rnd.normal(ki[12],(N_A,H))
W_full0 = jnp.sqrt(INIT/(N_F+H))*rnd.normal(ki[13],(N_F,H))

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
    ## "DOT_0" : DOT_0,
    ## "DOT_VEC" : DOT_VEC,
    "SIGMA_A" : SIGMA_A,
    # "SIGMA_D" : SIGMA_D,
    "SIGMA_N" : SIGMA_N,
    "COLORS" : COLORS,
    ## "SAMPLES" : SAMPLES,
    # "STEP_ARRAY" : STEP_ARRAY,
    ## "POS_0" : POS_0,
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
    "SWITCH_PROB" : SWITCH_PROB,
    "MAX_PLAN_LENGTH" : MAX_PLAN_LENGTH,
    "PLAN_RATIO" : PLAN_RATIO,
    "PRIOR_VEC" : PRIOR_VEC,
    "N_ARRAY" : N_ARRAY,
    "ZERO_VEC_IND" : ZERO_VEC_IND,
}

weights = {
    's_weights' : {

    },
    'p_weights' : {
        # "W_h1" : W_h10,
        # "W_r_f" : W_r_f0,
        # "W_r_a" : W_r_a0,
        # "W_r_full" : W_r_full0,
        # "W_v" : W_v0,
        # "W_p" : W_p0,
        # "W_m" : W_m0,
        # "U_vh" : U_vh0,
        # "b_vh" : b_vh0
        "W_h1_f" : W_h1_f0,
        "W_h1_hh" : W_h1_hh0,
        "W_p_f" : W_p_f0,
        "W_p_hh" : W_p_hh0,
        "W_m_f" : W_m_f0,
        "W_m_hh" : W_m_hh0,
        "W_v_f" : W_v_f0,
        "W_v_hh" : W_v_hh0,
        "U_f" : U_f0,
        "U_hh" : U_hh0,
        "b_f" : b_f0,
        "b_hh" : b_hh0,        
        "W_ap" : W_ap0,
        "W_full" : W_full0,
    },
    'r_weights' : {

    }
}
# opt_state,p_weights
###
_,(*_,opt_state,p_weights) = load_('/sc_project/test_data/forward_new_v10_81M_144N_12_10-023341.pkl') # '/sc_project/test_data/forward_new_v10_81M_144N_12_10-130123.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_11_10-234704.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_11_10-203208.pkl') #'/sc_project/test_data/forward_new_v8_81M_144N_04_10-233819.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_08_10-044429.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_07_10-212430.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_07_10-165129.pkl') #'/sc_project/test_data/forward_new_training_v10_07_10-165129.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_05_10-070151.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_04_10-203804.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_04_10-050644.pkl') # '/sc_project/test_data/forward_new_v10_81M_144N_03_10-183817.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_03_10-213835.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_03_10-175825.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_03_10-042039.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_03_10-035421.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_02_10-195857.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_02_10-053656.pkl') #'/sc_project/test_data/forward_new_v10_81M_144N_01_10-221031.pkl') # '/sc_project/test_data/forward_new_v10_81M_144N_01_10-182122.pkl') # /sc_project/test_data/forward_new_v10_81M_144N_30_09-051139.pkl /sc_project/test_data/forward_new_v10_81M_144N_23_09-004216.pkl, /sc_project/test_data/forward_new_v10_81M_144N_22_09-071717.pkl') #'/sc_project/test_data/forward_new_v6_81M_144N_22_08-082725.pkl') # ...14-06...,opt_state'/sc_project/pkl/forward_v9_225_13_06-0014.pkl','/pkl/forward_v8M_08_06-1857.pkl'
weights["p_weights"] = p_weights
###
# p_weights["W_p"] = W_p0
# p_weights["W_m"] = W_m0
#
# full_loop(); opt_state init; load/weights['s'] = weights_s; full_loop() call;
startTime = datetime.now()
arrs,aux = forward_model_loop(SC,weights,params,INIT_0,opt_state) ###
(loss_arr,loss_std,loss_v_arr,v_std_arr,loss_c_arr,c_std_arr) = arrs
(v_pred_arr,v_t_arr,loss_v_arr_,loss_c_arr_,pos_plan_arr,pos_arr,dot_arr,binary_arr,action_arr,opt_state,p_weights) = aux
print('binary_arr=',binary_arr.shape)

# plot training loss
print("Training time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
print("Time finished:",datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
plt.figure(figsize=(12,6))
title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, init={INIT:.2f}, update={LR:.6f}, WD={WD:.5f}, TOT_STEPS={TOT_STEPS}, MAX_PLAN_LENGTH={MAX_PLAN_LENGTH:.2f}, MAX_DOT_SPEED={MAX_DOT_SPEED:.2f} \n N_ARRAY={N_ARRAY}, PLAN_RATIO={PLAN_RATIO}, APERTURE={APERTURE:.2f}, PLAN_SPACE={PLAN_SPACE:.2f}, ACTION_SPACE={ACTION_SPACE:.2f}, SIGMA_A={SIGMA_A:.1f}, SIGMA_N={SIGMA_N:.1f} NEURONS_FULL={NEURONS_FULL**2}, MODULES={M}, H={H}'
fig,ax = plt.subplots(1,3,figsize=(13,6))
plt.suptitle('forward_new_training_v10, '+title__,fontsize=10)
plt.subplot(1,3,1)
plt.errorbar(jnp.arange(TOT_EPOCHS),loss_arr,yerr=loss_std,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Total loss',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.subplot(1,3,2)
plt.errorbar(jnp.arange(TOT_EPOCHS),loss_v_arr,yerr=v_std_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Visual model loss',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.subplot(1,3,3)
plt.errorbar(jnp.arange(TOT_EPOCHS),loss_c_arr,yerr=c_std_arr,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
plt.xscale('log')
# plt.yscale('log')
plt.ylabel(r'cosine similarity',fontsize=15)
plt.xlabel(r'Iteration',fontsize=15)
plt.tight_layout()
plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_ + 'forward_new_training_v10_' + dt + '.png')

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
# plt.suptitle('forward_new_training_v8, '+title__,fontsize=10)
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
# plt.savefig(path_ + 'figs_forward_new_training_v8_' + dt + '.png')

save_pkl((arrs,aux),'forward_new_v10_'+str(M)+'M_'+str(N_F)+'N') # (no rel_vec / loss_v / loss_d)