# -*- coding: utf-8 -*-
# simple test using constant policy
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
	dt = datetime.now().strftime("%d_%m-%H_S%M")
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

@jit
def neuron_act(COLORS,THETA,SIGMA_A,dots,pos): #e_t_1,th_j,th_i,SIGMA_A,COLORS # COLORS,THETA,SIGMA_A
    # COLORS = params["COLORS"]
    # THETA = params["THETA"]
    # SIGMA_A = params["SIGMA_A"]
    D_ = COLORS.shape[0]
    N_ = THETA.size
    th_x = jnp.tile(THETA,(N_,1)).reshape((N_**2,))
    th_y = jnp.tile(jnp.flip(THETA).reshape(N_,1),(1,N_)).reshape((N_**2,))
    G_0 = jnp.vstack([th_x,th_y])
    G = jnp.tile(G_0.reshape(2,N_**2,1),(1,1,D_))
    C = (COLORS/255).transpose((1,0))
    E = G.transpose((1,0,2)) - ((dots-pos).T) #.reshape((2,1))
    act = jnp.exp((jnp.cos(E[:,0,:]) + jnp.cos(E[:,1,:]) - 2)/SIGMA_A**2).T #.reshape((D_,N_**2))
    act_r,act_g,act_b = jnp.matmul(C,act) #.reshape((3*N_**2,))
    act_rgb = jnp.concatenate((act_r,act_g,act_b))
    return act_rgb

def mod_(val):
    return (val+jnp.pi)%(2*jnp.pi)-jnp.pi

@jit
def loss_obj(dots,sel,pos,SIGMA_R): # R_t
    dis = dots - pos
    obj = jnp.exp((jnp.cos(dis[:,0]) + jnp.cos(dis[:,1]) - 2)/SIGMA_R**2) ### (positive), standard loss (-) (1/(sigma_e*jnp.sqrt(2*jnp.pi)))*
    R_obj = jnp.dot(obj,sel)
    return R_obj #sigma_e

# @jit
def sample_action(r,ind,params):
    key = rnd.PRNGKey(ind) ##
    p_ = 1/(1+jnp.exp(-(r-params["SIGMOID_MEAN"])/params["SIGMA_S"])) # (shifted sigmoid)
    return jnp.int32(rnd.bernoulli(key,p=p_)) # 1=action, 0=plan

@jit
def sample_vec_rand(SC,ind):
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

def new_params(params, e): # Modify in place
    VMAPS = params["VMAPS"]
    H_S = params["H_S"]
    H_R = params["H_R"]
    H_P = params["H_P"]
    APERTURE = params["APERTURE"]
    MODULES = params["MODULES"]
    N = params["N"]
    M = params["M"]
    T = params["TRIAL_LENGTH"]
    N_DOTS = params["N_DOTS"]
    ki = rnd.split(rnd.PRNGKey(e), num=10)
    params["HS_0"] = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[0],(VMAPS,H_S))
    params["HR_0"] = jnp.zeros((VMAPS,H_R))
    params["HP_0"] = jnp.sqrt(INIT_P/(H_P))*rnd.normal(ki[1],(VMAPS,H_P))
    params["POS_0"] = rnd.choice(ke[2],jnp.arange(-APERTURE,APERTURE,0.01),(VMAPS,2))
    params["DOTS"] = gen_dots(ki[3],VMAPS,N_DOTS,APERTURE)
    params["SELECT"] = jnp.eye(N_DOTS)[rnd.choice(ki[4],N_DOTS,(VMAPS,))]
    params["IND"] = rnd.randint(ki[5],(VMAPS,T),minval=0,maxval=M,dtype=jnp.int32)

# @partial(jax.jit,static_argnums=(1,))
def kl_loss(policy,consts):
    *_,PRIOR_STAT,PRIOR_PLAN,_,_ = consts
    len_ = len(policy[0])
    val_ = (1-PRIOR_STAT)/(len_-1)
    vec_prior = jnp.full(len_,val_).at[len_//2].set(PRIOR_STAT)
    vec_prior = jnp.full(len_,1/len_)
    vec_kl = optax.kl_divergence(jnp.log(policy[0]),vec_prior)
    act_prior = jnp.array([PRIOR_PLAN,1-PRIOR_PLAN])
    act_kl = optax.kl_divergence(jnp.log(policy[1]),act_prior)
    return vec_kl,act_kl

@jit
def get_policy(args_t,weights_s): # hs_t_1,v_t,r_t,rp_t_1,rm_t_1,weights_s,params
    (hs_t_1,*_,rp_t_1,rm_t_1,v_t,r_t,r_tot) = args_t
    Ws_vt_z = weights_s["Ws_vt_z"]
    Ws_vt_f = weights_s["Ws_vt_f"]
    Ws_vt_h = weights_s["Ws_vt_h"]
    Ws_rt_z = weights_s["Ws_rt_z"]
    Ws_rt_f = weights_s["Ws_rt_f"]
    Ws_rt_h = weights_s["Ws_rt_h"]
    Ws_pt_1z = weights_s["Ws_pt_1z"]
    Ws_pt_1f = weights_s["Ws_pt_1f"]
    Ws_pt_1h = weights_s["Ws_pt_1h"]
    Ws_mt_1z = weights_s["Ws_mt_1z"]
    Ws_mt_1f = weights_s["Ws_mt_1f"]
    Ws_mt_1h = weights_s["Ws_mt_1h"]
    Us_z = weights_s["Us_z"]
    Us_f = weights_s["Us_f"]
    Us_h = weights_s["Us_h"]
    bs_z = weights_s["bs_z"]
    bs_f = weights_s["bs_f"]
    bs_h = weights_s["bs_h"]
    Ws_vec = weights_s["Ws_vec"]
    Ws_act = weights_s["Ws_act"]
    Ws_val = weights_s["Ws_val"]
    z_t = jax.nn.sigmoid(jnp.matmul(Ws_vt_z,v_t) + Ws_rt_z*r_t + Ws_pt_1z*rp_t_1 + Ws_mt_1z*rm_t_1 + jnp.matmul(Us_z,hs_t_1) + bs_z)
    f_t = jax.nn.sigmoid(jnp.matmul(Ws_vt_f,v_t) + Ws_rt_f*r_t + Ws_pt_1f*rp_t_1 + Ws_mt_1f*rm_t_1 + jnp.matmul(Us_f,hs_t_1) + bs_f)
    hhat_t = jax.nn.tanh(jnp.matmul(Ws_vt_h,v_t) + Ws_rt_h*r_t + Ws_pt_1h*rp_t_1 + Ws_mt_1h*rm_t_1 + jnp.matmul(Us_h,jnp.multiply(f_t,hs_t_1)) + bs_h)
    hs_t = jnp.multiply(1-z_t,hs_t_1) + jnp.multiply(z_t,hhat_t) #
    vec_t = jnp.matmul(Ws_vec,hs_t)
    act_t = jnp.matmul(Ws_act,hs_t)
    val_t = jnp.matmul(Ws_val,hs_t)
    # jax.debug.print('vec_t={}',vec_t)
    # jax.debug.print('act_t={}',act_t)
    vectors_t = jax.nn.softmax(vec_t)
    actions_t = jax.nn.softmax(act_t)
    return (vectors_t,actions_t),val_t,hs_t

def sample_policy(policy,SC,ind):
    vectors,actions = policy
    ID_ARR,VEC_ARR,H1VEC_ARR = SC
    keys = rnd.split(rnd.PRNGKey(ind),num=2)
    vec_ind = rnd.choice(key=keys[0],a=jnp.arange(len(vectors)),p=vectors)
    h1vec = H1VEC_ARR[:,vec_ind]
    vec = VEC_ARR[:,vec_ind]
    act_ind = rnd.choice(key=keys[1],a=jnp.arange(len(actions)),p=actions)
    act = jnp.eye(len(actions))[act_ind]
    return h1vec,vec,jnp.log(vectors[vec_ind]),act[0],act[1],jnp.log(actions[act_ind])

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

@jit
def v_predict(h1vec_t,v_t_1,hv_t_1,p_weights,NONE_PLAN): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    def scan_fnc(carry,x):
        hv_t_1,h1vec_t,v_t_1,W_h1,W_v,U_vh,b_vh = carry
        hv_t = jax.nn.sigmoid(jnp.matmul(W_h1,h1vec_t) + jnp.matmul(W_v,v_t_1) + jnp.matmul(U_vh,hv_t_1) + b_vh)
        return (hv_t,h1vec_t,v_t_1,W_h1,W_v,U_vh,b_vh),None
    W_h1 = p_weights["W_h1"]
    W_v = p_weights["W_v"]
    U_vh = p_weights["U_vh"]
    b_vh = p_weights["b_vh"]
    W_r = p_weights["W_r"]
    (hv_t,*_),_ = jax.lax.scan(scan_fnc,(hv_t_1,h1vec_t,v_t_1,W_h1,W_v,U_vh,b_vh),NONE_PLAN) # 10### CHANGE - MAKE EQUAL TO SHAPE OF None INPUT ARR
    v_t = jnp.matmul(W_r,hv_t)
    return v_t,hv_t

def plan(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,weights,NONE_PLAN):#,pos_t_1,dots,vec_t,sel(vec_t,r_t_1,hr_t_1,v_t_1,hv_t_1,weights,params):
    pos_t = pos_t_1 + vec_t
    v_t,hv_t = v_predict(h1vec_t,v_t_1,hv_t_1,weights["p"],NONE_PLAN)### #(hr_t_1,v_t_1,pos_t_1,weights["v"],params)
    r_t,hr_t = r_predict(v_t,v_t_1,r_t_1,hr_t_1,weights["r"])
    return r_t,hr_t,v_t,pos_t # ,r_tp,v_tp #,hv_t # predictions

def move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,consts): # shouldnt be random; should take action dictated by plan...
    INIT_LENGTH,TEST_LENGTH,NONE_PLAN,SIGMA_R,COLORS,THETA,SIGMA_A,PRIOR_STAT,PRIOR_PLAN,C_MOVE,C_PLAN = consts
    pos_t = pos_t_1 + vec_t
    v_t = neuron_act(COLORS,THETA,SIGMA_A,dots,pos_t)
    r_t = loss_obj(dots,sel,pos_t,SIGMA_R)###
    vhat_t_,_ = v_predict(h1vec_t,v_t_1,hv_t_1,weights["p"],NONE_PLAN)###
    rhat_t_,hr_t = r_predict(vhat_t_,v_t_1,r_t_1,hr_t_1,weights["r"])
    return r_t,hr_t,v_t,pos_t #,rhat_t_,vhat_t_ # hv_t, true

# control flow fncs
def dynamic_scan(carry_0):
    @jit
    def scan_body(carry_t_1,x):
        t,args_t_1,theta = carry_t_1
        (hs_t_1,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t_1,rm_t_1,v_t_1,r_t_1,r_tot_1) = args_t_1
        (SC,weights,weights_s,consts) = theta
        policy_t,val_t,hs_t = get_policy(args_t_1,weights_s) # hs_t_1,v_t_1,r_t_1,rp_t_1,rm_t_1,weights_s,params
        # vec_kl = -jnp.dot(policy_t[0],jnp.log(policy_t[0]))
        vec_kl,act_kl = kl_loss(policy_t,consts) #-jnp.dot(policy_t[1],jnp.log(policy_t[1]))
        h1vec_t,vec_t,lp_vec,rp_t,rm_t,lp_rpm = sample_policy(policy_t,SC,ind[t])
        lp_t = lp_vec + lp_rpm # lp_arr = lp_arr.at[t].set(lp_vec+lp_rpm) # lp_arr[t] = lp_vec+lp_rpm # lp_arr = lp_arr.at[t].set(lp_vec+lp_rpm) # jnp.append(arrs[0],lp_vec+lp_rpm)
        args_t = (hs_t,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t,rm_t,v_t_1,r_t_1,r_tot_1) # update rp/rm
        carry_args = (h1vec_t,vec_t,t,args_t,lp_t,theta,vec_kl,act_kl,val_t) # (lp_arr,r_arr,sample_arr) assemble carry with sampled vecs and updated args
        t,args_t,arrs_t = jax.lax.cond(rp_t == 1,plan_fnc,move_fnc,(carry_args))###DEBUG;CHANGE
        # t,args_t,arrs_t = jax.lax.cond(t < params["TEST_LENGTH"],action_fnc,no_op_fnc,(carry_args)) ### CHANGE 300
        return (t,args_t,theta),arrs_t
    def plan_fnc(carry_args):
        (h1vec_t,vec_t,t,args_t,lp_t,theta,vec_kl,act_kl,val_t) = carry_args
        (hs_t,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t,rm_t,v_t_1,r_t_1,r_tot_1) = args_t
        (SC,weights,weights_s,consts) = theta
        _,_,NONE_PLAN,*_,C_PLAN = consts
        r_t,_,v_t,pos_t = plan(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,weights,NONE_PLAN)
        r_tot = 0 - C_PLAN ### # r_t
        t += 1 # params["T_PLAN"]
        args_t = (hs_t,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t,rm_t,v_t,r_t,r_tot) # update v,r;dont update h
        return (t,args_t,jnp.concatenate([jnp.array([lp_t]),jnp.array([r_tot]),jnp.array([r_t]),val_t,jnp.array([1]),jnp.array([vec_kl]),jnp.array([act_kl]),pos_t]))
    def move_fnc(carry_args):
        (h1vec_t,vec_t,t,args_t,lp_t,theta,vec_kl,act_kl,val_t) = carry_args
        (hs_t,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t,rm_t,v_t_1,r_t_1,r_tot_1) = args_t
        (SC,weights,weights_s,consts) = theta
        *_,C_MOVE,_ = consts
        r_t,hr_t,v_t,pos_t = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,consts)
        r_tot = r_t - C_MOVE ###
        t += 1 # params["T_MOVE"]
        args_t = (hs_t,hr_t,hv_t_1,pos_t,dots,sel,ind,rp_t,rm_t,v_t,r_t,r_tot) # update v,r,hr,pos
        return (t,args_t,jnp.concatenate([jnp.array([lp_t]),jnp.array([r_tot]),jnp.array([r_t]),val_t,jnp.array([0]),jnp.array([vec_kl]),jnp.array([act_kl]),pos_t]))
        
    *_,(*_,(_,TEST_LENGTH,*_)) = carry_0
    (t,args_final,_),arrs_stack = jax.lax.scan(scan_body,carry_0,None,TEST_LENGTH)### # CHANGE 210
    scan_body._clear_cache()
    return t,args_final,arrs_stack

def init_scan(carry_0):
    def scan_body(carry_t_1,x):
        t,args_t_1,theta = carry_t_1
        (hs_t_1,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t_1,rm_t_1,v_t_1,r_t_1,r_tot_1) = args_t_1
        (SC,weights,weights_s,consts) = theta
        policy_t,val_t,hs_t = get_policy(args_t_1,weights_s) # hs_t_1,v_t_1,r_t_1,rp_t_1,rm_t_1,weights_s,params
        h1vec_t,vec_t = sample_vec_rand(SC,ind[t])
        r_t,hr_t,v_t,pos_t = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,consts)
        t += 1 # params["T_MOVE"]
        args_t = (hs_t,hr_t,hv_t_1,pos_t,dots,sel,ind,rp_t_1,rm_t_1,v_t,r_t,r_tot_1) 
        return (t,args_t,theta),jnp.concatenate([jnp.array([r_t]),pos_t]) # jnp.array([val_t]) , arrs_t

    *_,(*_,(INIT_LENGTH,*_)) = carry_0
    (t,args_init,_),rpos_stack = jax.lax.scan(scan_body,carry_0,None,INIT_LENGTH)### # arrs_stack
    return t,args_init,rpos_stack # ,arrs_stack

def body_fnc(SC,hs_t_1,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,weights,weights_s,params): # PLAN_ITS
    t = 0
    rp_t_1 = 0
    rm_t_1 = 1
    consts = params["INIT_LENGTH"],params["TEST_LENGTH"],params["NONE_PLAN"],params["SIGMA_R"],params["COLORS"],params["THETA"],params["SIGMA_A"],params["PRIOR_STAT"],params["PRIOR_PLAN"],params["C_MOVE"],params["C_PLAN"]
    v_t = neuron_act(params["COLORS"],params["THETA"],params["SIGMA_A"],dots,pos_t_1)
    r_t = loss_obj(dots,sel,pos_t_1,params["SIGMA_R"])###
    args_0 = (hs_t_1,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t_1,rm_t_1,v_t,r_t,0) # r_tot_1 = 0
    theta = (SC,weights,weights_s,consts)
    t,args_init,rpos_init_arr = init_scan((t,args_0,theta))
    r_init_arr,pos_init_arr = rpos_init_arr[:,0],rpos_init_arr[:,1:]
    t,args_final,arrs_stack = dynamic_scan((0,args_init,theta)) # arrs_0
    lp_arr,r_arr,rt_arr,val_arr,sample_arr,vec_kl_arr,act_kl_arr = (arrs_stack[:,i] for i in range(7)) #,t_arr
    pos_arr = arrs_stack[:,7:]
    return r_init_arr,pos_init_arr,lp_arr,r_arr,rt_arr,val_arr,sample_arr,vec_kl_arr,act_kl_arr,pos_arr #,t_arr #

def pg_obj(SC,hs_0,hr_0,hp_0,pos_0,dots,sel,ind,weights,weights_s,params):
    body_fnc_vmap = jax.vmap(body_fnc,in_axes=(None,0,0,0,0,0,0,0,None,None,None),out_axes=(1,1,1,1,1,1,1,1,1,1))
    r_init_arr,pos_init_arr,lp_arr,r_arr,rt_arr,val_arr,sample_arr,vec_kl_arr,act_kl_arr,pos_arr = body_fnc_vmap(SC,hs_0,hr_0,hp_0,pos_0,dots,sel,ind,weights,weights_s,params) # ,params["PLAN_ITS"] [TEST_LENGTH,VMAPS] arrs_(lp,r),aux
    r_to_go = jnp.cumsum(r_arr.T[:,::-1],axis=1)[:,::-1] # reward_to_go(r_arr.T)
    adv_arr = r_to_go - val_arr.T
    adv_norm = (adv_arr-jnp.mean(adv_arr,axis=None))/jnp.std(adv_arr,axis=None) ###
    actor_loss_arr = -(jnp.multiply(lp_arr.T,adv_norm)) # negative for adam
    actor_loss = jnp.mean(jnp.sum(actor_loss_arr,axis=1))
    std_actor = jnp.std(actor_loss_arr)
    critic_loss = jnp.mean(0.5*jnp.square(adv_arr),axis=None)
    std_critic = jnp.std(0.5*jnp.square(adv_arr),axis=None)
    vec_kl_loss = jnp.mean(vec_kl_arr,axis=None)
    std_vec_kl = jnp.std(vec_kl_arr,axis=None)
    act_kl_loss = jnp.mean(act_kl_arr,axis=None)
    std_act_kl = jnp.std(act_kl_arr,axis=None)
    tot_loss = actor_loss + params["LAMBDA_CRITIC"]*critic_loss + params["LAMBDA_VEC_KL"]*vec_kl_loss #+ params["LAMBDA_ACT_KL"]*act_kl_loss
    std_loss = (std_actor**2+(params["LAMBDA_CRITIC"]**2)*std_critic**2+(params["LAMBDA_VEC_KL"]**2)*(std_vec_kl**2+std_act_kl**2))**0.5
    sem_loss = std_loss/(params["VMAPS"]**0.5)
    r_tot = jnp.mean(r_to_go[:,0])
    std_r = jnp.std(r_to_go[:,0])
    plan_rate = jnp.mean(sample_arr)
    std_plan_rate = jnp.std(sample_arr)
    losses = (actor_loss,critic_loss,vec_kl_loss,act_kl_loss,r_tot,plan_rate)
    stds = (sem_loss,std_actor,std_critic,std_act_kl,std_vec_kl,std_r,std_plan_rate)
    other = (r_init_arr.T,r_arr.T,rt_arr.T,sample_arr.T,pos_init_arr.transpose([1,0,2]),pos_arr.transpose([1,0,2]))
    return tot_loss,(losses,stds,other)
    # tot_loss,(actor_loss,std_actor,critic_loss,std_critic,avg_vec_kl,std_vec_kl,avg_act_kl,std_act_kl,r_std,l_sem,plan_rate,avg_tot_r,kl_loss,r_init_arr.T,r_arr.T,rt_arr.T,sample_arr.T,pos_init_arr.transpose([1,0,2]),pos_arr.transpose([1,0,2])) # ,t_arr.T

def full_loop(SC,weights,params):
    # r_arr,sample_arr = (jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"])) for _ in range(2)) # ,params["VMAPS"]
    # pos_arr = jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"],2)) # ,params["VMAPS"]
    # dots_arr = jnp.zeros((params["TOT_EPOCHS"],3,2))
    loss_arr,sem_loss_arr,actor_loss_arr,std_actor_arr,critic_loss_arr,std_critic_arr,vec_kl_arr,std_vec_kl_arr,act_kl_arr,std_act_kl_arr,r_tot_arr,std_r_arr,plan_rate_arr,std_plan_rate_arr = (jnp.zeros((params["TOT_EPOCHS"],)) for _ in range(14)) #jnp.zeros((params["TOT_EPOCHS"]))
    weights_s = weights["s"]
    E = params["TOT_EPOCHS"]
    # optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    optimizer = optax.chain(
    optax.clip_by_global_norm(params["GRAD_CLIP"]),  # Add gradient clipping here
    optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"]),
    )
    opt_state = optimizer.init(weights_s)
    for e in range(E):
        new_params(params,e)
        hs_0 = params["HS_0"]
        hr_0 = params["HR_0"]
        hp_0 = params["HP_0"]
        pos_0 = params["POS_0"]
        dots = params["DOTS"]
        sel = params["SELECT"]
        ind = params["IND"]
        loss_grad = jax.value_and_grad(pg_obj,argnums=9,has_aux=True)
        (loss,(losses,stds,other)),grads = loss_grad(SC,hs_0,hr_0,hp_0,pos_0,dots,sel,ind,weights,weights_s,params) # (loss,aux),grads
        leaves,_ = jax.tree_util.tree_flatten(grads)
        # jax.debug.print("all_grads={}",grads)
        # jax.debug.print("max_grad={}",jnp.max(jnp.concatenate(jnp.array(leaves)),axis=None))
        (actor_loss,critic_loss,vec_kl_loss,act_kl_loss,r_tot,plan_rate) = losses
        (sem_loss,std_actor,std_critic,std_act_kl,std_vec_kl,std_r,std_plan_rate) = stds
        (r_init_arr,r_arr,rt_arr,sample_arr,pos_init_arr,pos_arr) = other
        opt_update,opt_state = optimizer.update(grads,opt_state,weights_s)
        weights_s = optax.apply_updates(weights_s,opt_update)
        loss_arr = loss_arr.at[e].set(loss)
        sem_loss_arr = sem_loss_arr.at[e].set(sem_loss)
        actor_loss_arr = actor_loss_arr.at[e].set(actor_loss)
        std_actor_arr = std_actor_arr.at[e].set(std_actor)
        critic_loss_arr = critic_loss_arr.at[e].set(critic_loss)
        std_critic_arr = std_critic_arr.at[e].set(std_critic)
        vec_kl_arr = vec_kl_arr.at[e].set(vec_kl_loss)
        std_vec_kl_arr = std_vec_kl_arr.at[e].set(std_vec_kl)
        act_kl_arr = act_kl_arr.at[e].set(act_kl_loss)
        std_act_kl_arr = std_act_kl_arr.at[e].set(std_act_kl)
        r_tot_arr = r_tot_arr.at[e].set(r_tot)
        std_r_arr = std_r_arr.at[e].set(std_r)
        plan_rate_arr = plan_rate_arr.at[e].set(plan_rate)
        std_plan_rate_arr = std_plan_rate_arr.at[e].set(std_plan_rate)
        # pos_init_arr = pos_init_arr.at[e].set(pos_init_arr)
        # pos_arr = pos_arr.at[e].set(pos_arr)
        # r_arr = r_arr.at[e,:].set(r_arr_) # r_std?
        # pos_arr = pos_arr.at[e,:,:].set(pos_arr_)
        # sample_arr = sample_arr.at[e,:].set(sample_arr_)
        # dots_arr = dots_arr.at[e,:,:].set(dots)
        print("e=",e,"r=",r_tot,"r_std=",std_r,"loss=",loss,"sem_loss=",sem_loss)
        print("actor_loss=",actor_loss,"std_actor=",std_actor,"critic_loss=",critic_loss,"std_critic=",std_critic)
        print("vec_kl=",vec_kl_loss,"std_vec=",std_vec_kl,"act_kl=",act_kl_loss,"std_act=",std_act_kl,"plan=",plan_rate,'\n')
        if e == E-1:
            pass
        ### loss.block_until_ready()
        ### jax.profiler.save_device_memory_profile(f"memory{e}.prof")
        if e>0 and (e % 50) == 0:
            print("*clearing cache*")
            jax.clear_caches()
    losses = (loss_arr,actor_loss_arr,critic_loss_arr,vec_kl_arr,act_kl_arr,r_tot_arr,plan_rate_arr)
    stds = (sem_loss_arr,std_actor_arr,std_critic_arr,std_act_kl_arr,std_vec_kl_arr,std_r_arr,std_plan_rate_arr)
    other = (r_init_arr,r_arr,rt_arr,sample_arr,pos_init_arr,pos_arr,dots,sel)
    return losses,stds,other,weights #r_arr,pos_arr,sample_arr,dots_arr

# hyperparams ###
TOT_EPOCHS = 15000 ## 1000
# EPOCHS = 1
PLOTS = 5
# LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 1500 ## 2000,500,1100,1000,800,500
LR = 0.003 # 0.001,0.0008,0.0005,0.001,0.000001,0.0001
WD = 0.0001 # 0.0001
GRAD_CLIP = 0.5 #0.5 1.0
INIT_S = 2
INIT_P = 2 # 0.5
INIT_R = 3 # 5,2
H_S = 100
H_P = 500 # 500,300
H_R = 100
PLAN_ITS = 10
NONE_PLAN = jnp.zeros((PLAN_ITS,)) #[None] * PLAN_ITS
TRIAL_LENGTH = 90 ## 90 120 100
INIT_LENGTH = 30
TEST_LENGTH = TRIAL_LENGTH - INIT_LENGTH

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
C_MOVE = 0.30 #0.28 0.30
C_PLAN = 0.0 # 0.05 # 0.0
PRIOR_STAT = 0.2 # 1 # 0.2
PRIOR_PLAN = 0.2
MODULES = 15 # 10,8
M = MODULES**2
NEURONS = 10
N = 3*(NEURONS**2)
SIGMA_R = 1 # 1.2,1.5,1
SIGMA_A = 0.5 # 0.5
SIGMA_S = 0.1
LAMBDA_CRITIC = 0.1 # 0.01
LAMBDA_VEC_KL = 0.5 #0.5
LAMBDA_ACT_KL = 0.5
SIGMOID_MEAN = 0.5
APERTURE = jnp.pi
THETA = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]]) # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = COLORS.shape[0]
HS_0 = None
HP_0 = None #jnp.sqrt(INIT/(H_S))*rnd.normal(ke[4],(EPOCHS,VMAPS,H_S)) # [E,V,H_S]
HR_0 = None #jnp.sqrt(INIT/(H_S))*rnd.normal(ke[4],(EPOCHS,VMAPS,H_S)) # [E,V,H_S]
POS_0 = None #rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
DOTS = None #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
SELECT = None #jnp.eye(N_DOTS)[rnd.choice(ke[1],N_DOTS,(EPOCHS,VMAPS))]
IND = None
ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True)
VEC_ARR = gen_vectors(MODULES,APERTURE)
H1VEC_ARR = jnp.diag(jnp.ones(M))[:,ID_ARR]
SC = (ID_ARR,VEC_ARR,H1VEC_ARR)

# INITIALIZATION
ki = rnd.split(rnd.PRNGKey(1),num=20)
Ws_vt_z0 = jnp.sqrt(INIT_S/(H_S+N))*rnd.normal(ki[0],(H_S,N))
Ws_vt_f0 = jnp.sqrt(INIT_S/(H_S+N))*rnd.normal(ki[1],(H_S,N))
Ws_vt_h0 = jnp.sqrt(INIT_S/(H_S+N))*rnd.normal(ki[2],(H_S,N))
Ws_rt_z0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[3],(H_S,))
Ws_rt_f0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[4],(H_S,))
Ws_rt_h0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[5],(H_S,))
Ws_pt_1z0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[6],(H_S,))
Ws_pt_1f0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[7],(H_S,))
Ws_pt_1h0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[8],(H_S,))
Ws_mt_1z0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[9],(H_S,))
Ws_mt_1f0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[10],(H_S,))
Ws_mt_1h0 = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[11],(H_S,))
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
    # "EPOCHS" : EPOCHS,
    # "EPOCHS_SWITCH" : EPOCHS_SWITCH,
    # "LOOPS" : LOOPS,
    "MODULES" : MODULES,
    "NEURONS" : NEURONS,
    "APERTURE" : APERTURE,
    "LR" : LR,
    "WD" : WD,
    "GRAD_CLIP" : GRAD_CLIP,
    "H_S" : H_S,
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
    "THETA" : THETA,
    "COLORS" : COLORS,
    # "SAMPLES" : SAMPLES,
    "POS_0" : POS_0,
    "HP_0" : HP_0,
    "HR_0" : HR_0,
    "IND" : IND,
    "INIT_LENGTH" : INIT_LENGTH,
    "TRIAL_LENGTH" : TRIAL_LENGTH,
    "TEST_LENGTH" : TEST_LENGTH,
    "PLAN_ITS" : PLAN_ITS,
    "NONE_PLAN" : NONE_PLAN,
    "SIGMOID_MEAN" : SIGMOID_MEAN,
    "SIGMA_R" : SIGMA_R,
    "SIGMA_A" : SIGMA_A,
    "SIGMA_S" : SIGMA_S,
    "LAMBDA_CRITIC" : LAMBDA_CRITIC,
    "LAMBDA_VEC_KL" : LAMBDA_VEC_KL,
    "LAMBDA_ACT_KL" : LAMBDA_ACT_KL,
    # "ALPHA" : ALPHA,
    "C_MOVE" : C_MOVE,
    "C_PLAN" : C_PLAN,
    "PRIOR_STAT" : PRIOR_STAT,
    "PRIOR_PLAN" : PRIOR_PLAN,
    }

weights = {
    "s" : { # policy weights
    "Ws_vt_z" : Ws_vt_z0,
    "Ws_vt_f" : Ws_vt_f0,
    "Ws_vt_h" : Ws_vt_h0,
    "Ws_rt_z" : Ws_rt_z0,
    "Ws_rt_f" : Ws_rt_f0,
    "Ws_rt_h" : Ws_rt_h0,
    "Ws_pt_1z" : Ws_pt_1z0,
    "Ws_pt_1f" : Ws_pt_1f0,
    "Ws_pt_1h" : Ws_pt_1h0,
    "Ws_mt_1z" : Ws_mt_1z0,
    "Ws_mt_1f" : Ws_mt_1f0,
    "Ws_mt_1h" : Ws_mt_1h0,
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
    "p" : { # p_weights
    # LOAD IN
    },
    "r" : { # r_weights
    # LOAD IN
    }
}

###
*_,p_weights = load_('/sc_project/pkl_sc/forward_v9_225_17_06-2358.pkl') # /pkl/forward_v8M_08_06-1857.pkl
*_,r_weights = load_('/sc_project/pkl_sc/value_v5_225_13_06-1849.pkl') # sc_project/pkl/value_v5_225_10_06-0722.pkl /homes/lrj34/projects/meta_rl_ego_sim/sc_project/pkl/value_v5_225_09_06-1830.pkl
weights['p'] = p_weights
weights['r'] = r_weights
###
startTime = datetime.now()
losses,stds,other,weights = full_loop(SC,weights,params) # (loss_arr,actor_loss_arr,critic_loss_arr,kl_loss_arr,vec_kl_arr,act_kl_arr,r_std_arr,l_sem_arr,plan_rate_arr,avg_tot_r_arr,avg_pol_kl_arr,r_init_arr,r_arr,rt_arr,sample_arr,pos_init_arr,pos_arr,dots,sel)
print("Sim time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
(loss_arr,actor_loss_arr,critic_loss_arr,vec_kl_arr,act_kl_arr,r_tot_arr,plan_rate_arr) = losses
(sem_loss_arr,std_actor_arr,std_critic_arr,std_act_kl_arr,std_vec_kl_arr,std_r_arr,std_plan_rate_arr) = stds
(r_init_arr,r_arr,rt_arr,sample_arr,pos_init_arr,pos_arr,dots,sel) = other
print('pos_arr=',pos_arr,'pos_arr.shape=',pos_arr.shape)

# plot loss_arr:
#####
legend_handles = [
    plt.Line2D([], [], color='k', label='loss'), # 
    plt.Line2D([], [], color='b', label='plan rate'), # 
]

fig,axes = plt.subplots(2,3,figsize=(12,9))
title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, TEST_LENGTH={TEST_LENGTH}, INIT_LENGTH={INIT_LENGTH}, update={LR:.6f}, WD={WD:.5f} \n C_MOVE={C_MOVE:.2f}, C_PLAN={"N/A"}, L_CRITIC={LAMBDA_CRITIC}, L_VEC_KL={LAMBDA_VEC_KL}, L_ACT_KL={LAMBDA_ACT_KL}, PRIOR_PLAN={PRIOR_PLAN}, PRIOR_STAT={PRIOR_STAT}, GRAD_CLIP={GRAD_CLIP}'
plt.suptitle('outer_loop_pg_training_v6, '+title__,fontsize=10)
axes[0,0].errorbar(np.arange(TOT_EPOCHS),r_tot_arr,yerr=std_r_arr/2,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[0,0].set_xlabel('iteration')
axes[0,0].set_ylabel('R_tot')
axes[0,1].errorbar(np.arange(TOT_EPOCHS),plan_rate_arr,yerr=std_plan_rate_arr/2,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[0,1].set_xlabel('iteration')
axes[0,1].set_ylabel('plan rate')
axes[0,2].errorbar(np.arange(TOT_EPOCHS),loss_arr,yerr=(sem_loss_arr*jnp.sqrt(VMAPS))/2,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[0,2].set_xlabel('iteration')
axes[0,2].set_ylabel('loss')
axes[1,0].errorbar(np.arange(TOT_EPOCHS),actor_loss_arr,yerr=std_actor_arr/2,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[1,0].set_xlabel('iteration')
axes[1,0].set_ylabel('actor loss')
axes[1,1].errorbar(np.arange(TOT_EPOCHS),critic_loss_arr,yerr=std_critic_arr/2,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[1,1].set_xlabel('iteration')
axes[1,1].set_ylabel('critic loss')
line1,*_ = axes[1,2].errorbar(np.arange(TOT_EPOCHS),vec_kl_arr,yerr=std_vec_kl_arr/2,color='blue',ecolor='lightgray',elinewidth=2,capsize=0)
axes[1,2].set_xlabel('iteration')
axes[1,2].set_ylabel('vector kl')
# ax12_2 = axes[1,2].twinx()
# line2,*_ = ax12_2.errorbar(np.arange(TOT_EPOCHS),act_kl_arr,yerr=std_act_kl_arr/2,color='red',ecolor='lightgray',elinewidth=2,capsize=0)
# ax12_2.set_ylabel('action kl')
# plt.legend([line1,line2],['vector kl','action kl'])

for ax in axes.flatten():
    ax.ticklabel_format(style='plain', useOffset=False)

plt.tight_layout()
plt.subplots_adjust(top=0.9)

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_+'outer_loop_pg_training_v6_'+dt+'.png')

# PLOT TESTING DATA
r_init_arr = r_init_arr[-PLOTS:,:] # [PLOTS,TEST_LENGTH]
rt_arr = rt_arr[-PLOTS:,:] # [PLOTS,TEST_LENGTH]
sample_arr = sample_arr[-PLOTS:,:] # [PLOTS,TEST_LENGTH]
pos_init_arr = pos_init_arr[-PLOTS:,:,:] # [PLOTS,TEST_LENGTH,2]
pos_arr = pos_arr[-PLOTS:,:,:] # [PLOTS,TEST_LENGTH,2]
print('new pos_arr=',pos_arr,'new pos_arr.shape=',pos_arr.shape)
dots = dots[-PLOTS:,:,:] # [PLOTS,3,2]
sel = sel[-PLOTS:,:] # [PLOTS,3]

# # Manually define legend
# legend_handles = [
#     plt.Line2D([], [], color='r', marker='x', label='r_t'),
#     plt.Line2D([], [], color='k', marker='o', label='r_hat_t'),
#     plt.Line2D([], [], color='b', label='v_loss'), # marker='.', 
#     plt.Line2D([], [], color='purple', label='r_loss'), # marker='.', 
#     plt.Line2D([], [], color='g', marker='o', label='r_true'), # marker='.',
#     # plt.Line2D([], [], color='orange', marker='o', label='r_tp'), # marker='.',
# ]

colors_ = np.float32([[255,0,0],[0,255,0],[0,0,255]])/255 # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])/255

# plot timeseries of true/planned reward
fig,axis = plt.subplots(2*PLOTS,4,figsize=(15,6*PLOTS)) #9,4x plt.figure(figsize=(12,6))
# title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, INIT_STEPS={INIT_STEPS}, TRIAL_LENGTH={TRIAL_LENGTH} \n SIGMA_R={SIGMA_R:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H_P={H_P}, H_R={H_R}'
plt.suptitle('outer_loop_pg_v6_test_, '+title__,fontsize=10)
for i in range(params["PLOTS"]):
    ax0 = plt.subplot2grid((2*PLOTS,4),(2*i,0),colspan=2,rowspan=1)
    # ax0.set_title('r_t') # v_arr
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.set_ylabel('r_t',fontsize=15)
    ax0.set_xlabel('t',fontsize=15)
    ax0.ticklabel_format(useOffset=False)
    for t in range(INIT_LENGTH):
        ax0.scatter(x=t,y=r_init_arr[i,t],color='k',s=5,marker='x')
    for t in range(TEST_LENGTH):
        if sample_arr[i,t] == 1:
            ax0.scatter(x=t+INIT_LENGTH,y=rt_arr[i,t],color='red',s=5,marker='x')
        else:
            ax0.scatter(x=t+INIT_LENGTH,y=rt_arr[i,t],color='k',s=5,marker='x')
    ax0.axvline(x=INIT_LENGTH,color='k',linestyle='--',linewidth=1)
    ax0.set_ylim([-(C_MOVE+0.1),1.1])
    
    ax1 = plt.subplot2grid((2*PLOTS,4),(2*i,2),colspan=2,rowspan=2)
    # for i in range(M):
    #     ax1.scatter(x=VEC_ARR[i,0],y=VEC_ARR[i,1],color='light grey',s=20,marker='o',label='modules')
    # for t in range(INIT_LENGTH):
    #     ax1.scatter(x=pos_init_arr[i,t,0],y=pos_init_arr[i,t,1],color='black',alpha=0.5,s=60,marker='+')
    for t in range(TEST_LENGTH):
        if sample_arr[i,t] == 1:
            ax1.scatter(x=mod_(pos_arr[i,t,0]),y=mod_(pos_arr[i,t,1]),color='red',alpha=0.4,s=60,marker='o')
        else:
            ax1.scatter(x=mod_(pos_arr[i,t,0]),y=mod_(pos_arr[i,t,1]),color='black',alpha=0.2,s=60,marker='o')
    for d in range(N_DOTS):
        ax1.scatter(x=dots[i,d,0],y=dots[i,d,1],color=colors_[d,:],s=120,marker='x')
    ax1.set_xlim(-jnp.pi,jnp.pi)
    ax1.set_ylim(-jnp.pi,jnp.pi)
    ax1.set_xticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
    ax1.set_xticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=14)
    ax1.set_yticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
    ax1.set_yticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=14)
    ax1.set_aspect('equal')
    ax1.set_title(f'vector heatmap, sel={sel[i,:]}',fontsize=14)

    ax2 = plt.subplot2grid((2*PLOTS,4),(2*i+1,0),colspan=2,rowspan=1)
    ax2.axis('off')
plt.tight_layout()
plt.subplots_adjust(top=0.94)

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

dt = datetime.now().strftime("%d_%m-%H%M%S")
path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
plt.savefig(path_ + 'figs_outer_loop_pg_v6_' + dt + '.png') # ctrl_v7_(v,r,h normal, r_tp changed)

save_pkl(weights,'value_v6_')