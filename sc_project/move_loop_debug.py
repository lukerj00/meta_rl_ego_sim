# admin
# train
# test
# admin (sav; output data)

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
	path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/pkl/' # '/scratch/lrj34/'
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

@jit
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

def mod_(val):
    return (val+jnp.pi)%(2*jnp.pi)-jnp.pi

@jit
def loss_obj(dot,sel,pos,SIGMA_R): # R_t
    dis = dot - pos
    obj = jnp.exp((jnp.cos(dis[:,0]) + jnp.cos(dis[:,1]) - 2)/SIGMA_R**2) ### (positive), standard loss (-) (1/(sigma_e*jnp.sqrt(2*jnp.pi)))*
    R_obj = jnp.dot(obj,sel)
    return R_obj #sigma_e

# @jit
def sample_action(r,ind,params): # change to key ?
    key = rnd.PRNGKey(ind) ##
    p_ = 1/(1+jnp.exp(-(r-params["SIGMOID_MEAN"])/params["SIGMA_S"])) # (shifted sigmoid)
    return jnp.int32(rnd.bernoulli(key,p=p_)) # 1=action, 0=plan

# @partial(jax.jit,static_argnums=(1,2,3))
def gen_dots(key,VMAPS,N_DOTS,APERTURE):
    return rnd.uniform(key,shape=(VMAPS,1,2),minval=-jnp.pi,maxval=jnp.pi) #minval=jnp.array([APERTURE/4,APERTURE/4]),maxval=jnp.array([3*APERTURE/4,3*APERTURE/4]))

def new_params(params,e): # Modify in place
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
    ki = rnd.split(rnd.PRNGKey(e),num=10)
    params["HS_0"] = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[0],(VMAPS,H_S))
    params["HR_0"] = jnp.zeros((VMAPS,H_R))
    params["HP_0"] = jnp.sqrt(INIT_P/(H_P))*rnd.normal(ki[1],(VMAPS,H_P))
    params["POS_0"] = rnd.choice(ke[2],jnp.arange(-APERTURE,APERTURE,0.01),(VMAPS,2))
    params["DOT"] = gen_dots(ki[3],VMAPS,N_DOTS,APERTURE)
    params["DOT_VEC"] = gen_dots(ki[4],VMAPS,N_DOTS,APERTURE)
    params["SELECT"] = jnp.eye(N_DOTS)[rnd.choice(ki[5],N_DOTS,(VMAPS,))]
    params["IND"] = rnd.randint(ki[6],(VMAPS,T),minval=0,maxval=M,dtype=jnp.int32)

# @partial(jax.jit,static_argnums=(1,))
def kl_loss(policy,consts,SC):
    *_,PRIOR_SIGMA,PRIOR_STAT,PRIOR_PLAN,_,_ = consts
    _,VEC_ARR,_ = SC
    len_ = len(policy[0])
    vec_prior_vals = jax.scipy.stats.norm.pdf(jnp.linalg.norm(VEC_ARR,axis=0),loc=0,scale=PRIOR_SIGMA) # jnp.linspace(-3*PRIOR_SIGMA,3*PRIOR_SIGMA,len_))
    vec_prior = vec_prior_vals/jnp.sum(vec_prior_vals)
    jax.debug.print('vec_prior={}',vec_prior)
    jax.debug.print('policy[0]={}',policy[0])
    vec_kl = jnp.dot(policy[0],jnp.log(policy[0]/vec_prior))
    # vec_kl = optax.kl_divergence(jnp.log(vec_prior),policy[0])
    act_prior = jnp.array([PRIOR_PLAN,1-PRIOR_PLAN])
    act_kl = optax.kl_divergence(jnp.log(act_prior),policy[1])
    return vec_kl,act_kl,vec_prior,policy[0]

@jit
def new_dot(dot_t,dot_vec,r_t,ALPHA,BETA,DOT_SPEED,key_t_1):
    def gen_dot(args):
        *_,key_t_1 = args
        key_t,subkey_t = rnd.split(key_t_1,2)
        dot_0 = rnd.uniform(key_t,shape=(1,2),minval=-jnp.pi,maxval=jnp.pi)
        dot_vec = rnd.uniform(subkey_t,shape=(1,2),minval=-jnp.pi,maxval=jnp.pi)
        return dot_0,dot_vec,key_t
    def step_dot(args):
        dot_t,dot_vec,DOT_SPEED,key_t_1 = args
        dot_t += DOT_SPEED*dot_vec
        return dot_t,dot_vec,key_t_1
    p_ = ALPHA*(r_t - BETA*(r_t)**2)
    sample_ = rnd.bernoulli(key_t_1,p=p_)
    dot_t,dot_vec,key_t_1 = jax.lax.cond(sample_==1,gen_dot,step_dot,(dot_t,dot_vec,DOT_SPEED,key_t_1))
    return dot_t,dot_vec,key_t_1

@jit
def get_policy(args_t,weights_s): # hs_t_1,v_t,r_t,rp_t_1,rm_t_1,weights_s,params
    (hs_t_1,*_,rp_t_1,rm_t_1,v_t,r_t,r_tot,key_t_1) = args_t
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

# @jit
# def v_predict(h1vec_t,v_t_1,hv_t_1,p_weights,NONE_PLAN): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
#     def scan_fnc(carry,x):
#         hv_t_1,h1vec_t,v_t_1,W_h1,W_v,U_vh,b_vh = carry
#         hv_t = jax.nn.sigmoid(jnp.matmul(W_h1,h1vec_t) + jnp.matmul(W_v,v_t_1) + jnp.matmul(U_vh,hv_t_1) + b_vh)
#         return (hv_t,h1vec_t,v_t_1,W_h1,W_v,U_vh,b_vh),None
#     W_h1 = p_weights["W_h1"]
#     W_v = p_weights["W_v"]
#     U_vh = p_weights["U_vh"]
#     b_vh = p_weights["b_vh"]
#     W_r = p_weights["W_r"]
#     (hv_t,*_),_ = jax.lax.scan(scan_fnc,(hv_t_1,h1vec_t,v_t_1,W_h1,W_v,U_vh,b_vh),NONE_PLAN) # 10### CHANGE - MAKE EQUAL TO SHAPE OF None INPUT ARR
#     v_t = jnp.matmul(W_r,hv_t)
#     return v_t,hv_t

# def plan_old(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,weights,NONE_PLAN):#,pos_t_1,dots,vec_t,sel(vec_t,r_t_1,hr_t_1,v_t_1,hv_t_1,weights,params):
#     pos_t = pos_t_1 + vec_t
#     v_t,hv_t = v_predict(h1vec_t,v_t_1,hv_t_1,weights["p"],NONE_PLAN)### #(hr_t_1,v_t_1,pos_t_1,weights["v"],params)
#     r_t,hr_t = r_predict(v_t,v_t_1,r_t_1,hr_t_1,weights["r"])
#     return r_t,hr_t,v_t,pos_t # ,r_tp,v_tp #,hv_t # predictions

def plan(vec_t,pos_t_1,dot_t,sel,consts): ### same as move but 0 reward, old pos
    ALPHA,BETA,DOT_SPEED,TEST_LENGTH,NONE_PLAN,SIGMA_R,COLORS,THETA,SIGMA_A,PRIOR_SIGMA,PRIOR_STAT,PRIOR_PLAN,C_MOVE,C_PLAN = consts
    pos_t = pos_t_1 + vec_t
    v_t = neuron_act(COLORS,THETA,SIGMA_A,dot_t,pos_t)
    r_t = loss_obj(dot_t,sel,pos_t,SIGMA_R)###
    return jnp.float32(0),v_t,pos_t_1 #

def move(vec_t,pos_t_1,dot_t,sel,consts): # shouldnt be random; should take action dictated by plan...
    ALPHA,BETA,DOT_SPEED,TEST_LENGTH,NONE_PLAN,SIGMA_R,COLORS,THETA,SIGMA_A,PRIOR_SIGMA,PRIOR_STAT,PRIOR_PLAN,C_MOVE,C_PLAN = consts
    pos_t = pos_t_1 + vec_t
    v_t = neuron_act(COLORS,THETA,SIGMA_A,dot_t,pos_t)
    r_t = loss_obj(dot_t,sel,pos_t,SIGMA_R)###
    return r_t,v_t,pos_t #,rhat_t_,vhat_t_ # hv_t, true

# control flow fncs
def dynamic_scan(carry_0):
    @jit
    def scan_body(carry_t_1,x):
        t,args_t_1,theta = carry_t_1
        (hs_t_1,pos_t_1,dot_t_1,dot_vec,sel,ind,rp_t_1,rm_t_1,v_t_1,r_t_1,r_tot_1,key_t_1) = args_t_1
        (SC,weights_s,consts) = theta
        policy_t,val_t,hs_t = get_policy(args_t_1,weights_s) # hs_t_1,v_t_1,r_t_1,rp_t_1,rm_t_1,weights_s,params
        vec_kl,act_kl,vec_prior,policy_0 = kl_loss(policy_t,consts,SC)
        jax.debug.print('vec_prior={}',vec_prior)
        jax.debug.print('sum_vp={}',jnp.sum(vec_prior))
        jax.debug.print('policy[0]={}',policy_0)
        jax.debug.print('sum_p0={}',jnp.sum(policy_0))
        h1vec_t,vec_t,lp_vec,rp_t,rm_t,lp_rpm = sample_policy(policy_t,SC,ind[t])
        lp_t = lp_vec + lp_rpm # lp_arr = lp_arr.at[t].set(lp_vec+lp_rpm) # lp_arr[t] = lp_vec+lp_rpm # lp_arr = lp_arr.at[t].set(lp_vec+lp_rpm) # jnp.append(arrs[0],lp_vec+lp_rpm)
        args_t = (hs_t,pos_t_1,dot_t_1,dot_vec,sel,ind,rp_t,rm_t,v_t_1,r_t_1,r_tot_1,key_t_1) # update rp/rm
        carry_args = (h1vec_t,vec_t,t,args_t,lp_t,theta,vec_kl,act_kl,val_t,vec_prior,policy_0) # (lp_arr,r_arr,sample_arr) assemble carry with sampled vecs and updated args
        t,args_t,arrs_t = jax.lax.cond(rp_t == 2,plan_fnc,move_fnc,(carry_args))###DEBUG;CHANGE
        return (t,args_t,theta),arrs_t
    def plan_fnc(carry_args):
        (h1vec_t,vec_t,t,args_t,lp_t,theta,vec_kl,act_kl,val_t,vec_prior,policy_0) = carry_args
        (hs_t,pos_t_1,dot_t_1,dot_vec,sel,ind,rp_t,rm_t,v_t_1,r_t_1,r_tot_1,key_t_1) = args_t
        *_,consts = theta
        ALPHA,BETA,DOT_SPEED,*_,C_PLAN = consts
        dot_t,dot_vec,key_t = new_dot(dot_t_1,dot_vec,r_t_1,ALPHA,BETA,DOT_SPEED,key_t_1)
        r_t,v_t,pos_t_1 = plan(vec_t,pos_t_1,dot_t,sel,consts)
        r_tot = r_t - C_PLAN ### r_t
        t += 1 # params["T_PLAN"]
        args_t = (hs_t,pos_t_1,dot_t,dot_vec,sel,ind,rp_t,rm_t,v_t,r_t,r_tot,key_t) # update v,r;dont update h
        return (t,args_t,jnp.concatenate([jnp.array([lp_t]),jnp.array([r_tot]),jnp.array([r_t]),val_t,jnp.array([1]),jnp.array([vec_kl]),jnp.array([act_kl]),pos_t_1,dot_t.reshape(2,),jnp.array([vec_prior]).reshape(225,),jnp.array([policy_0]).reshape(225,)]))
    def move_fnc(carry_args):
        (h1vec_t,vec_t,t,args_t,lp_t,theta,vec_kl,act_kl,val_t,vec_prior,policy_0) = carry_args
        (hs_t,pos_t_1,dot_t_1,dot_vec,sel,ind,rp_t,rm_t,v_t_1,r_t_1,r_tot_1,key_t_1) = args_t
        *_,consts = theta
        ALPHA,BETA,DOT_SPEED,*_,C_MOVE,_ = consts
        dot_t,dot_vec,key_t = new_dot(dot_t_1,dot_vec,r_t_1,ALPHA,BETA,DOT_SPEED,key_t_1)
        r_t,v_t,pos_t = move(vec_t,pos_t_1,dot_t,sel,consts)
        r_tot = r_t - C_MOVE ###
        t += 1 # params["T_MOVE"]
        args_t = (hs_t,pos_t,dot_t,dot_vec,sel,ind,rp_t,rm_t,v_t,r_t,r_tot,key_t) # update v,r,hr,pos
        return (t,args_t,jnp.concatenate([jnp.array([lp_t]),jnp.array([r_tot]),jnp.array([r_t]),val_t,jnp.array([0]),jnp.array([vec_kl]),jnp.array([act_kl]),pos_t,dot_t.reshape(2,),jnp.array([vec_prior]).reshape(225,),jnp.array([policy_0]).reshape(225,)]))
        
    *_,(*_,(_,_,_,TEST_LENGTH,*_)) = carry_0 # t,args_0,theta
    (t,args_final,_),arrs_stack = jax.lax.scan(scan_body,carry_0,None,TEST_LENGTH)### # CHANGE 210
    scan_body._clear_cache()
    return t,args_final,arrs_stack

def body_fnc(SC,hs_0,pos_0,dot_0,dot_vec_0,key_0,sel,ind,weights_s,params): # PLAN_ITS
    rp_0 = 0
    rm_0 = 1
    r_tot_0 = 0
    consts = params["ALPHA"],params["BETA"],params["DOT_SPEED"],params["TEST_LENGTH"],params["NONE_PLAN"],params["SIGMA_R"],params["COLORS"],params["THETA"],params["SIGMA_A"],params["PRIOR_SIGMA"],params["PRIOR_STAT"],params["PRIOR_PLAN"],params["C_MOVE"],params["C_PLAN"]
    v_0 = neuron_act(params["COLORS"],params["THETA"],params["SIGMA_A"],dot_0,pos_0)
    r_0 = loss_obj(dot_0,sel,pos_0,params["SIGMA_R"])###
    args_0 = (hs_0,pos_0,dot_0,dot_vec_0,sel,ind,rp_0,rm_0,v_0,r_0,r_tot_0,key_0)
    theta = (SC,weights_s,consts)
    t,args_final,arrs_stack = dynamic_scan((0,args_0,theta)) # (hs_t_1,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t_1,rm_t_1,v_t_1,r_t_1,r_tot_1)
    lp_arr,r_arr,rt_arr,val_arr,sample_arr,vec_kl_arr,act_kl_arr = (arrs_stack[:,i] for i in range(7)) #,t_arr
    pos_arr = arrs_stack[:,7:9]
    dot_arr = arrs_stack[:,9:11]
    vec_prior_pol = arrs_stack[:,11:]
    return lp_arr,r_arr,rt_arr,val_arr,sample_arr,vec_kl_arr,act_kl_arr,pos_arr,dot_arr,vec_prior_pol #,t_arr #

def pg_obj(SC,hs_0,pos_0,dot_0,dot_vec_0,key_0,sel,ind,weights_s,params):
    body_fnc_vmap = jax.vmap(body_fnc,in_axes=(None,0,0,0,0,None,0,0,None,None),out_axes=(1,1,1,1,1,1,1,1,1,1))
    lp_arr,r_arr,rt_arr,val_arr,sample_arr,vec_kl_arr,act_kl_arr,pos_arr,dot_arr,vec_prior_pol = body_fnc_vmap(SC,hs_0,pos_0,dot_0,dot_vec_0,key_0,sel,ind,weights_s,params) # ,params["PLAN_ITS"] [TEST_LENGTH,VMAPS] arrs_(lp,r),aux
    jax.debug.print('vpp.shape={}',vec_prior_pol.shape)
    jax.debug.print('vecp={}',vec_prior_pol[:,0,:225])
    jax.debug.print('sum_vp={}',jnp.sum(vec_prior_pol[:,0,:225]))
    jax.debug.print('pp={}',vec_prior_pol[:,0,225:])
    jax.debug.print('sum_pp={}',jnp.sum(vec_prior_pol[:,0,225:]))
    # print('vpp=',vec_prior_pol,'shape=',vec_prior_pol.shape)
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
    r_true = jnp.mean(jnp.sum(rt_arr.T,axis=1),axis=0)
    std_r = jnp.std(r_to_go[:,0])
    plan_rate = jnp.mean(sample_arr)
    std_plan_rate = jnp.std(sample_arr)
    losses = (actor_loss,critic_loss,vec_kl_loss,act_kl_loss,r_tot,r_true,plan_rate)
    stds = (sem_loss,std_actor,std_critic,std_act_kl,std_vec_kl,std_r,std_plan_rate)
    other = (r_arr.T,rt_arr.T,sample_arr.T,pos_arr.transpose([1,0,2]),dot_arr.transpose([1,0,2]))
    return tot_loss,(losses,stds,other)

def train_loop(SC,weights,params):
    loss_arr,sem_loss_arr,actor_loss_arr,std_actor_arr,critic_loss_arr,std_critic_arr,vec_kl_arr,std_vec_kl_arr,act_kl_arr,std_act_kl_arr,r_tot_arr,std_r_arr,r_true_arr,plan_rate_arr,std_plan_rate_arr = (jnp.zeros((params["TOT_EPOCHS"],)) for _ in range(15)) #jnp.zeros((params["TOT_EPOCHS"]))
    weights_s = weights["s"]
    E = params["TOT_EPOCHS"]
    optimizer = optax.chain(
    optax.clip_by_global_norm(params["GRAD_CLIP"]),
    optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"]),
    )
    opt_state = optimizer.init(weights_s)
    for e in range(E):
        new_params(params,e)
        hs_0 = params["HS_0"]
        pos_0 = params["POS_0"]
        dot_0 = params["DOT"]
        dot_vec_0 = params["DOT_VEC"]
        sel = params["SELECT"]
        ind = params["IND"]
        key_0 = rnd.PRNGKey(e)
        loss_grad = jax.value_and_grad(pg_obj,argnums=8,has_aux=True)
        (loss,(losses,stds,other)),grads = loss_grad(SC,hs_0,pos_0,dot_0,dot_vec_0,key_0,sel,ind,weights_s,params) # (loss,aux),grads
        # leaves,_ = jax.tree_util.tree_flatten(grads)
        # jax.debug.print("all_grads={}",grads)
        # jax.debug.print("max_grad={}",jnp.max(jnp.concatenate(jnp.array(leaves)),axis=None))
        (actor_loss,critic_loss,vec_kl_loss,act_kl_loss,r_tot,r_true,plan_rate) = losses
        (sem_loss,std_actor,std_critic,std_act_kl,std_vec_kl,std_r,std_plan_rate) = stds
        (r_arr,rt_arr,sample_arr,pos_arr,dot_arr) = other
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
        r_true_arr = r_true_arr.at[e].set(r_true)
        plan_rate_arr = plan_rate_arr.at[e].set(plan_rate)
        std_plan_rate_arr = std_plan_rate_arr.at[e].set(std_plan_rate)
        print("e=",e,"r_tot=",r_tot,"r_true=",r_true,"std_r=",std_r,"loss=",loss,"sem_loss=",sem_loss)
        print("actor_loss=",actor_loss,"std_actor=",std_actor,"critic_loss=",critic_loss,"std_critic=",std_critic)
        print("vec_kl=",vec_kl_loss,"std_vec=",std_vec_kl,"act_kl=",act_kl_loss,"std_act=",std_act_kl,"plan=",plan_rate,'\n')
        if e == E-1:
            pass
        ### loss.block_until_ready()
        ### jax.profiler.save_device_memory_profile(f"memory{e}.prof")
        if e>0 and (e % 50) == 0:
            print("*clearing cache*")
            jax.clear_caches()
    losses = (loss_arr,actor_loss_arr,critic_loss_arr,vec_kl_arr,act_kl_arr,r_tot_arr,r_true_arr,plan_rate_arr)
    stds = (sem_loss_arr,std_actor_arr,std_critic_arr,std_act_kl_arr,std_vec_kl_arr,std_r_arr,std_plan_rate_arr)
    # other = (r_arr,rt_arr,sample_arr,pos_arr,dot_arr,sel)
    return losses,stds,weights_s #r_arr,pos_arr,sample_arr,dots_arr

def test_loop(SC,weights_s,params):
    T = params["TESTS"]
    for e in range(T):
        new_params(params,e)
        hs_0 = params["HS_0"]
        pos_0 = params["POS_0"]
        dot_0 = params["DOT"]
        dot_vec_0 = params["DOT_VEC"]
        sel = params["SELECT"]
        ind = params["IND"]
        key_0 = rnd.PRNGKey(e)
        loss,(losses,stds,other) = pg_obj(SC,hs_0,pos_0,dot_0,dot_vec_0,key_0,sel,ind,weights_s,params)
        (actor_loss,critic_loss,vec_kl_loss,act_kl_loss,r_tot,r_true,plan_rate) = losses
        (sem_loss,std_actor,std_critic,std_act_kl,std_vec_kl,std_r,std_plan_rate) = stds
        (r_arr,rt_arr,sample_arr,pos_arr,dot_arr) = other
    # losses = (loss_arr,actor_loss_arr,critic_loss_arr,vec_kl_arr,act_kl_arr,r_tot_arr,plan_rate_arr)
    # stds = (sem_loss_arr,std_actor_arr,std_critic_arr,std_act_kl_arr,std_vec_kl_arr,std_r_arr,std_plan_rate_arr)
    # other = (r_arr,rt_arr,sample_arr,pos_arr,dot_arr,sel)
    test_data = (r_arr,rt_arr,sample_arr,pos_arr,dot_arr,sel)
    return test_data

def full_loop(SC,weights,params):
    losses,stds,weights_s = train_loop(SC,weights,params)
    test_data = test_loop(SC,weights_s,params)
    return losses,stds,weights_s,test_data

# hyperparams ###
TOT_EPOCHS = 2000 # 2000 ## 1000
TESTS = 1
PLOTS = 5
VMAPS = 1000 ## 2000,500,1100,1000,800,500
LAMBDA_CRITIC = 0.1 # 0.01
LAMBDA_VEC_KL = 0.1 #0.5
LAMBDA_ACT_KL = 0.5
LR = 0.001 # 0.001,0.0008,0.0005,0.001,0.000001,0.0001
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
TRIAL_LENGTH = 30 ## 90 120 100
INIT_LENGTH = 0 ###
TEST_LENGTH = TRIAL_LENGTH - INIT_LENGTH

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
C_MOVE = 0.0 #0.28 0.30
C_PLAN = 0.0 # 0.05 # 0.0
PRIOR_STAT = 0.2 # 1 # 0.2
PRIOR_PLAN = 0.2
PRIOR_SIGMA = 2.5
MODULES = 15 # 10,8
M = MODULES**2
NEURONS = 10
N = 3*(NEURONS**2)
SIGMA_R = 0.4 # 1.2,1.5,1
SIGMA_A = 0.5 # 0.5
SIGMA_S = 0.1
ALPHA = 0.7
BETA = 0.3
DOT_SPEED = 0.2
SIGMOID_MEAN = 0.5
APERTURE = jnp.pi/2 #
THETA = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
COLORS = jnp.array([[255,255,255]]).reshape((1,3)) ### ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = 1 #COLORS.shape[0]
HS_0 = None
HP_0 = None #jnp.sqrt(INIT/(H_S))*rnd.normal(ke[4],(EPOCHS,VMAPS,H_S)) # [E,V,H_S]
HR_0 = None #jnp.sqrt(INIT/(H_S))*rnd.normal(ke[4],(EPOCHS,VMAPS,H_S)) # [E,V,H_S]
POS_0 = None #rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
DOT = None #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
DOT_VEC = None
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
    "TESTS" : TESTS,
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
    "DOT" : DOT,
    "DOT_VEC" : DOT_VEC,
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
    "ALPHA" : ALPHA,
    "DOT_SPEED" : DOT_SPEED,
    "LAMBDA_CRITIC" : LAMBDA_CRITIC,
    "LAMBDA_VEC_KL" : LAMBDA_VEC_KL,
    "LAMBDA_ACT_KL" : LAMBDA_ACT_KL,
    "ALPHA" : ALPHA,
    "BETA" : BETA,
    "C_MOVE" : C_MOVE,
    "C_PLAN" : C_PLAN,
    "PRIOR_STAT" : PRIOR_STAT,
    "PRIOR_PLAN" : PRIOR_PLAN,
    "PRIOR_SIGMA" : PRIOR_SIGMA,
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
    # "p" : { # p_weights
    # # LOAD IN
    # },
    # "r" : { # r_weights
    # # LOAD IN
    # }
}

###
startTime = datetime.now()
losses,stds,weights_s,test_data = full_loop(SC,weights,params) # (loss_arr,actor_loss_arr,critic_loss_arr,kl_loss_arr,vec_kl_arr,act_kl_arr,r_std_arr,l_sem_arr,plan_rate_arr,avg_tot_r_arr,avg_pol_kl_arr,r_init_arr,r_arr,rt_arr,sample_arr,pos_init_arr,pos_arr,dots,sel)
print("Sim time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
(loss_arr,actor_loss_arr,critic_loss_arr,vec_kl_arr,act_kl_arr,r_tot_arr,r_true_arr,plan_rate_arr) = losses
(sem_loss_arr,std_actor_arr,std_critic_arr,std_act_kl_arr,std_vec_kl_arr,std_r_arr,std_plan_rate_arr) = stds
(r_arr,rt_arr,sample_arr,pos_arr,dot_arr,sel) = test_data
print('pos_arr=',pos_arr.shape,pos_arr,'dot_arr=',dot_arr.shape,dot_arr)

# plot loss_arr:
#####
legend_handles = [
    plt.Line2D([], [], color='k', label='loss'), # 
    plt.Line2D([], [], color='b', label='plan rate'), # 
]

fig,axes = plt.subplots(2,3,figsize=(12,9))
title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, TEST_LENGTH={TEST_LENGTH}, INIT_LENGTH={INIT_LENGTH}, update={LR:.6f}, WD={WD:.5f} \n C_MOVE={C_MOVE:.2f}, C_PLAN={"N/A"}, L_CRITIC={LAMBDA_CRITIC}, L_VEC_KL={LAMBDA_VEC_KL}, L_ACT_KL={LAMBDA_ACT_KL}, PRIOR_PLAN={PRIOR_PLAN}, PRIOR_STAT={PRIOR_STAT}, ALPHA={ALPHA}, BETA={BETA}'
plt.suptitle('move_loop_training_v1, '+title__,fontsize=10)
line_r_tot,*_ = axes[0,0].errorbar(np.arange(TOT_EPOCHS),r_tot_arr,yerr=std_r_arr/2,color='black',ecolor='lightgray',elinewidth=2,capsize=0)
axes[0,0].set_xlabel('iteration')
axes[0,0].set_ylabel('reward')
ax00_2 = axes[0,0].twinx()
line_r_true, = ax00_2.plot(np.arange(TOT_EPOCHS),r_true_arr,color='blue')
ax00_2.legend([line_r_tot,line_r_true],['r_tot','r_true'])
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

path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs_move/'
dt = datetime.now().strftime("%d_%m-%H%M%S")
plt.savefig(path_+'move_loop_training_v1_'+dt+'.png')

save_pkl((test_data,weights_s),'move_loop_v1_')

# PLOT TESTING DATA
# r_init_arr = r_init_arr[-PLOTS:,:] # [PLOTS,TEST_LENGTH]
# rt_arr = rt_arr[-PLOTS:,:] # [PLOTS,TEST_LENGTH]
# sample_arr = sample_arr[-PLOTS:,:] # [PLOTS,TEST_LENGTH]
# pos_init_arr = pos_init_arr[-PLOTS:,:,:] # [PLOTS,TEST_LENGTH,2]
# pos_arr = pos_arr[-PLOTS:,:,:] # [PLOTS,TEST_LENGTH,2]
# print('new pos_arr=',pos_arr,'new pos_arr.shape=',pos_arr.shape)
# dots = dot_arr[-PLOTS:,:,:] # [PLOTS,3,2]
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
# plt.suptitle('move_move_v1_test_, '+title__,fontsize=10)
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

# dt = datetime.now().strftime("%d_%m-%H%M%S")
# path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs_move/'
# plt.savefig(path_ + 'figs_move_loop_v1_' + dt + '.png') # ctrl_v7_(v,r,h normal, r_tp changed)