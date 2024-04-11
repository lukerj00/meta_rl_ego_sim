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
	dt = datetime.now().strftime("%d_%m-%H_S%M")
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
    dots = dots - pos
    obj = jnp.exp((jnp.cos(dots[:,0]) + jnp.cos(dots[:,1]) - 2)/SIGMA_R**2) ### (positive), standard loss (-) (1/(sigma_e*jnp.sqrt(2*jnp.pi)))*
    R_obj = jnp.dot(obj,sel)
    return R_obj #sigma_e

# @jit
def sample_action(r,ind,params):
    key = rnd.PRNGKey(ind) ##
    p_ = 1/(1+jnp.exp(-(r-params["SIGMOID_MEAN"])/params["SIGMA_S"])) # (shifted sigmoid)
    return jnp.int32(rnd.bernoulli(key,p=p_)) # 1=action, 0=plan

# @jit
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
    ki = rnd.split(rnd.PRNGKey(e),num=10) #l/0 , key_ = rnd.PRNGKey(0)
    params_["HS_0"] = jnp.sqrt(INIT_S/(H_S))*rnd.normal(ki[0],(VMAPS,H_S))
    params_["HR_0"] = jnp.zeros((VMAPS,H_R)) #jnp.sqrt(INIT_R/(H_R))*rnd.normal(ki[0],(VMAPS,H_R))
    params_["HP_0"] = jnp.sqrt(INIT_P/(H_P))*rnd.normal(ki[1],(VMAPS,H_P)) ### CHECK/CHANGE
    params_["POS_0"] = rnd.choice(ke[2],jnp.arange(-APERTURE,APERTURE,0.01),(VMAPS,2))
    params_["DOTS"] = gen_dots(ki[3],VMAPS,N_DOTS,APERTURE) #key_,rnd.uniform(ki[0],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #jnp.tile(jnp.array([[2,3]]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,1)) #
    params_["SELECT"] = jnp.eye(N_DOTS)[rnd.choice(ki[4],N_DOTS,(VMAPS,))] #rnd.choice(ki[1],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2))
    params_["IND"] = rnd.randint(ki[5],(VMAPS,T),minval=0,maxval=M,dtype=jnp.int32)
    return params_

# @jit
def get_policy(args_t,theta): # hs_t_1,v_t,r_t,rp_t_1,rm_t_1,weights_s,params
    (hs_t_1,*_,rp_t_1,rm_t_1,v_t,r_t) = args_t
    (_,_,weights_s,_) = theta
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
    z_t = jax.nn.sigmoid(jnp.matmul(Ws_vt_z,v_t) + Ws_rt_z*r_t + Ws_pt_1z*rp_t_1 + Ws_mt_1z*rm_t_1 + jnp.matmul(Us_z,hs_t_1) + bs_z)
    f_t = jax.nn.sigmoid(jnp.matmul(Ws_vt_f,v_t) + Ws_rt_f*r_t + Ws_pt_1f*rp_t_1 + Ws_mt_1f*rm_t_1 + jnp.matmul(Us_f,hs_t_1) + bs_f)
    hhat_t = jax.nn.tanh(jnp.matmul(Ws_vt_h,v_t) + Ws_rt_h*r_t + Ws_pt_1h*rp_t_1 + Ws_mt_1h*rm_t_1 + jnp.matmul(Us_h,jnp.multiply(f_t,hs_t_1)) + bs_h)
    hs_t = jnp.multiply(1-z_t,hs_t_1) + jnp.multiply(z_t,hhat_t) #
    vec_t = jnp.matmul(Ws_vec,hs_t)
    act_t = jnp.matmul(Ws_act,hs_t)
    vectors_t = jax.nn.softmax(vec_t)
    actions_t = jax.nn.softmax(act_t)
    return (vectors_t,actions_t),hs_t

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

# @partial(jax.jit,static_argnums=(5,))
def v_predict(h1vec_t,v_t_1,hv_t_1,p_weights,params,PLAN_ITS): # self,hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params
    def loop_fnc(i,carry):
        hv_t_1,h1vec_t,v_t_1,W_h1,W_v,U_vh,b_vh = carry
        hv_t = jax.nn.sigmoid(jnp.matmul(W_h1,h1vec_t) + jnp.matmul(W_v,v_t_1) + jnp.matmul(U_vh,hv_t_1) + b_vh)
        return hv_t,h1vec_t,v_t_1,W_h1,W_v,U_vh,b_vh
    W_h1 = p_weights["W_h1"]
    W_v = p_weights["W_v"]
    U_vh = p_weights["U_vh"]
    b_vh = p_weights["b_vh"]
    W_r = p_weights["W_r"]
    hv_t,*_ = jax.lax.fori_loop(0,PLAN_ITS,loop_fnc,(hv_t_1,h1vec_t,v_t_1,W_h1,W_v,U_vh,b_vh))
    v_t = jnp.matmul(W_r,hv_t)
    return v_t,hv_t

def plan(h1vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,weights,params):#,pos_t_1,dots,vec_t,sel(vec_t,r_t_1,hr_t_1,v_t_1,hv_t_1,weights,params):
    v_t,hv_t = v_predict(h1vec_t,v_t_1,hv_t_1,weights["p"],params,params["PLAN_ITS"])#(hr_t_1,v_t_1,pos_t_1,weights["v"],params)
    r_t,hr_t = r_predict(v_t,v_t_1,r_t_1,hr_t_1,weights["r"])
    return r_t,hr_t,v_t # ,r_tp,v_tp #,hv_t # predictions

def move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,params): # shouldnt be random; should take action dictated by plan...
    pos_t = pos_t_1 + vec_t
    v_t = neuron_act(params,dots,pos_t)
    r_t = loss_obj(dots,sel,pos_t,params["SIGMA_R"])
    vhat_t_,_ = v_predict(h1vec_t,v_t_1,hv_t_1,weights["p"],params,params["PLAN_ITS"])
    rhat_t_,hr_t = r_predict(vhat_t_,v_t_1,r_t_1,hr_t_1,weights["r"])
    return r_t,hr_t,v_t,pos_t #,rhat_t_,vhat_t_ # hv_t, true

# control flow fncs
def dynamic_while_loop(carry_0):
    def loop_body(carry_t_1):
        t,args_t_1,arrs,theta = carry_t_1
        (hs_t_1,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t_1,rm_t_1,v_t_1,r_t_1) = args_t_1
        (SC,weights,weights_s,params) = theta
        (lp_arr,r_arr,sample_arr) = arrs
        # jax.debug.print('LB********type_arr={}',type(lp_arr))
        policy_t,hs_t = get_policy(args_t_1,theta) # hs_t_1,v_t_1,r_t_1,rp_t_1,rm_t_1,weights_s,params
        h1vec_t,vec_t,lp_vec,rp_t,rm_t,lp_rpm = sample_policy(policy_t,SC,ind[t])
        lp_arr = lp_arr.at[t].set(lp_vec+lp_rpm) # lp_arr[t] = lp_vec+lp_rpm # lp_arr = lp_arr.at[t].set(lp_vec+lp_rpm) # jnp.append(arrs[0],lp_vec+lp_rpm)
        jax.debug.print('both, lp_arr={}',lp_arr)
        args_t = (hs_t,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t,rm_t,v_t_1,r_t_1) # update rp/rm
        carry_args = (h1vec_t,vec_t,t,args_t,(lp_arr,r_arr,sample_arr),theta) # assemble carry with sampled vecs and updated args
        t,args_t,arrs = jax.lax.cond(rp_t == 1, plan_fnc, move_fnc, (carry_args))
        return (t,args_t,arrs,theta)
    def dyn_cnd(carry_t_1):
        t,*_,theta = carry_t_1
        *_,params = theta
        return t < params["TRIAL_LENGTH"]

    (t,args,arrs,_) = jax.lax.while_loop(dyn_cnd,loop_body,(carry_0))
    return (t,args,arrs,_)

def plan_fnc(carry_args):
    (h1vec_t,vec_t,t,args_t,arrs,theta) = carry_args
    (hs_t,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t,rm_t,v_t_1,r_t_1) = args_t
    (SC,weights,weights_s,params) = theta
    (lp_arr,r_arr,sample_arr) = arrs
    # jax.debug.print('PF********type_arr={}',type(lp_arr))
    # jax.debug.print('plan, t={}',t)
    # plan
    r_t,_,v_t = plan(h1vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,weights,params)
    r_arr = r_arr.at[t].set(0) # r_arr[t] = 0 # r_arr = r_arr.at[t].set(0) # jnp.append(arrs[1],0)
    sample_arr = sample_arr.at[t].set(0) # sample_arr[t] = 0 # sample_arr = sample_arr.at[t].set(0) # jnp.append(arrs[2],0)
    # jax.debug.print('plan, r_arr={}',r_arr)
    # jax.debug.print('plan, sample_arr={}',sample_arr)
    t += params["T_PLAN"]
    #
    # hs_t,hr_t_1,v_t,r_t,pos_t_1,rp_t,rm_t = hs_t,hr_t_1,v_t,r_t,pos_t_1,rp_t,rm_t 
    args_t = (hs_t,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t,rm_t,v_t,r_t) # update v,r;dont update h
    return (t,args_t,(lp_arr,r_arr,sample_arr))

def move_fnc(carry_args):
    (h1vec_t,vec_t,t,args_t,arrs,theta) = carry_args
    (hs_t,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t,rm_t,v_t_1,r_t_1) = args_t
    (SC,weights,weights_s,params) = theta
    (lp_arr,r_arr,sample_arr) = arrs
    # jax.debug.print('MF********type_arr={}',type(lp_arr))
    # jax.debug.print('move, t={}',t)
    # move
    r_t,hr_t,v_t,pos_t = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,params)
    r_arr = r_arr.at[t].set(r_t) # r_arr[t] = r_t # r_arr = r_arr.at[t].set(r_t) # jnp.append(arrs[1],r_t)
    sample_arr = sample_arr.at[t].set(1) # sample_arr[t] = 1 # sample_arr = sample_arr.at[t].set(1) # jnp.append(arrs[2],1)
    # jax.debug.print('move, r_arr={}',r_arr)
    # jax.debug.print('move, sample_arr={}',sample_arr)
    t += params["T_MOVE"]
    #
    # hs_t_1,hr_t_1,v_t_1,r_t_1,pos_t_1,rp_t_1,rm_t_1 = hs_t,hr_t,v_t,r_t,pos_t,rp_t,rm_t
    args_t = (hs_t,hr_t,hv_t_1,pos_t,dots,sel,ind,rp_t,rm_t,v_t,r_t) # update v,r,hr,pos
    return (t,args_t,(lp_arr,r_arr,sample_arr))

def init_while_loop(carry_0):
    def loop_body(carry_t_1):
        t,args_t_1,theta = carry_t_1
        (hs_t_1,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t_1,rm_t_1,v_t_1,r_t_1) = args_t_1
        (SC,weights,weights_s,params) = theta
        # move (rand)
        policy_t,hs_t = get_policy(args_t_1,theta) # hs_t_1,v_t_1,r_t_1,rp_t_1,rm_t_1,weights_s,params
        h1vec_t,vec_t = sample_vec_rand(SC,ind[t])
        r_t,hr_t,v_t,pos_t = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,params)
        t += params["T_MOVE"]
        #
        args_t = (hs_t,hr_t,hv_t_1,pos_t,dots,sel,ind,rp_t_1,rm_t_1,v_t,r_t) 
        return (t,args_t,theta)
    def init_cnd(carry_t_1):
        t,*_,theta = carry_t_1
        *_,params = theta
        return t < params["INIT_LENGTH"]

    (t,args_init,_) = jax.lax.while_loop(init_cnd,loop_body,(carry_0))
    return t,args_init

def body_fnc(SC,hs_t_1,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,weights,weights_s,params):
    lp_arr,r_arr,sample_arr = (jnp.nan*jnp.empty((params["TEST_LENGTH"],)) for _ in range(3)) # jnp.array([])
    # jax.debug.print('BF********type_arr={}',type(lp_arr))
    arrs_0 = (lp_arr,r_arr,sample_arr)
    t = 0
    rp_t_1 = 0
    rm_t_1 = 1
    v_t = neuron_act(params,dots,pos_t_1)
    r_t = loss_obj(dots,sel,pos_t_1,params["SIGMA_R"])
    args = (hs_t_1,hr_t_1,hv_t_1,pos_t_1,dots,sel,ind,rp_t_1,rm_t_1,v_t,r_t)
    theta = (SC,weights,weights_s,params)
    t,args_init = init_while_loop((t,args,theta))
    _,args,arrs,_ = dynamic_while_loop((t,args_init,arrs_0,theta))
    (lp_arr,r_arr,sample_arr) = arrs
    # jax.debug.print('type(lp_arr)={}',type(lp_arr))
    # jax.debug.print('type(~jnp.isnan(lp_arr))={}',type(~jnp.isnan(lp_arr)))
    # lp_mask = jax.lax.convert_element_type(~jnp.isnan(lp_arr), jnp.bool_)
    # jax.debug.print('type(lp_mask)={}',type(lp_mask))
    # lp_arr = lp_arr[~np.isnan(lp_arr)] # jnp.extract(lp_mask,lp_arr)
    # r_mask = jax.lax.convert_element_type(~jnp.isnan(r_arr), jnp.bool_)
    # r_arr = r_arr[~np.isnan(r_arr)] # jnp.extract(r_mask,r_arr)
    # sample_mask = jax.lax.convert_element_type(~jnp.isnan(sample_arr), jnp.bool_)
    # sample_arr = sample_arr[~np.isnan(sample_arr)] # jnp.extract(sample_mask,sample_arr) # 
    return lp_arr,r_arr # ,sample_arr
    # r_mat = jnp.triu(jnp.ones((params["TEST_LENGTH"],params["TEST_LENGTH"]))*jnp.array(r_arr)) #-cost_arr)) # - jnp.mean(r_arr-cost_arr)) )
    # qhat_arr = jnp.sum(r_mat,axis=1)
    # return jnp.array(lp_arr),qhat_arr

    # _,(lp_arr,r_arr) = jax.lax.while_loop(cond_fnc,test_fnc,(0,(None,None)))
    # while t < params["TRIAL_LENGTH"]:
    #     if t < params["INIT_STEPS"]:
    #         h1vec_t,vec_t = sample_vec_rand(SC,ind[t])
    #         r_t,hr_t,v_t,pos_t = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,params)
    #         # r_arr.append(r_t)
    #         # cost_arr.append(params["C"])
    #         # sample_arr.append(1)
    #         # pos_arr.append(pos_t)
    #         r_t_1,hr_t_1,v_t_1,pos_t_1 = r_t,hr_t,v_t,pos_t
    #         t += 3
    #     else:
    #         policy_t,hs_t = get_policy(hs_t_1,v_t_1,r_t_1,rp_t_1,rm_t_1,weights_s,params)
    #         h1vec_t,vec_t,lp_vec,rp_t,rm_t,lp_rpm = sample_policy(policy_t,SC,ind[t])
    #         lp_arr.append(lp_vec+lp_rpm)
    #         if rp_t == 1: # planning
    #             r_t,_,v_t = plan(h1vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,weights,params)
    #             r_arr.append(0)
    #             # cost_arr.append(0)
    #             sample_arr.append(0)
    #             hs_t_1,hr_t_1,v_t_1,r_t_1,pos_t_1,rp_t_1,rm_t_1 = hs_t,hr_t,v_t,r_t,pos_t,rp_t,rm_t      
    #             t += 1
    #         else: # rm_t == 1: # moving
    #             r_t,hr_t,v_t,pos_t = move(h1vec_t,vec_t,v_t_1,hv_t_1,r_t_1,hr_t_1,pos_t_1,dots,sel,weights,params)
    #             r_arr.append(r_t)
    #             # cost_arr.append(params["C"])
    #             sample_arr.append(1)
    #             hs_t_1,hr_t_1,v_t_1,r_t_1,pos_t_1,rp_t_1,rm_t_1 = hs_t,hr_t,v_t,r_t,pos_t,rp_t,rm_t      
    #             t += 3
    #         # hs_t_1,hr_t_1,v_t_1,r_t_1,pos_t_1,rp_t_1,rm_t_1 = hs_t,hr_t,v_t,r_t,pos_t,rp_t,rm_t      
    # # ### reverse accumulate rewards to go
    # r_mat = jnp.triu(jnp.ones((params["TEST_LENGTH"],params["TEST_LENGTH"]))*jnp.array(r_arr)) #-cost_arr)) # - jnp.mean(r_arr-cost_arr)) )
    # qhat_arr = jnp.sum(r_mat,axis=1)
    # # # return obj,(r_arr,rtg_arr,pos_arr,sample_arr)
    # return jnp.array(lp_arr),qhat_arr #rewards_tg #jnp.array(r_arr[:params["TRIAL_LENGTH"]]),jnp.array(pos_arr[:params["TRIAL_LENGTH"]]),jnp.array(sample_arr[:params["TRIAL_LENGTH"]]) #,jnp.array(r_true_arr)

def interp_fnc(r_arr_):
    def interpolate(indices,values,N):
        # jax.debug.print('typeindices={}',type(indices))
        # jax.debug.print('typevalues={}',type(values))
        indices = jnp.append(indices,N-1)
        values = jnp.append(values,0)
        result = jnp.zeros(N)
        result = result.at[indices].set(values)
        nonzero_indices = jnp.nonzero(result)[0]
        zero_indices = jnp.where(result == 0)[0]
        if zero_indices.size > 0:
            left_indices = jnp.searchsorted(nonzero_indices, zero_indices, side='left')
            right_indices = jnp.searchsorted(nonzero_indices, zero_indices, side='right')
            left_values = result[nonzero_indices[left_indices]]
            right_values = result[nonzero_indices[right_indices]]
            result = result.at[zero_indices].set((left_values + right_values) / 2)
        return result
    qhat_interp_arr = jnp.zeros((r_arr_.shape[0],r_arr_.shape[1]))
    r_locs_arr = jnp.nan*jnp.empty((r_arr_.shape[0],r_arr_.shape[1]))
    for i,r_vec_ in enumerate(r_arr_):
        r_locs = ~jnp.isnan(r_vec_)
        r_vec = r_vec_[r_locs]
        l_vec = len(r_vec_)
        # jax.debug.print('l_vec={}',l_vec)
        l_red = len(r_vec)
        # jax.debug.print('l_red={}',l_red)
        r_mat = jnp.multiply(jnp.triu(jnp.ones((l_red,l_red))),r_vec)
        qhat_arr = jnp.sum(r_mat,axis=1)
        # jax.debug.print('shape_qhat_arr={}',qhat_arr.shape)
        qhat_interp = interpolate(jnp.array(jnp.where(r_locs)),qhat_arr,l_vec)
        qhat_interp_arr = qhat_interp_arr.at[i,:].set(qhat_interp)
        r_locs_arr = r_locs_arr.at[i,:].set(r_locs)
    return qhat_interp_arr #,r_locs_arr

def pg_obj(SC,hs_0,hr_0,hp_0,pos_0,dots,sel,ind,weights,weights_s,params):
    body_fnc_vmap = jax.vmap(body_fnc,in_axes=(None,0,0,0,0,0,0,0,None,None,None),out_axes=(0,0))
    lp_arr_,r_arr_ = body_fnc_vmap(SC,hs_0,hr_0,hp_0,pos_0,dots,sel,ind,weights,weights_s,params) # lp_arr_,qhat_arr_,aux
    qhat_interp_arr = interp_fnc(r_arr_) # r_locs_arr
    b_t = jnp.mean(qhat_interp_arr,axis=0)
    A_arr_ = r_arr_ - b_t
    # A_arr = A_arr_[locs_arr]
    # lp_arr = lp_arr_[locs_arr]
    obj = -jnp.mean(jnp.nansum(jnp.multiply(lp_arr_,A_arr_),axis=1)) # negative for adam
    return obj #,aux

def full_loop(SC,weights,params):
    r_arr,sample_arr = (jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"])) for _ in range(2)) # ,params["VMAPS"]
    pos_arr = jnp.zeros((params["TOT_EPOCHS"],params["TRIAL_LENGTH"],2)) # ,params["VMAPS"]
    dots_arr = jnp.zeros((params["TOT_EPOCHS"],3,2))
    loss_arr,std_arr = ([] for _ in range(2)) #jnp.zeros((params["TOT_EPOCHS"]))
    weights_s = weights["s"]
    E = params["TOT_EPOCHS"]
    optimizer = optax.adamw(learning_rate=params["LR"],weight_decay=params["WD"])
    opt_state = optimizer.init(weights)
    for e in range(E):
        params = new_params(params,e)
        hs_0 = params["HS_0"]
        hr_0 = params["HR_0"]
        hp_0 = params["HP_0"]
        pos_0 = params["POS_0"]
        dots = params["DOTS"]
        sel = params["SELECT"]
        ind = params["IND"]
        loss_grad = jax.value_and_grad(pg_obj,argnums=9)
        loss,grads = loss_grad(SC,hs_0,hr_0,hp_0,pos_0,dots,sel,ind,weights,weights_s,params) # (loss,aux),grads
        opt_update,opt_state = optimizer.update(grads,opt_state,weights_s)
        weights_s = optax.apply_updates(weights_s,opt_update)
        loss_arr.append(loss)
        # (from aux)
        # std_arr.append(jnp.std(grads))
        # r_arr = r_arr.at[e,:].set(r_arr_) # r_std?
        # pos_arr = pos_arr.at[e,:,:].set(pos_arr_)
        # sample_arr = sample_arr.at[e,:].set(sample_arr_)
        # dots_arr = dots_arr.at[e,:,:].set(dots)
        print("e=",e,"grads=",grads,"loss=",loss)
        if e == E-1:
            pass
    return loss_arr,weights #r_arr,pos_arr,sample_arr,dots_arr

# hyperparams ### CHANGE BELOW (no train_length, now care abt init_steps etc)
TOT_EPOCHS = 10
# EPOCHS = 1
PLOTS = 5
# LOOPS = TOT_EPOCHS//EPOCHS
VMAPS = 10 # 1100,1000,800,500
LR = 0.0001 # 0.000001,0.0001
WD = 0.0001 # 0.0001
INIT_S = 2
INIT_P = 2 # 0.5
INIT_R = 3 # 5,2
H_S = 100
H_P = 500 # 500,300
H_R = 100
PLAN_ITS = 10
INIT_LENGTH = 90
TRIAL_LENGTH = 200 # 100
TEST_LENGTH = TRIAL_LENGTH - INIT_LENGTH

# ENV/sc params
ke = rnd.split(rnd.PRNGKey(0),10)
T_MOVE = 3
T_PLAN = 1 # 0
C = 0.5
MODULES = 15 # 10,8
M = MODULES**2
NEURONS = 10
N = 3*(NEURONS**2)
SIGMA_R = 1 # 1.2,1.5,1
SIGMA_A = 0.5 # 0.5
SIGMA_S = 0.1
SIGMOID_MEAN = 0.5
APERTURE = jnp.pi
THETA_X = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
THETA_Y = jnp.linspace(-(APERTURE-APERTURE/NEURONS),(APERTURE-APERTURE/NEURONS),NEURONS)
COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]]) # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])
N_DOTS = COLORS.shape[0]
HS_0 = None
HP_0 = None #jnp.sqrt(INIT/(H_S))*rnd.normal(ke[4],(EPOCHS,VMAPS,H_S)) # [E,V,H_S]
HR_0 = None #jnp.sqrt(INIT/(H_S))*rnd.normal(ke[4],(EPOCHS,VMAPS,H_S)) # [E,V,H_S]
POS_0 = None #rnd.choice(ke[3],jnp.arange(-APERTURE,APERTURE,0.01),(EPOCHS,VMAPS,2)) #jnp.array([-0.5,0.5]) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
DOTS = None #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) # jnp.array([[-2,-2],[0,0],[2,2]]) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE) jnp.tile(jnp.array([1,2]).reshape(1,1,1,2),(EPOCHS,VMAPS,1,2)) #rnd.uniform(ke[6],shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE) #gen_dots(ke[0],EPOCHS,VMAPS,N_DOTS,APERTURE)
SELECT = None #jnp.eye(N_DOTS)[rnd.choice(ke[1],N_DOTS,(EPOCHS,VMAPS))]
IND = None
ID_ARR = rnd.permutation(ke[5],jnp.arange(0,M),independent=True) # CHANGE TO PERMUATION
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
    # "SAMPLES" : SAMPLES,
    "POS_0" : POS_0,
    "HP_0" : HP_0,
    "HR_0" : HR_0,
    "IND" : IND,
    "INIT_LENGTH" : INIT_LENGTH,
    "TRIAL_LENGTH" : TRIAL_LENGTH,
    "TEST_LENGTH" : TEST_LENGTH,
    "PLAN_ITS" : PLAN_ITS,
    "SIGMOID_MEAN" : SIGMOID_MEAN,
    "SIGMA_R" : SIGMA_R,
    "SIGMA_A" : SIGMA_A,
    "SIGMA_S" : SIGMA_S,
    # "LAMBDA_S" : LAMBDA_S,
    # "ALPHA" : ALPHA,
    "C" : C,
    "T_MOVE" : T_MOVE,
    "T_PLAN" : T_PLAN,
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
(loss_arr,weights) = full_loop(SC,weights,params) # r_arr,pos_arr,sample_arr,dots_arr,v_loss_arr,r_loss_arr,r_true_arr

# plot training loss
print("Sim time: ",datetime.now()-startTime,"s/epoch=",((datetime.now()-startTime)/TOT_EPOCHS).total_seconds())
print(loss_arr)

#####
# r_arr = r_arr[-PLOTS:,:] # (3,0,T)
# pos_arr = pos_arr[-PLOTS:,:,:] # (3,0,T),((-1 if e))
# dots_arr = dots_arr[-PLOTS:,:,:] # (3,0,T,N,2)
# sample_arr = sample_arr[-PLOTS:,:]
# v_loss_arr = v_loss_arr[-PLOTS:,:]
# r_loss_arr = r_loss_arr[-PLOTS:,:]
# r_true_arr = r_true_arr[-PLOTS:,:]

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
# T_ = [15,35,90] # ,[5,15,25]
# neuron_locs = gen_vectors(NEURONS,APERTURE)

# # plot timeseries of true/planned reward
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

# plt.axis('tight')
# plt.subplots_adjust(top=0.95)

# dt = datetime.now().strftime("%d_%m-%H_S%M%S")
# path_ = str(Path(__file__).resolve().parents[1]) + '/sc_project/figs/'
# plt.savefig(path_ + 'figs_outer_loop_const_fixed_' + dt + '.png') # ctrl_v7_(v,r,h normal, r_tp changed)

# # save_pkl((loss_arr,loss_std,p_weights_trained,v_arr,r_arr,r_hat_arr),'value_v3_')
