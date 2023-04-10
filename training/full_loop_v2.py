# -*- coding: utf-8 -*-
"""
Created on Mon Apr 3 2023

@author: lukej
"""
import jax
import jax.numpy as jnp
from jax import jit
import jax.random as rnd
# from jax.experimental.host_callback import id_print
# from jax.experimental.host_callback import call
import optax
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from drawnow import drawnow
import numpy as np
import csv
import pickle
from pathlib import Path
from datetime import datetime
import re
import os
import sys
# from os.path import dirname, abspath
# jax.config.update('jax_platform_name', 'cpu')

# fnc definitions

# def save_params(param,str_):  # can't jit (can't pickle jax tracers)
# 	path_ = '/homes/lrj34/projects/meta_rl_ego_sim/pkl/' # str(Path(__file__).resolve().parents[1]) + '/pkl/'
# 	dt = datetime.now().strftime("%d_%m-%H%M")
# 	# file_ = os.path.basename(__file__).split('.')[0]
# 	with open(path_+str_+dt+'.pkl','wb') as file:
# 		pickle.dump(param,file,pickle.HIGHEST_PROTOCOL)

# def save_npy(param,str_):
# 	path_ = '/homes/lrj34/projects/meta_rl_ego_sim/pkl/' # str(Path(__file__).resolve().parents[1]) + '/pkl/'
# 	dt = datetime.now().strftime("%d_%m-%H%M")
# 	with open(path_+str_+'_'+dt+'.npy','wb') as file:
# 		jnp.save(file,param,allow_pickle=False)

def gen_neurons(NEURONS,APERTURE):
	return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32")
gen_neurons = jit(gen_neurons,static_argnums=(0,1))  

def create_dots(N_DOTS,KEY_DOT,VMAPS,EPOCHS):
	return rnd.uniform(KEY_DOT,shape=[EPOCHS,N_DOTS,2,VMAPS],minval=-jnp.pi,maxval=jnp.pi,dtype="float32")
create_dots = jit(create_dots,static_argnums=(0,2,3))

@jit
def neuron_act(e_t_1,th_j,th_i,SIGMA_A,COLORS,pos):
	D_ = COLORS.shape[0]
	N_ = th_j.size
	G_0 = jnp.vstack((jnp.tile(th_j,N_),jnp.tile(th_i,N_)))
	G = jnp.tile(G_0.reshape(2,N_**2,1),(1,1,D_))
	C = (COLORS/255).transpose((1,0))
	E = G.transpose((1,0,2)) - (e_t_1-pos).T
	act = jnp.exp((jnp.cos(E[:,0,:]) + jnp.cos(E[:,1,:]) - 2)/SIGMA_A**2).reshape((D_,N_**2))
	act_C = jnp.matmul(C,act)
	return act_C

@jit
def sigma_fnc(SIGMA_R0,SIGMA_RINF,TAU,e):
    sigma_e = SIGMA_RINF*(1-jnp.exp(-e/TAU))+SIGMA_R0*jnp.exp(-e/TAU) # exp decay to 1/e mag in 1/e time
    return sigma_e

@jit 
def loss_env(pos_hat,pos_t): 
    theta_e_0 = jnp.arctan2(pos_hat[0],pos_hat[1]) #x=atan(sinx/cosx)
    theta_e_1 = jnp.arctan2(pos_hat[2],pos_hat[3]) #y=atan(siny/cosy)
    R_env = jnp.exp((jnp.cos(theta_e_0 - pos_t[0]) + jnp.cos(theta_e_1 - pos_t[1]) - 2)) # sigma?
    return R_env

@jit
def loss_dot(dot_hat,e_t_1,sel):
    # sel = sel.reshape((1,sel.size))
    # dot = jnp.matmul(sel,e_t_1).reshape([2,])
    dot = jnp.dot(sel,e_t_1)
    theta_d_0 = jnp.arctan2(dot_hat[0],dot_hat[1]) #x=atan(sinx/cosx)
    theta_d_1 = jnp.arctan2(dot_hat[2],dot_hat[3]) #y=atan(siny/cosy)
    R_dot = jnp.exp((jnp.cos(theta_d_0 - dot[0]) + jnp.cos(theta_d_1 - dot[1]) - 2)) # sigma?
    return R_dot

@jit
def loss_sel(sel_hat,sel):
    R_sel_ = optax.softmax_cross_entropy(logits=sel_hat,labels=sel)
    return R_sel_

@jit
def loss_obj(e_t_1,sel,e,pos,SIGMA_R0,SIGMA_RINF,TAU,LAMBDA_N,LAMBDA_E,LAMBDA_D,LAMBDA_S): # R_t
    sigma_e = sigma_fnc(SIGMA_R0,SIGMA_RINF,TAU,e)
    e_rel = e_t_1-pos
    obj = -jnp.exp((jnp.cos(e_rel[:,0]) + jnp.cos(e_rel[:,1]) - 2)/sigma_e**2) ### standard loss (-)
    R_obj = jnp.dot(obj,sel)
    return(R_obj,sigma_e)

# @jit
def switch_dots(evrnve):###change to distance-based
    (e_t_1,v_t,dot,pos_t,ALPHA,epoch) = evrnve
    jax.debug.print('\n *****SWITCH***** \n')
    jax.debug.print('epoch: {}',epoch)
    # jax.debug.print('e_t_1: {}',e_t_1)
    # jax.debug.print('v_t: {}',v_t)
    jax.debug.print('dot-pos: {}',dot-pos_t)
    key1 = rnd.PRNGKey(jnp.int32(jnp.floor(1000*(v_t[0]+v_t[1]))))
    key2 = rnd.PRNGKey(jnp.int32(jnp.floor(1000*(v_t[0]-v_t[1]))))
    del_ = e_t_1[1:,:] - e_t_1[0,:] # [N_DOTS-1,2]
    e_t_th = jnp.arctan(del_[:,1]/del_[:,0])
    e_t_abs = jnp.linalg.norm(del_,axis=1)
    abs_transform = jnp.diag(e_t_abs)
    theta_rnd = rnd.uniform(key1,minval=-jnp.pi,maxval=jnp.pi)
    theta_transform = jnp.vstack((jnp.cos(e_t_th+theta_rnd),jnp.sin(e_t_th+theta_rnd))).T
    e_t_1 = e_t_1.at[0,:].set(rnd.uniform(key2,shape=[2,],minval=-jnp.pi,maxval=jnp.pi,dtype=jnp.float32))
    e_t_1 = e_t_1.at[1:,:].set(e_t_1[0,:] + jnp.matmul(abs_transform, theta_transform))
    e_t_1 = (e_t_1+jnp.pi)%(2*jnp.pi)-jnp.pi # reset back to [-pi,pi]
    return e_t_1

# @jit
def keep_dots(evrnve):
    (e_t_1,v_t,dot,pos_t,ALPHA,epoch) = evrnve # e_t_1,v_t,R_t,ALPHA,N_DOTS,VMAPS,EPOCHS,epoch,dot,pos_t
    return e_t_1 # - v_t

# @jit
def new_env(e_t_1,v_t,dot,pos_t,ALPHA,epoch): #change switch condition, e_t_1,v_t,R_t,ALPHA,N_DOTS,VMAPS,EPOCHS,epoch,dot,pos_t
    evrnve = (e_t_1,v_t,dot,pos_t,ALPHA,epoch)
    e_t = jax.lax.cond((jnp.linalg.norm((dot-pos_t),2)<=ALPHA),switch_dots,keep_dots,evrnve) # (epoch>=4000)and(jnp.linalg.norm((dot-pos_t),2)<=ALPHA) , jnp.abs(R_t)>ALPHA
    return e_t

@jit
def abs_dist(e_t,pos):###CHECK behaviour at pi
	# e_t_ = (e_t + jnp.pi)%(2*jnp.pi)-jnp.pi
    pos_ = (pos+jnp.pi)%(2*jnp.pi)-jnp.pi
    dis_rel = e_t-pos_
    return jnp.sqrt(dis_rel[:,0]**2+dis_rel[:,1]**2)

@jit
def true_debug(esdr): # debug
    (epoch,sel,R_obj,R_env,R_dot,R_sel,R_tot,dis,dot,pos_t) = esdr
    # path_ = str(Path(__file__).resolve().parents[1]) + '/stdout/'
    # dt = datetime.now().strftime("%d_%m-%H%M")
    jax.debug.print('epoch = {}', epoch)
    jax.debug.print('sel = {}', sel)
    jax.debug.print('dis = {}', dis)
    jax.debug.print('pos_t = {}', pos_t)
    jax.debug.print('R_obj = {}', R_obj)
    jax.debug.print('R_env = {}', R_env)
    jax.debug.print('R_dot = {}', R_dot)
    jax.debug.print('R_sel = {}', R_sel)
    jax.debug.print('R_tot = {}', R_tot)
    jax.debug.print('dot = {}', dot)
    # jax.debug.print('R_tot={}', R_tot)
    # jax.debug.callback(callback_debug,R_tot)
    # jax.debug.print('sigma_e={}', sigma_e)

@jit
def false_debug(esdr):
	return

@jit
def single_step(EHT_t_1,eps):
    # unpack values
    e_t_1,h_t_1,theta,pos_t,sel,epoch,dot,R_tot,R_obj,R_env,R_dot,R_sel = EHT_t_1

    # extract data from theta
    Wr_z = theta["GRU"]["Wr_z"]
    Wg_z = theta["GRU"]["Wg_z"]
    Wb_z = theta["GRU"]["Wb_z"]
    U_z = theta["GRU"]["U_z"]
    b_z = theta["GRU"]["b_z"]
    Wr_r = theta["GRU"]["Wr_r"]
    Wg_r = theta["GRU"]["Wg_r"]
    Wb_r = theta["GRU"]["Wb_r"]
    U_r = theta["GRU"]["U_r"]
    b_r = theta["GRU"]["b_r"]
    Wr_h = theta["GRU"]["Wr_h"]
    Wg_h = theta["GRU"]["Wg_h"]
    Wb_h = theta["GRU"]["Wb_h"]
    W_r = theta["GRU"]["W_r"]
    U_h = theta["GRU"]["U_h"]
    b_h = theta["GRU"]["b_h"]
    C = theta["GRU"]["C"]
    E = theta["GRU"]["E"]
    D = theta["GRU"]["D"]
    S = theta["GRU"]["S"]
    
    THETA_J = theta["ENV"]["THETA_J"]
    THETA_I = theta["ENV"]["THETA_I"]
    SIGMA_A = theta["ENV"]["SIGMA_A"]
    SIGMA_N = theta["ENV"]["SIGMA_N"]
    SIGMA_R0 = theta["ENV"]["SIGMA_R0"]
    SIGMA_RINF = theta["ENV"]["SIGMA_RINF"]
    TAU = theta["ENV"]["TAU"]
    LAMBDA_N = theta["ENV"]["LAMBDA_N"]
    LAMBDA_E = theta["ENV"]["LAMBDA_E"]
    LAMBDA_D = theta["ENV"]["LAMBDA_D"]
    LAMBDA_S = theta["ENV"]["LAMBDA_S"]
    COLORS = theta["ENV"]["COLORS"]
    STEP = theta["ENV"]["STEP"]
    N_DOTS = theta["ENV"]["N_DOTS"]
    VMAPS = theta["ENV"]["VMAPS"]
    EPOCHS = theta["ENV"]["EPOCHS"]
    ALPHA = theta["ENV"]["ALPHA"]
    
    # neuron activations
    (act_r,act_g,act_b) = neuron_act(e_t_1,THETA_J,THETA_I,SIGMA_A,COLORS,pos_t)
    
    # reward from neurons
    (R_obj,_) = loss_obj(e_t_1,sel,epoch,pos_t,SIGMA_R0,SIGMA_RINF,TAU,LAMBDA_N,LAMBDA_E,LAMBDA_D,LAMBDA_S)
    
    # minimal GRU equations
    z_t = jax.nn.sigmoid(jnp.matmul(Wr_z,act_r) + jnp.matmul(Wg_z,act_g) + jnp.matmul(Wb_z,act_b) + R_obj*W_r + jnp.matmul(U_z,h_t_1) + b_z) # matmul(W_r,R_t)
    f_t = jax.nn.sigmoid(jnp.matmul(Wr_r,act_r) + jnp.matmul(Wg_r,act_g) + jnp.matmul(Wb_r,act_b) + R_obj*W_r + jnp.matmul(U_r,h_t_1) + b_r)
    hhat_t = jnp.tanh(jnp.matmul(Wr_h,act_r)  + jnp.matmul(Wg_h,act_g) + jnp.matmul(Wb_h,act_b) + R_obj*W_r + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply(z_t,h_t_1) + jnp.multiply((1-z_t),hhat_t)# ((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t)
    
    # env, dot, sel readouts
    e_t_hat = jnp.matmul(E,h_t) # [4,1]
    dot_hat = jnp.matmul(D,h_t) # [4,1]
    sel_hat = jnp.matmul(S,h_t) # [DOTS,1]

    # v_t readout
    v_t = STEP*(jnp.matmul(C,h_t) + SIGMA_N*eps) # 'motor noise'
    
    # new env
    # sel_ = sel.reshape((1,sel.size))
    # dot = jnp.matmul(sel_,e_t).reshape((2,))
    dot = jnp.dot(sel,e_t_1)
    pos_t += v_t
    ### e_t = new_env(e_t_1,v_t,dot,pos_t,ALPHA,epoch) #check, e0,v_t,R_obj,ALPHA,N_DOTS,VMAPS,EPOCHS,epoch,dot,pos_t

    # accumulate rewards    
    R_env += LAMBDA_E*loss_env(e_t_hat,pos_t) ###
    R_dot += LAMBDA_D*loss_dot(dot_hat,e_t_1,sel) ###
    R_sel += LAMBDA_S*loss_sel(sel_hat,sel) ###
    R_tot += R_obj #5* + R_env + R_dot + R_sel ### 

    # abs distance
    dis_t = abs_dist(e_t_1,pos_t)
    
    # assemble output
    EHT_t = (e_t_1,h_t,theta,pos_t,sel,epoch,dot,R_tot,R_obj,R_env,R_dot,R_sel)
    pos_dis = (pos_t,dis_t) ###

    return (EHT_t,pos_dis)

@jit
def tot_reward(e0,h0,theta,sel,eps,epoch):
    pos_t=jnp.array([0,0],dtype=jnp.float32)
    dot=jnp.array([0,0],dtype=jnp.float32)
    R_tot,R_obj,R_env,R_dot,R_sel = (jnp.float32(0) for _ in range(5))
    # R_obj=jnp.float32(0)
    # R_env=jnp.float32(0)
    # R_dot=jnp.float32(0)
    # R_sel=jnp.float32(0)
    EHT_0 = (e0,h0,theta,pos_t,sel,epoch,dot,R_tot,R_obj,R_env,R_dot,R_sel)
    EHT_,pos_dis_ = jax.lax.scan(single_step,EHT_0,eps)
    *_,dot,R_tot_,R_obj_,R_env_,R_dot_,R_sel_ = EHT_
    pos_t,dis_t = pos_dis_
    esdr=(epoch,sel,R_tot_,R_obj_,R_env_,R_dot_,R_sel_,dis_t,dot,pos_t)
    jax.lax.cond(((epoch%500==0)),true_debug,false_debug,esdr)
    R_aux = (pos_t,dis_t,R_obj_,R_env_,R_dot_,R_sel_)
    return R_tot_,R_aux

@jit
def train_body(e,LTORS): # (body_fnc) returns theta etc after each trial
    (loop_params,theta,opt_state,R_arr,sd_arr) = LTORS
    (R_tot,R_obj,R_env,R_dot,R_sel) = R_arr
    (sd_tot,sd_obj,sd_env,sd_dot,sd_sel) = sd_arr
    optimizer = optax.adamw(learning_rate=loop_params["UPDATE"],weight_decay=loop_params["WD"])
    ###
    e0 = theta_0["ENV"]["DOTS"][e,:,:,:]
    h0 = theta_0["GRU"]["h0"]
    SELECT = theta_0["ENV"]["SELECT"][e,:,:]
    EPS = theta_0["ENV"]["EPS"][e,:,:,:]
    val_grad = jax.value_and_grad(tot_reward,argnums=2,allow_int=True,has_aux=True)
    val_grad_vmap = jax.vmap(val_grad,in_axes=(2,None,None,0,2,None),out_axes=(0,0))#(stack over axes 0 for both outputs)
    values,grads = val_grad_vmap(e0,h0,theta_0,SELECT,EPS,e)#((R_tot_,R_aux),grads))[vmap'd]
    R_tot_,R_aux = values
    (*_,R_obj_,R_env_,R_dot_,R_sel_) = R_aux
    grads_ = jax.tree_util.tree_map(lambda g: jnp.mean(g,axis=0), grads["GRU"])
    R_tot = R_tot.at[e].set(jnp.mean(R_tot_))
    sd_tot = sd_tot.at[e].set(jnp.std(R_tot_))#
    R_obj = R_obj.at[e].set(jnp.mean(R_obj_))
    sd_obj = sd_obj.at[e].set(jnp.std(R_obj_))#
    R_env = R_env.at[e].set(jnp.mean(R_env_))
    sd_env = sd_env.at[e].set(jnp.std(R_env_))#
    R_dot = R_dot.at[e].set(jnp.mean(R_dot_))
    sd_dot = sd_dot.at[e].set(jnp.std(R_dot_))#
    R_sel = R_sel.at[e].set(jnp.mean(R_sel_))
    sd_sel = sd_sel.at[e].set(jnp.std(R_sel_))#
    R_arr = (R_tot,R_obj,R_env,R_dot,R_sel)
    sd_arr = (sd_tot,sd_obj,sd_env,sd_dot,sd_sel)#

    # update
    opt_update,opt_state = optimizer.update(grads_,opt_state,theta["GRU"])
    theta["GRU"] = optax.apply_updates(theta["GRU"], opt_update)
    LTORS = (loop_params,theta,opt_state,R_arr,sd_arr)

    return LTORS # becomes new input

def test_body(e,LTP):
    (loop_params,theta_test,pos_arr,dis_arr) = LTP #,R_arr,sd_arr)
    # R_arr = (R_obj,R_env,R_dot,R_sel,R_tot)
    # sd_arr = (sd_obj,sd_env,sd_dot,sd_sel,R_tot)
    e0 = theta_test["ENV"]["DOTS"][e,:,:,:]
    h0 = theta_test["GRU"]["h0"]
    SELECT = theta_test["ENV"]["SELECT"][e,:,:]
    EPS = theta_test["ENV"]["EPS"][e,:,:,:]
    tot_reward_vmap = jax.vmap(tot_reward,in_axes=(2,None,None,0,2,None),out_axes=(0,0))
    R_tot_,R_aux = tot_reward_vmap(e0,h0,theta_test,SELECT,EPS,e)
    pos_t,dis_t = R_aux[0:2]
    pos_arr.at[e,:,:,:].set(pos_t) #vmap'd tuple may be hard to unpack; check
    dis_arr.at[e,:,:,:].set(dis_t) #arr[e,:,:] = R_aux[0]
    LTP = (loop_params,theta_test,pos_arr,dis_arr)#,pos_arr,dis_arr,R_arr,sd_arr)
    return LTP

def train_loop(loop_params,theta_0):
    R_tot,sd_tot,R_obj,sd_obj,R_env,sd_env,R_dot,sd_dot,R_sel,sd_sel=(jnp.empty([loop_params["EPOCHS"]]) for _ in range(10))
    R_arr = (R_tot,R_obj,R_env,R_dot,R_sel)
    sd_arr = (sd_tot,sd_obj,sd_env,sd_dot,sd_sel)
    optimizer = optax.adamw(learning_rate=loop_params['UPDATE'],weight_decay=loop_params['WD'])
    opt_state = optimizer.init(theta_0["GRU"])
    LTORS_0 = (loop_params,theta_0,opt_state,R_arr,sd_arr)
    (loop_params_,theta_test,opt_state_,R_arr_,sd_arr_) = jax.lax.fori_loop(0,loop_params["EPOCHS"],train_body,LTORS_0) #theta_,opt_state_,R/sd arrays
    vals_train = (R_arr_,sd_arr_)#R_obj,R_env,R_dot,R_sel
    return theta_test,vals_train

def test_loop(loop_params,theta_test):
    pos_arr=jnp.empty([loop_params["TESTS"],loop_params["VMAPS"],loop_params["IT"],2])
    dis_arr=jnp.empty([loop_params["TESTS"],loop_params["VMAPS"],loop_params["IT"],3])
    LTP_0 = (loop_params,theta_test,pos_arr,dis_arr)
    loop_params_,theta_test_,pos_arr,dis_arr = jax.lax.fori_loop(0,loop_params["TESTS"],test_body,LTP_0)
    vals_test = (pos_arr,dis_arr)
    return vals_test

def full_loop(loop_params,theta_0): # main routine: R_arr, std_arr = full_loop(params)
    theta_test,vals_train = train_loop(loop_params,theta_0)
    vals_test = test_loop(loop_params,theta_test) #check inputs/outputs
    return (vals_train,vals_test)

# ENV parameters
SIGMA_A = jnp.float32(1) # 0.9
SIGMA_R0 = jnp.float32(1) # 0.5
SIGMA_RINF = jnp.float32(0.8) # 0.3
SIGMA_N = jnp.float32(1.8) # 1.6
LAMBDA_N = jnp.float32(0.0001)
LAMBDA_E = jnp.float32(0.01) ### 0.1
LAMBDA_D = jnp.float32(0.01) ###0.001 
LAMBDA_S = jnp.float32(0.0001) ### 0.00001
ALPHA = jnp.float32(0.001) # 0.1,0.7,0.99
STEP = jnp.float32(0.002) # play around with! 0.005
APERTURE = jnp.pi/2 #pi/3
COLORS = jnp.float32([[255,0,0],[0,255,0],[0,0,255]]) # ,[100,100,100],[200,200,200]]) # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[50,0,200]]) # [[255,0,0],[0,200,200],[100,100,100]]
N_DOTS = COLORS.shape[0]
NEURONS = 11

# GRU parameters
N = NEURONS**2
G = 80 # size of GRU
KEY_INIT = rnd.PRNGKey(0) # 0
INIT = jnp.float32(0.1) # 0.1

# loop params
EPOCHS = 1000
# EPOCHS_TEST = 5
IT = 50
VMAPS = 500 # 500
TESTS = 2
UPDATE = jnp.float32(0.0002) # 0.0001,0.00008
WD = jnp.float32(0.00015) # 0.00001
TAU = jnp.float32((1-1/jnp.e)*EPOCHS) # 0.01
optimizer = optax.adamw(learning_rate=UPDATE,weight_decay=WD) #optax.adam(learning_rate=UPDATE)#

# assemble loop_params pytree
loop_params = {
        'EPOCHS':   EPOCHS,
        # 'EPOCHS_TEST': EPOCHS_TEST,
        'VMAPS' :   VMAPS,
        'IT'    :   IT,
        'TESTS' :   TESTS,
    	'UPDATE':   jnp.float32(UPDATE),
        'WD'    :   jnp.float32(WD)
	}

# generate initial values
ki = rnd.split(KEY_INIT,num=20)
h0 = rnd.normal(ki[0],(G,),dtype=jnp.float32)
Wr_z0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype=jnp.float32)
Wg_z0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype=jnp.float32)
Wb_z0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype=jnp.float32)
U_z0 = (INIT/G*G)*rnd.normal(ki[2],(G,G),dtype=jnp.float32)
b_z0 = (INIT/G)*rnd.normal(ki[3],(G,),dtype=jnp.float32)
Wr_r0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype=jnp.float32)
Wg_r0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype=jnp.float32)
Wb_r0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype=jnp.float32)
U_r0 = (INIT/G*G)*rnd.normal(ki[5],(G,G),dtype=jnp.float32)
b_r0 = (INIT/G)*rnd.normal(ki[6],(G,),dtype=jnp.float32)
Wr_h0 = (INIT/G*N)*rnd.normal(ki[7],(G,N),dtype=jnp.float32)
Wg_h0 = (INIT/G*N)*rnd.normal(ki[7],(G,N),dtype=jnp.float32)
Wb_h0 = (INIT/G*N)*rnd.normal(ki[7],(G,N),dtype=jnp.float32)
U_h0 = (INIT/G*G)*rnd.normal(ki[8],(G,G),dtype=jnp.float32)
b_h0 = (INIT/G)*rnd.normal(ki[9],(G,),dtype=jnp.float32)
W_r0 = (INIT/G)*rnd.normal(ki[10],(G,),dtype=jnp.float32)
C0 = (INIT/2*G)*rnd.normal(ki[11],(2,G),dtype=jnp.float32)
E0 = (INIT/4*G)*rnd.normal(ki[12],(4,G),dtype=jnp.float32)
D0 = (INIT/4*G)*rnd.normal(ki[13],(4,G),dtype=jnp.float32)
S0 = (INIT/N_DOTS*G)*rnd.normal(ki[14],(N_DOTS,G),dtype=jnp.float32) # (check produce logits)
THETA_I = gen_neurons(NEURONS,APERTURE)
THETA_J = gen_neurons(NEURONS,APERTURE)
DOTS = create_dots(N_DOTS,ki[15],VMAPS,EPOCHS) # [EPOCHS,N_DOTS,2,VMAPS]
EPS = rnd.normal(ki[16],shape=[EPOCHS,IT,2,VMAPS],dtype=jnp.float32)
SELECT = jnp.eye(N_DOTS)[rnd.choice(ki[16],N_DOTS,(EPOCHS,VMAPS))]

# assemble theta pytree
theta_0 = { "GRU" : {
    	"h0"     : h0,
    	"Wr_z"   : Wr_z0,
		"Wg_z"   : Wg_z0,
		"Wb_z"   : Wb_z0,
    	"U_z"    : U_z0,
    	"b_z"    : b_z0,
    	"Wr_r"   : Wr_r0,
		"Wg_r"   : Wg_r0,
		"Wb_r"   : Wb_r0,
    	"U_r"    : U_r0,
    	"b_r"    : b_r0,
    	"Wr_h"   : Wr_h0,
		"Wg_h"   : Wg_h0,
		"Wb_h"   : Wb_h0,
    	"U_h"    : U_h0,
    	"b_h"    : b_h0,
		"W_r"    : W_r0,
    	"C"	     : C0,
        "E"      : E0,
        "D"      : D0,
        "S"      : S0
	},
        	"ENV" : {
    	"THETA_I"  	: THETA_I,
    	"THETA_J"  	: THETA_J,
    	"COLORS"   	: COLORS,
    	"SIGMA_N"  	: SIGMA_N,
    	"SIGMA_A"  	: SIGMA_A,
    	"SIGMA_R0"  : SIGMA_R0,
        "SIGMA_RINF": SIGMA_RINF,
        "TAU"       : TAU,
    	"STEP"     	: STEP,
    	"DOTS"     	: DOTS,
    	"EPS"      	: EPS,
    	"SELECT"   	: SELECT,
        "N_DOTS"    : N_DOTS,
        "VMAPS"     : VMAPS,
        "EPOCHS"    : EPOCHS,
        "LAMBDA_N"  : LAMBDA_N,
        "LAMBDA_E"  : LAMBDA_E,
        "LAMBDA_D"  : LAMBDA_D,
        "LAMBDA_S"  : LAMBDA_S,
        "ALPHA"     : ALPHA
	}
        	}
theta_0["ENV"] = jax.lax.stop_gradient(theta_0["ENV"])

###
startTime = datetime.now()
(vals_train,vals_test) = full_loop(loop_params,theta_0) #,vals_test
time_elapsed = datetime.now() - startTime
print(f'Completed in: {time_elapsed}, {time_elapsed/EPOCHS} s/epoch')

# plot training
(R_tot,R_obj,R_env,R_dot,R_sel),(sd_tot,sd_obj,sd_env,sd_dot,sd_sel) = vals_train
fig = plt.figure()
plt.subplots(2,2,figsize=(10,10))
title__ = f'v1 training, epochs={EPOCHS}, it={IT}, vmaps={VMAPS}, update={UPDATE:.4f}, SIGMA_A={SIGMA_A:.1f}, SIGMA_RINF={SIGMA_RINF:.1f}, STEP={STEP:.3f} \n WD={WD:.5f}, LAMBDA_D={LAMBDA_D:.4f}, LAMBDA_E={LAMBDA_E:.4f}, LAMBDA_S={LAMBDA_S:.4f}' # \n colors={jnp.array_str(COLORS[0][:]) + jnp.array_str(COLORS[1][:]) + jnp.array_str(COLORS[2][:])}' #  + jnp.array_str(COLORS[3][:]) + jnp.array_str(COLORS[4][:])}'
fig.suptitle(title__,fontsize=8)
plt.subplot(2,2,1)
plt.errorbar(jnp.arange(len(R_tot)),R_tot,yerr=sd_obj/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.ylabel(r'$R_{obj}$')
plt.xlabel(r'Iteration')
plt.subplot(2,2,2)
plt.errorbar(jnp.arange(len(R_obj)),R_obj,yerr=sd_obj/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.ylabel(r'$R_{env}$')
plt.xlabel(r'Iteration')
plt.subplot(2,2,3)
plt.errorbar(jnp.arange(len(R_dot)),R_dot,yerr=sd_dot/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.ylabel(r'$R_{dot}$')
plt.xlabel(r'Iteration')
plt.subplot(2,2,4)
plt.errorbar(jnp.arange(len(R_sel)),R_sel,yerr=sd_sel/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.ylabel(r'$R_{sel}$')
plt.xlabel(r'Iteration')
# plt.show()

#plot testing
pos_arr,dis_arr = vals_test # R_obj_t,R_env_t,R_dot_t,R_sel_t,dis_t,pos_t = vals_test
print('pos_arr=',pos_arr,pos_arr.shape,'dis_arr=',dis_arr,dis_arr.shape)
# (array of tuples to array)
# for each 1d array of pos/dis
# fig,axis = plt.subplots(1,TESTS,figsize=(10,5))
# title__ = f'v1 testing, epochs={EPOCHS}, it={IT}, vmaps={VMAPS}, update={UPDATE:.4f}, SIGMA_A={SIGMA_A:.1f}, SIGMA_RINF={SIGMA_RINF:.1f}, STEP={STEP:.3f} \n WD={WD:.5f}, LAMBDA_D={LAMBDA_D:.4f}, LAMBDA_E={LAMBDA_E:.4f}, LAMBDA_S={LAMBDA_S:.4f}'
# for i in range(TESTS):
    # jnp.random.choice(dis)
    # axis[1,i].plot...
     
#plot |R_obj_t| vs time 
#plot dis_t vs time
# plt.subplot(2,1,1)
# plt.plot()

#plot pos_t vs time (2D)
#(plot dots using sel, e_t_1)
#(plot pos_t vs time using colormap)

# print(f'R_arr: {R_arr} \n std_arr: {std_arr}')

path_ = str(Path(__file__).resolve().parents[1]) + '/figs/task8/'
dt = datetime.now().strftime("%d_%m-%H%M")
plt.savefig(path_ + 'fig_' + dt + '.png')