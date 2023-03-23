# -*- coding: utf-8 -*-
"""
Created on 10 March 2023

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
matplotlib.use('Agg')
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

### differences
# agent has position; velocity added to this
# objective then takes in (dot-pos)

# fnc definitions

def gen_neurons(NEURONS,APERTURE):
	return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype=jnp.float32)
gen_neurons = jit(gen_neurons,static_argnums=(0,1))  

def create_dots(N_DOTS,KEY_DOT,VMAPS,EPOCHS):
	return rnd.uniform(KEY_DOT,shape=[EPOCHS,N_DOTS,2,VMAPS],minval=-jnp.pi,maxval=jnp.pi,dtype=jnp.float32)
create_dots = jit(create_dots,static_argnums=(0,2,3))

@jit
def neuron_act(e_t_1,th_j,th_i,SIGMA_A,COLORS,pos): ### change to be fnc of pos
	D_ = COLORS.shape[0] # no dots
	N_ = th_j.size # number of neurons
	G_0 = jnp.vstack((jnp.tile(th_j,N_),jnp.tile(th_i,N_)))
	G = jnp.tile(G_0.reshape(2,N_**2,1),(1,1,D_))
	C = (COLORS/255).transpose((1,0))
	E = G.transpose((1,0,2)) - (e_t_1-pos).T # diff between neurons and dots; e=[N,2] so do e-pos to get relative positions of neurons
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
    sel = sel.reshape((1,sel.size))
    dot = jnp.matmul(sel,e_t_1)
    theta_d_0 = jnp.arctan2(dot[0],dot[1]) #x=atan(sinx/cosx)
    theta_d_1 = jnp.arctan2(dot[2],dot[3]) #y=atan(siny/cosy)
    R_env = jnp.exp((jnp.cos(theta_d_0 - dot[0]) + jnp.cos(theta_d_1 - dot[1]) - 2)) # sigma?
    return R_env

@jit
def loss_sel(sel_hat,sel):
    R_sel_ = optax.softmax_cross_entropy(logits=sel_hat,labels=sel)
    return R_sel_

@jit
def loss_obj(e_t_1,sel,e,pos,SIGMA_R0,SIGMA_RINF,TAU,LAMBDA_N,LAMBDA_E,LAMBDA_D,LAMBDA_S): ###
    SIGMA_R0 = theta["ENV"]["SIGMA_R0"]
    SIGMA_RINF = theta["ENV"]["SIGMA_RINF"]
    TAU = theta["ENV"]["TAU"]
    # LAMBDA_N = theta["ENV"]["LAMBDA_N"]
    sigma_e = sigma_fnc(SIGMA_R0,SIGMA_RINF,TAU,e)
    e_rel = e_t_1-pos
    obj = -jnp.exp((jnp.cos(e_rel[:,0]) + jnp.cos(e_rel[:,1]) - 2)/sigma_e**2) ### standard loss (-)
    R_obj = jnp.dot(obj,sel)
    # loss_norm = LAMBDA_N*jnp.linalg.norm(theta["GRU"],2)
    # R_t = R_t_ + LAMBDA_E*loss_env + LAMBDA_D*loss_dot + LAMBDA_S*loss_sel + LAMBDA_N*loss_norm
    # R_obj = R_obj_
    return(R_obj,sigma_e)

# @jit
def switch_dots(evrnve): ### change to teleport pos, not e_t
    (e_t_1,v_t,R_t,N_DOTS,VMAPS,EPOCHS) = evrnve # N_DOTS = e_t_1.shape[0]?
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
    e_t_1 = (e_t_1 + jnp.pi)%(2*jnp.pi)-jnp.pi # reset back to [-pi,pi]
    return e_t_1

@jit
def keep_dots(evrnve): ###
    (e_t_1,v_t,R_t,N_DOTS,VMAPS,EPOCHS) = evrnve
    return e_t_1 - v_t

@jit
def new_env(e_t_1,v_t,R_t,ALPHA,N_DOTS,VMAPS,EPOCHS,epoch): ###
    evrnve = (e_t_1,v_t,R_t,N_DOTS,VMAPS,EPOCHS)
    e_t = jax.lax.cond((jnp.abs(R_t)>ALPHA)&(epoch>=4000),switch_dots,keep_dots,evrnve)
    return e_t

@jit
def abs_dist(e_t,pos): ###
	# e_t_ = (e_t + jnp.pi)%(2*jnp.pi)-jnp.pi
    pos_ = (pos + jnp.pi)%(2*jnp.pi)-jnp.pi
    dis_rel = e_t-pos_
    return jnp.sqrt(dis_rel[:,0]**2+dis_rel[:,1]**2)

@jit
def single_step(EHT_t_1,eps):
    # unpack values
    e_t_1,h_t_1,theta,pos_t,sel,epoch = EHT_t_1
    
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
    act_r,act_g,act_b = neuron_act(e_t_1,THETA_J,THETA_I,SIGMA_A,COLORS,pos_t) ###
    
    # reward from neurons
    R_obj,sigma_e = loss_obj(e_t_1,sel,epoch,pos_t,SIGMA_R0,SIGMA_RINF,TAU,LAMBDA_N,LAMBDA_E,LAMBDA_D,LAMBDA_S) ###
    
    # minimal GRU equations
    z_t = jax.nn.sigmoid(jnp.matmul(Wr_z,act_r) + jnp.matmul(Wg_z,act_g) + jnp.matmul(Wb_z,act_b) + R_obj*W_r + jnp.matmul(U_z,h_t_1) + b_z) # matmul(W_r,R_t)
    f_t = jax.nn.sigmoid(jnp.matmul(Wr_r,act_r) + jnp.matmul(Wg_r,act_g) + jnp.matmul(Wb_r,act_b) + R_obj*W_r + jnp.matmul(U_r,h_t_1) + b_r)
    hhat_t = jnp.tanh(jnp.matmul(Wr_h,act_r) + jnp.matmul(Wg_h,act_g) + jnp.matmul(Wb_h,act_b) + R_obj*W_r + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply(z_t,h_t_1) + jnp.multiply((1-z_t),hhat_t) # ((1-f_t),h_t_1) + jnp.multiply(f_t,hhat_t)
    
    # env, dot, sel readouts
    # pos_hat = jnp.matmul(E,h_t)
    # dot_hat = jnp.matmul(D,h_t)
    # sel_hat = jnp.matmul(S,h_t)

    # v_t readout
    v_t = STEP*(jnp.matmul(C,h_t) + SIGMA_N*eps)
    
    # new env
    pos_t += v_t
    # e_t = new_env(e_t_1,v_t,R_t,ALPHA,N_DOTS,VMAPS,EPOCHS,epoch) ###

    # dot/env/sel prediction losses
    # R_env = 0 #LAMBDA_E*loss_env(pos_hat,pos_t,) ###
    # R_dot = 0 #LAMBDA_D*loss_dot(dot_hat,e_t_1,sel) ###
    # R_sel = 0 #LAMBDA_S*loss_sel(sel_hat,sel) ###
    R_t = R_obj #+ R_env + R_dot + R_sel

    # abs distance
    dis = abs_dist(e_t_1,pos_t) ###
    
    # assemble output
    EHT_t = (e_t_1,h_t,theta,pos_t,sel,epoch) # carry
    R_dis = (R_t,R_obj,dis,sigma_e,pos_t) # ,R_env,R_dot,R_sel; print at every timestep (vmap'd)
    
    return (EHT_t,R_dis)

@jit
def true_debug(esdr): # debug
    (sel,epoch,R_t,R_obj,dis,sigma_e,pos_t) = esdr #R_env,R_dot,R_sel
    path_ = str(Path(__file__).resolve().parents[1]) + '/stdout/'
    dt = datetime.now().strftime("%d_%m-%H%M")
    jax.debug.print('epoch = {}', epoch)
    jax.debug.print('sel = {}', sel)
    jax.debug.print('dis={}', dis)
    jax.debug.print('pos_t={}', pos_t)
    jax.debug.print('R_t={}', R_t)
    jax.debug.print('R_obj={}', R_obj)
    # jax.debug.print('R_env={}', R_env)
    # jax.debug.print('R_dot={}', R_dot)
    # jax.debug.print('R_sel={}', R_sel)
    # jax.debug.print('dis2={}', dis2)
    # jax.debug.print('dots={}', e0) #wrong; need to extract at each timestep
    # jax.debug.print('R_tot={}', R_tot)
    # jax.debug.callback(callback_debug,R_tot)
    # jax.debug.print('sigma_e={}', sigma_e)

@jit
def false_debug(esdr):
	return

@jit
def tot_reward(e0,h0,theta,sel,eps,epoch): # ehtsee
    pos_t = jnp.array([0,0],dtype=jnp.float32)
    EHT_0 = (e0,h0,theta,pos_t,sel,epoch)
    EHT_,R_dis = jax.lax.scan(single_step,EHT_0,eps)
    (R_t,R_obj,dis,sigma_e,pos_t) = R_dis # R_env,R_dot,R_sel,dis=[1,IT*N_DOTS[VMAPS]]
    esdr=(sel,epoch,R_t,R_obj,dis,sigma_e,pos_t) # R_env,R_dot,R_sel
    jax.lax.cond(((epoch%1000==0)|((epoch>=4000)&(epoch%500==0))),true_debug,false_debug,esdr)
    return jnp.sum(R_t) # R_tot

@jit
def RG_no_vmap(ehtsee): # test (no grads) - something abt scanning this fnc means that output shape doesnt match input
    (e0,h0,theta,SELECT,EPS,e) = ehtsee
    shape_e = e0.shape
    shape_s = SELECT.shape
    e0 = jnp.resize(e0[:,:,0],(shape_e[2],shape_e[0],shape_e[1])).transpose(1,2,0)
    SELECT = jnp.resize(SELECT[0,:],(shape_s[0],shape_s[1]))
    val_grad = jax.value_and_grad(tot_reward,argnums=2,allow_int=True)
    val_grad_vmap = jax.vmap(val_grad,in_axes=(2,None,None,0,2,None),out_axes=(0,0))
    R_tot,grads = val_grad_vmap(e0,h0,theta,SELECT,EPS,e) # R_tot,grads = val_grad(e0,h0,theta,SELECT,EPS,e)
    grads_ = jax.tree_util.tree_map(lambda g: (0)*jnp.mean(g,axis=0), grads["GRU"])
    return (R_tot,grads_)

@jit
def RG_vmap(ehtsee): # train
    (e0,h0,theta,SELECT,EPS,e) = ehtsee
    val_grad = jax.value_and_grad(tot_reward,argnums=2,allow_int=True)
    val_grad_vmap = jax.vmap(val_grad,in_axes=(2,None,None,0,2,None),out_axes=(0,0)) # (0,0)
    R_tot,grads = val_grad_vmap(e0,h0,theta,SELECT,EPS,e)
    grads_ = jax.tree_util.tree_map(lambda g: jnp.mean(g,axis=0), grads["GRU"])
    return (R_tot,grads_)

@jit
def RG_vmap_TEST(ehtsee): # allow for multiple episodes at testing by disabling vmap
    (e0,h0,theta,SELECT,EPS,e) = ehtsee
    val_grad = jax.value_and_grad(tot_reward,argnums=2,allow_int=True)
    val_grad_vmap = jax.vmap(val_grad,in_axes=(2,None,None,0,2,None),out_axes=(0,0)) # (0,0)
    R_tot,grads = val_grad_vmap(e0,h0,theta,SELECT,EPS,e)
    grads_ = jax.tree_util.tree_map(lambda g: 0*jnp.mean(g,axis=0), grads["GRU"])
    return (R_tot,grads_)

@jit
def body_fnc(e,UTORR): # returns theta
    # unpack
    (UPDATE,WD,theta,opt_state,R_arr,std_arr) = UTORR #opt_state
    optimizer = optax.adam(learning_rate=UPDATE) #optax.adamw(learning_rate=UPDATE,weight_decay=WD) # theta["ENV"]["OPT"] # put in GRU?
    e0 = theta["ENV"]["DOTS"][e,:,:,:]
    h0 = theta["GRU"]["h0"]
    SELECT = theta["ENV"]["SELECT"][e,:,:]
    EPS = theta["ENV"]["EPS"][e,:,:,:]

    # each iteration effects next LTRR (L{R_arr,std_arr},T{GRU}) # vmap tot_reward over dots (e0), eps (EPS) and sel (SELECT)); find avg r_tot, grad
    ehtsee = (e0,h0,theta,SELECT,EPS,e)
    (R_tot,grads_) = jax.lax.cond((e%1000==0)|(e>=4000),RG_vmap,RG_vmap,ehtsee) # no_vmap,vmap #jax.lax.cond((e>=7000),RG_no_vmap,RG_vmap,ehtsee) # jax.lax.cond((e%1000==0)|(e>=4000),RG_no_vmap,RG_vmap,ehtsee) # ,RG_no_vmap,RG_vmap,ehtsee)
    R_arr = R_arr.at[e].set(jnp.mean(R_tot))
    std_arr = std_arr.at[e].set(jnp.std(R_tot))
    # update theta
    opt_update, opt_state = optimizer.update(grads_, opt_state, theta["GRU"])
    theta["GRU"] = optax.apply_updates(theta["GRU"], opt_update)
    UTORR = (UPDATE,WD,theta,opt_state,R_arr,std_arr) # must be same shape as input...

    return UTORR # becomes new input

@jit
def full_loop(loop_params,theta): # main routine: R_arr, std_arr = full_loop(params)
    UPDATE = loop_params['UPDATE']
    WD = loop_params['WD']
    R_arr = loop_params['R_arr']
    std_arr = loop_params['std_arr']
    optimizer = optax.adam(learning_rate=UPDATE)#optax.adamw(learning_rate=UPDATE,weight_decay=WD) ### add weight decay
    opt_state = optimizer.init(theta["GRU"])
    UTORR_0 = (UPDATE,WD,theta,opt_state,R_arr,std_arr)
    UPDATE_,WD_,theta_,opt_state_,R_arr,std_arr = jax.lax.fori_loop(0, EPOCHS, body_fnc, UTORR_0) # 
    return (R_arr,std_arr)

startTime = datetime.now()
# ENV parameters
SIGMA_A = jnp.float32(1) # 0.9
SIGMA_R0 = jnp.float32(1) # 0.5
SIGMA_RINF = jnp.float32(0.3) # 0.3
SIGMA_N = jnp.float32(1.8) # 1.6
LAMBDA_N = jnp.float32(0.00001)
LAMBDA_E = jnp.float32(0.001) ### check
LAMBDA_D = jnp.float32(0.001) ### check
LAMBDA_S = jnp.float32(0.001) ### check
ALPHA = jnp.float32(0.7) # 0.9
STEP = jnp.float32(0.005) # play around with! 0.005
APERTURE = jnp.pi/3
COLORS = jnp.float32([[255,0,0],[0,255,0],[0,0,255]]) # ,[100,100,100],[200,200,200]]) # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[50,0,200]]) # [[255,0,0],[0,200,200],[100,100,100]]
N_DOTS = COLORS.shape[0]
NEURONS = 11

# GRU parameters
N = NEURONS**2
G = 80 # size of GRU
KEY_INIT = rnd.PRNGKey(0) # 0
INIT = jnp.float32(100) # 0.1

# loop params
EPOCHS = 501 # 1001
IT = 30 # 50
VMAPS = 400 # 500
UPDATE = jnp.float32(0.0001) # 0.001
WD = jnp.float32(0) # 0.0005,0.0000001
TAU = jnp.float32((1-1/jnp.e)*EPOCHS) # 0.01
R_arr = jnp.empty(EPOCHS)*jnp.nan
std_arr = jnp.empty(EPOCHS)*jnp.nan
optimizer = optax.adam(learning_rate=UPDATE) #optax.adamw(learning_rate=UPDATE,weight_decay=WD)

# assemble loop_params pytree
loop_params = {
    	'UPDATE' :  jnp.float32(UPDATE),
        'WD'     :  jnp.float32(WD),
    	'R_arr'  :	jnp.zeros(EPOCHS)*jnp.nan,
    	'std_arr':  jnp.zeros(EPOCHS)*jnp.nan
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
W_r0 = (INIT/G)*rnd.normal(ki[10],(G,),dtype=jnp.float32) # W_s = [G,N_DOTS], W_r = [G,] (needs to be 1d)
C0 = (INIT/2*G)*rnd.normal(ki[11],(2,G),dtype=jnp.float32)
E0 = (INIT/4*G)*rnd.normal(ki[12],(4,G),dtype=jnp.float32)
D0 = (INIT/4*G)*rnd.normal(ki[13],(4,G),dtype=jnp.float32)
S0 = (INIT/N_DOTS*G)*rnd.normal(ki[14],(3,G),dtype=jnp.float32) # check... (produce logits?)
THETA_I = gen_neurons(NEURONS,APERTURE)
THETA_J = gen_neurons(NEURONS,APERTURE)
DOTS = create_dots(N_DOTS,ki[15],VMAPS,EPOCHS) # [EPOCHS,N_DOTS,2,VMAPS]
EPS = rnd.normal(ki[16],shape=[EPOCHS,IT,2,VMAPS],dtype=jnp.float32)
SELECT = jnp.eye(N_DOTS)[rnd.choice(ki[16],N_DOTS,(EPOCHS,VMAPS))]

# assemble theta pytree
theta = { "GRU" : {
    	"h0"     : h0, # ?
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
theta["ENV"] = jax.lax.stop_gradient(theta["ENV"])
R_arr,std_arr = full_loop(loop_params,theta)
R_arr = jnp.delete(R_arr,jnp.array(list(range(0,EPOCHS,1000))))
std_arr = jnp.delete(std_arr,jnp.array(list(range(0,EPOCHS,1000))))
# sys.stdout = sys.__stdout__
path_ = str(Path(__file__).resolve().parents[1]) + '/stdout/'
dt = datetime.now().strftime("%d_%m-%H%M")
# with open(path_+'dump_'+dt,'a') as sys.stdout:
# 	print('R_arr: {} \n std_arr: {}',R_arr,std_arr)
print(f'R_arr: {R_arr} \n std_arr: {std_arr}')
time_elapsed = datetime.now() - startTime
print(f'Completed in: {time_elapsed}, {time_elapsed/EPOCHS} s/epoch')

#figure
plt.figure()
plt.errorbar(jnp.arange(len(R_arr)),R_arr,yerr=std_arr/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.show(block=False)
title__ = f'epochs={EPOCHS}, it={IT}, vmaps={VMAPS}, update={UPDATE:.4f}, SIGMA_A={SIGMA_A:.1f}, SIGMA_RINF={SIGMA_RINF:.1f}, SIGMA_N={SIGMA_N:.1f} \n colors={jnp.array_str(COLORS[0][:]) + jnp.array_str(COLORS[1][:]) + jnp.array_str(COLORS[2][:])}' #  + jnp.array_str(COLORS[3][:]) + jnp.array_str(COLORS[4][:])}'
plt.title(title__,fontsize=8)
plt.xlabel('Iteration')
plt.ylabel('Reward')
# plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/figs/task7/'
dt = datetime.now().strftime("%d_%m-%H%M")
plt.savefig(path_ + 'fig_' + dt + '.png')

#FOR DEBUGGING
# csv_write(R_arr,2)
# csv_write(std_arr,3)
# save_params(R_arr,'R_arr')
# save_params(std_arr,'std_arr')
# save_npy(R_arr,'R_arr')
# save_npy(std_arr,'std_arr')
# save_npy(COLORS,'COLORS')
# save_npy(SELECT,'SELECT')