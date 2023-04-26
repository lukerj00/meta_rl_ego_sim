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
from matplotlib.pyplot import cm
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
import scipy
# jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')

# fnc definitions

def load_(str_):
    path_ = str(Path(__file__).resolve().parents[1]) + '/pkl/'
    with open(path_+str_,'rb') as file_:
        # param = pickle.load(file_)
        param_ = jnp.load(file_,allow_pickle=True)
    return param_

def save_pkl(param,str_):  # can't jit (can't pickle jax tracers)
	path_ = str(Path(__file__).resolve().parents[1]) + '/pkl/'
	dt = datetime.now().strftime("%d_%m-%H%M")
	# file_ = os.path.basename(__file__).split('.')[0]
	with open(path_+str_+'_'+dt+'.pkl','wb') as file:
		pickle.dump(param,file,pickle.HIGHEST_PROTOCOL)

def save_(param,str_):
	path_ = '/homes/lrj34/projects/meta_rl_ego_sim/pkl/' # str(Path(__file__).resolve().parents[1]) + '/pkl/'
	dt = datetime.now().strftime("%d_%m-%H%M")
	with open(path_+str_+'_'+dt+'.npy','wb') as file:
		jnp.save(file,param,allow_pickle=False)

def gen_neurons(NEURONS,APERTURE):
	return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype=jnp.float32)
# gen_neurons = jit(gen_neurons,static_argnums=(0,1))  

def gen_dots(KEY_DOT,N_DOTS,VMAPS,EPOCHS):
	return rnd.uniform(KEY_DOT,shape=[EPOCHS,N_DOTS,2,VMAPS],minval=-jnp.pi,maxval=jnp.pi,dtype=jnp.float32)
# gen_dots = jit(gen_dots,static_argnums=(1,2,3))

def gen_eps(KEY_EPS,EPOCHS,IT,VMAPS):
    return rnd.normal(KEY_EPS,shape=[EPOCHS,IT,2,VMAPS],dtype=jnp.float32)
# gen_eps = jit(gen_eps,static_argnums=(1,2,3))

def gen_select(KEY_SEL,N_DOTS,EPOCHS,VMAPS):
    return jnp.eye(N_DOTS)[rnd.choice(KEY_SEL,N_DOTS,(EPOCHS,VMAPS))]
# gen_select = jit(gen_select,static_argnums=(1,2,3))

def gen_h0(KEY_H0,G,VMAPS,EPOCHS):
    return rnd.normal(KEY_H0,shape=[EPOCHS,VMAPS,G],dtype=jnp.float32)

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
def sigma_fnc(SIGMA_R0,SIGMA_RINF,TAU,EPOCHS,e,x):
    e_ = EPOCHS*x+e
    sigma_e = SIGMA_RINF*(1-jnp.exp(-e_/TAU))+SIGMA_R0*jnp.exp(-e_/TAU) # exp decay to 1/e mag in 1/e time
    return sigma_e

@jit 
def loss_env(pos_hat,pos_t): 
    theta_e_0 = jnp.arctan2(pos_hat[1],pos_hat[0]) #x=atan(sinx/cosx)
    theta_e_1 = jnp.arctan2(pos_hat[3],pos_hat[2]) #y=atan(siny/cosy)
    R_env = jnp.exp((jnp.cos(theta_e_0 - pos_t[0]) + jnp.cos(theta_e_1 - pos_t[1]) - 2)) # sigma?
    return R_env

@jit
def loss_dot(dot_hat,dot):
    theta_d_0 = jnp.arctan2(dot_hat[1],dot_hat[0]) #x=atan(sinx/cosx)
    theta_d_1 = jnp.arctan2(dot_hat[3],dot_hat[2]) #y=atan(siny/cosy)
    R_dot = jnp.exp((jnp.cos(theta_d_0 - dot[0]) + jnp.cos(theta_d_1 - dot[1]) - 2)) # sigma?
    return R_dot

@jit
def loss_sel(sel_hat,sel):
    R_sel_ = optax.softmax_cross_entropy(logits=sel_hat,labels=sel)
    return R_sel_

@jit
def loss_obj(e_t_1,sel,e,pos,SIGMA_R0,SIGMA_RINF,TAU,EPOCHS,x): # R_t
    sigma_e = sigma_fnc(SIGMA_R0,SIGMA_RINF,TAU,EPOCHS,e,x)
    # pos_ = (pos+jnp.pi)%(2*jnp.pi)-jnp.pi
    e_rel = e_t_1-pos
    obj = -jnp.exp((jnp.cos(e_rel[:,0]) + jnp.cos(e_rel[:,1]) - 2)/sigma_e**2) ### standard loss (-) (1/(sigma_e*jnp.sqrt(2*jnp.pi)))*
    R_obj = jnp.dot(obj,sel)
    p_ = (1/(-sigma_e*jnp.sqrt(2*jnp.pi)))*R_obj #- (1/8*sigma_e*jnp.sqrt(2*jnp.pi))*jnp.sqrt(-R_obj) #(check normalised correctly... bessel fncs?)
    sample_ = rnd.bernoulli(rnd.PRNGKey(jnp.int32(jnp.floor(1000*sigma_e))),p=p_/7)
    return obj,sample_ #sigma_e

# @jit
def keep_dots(evrnve):
    (e_t_1,v_t,dot,pos_t,ALPHA,epoch,R_temp) = evrnve # e_t_1,v_t,R_t,ALPHA,N_DOTS,VMAPS,EPOCHS,epoch,dot,pos_t
    return e_t_1,pos_t # - v_t

# @jit
# def new_env(e_t_1,v_t,dot,pos_t,ALPHA,epoch,R_temp): #change switch condition, e_t_1,v_t,R_t,ALPHA,N_DOTS,VMAPS,EPOCHS,epoch,dot,pos_t
#     evrnve = (e_t_1,v_t,dot,pos_t,ALPHA,epoch,R_temp)
#     e_t,pos_t = jax.lax.cond((epoch>=2000)&(jnp.linalg.norm((dot-pos_t),ord=2)<(-1-ALPHA)),switch_agent,keep_dots,evrnve) # (epoch>=4000)and(jnp.linalg.norm((dot-pos_t),2)<=ALPHA) , jnp.abs(R_t)>ALPHA
#     return e_t,pos_t

def abs_pos(pos_t):
    pos_t_ = (pos_t+jnp.pi)%(2*jnp.pi)-jnp.pi
    return pos_t_

def geod_dist(dots,pos):### calculate geodesic
    pos_ = pos #(pos+jnp.pi)%(2*jnp.pi)-jnp.pi # could cause problems...
    # del_x = jnp.abs(dot[0]-pos_[0]) #[3,]
    # del_y = jnp.abs(dot[1]-pos_[1]) #[3,]
    # jax.debug.print('pos={}',pos)
    # jax.debug.print('dot: {}',dot)
    # jax.debug.print('del_y: {}',del_y)
    # hav = (1-jnp.cos(del_y))/2 + jnp.cos(pos_[1])*jnp.cos(dot[1])*((1-jnp.cos(del_x))/2)#(jnp.sin(del_y/2))**2 + jnp.cos(pos_[1])*jnp.cos(dot[1])*(jnp.sin(del_x/2))**2
    # dist = 2*jnp.arcsin(jnp.sqrt(hav))
    ## th1 = jnp.pi/2 - pos_[1]
    ## th2 = jnp.pi/2 - dots[:,1]
    ## ld1 = pos_[0]
    ## ld2 = dots[:,0]
    ## s = jnp.arccos(jnp.cos(th1)*jnp.cos(th2)+jnp.sin(th1)*jnp.multiply(jnp.sin(th2),jnp.cos(ld1-ld2)))
    th1 = jnp.minimum(jnp.abs(pos_[1]-dots[:,1]),2*jnp.pi-jnp.abs(pos_[1]-dots[:,1]))
    th2 = jnp.minimum(jnp.abs(pos_[0]-dots[:,0]),2*jnp.pi-jnp.abs(pos_[0]-dots[:,0]))
    return jnp.sqrt(th1**2+th2**2)

# @jit
def arc(dis_t):
    return jnp.arccos(1-(dis_t**2)/2)

@jit
def abs_dist(e_t,pos):### calculate geodesic
	## e_t_ = (e_t + jnp.pi)%(2*jnp.pi)-jnp.pi
    # pos_ = (pos+jnp.pi)%(2*jnp.pi)-jnp.pi
    # # pos_ = pos
    # del_x = jnp.abs(e_t[:,0]-pos_[0]) #[3,]
    # del_y = jnp.abs(e_t[:,1]-pos_[1]) #[3,]
    # jax.debug.print('pos={}',pos)
    # jax.debug.print('del_x: {}',del_x)
    # jax.debug.print('del_y: {}',del_y)
    # hav = (jnp.sin(del_y/2))**2 + jnp.cos(pos_[1])*jnp.multiply(jnp.cos(e_t[:,1]),(jnp.sin(del_x/2))**2)
    # jax.debug.print('hav: {}',hav)
    # dis = 2*jnp.arcsin(jnp.sqrt(hav))
    pos_ = (pos+jnp.pi)%(2*jnp.pi)-jnp.pi
    dis_rel = e_t-pos_
    dis_rel_ = (dis_rel+jnp.pi)%(2*jnp.pi)-jnp.pi
    dis = jnp.linalg.norm(dis_rel_,ord=2,axis=1) #jnp.sqrt(dis_rel_[:,0]**2+dis_rel_[:,1]**2)
    return dis

@jit
def true_debug(esdr): # debug
    (epoch,sel,R_test,R_tot,R_obj,R_env,R_dot,R_sel,dis,dot,pos_t) = esdr
    # path_ = str(Path(__file__).resolve().parents[1]) + '/stdout/'
    # dt = datetime.now().strftime("%d_%m-%H%M")
    jax.debug.print('epoch = {}', epoch)
    jax.debug.print('sel = {}', sel)
    jax.debug.print('dis = {}', dis)
    jax.debug.print('pos_t = {}', pos_t)
    jax.debug.print('R_test = {}', R_test)
    jax.debug.print('R_tot = {}', R_tot)
    jax.debug.print('R_obj = {}', R_obj)
    jax.debug.print('R_env = {}', R_env)
    jax.debug.print('R_dot = {}', R_dot)
    jax.debug.print('R_sel = {}', R_sel)
    jax.debug.print('dot = {}', dot)
    # jax.debug.print('R_tot={}', R_tot)
    # jax.debug.callback(callback_debug,R_tot)
    # jax.debug.print('sigma_e={}', sigma_e)

def true_debug2(x,act_r,act_g,act_b,h_t_1,f_t,hhat_t,h_t):
    jax.debug.print('epoch={}',x)
    # jax.debug.print('act_r={}',act_r)#rnd.choice(rnd.PRNGKey(0),act_r,10))
    # jax.debug.print('act_g={}',act_g)#rnd.choice(rnd.PRNGKey(0),act_g,10))
    # jax.debug.print('act_b={}',act_b)#rnd.choice(rnd.PRNGKey(0),act_b,10))
    jax.debug.print('h_t_1={}',h_t_1)#rnd.choice(rnd.PRNGKey(0),h_t_1,20))
    jax.debug.print('f_t={}',f_t)#rnd.choice(rnd.PRNGKey(0),f_t,20))
    jax.debug.print('hhat_t={}',hhat_t)#rnd.choice(rnd.PRNGKey(0),hhat_t,20))
    jax.debug.print('h_t={}',h_t)#rnd.choice(rnd.PRNGKey(0),h_t,20))

def false_debug2(x,act_r,act_g,act_b,h_t_1,f_t,hhat_t,h_t):
    return

@jit
def false_debug(esdr):
	return

def new_theta(theta,x):
    theta_ = theta
    EPOCHS = theta_["ENV"]["EPOCHS"]
    IT = theta_["ENV"]["IT"]
    VMAPS = theta_["ENV"]["VMAPS"]
    N_DOTS = theta_["ENV"]["N_DOTS"]
    ki = rnd.split(rnd.PRNGKey(x),num=10)
    theta_["ENV"]["DOTS"] = gen_dots(ki[0],N_DOTS,VMAPS,EPOCHS)
    theta_["ENV"]["EPS"] = gen_eps(ki[1],EPOCHS,IT,VMAPS)
    theta_["ENV"]["SELECT"] = gen_select(ki[2],N_DOTS,EPOCHS,VMAPS)
    return theta_

# @jit
def switch_dots(evrnve):###change to distance-based
    (e_t_1,v_t,dot,pos_t,ALPHA,epoch,R_temp) = evrnve
    # jax.debug.print('\n *****SWITCH***** \n')
    # jax.debug.print('epoch: {}',epoch)
    # jax.debug.print('e_t_1: {}',e_t_1)
    # jax.debug.print('v_t: {}',v_t)
    # jax.debug.print('dot-pos: {}',dot-pos_t)
    key1 = rnd.PRNGKey(jnp.int32(jnp.floor(1000*(v_t[0]+v_t[1]))))
    key2 = rnd.PRNGKey(jnp.int32(jnp.floor(1000*(v_t[0]-v_t[1]))))
    del_ = e_t_1[1:,:] - e_t_1[0,:] # [N_DOTS-1,2]
    e_t_th = jnp.arctan2(del_[:,1],del_[:,0])
    e_t_abs = jnp.linalg.norm(del_,axis=1)
    abs_transform = jnp.diag(e_t_abs)
    theta_rnd = rnd.uniform(key1,minval=-jnp.pi,maxval=jnp.pi)
    theta_transform = jnp.vstack((jnp.cos(e_t_th+theta_rnd),jnp.sin(e_t_th+theta_rnd))).T
    e_t_1 = e_t_1.at[0,:].set(rnd.uniform(key2,shape=[2,],minval=-jnp.pi,maxval=jnp.pi,dtype=jnp.float32))
    e_t_1 = e_t_1.at[1:,:].set(e_t_1[0,:] + jnp.matmul(abs_transform, theta_transform))
    e_t_1 = (e_t_1+jnp.pi)%(2*jnp.pi)-jnp.pi # reset back to [-pi,pi]
    return e_t_1,pos_t

def switch_agent(pos_t):
    pos_t = rnd.uniform(rnd.PRNGKey(jnp.int32(jnp.floor(1000*(pos_t[0]+pos_t[1])))),shape=[2,],minval=-jnp.pi,maxval=jnp.pi,dtype=jnp.float32)
    return pos_t

def keep_agent(pos_t):
    return pos_t

@jit
def single_step(EHT_t_1,eps):
    # unpack values
    e_t_1,h_t_1,theta,pos_t,sel,epoch,dot,R_obj,R_env,R_dot,R_sel,x = EHT_t_1#R_tot

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
    W_r_z = theta["GRU"]["W_r_z"]
    W_r_f = theta["GRU"]["W_r_f"]
    W_r_hh = theta["GRU"]["W_r_hh"]
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
    R_rgb,sample_ = loss_obj(e_t_1,sel,epoch,pos_t,SIGMA_R0,SIGMA_RINF,TAU,EPOCHS,x)
    R_r,R_b,R_g = R_rgb
    R_temp = jnp.dot(R_rgb,sel)

    # GRU
    z_t = jax.nn.sigmoid(jnp.matmul(Wr_z,act_r) + jnp.matmul(Wg_z,act_g) + jnp.matmul(Wb_z,act_b) + R_temp*W_r_z + jnp.matmul(U_z,h_t_1) + b_z) # matmul(W_r,R_t)
    f_t = jax.nn.sigmoid(jnp.matmul(Wr_r,act_r) + jnp.matmul(Wg_r,act_g) + jnp.matmul(Wb_r,act_b) + R_temp*W_r_f + jnp.matmul(U_r,h_t_1) + b_r)
    hhat_t = jnp.tanh(jnp.matmul(Wr_h,act_r)  + jnp.matmul(Wg_h,act_g) + jnp.matmul(Wb_h,act_b) + R_temp*W_r_hh + jnp.matmul(U_h,(jnp.multiply(f_t,h_t_1))) + b_h )
    h_t = jnp.multiply(z_t,h_t_1) + jnp.multiply((1-z_t),hhat_t)

    # env, dot, sel readouts
    pos_hat = jnp.matmul(E,h_t) # [4,1]
    dot_hat = jnp.matmul(D,h_t) # [4,1]
    sel_hat = jnp.matmul(S,h_t) # [DOTS,1]
    
    # new env    
    dot = jnp.dot(sel,e_t_1)
    pos_t = jax.lax.cond((sample_==1)&(x>=0),switch_agent,keep_agent,pos_t) # (x>=1)|(jnp.linalg.norm((dot-pos_t),ord=2)<=ALPHA)
    # e_t,pos_t = new_env(e_t_1,v_t,dot,pos_t,ALPHA,epoch,R_temp) #check, e0,v_t,R_obj,ALPHA,N_DOTS,VMAPS,EPOCHS,epoch,dot,pos_t

    # v_t readout
    v_t = STEP*(jnp.matmul(C,h_t) + SIGMA_N*eps) # 'motor noise'
    pos_t += v_t

    # accumulate rewards
    R_obj += R_temp
    R_env += LAMBDA_E*loss_env(pos_hat,pos_t) ###
    R_dot += LAMBDA_D*loss_dot(dot_hat,dot) ###
    R_sel += LAMBDA_S*loss_sel(sel_hat,sel) ###
    
    # assemble output
    EHT_t = (e_t_1,h_t,theta,pos_t,sel,epoch,dot,R_obj,R_env,R_dot,R_sel,x)#R_tot
    pos_dis = (pos_t,sample_,R_temp,R_r,R_g,R_b) ### dis_t

    # jax.lax.cond(epoch%1000==0,true_debug2,false_debug2,x,act_r,act_g,act_b,h_t_1,f_t,hhat_t,h_t)

    return (EHT_t,pos_dis)

@jit
def tot_reward(e0,h0,theta,sel,eps,epoch,x):
    pos_t=jnp.array([0,0],dtype=jnp.float32)#jnp.float64
    dot=jnp.array([0,0],dtype=jnp.float32)#jnp.float64
    R_tot,R_obj,R_env,R_dot,R_sel = (jnp.float32(0) for _ in range(5))
    EHT_0 = (e0,h0,theta,pos_t,sel,epoch,dot,R_obj,R_env,R_dot,R_sel,x)#R_tot
    EHT_,pos_dis_ = jax.lax.scan(single_step,EHT_0,eps)
    *_,dot,R_obj_,R_env_,R_dot_,R_sel_,x_ = EHT_ #R_tot_
    R_tot_ = R_obj_ + R_sel_ + R_env_ + R_dot_
    pos_t,sample_,R_temp,R_r,R_g,R_b = pos_dis_ # dis_t,
    R_t = R_temp,R_r,R_g,R_b
    # esdr=(epoch,sel,R_temp,R_tot_,R_obj_,R_env_,R_dot_,R_sel_,dis_t,dot,pos_t)#R_tot_
    # jax.lax.cond(((epoch%20==0)),true_debug,false_debug,esdr)
    R_aux = (pos_t,sample_,R_t,R_obj_,R_env_,R_dot_,R_sel_) # dis_t,
    return R_tot_,R_aux

@jit
def train_body(e,LTORS): # (body_fnc) returns theta etc after each trial
    (loop_params,theta,opt_state,R_arr,sd_arr,x) = LTORS
    (R_tot,R_obj,R_env,R_dot,R_sel) = R_arr
    (sd_tot,sd_obj,sd_env,sd_dot,sd_sel) = sd_arr
    optimizer = optax.adamw(learning_rate=loop_params["UPDATE"],weight_decay=loop_params["WD"])
    ###
    e0 = theta["ENV"]["DOTS"][e,:,:,:]
    h0 = theta["ENV"]["H0"][e,:,:] # [VMAPS,G]
    SELECT = theta["ENV"]["SELECT"][e,:,:]
    EPS = theta["ENV"]["EPS"][e,:,:,:]
    val_grad = jax.value_and_grad(tot_reward,argnums=2,allow_int=True,has_aux=True)
    val_grad_vmap = jax.vmap(val_grad,in_axes=(2,0,None,0,2,None,None),out_axes=(0,0))#((0,0),0)(stack over axes 0 for both outputs)
    values,grads = val_grad_vmap(e0,h0,theta,SELECT,EPS,e,x)#((R_tot_,R_aux),grads))[vmap'd]
    R_tot_,R_aux = values #check unpacked correctly
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
    theta["GRU"] = optax.apply_updates(theta["GRU"],opt_update)
    LTORS = (loop_params,theta,opt_state,R_arr,sd_arr,x)
    return LTORS # becomes new input

def test_body(e,LTP):
    (loop_params,theta_test,pos_arr,switch_arr,R_t) = LTP #,dis_arr
    R_test,R_r,R_g,R_b = R_t
    e0 = theta_test["ENV"]["DOTS"][e,:,:,:]
    h0 = theta_test["ENV"]["H0"][e,:,:]
    SELECT = theta_test["ENV"]["SELECT"][e,:,:]
    EPS = theta_test["ENV"]["EPS"][e,:,:,:]
    tot_reward_vmap = jax.vmap(tot_reward,in_axes=(2,0,None,0,2,None,None),out_axes=(0,0))
    R_tot_,R_aux = tot_reward_vmap(e0,h0,theta_test,SELECT,EPS,0,loop_params["LOOPS"])
    pos_t,sample_,R_t,*_ = R_aux # dis_t,
    switch_arr = switch_arr.at[e,:,1:].set(sample_) #[TESTS,VMAPS,IT]
    R_test_,R_r_,R_g_,R_b_ = R_t
    pos_arr = pos_arr.at[e,:,1:,:].set(abs_pos(pos_t)) #[TESTS,VMAPS,IT,2], vmap'd tuple may be hard to unpack; check
    R_test = R_test.at[e,:,1:].set(R_test_) #[TESTS,VMAPS,IT]
    R_r = R_r.at[e,:,1:].set(R_r_) #[TESTS,VMAPS,IT]
    R_g = R_g.at[e,:,1:].set(R_g_) #[TESTS,VMAPS,IT]
    R_b = R_b.at[e,:,1:].set(R_b_) #[TESTS,VMAPS,IT]
    R_t = (R_test,R_r,R_g,R_b)
    LTP = (loop_params,theta_test,pos_arr,switch_arr,R_t)# ,dis_arr
    return LTP

def train_loop(loop_params,theta_0,opt_state,x):
    R_tot,sd_tot,R_obj,sd_obj,R_env,sd_env,R_dot,sd_dot,R_sel,sd_sel=(jnp.empty(theta_0["ENV"]["EPOCHS"]) for _ in range(10))
    R_arr = (R_tot,R_obj,R_env,R_dot,R_sel)
    sd_arr = (sd_tot,sd_obj,sd_env,sd_dot,sd_sel)
    LTORS_0 = (loop_params,theta_0,opt_state,R_arr,sd_arr,x)
    (loop_params_,theta_test,opt_state_,R_arr_,sd_arr_,x_) = jax.lax.fori_loop(0,loop_params["EPOCHS"],train_body,LTORS_0) #jnp.int32(x*E),jnp.int32((x+1)*E)x*loop_params["EPOCHS"],(x+1)*loop_params["EPOCHS"]theta_,opt_state_,R/sd arrays
    vals_train = (R_arr_,sd_arr_)#R_obj,R_env,R_dot,R_sel
    return theta_test,vals_train,opt_state_

def test_loop(loop_params,theta_test):
    pos_arr = jnp.empty([loop_params["TESTS"],loop_params["VMAPS"],loop_params["IT"]+1,2])
    switch_arr = jnp.empty([loop_params["TESTS"],loop_params["VMAPS"],loop_params["IT"]+1])
    R_test = jnp.empty([loop_params["TESTS"],loop_params["VMAPS"],loop_params["IT"]+1])
    R_r = jnp.empty([loop_params["TESTS"],loop_params["VMAPS"],loop_params["IT"]+1])
    R_g = jnp.empty([loop_params["TESTS"],loop_params["VMAPS"],loop_params["IT"]+1])
    R_b = jnp.empty([loop_params["TESTS"],loop_params["VMAPS"],loop_params["IT"]+1])
    R_t = (R_test,R_r,R_g,R_b)
    pos_arr = pos_arr.at[:,:,0,:].set(jnp.float32([0,0]))
    LTP_0 = (loop_params,theta_test,pos_arr,switch_arr,R_t) #,dis_arr
    *_,pos_arr,switch_arr,R_t = jax.lax.fori_loop(0,loop_params["TESTS"],test_body,LTP_0) # ,dis_arr
    vals_test = (pos_arr,switch_arr,R_t) #,dis_arr
    return vals_test

def train_outer_loop(loop_params,theta):
    E = loop_params["EPOCHS"]
    R_tot_,sd_tot_,R_obj_,sd_obj_,R_env_,sd_env_,R_dot_,sd_dot_,R_sel_,sd_sel_=(jnp.empty([loop_params["TOT_EPOCHS"]]) for _ in range(10))
    vals_train_tot = (R_tot_,R_obj_,R_env_,R_dot_,R_sel_),(sd_tot_,sd_obj_,sd_env_,sd_dot_,sd_sel_)
    optimizer = optax.adamw(learning_rate=loop_params['UPDATE'],weight_decay=loop_params['WD'])
    opt_state = optimizer.init(theta["GRU"])
    for x in range(loop_params["LOOPS"]):
        theta_,vals_train,opt_state = train_loop(loop_params,theta,opt_state,x)
        (R_tot,R_obj,R_env,R_dot,R_sel),(sd_tot,sd_obj,sd_env,sd_dot,sd_sel) = vals_train # unpack from training, R_arr_,sd_arr_
        R_tot_ = R_tot_.at[x*E:(x+1)*E].set(R_tot)
        sd_tot_ = sd_tot_.at[x*E:(x+1)*E].set(sd_tot)#
        R_obj_ = R_obj_.at[x*E:(x+1)*E].set(R_obj)
        sd_obj_ = sd_obj_.at[x*E:(x+1)*E].set(sd_obj)#
        R_env_ = R_env_.at[x*E:(x+1)*E].set(R_env)
        sd_env_ = sd_env_.at[x*E:(x+1)*E].set(sd_env)#
        R_dot_ = R_dot_.at[x*E:(x+1)*E].set(R_dot)
        sd_dot_ = sd_dot_.at[x*E:(x+1)*E].set(sd_dot)#
        R_sel_ = R_sel_.at[x*E:(x+1)*E].set(R_sel)
        sd_sel_ = sd_sel_.at[x*E:(x+1)*E].set(sd_sel)#
        theta = new_theta(theta_,x)
    vals_train_tot = (R_tot_,R_obj_,R_env_,R_dot_,R_sel_),(sd_tot_,sd_obj_,sd_env_,sd_dot_,sd_sel_)
    return theta,vals_train_tot

def full_loop(loop_params,theta_0): # main routine: R_arr, std_arr = full_loop(params)
    theta_test,vals_train_tot = train_outer_loop(loop_params,theta_0)
    vals_test = test_loop(loop_params,theta_test) #check inputs/outputs
    return (vals_train_tot,vals_test,theta_test)

# ENV parameters
SIGMA_A = jnp.float32(0.5) # 0.4,0.5,0.3,0.5,0.9
SIGMA_R0 = jnp.float32(0.3) # 0.5,0.7,1,0.5,,0.8,0.5,0.8,0.5
SIGMA_RINF = jnp.float32(0.3) # 0.15,0.3,0.6,1.8,0.1,,0.3
SIGMA_N = jnp.float32(1.2) # 1,2,0.3, 1.8,1.6
LAMBDA_N = jnp.float32(0.0001)
LAMBDA_E = jnp.float32(0.03) ### 0.008,0.04,0.1,0.05,0.01,0.1
LAMBDA_D = jnp.float32(0.06) ### 0.08,0.06,0.07,0.03,0.04,0.01,0.001 
LAMBDA_S = jnp.float32(0.015) ### 0.0024,0.0012,0.001,0.0001,0.00001
ALPHA = jnp.float32(0.8) # 0.1,0.7,0.99
STEP = jnp.float32(0.03) # 0.02,0.1, play around with! 0.05,,0.002,0.005
APERTURE = jnp.pi #pi/3
COLORS = jnp.float32([[255,0,0],[0,255,0],[0,0,255]]) # ,[100,100,100],[200,200,200]]) # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[50,0,200]]) # [[255,0,0],[0,200,200],[100,100,100]]
N_DOTS = COLORS.shape[0]
NEURONS = 21 # 11

# GRU parameters
N = NEURONS**2
G = 80 # size of GRU
INIT = jnp.float32(20) # 15-300..,0.3,0.5,0.1,0.2,0.3,,0.5,0.1

# loop params
TOT_EPOCHS = 2000
EPOCHS = 200
LOOPS = TOT_EPOCHS//EPOCHS # TOT_EPOCHS//EPOCHS
IT = 160
VMAPS = 1000 # 500
TESTS = 8
UPDATE = jnp.float32(0.00003) #0.00001,0.00002,0.0001,0.00005,,0.0001,0.00001,0.0005,0.0001,0.00001,0.00002,0.0001,0.00008
WD = jnp.float32(0.0005) # 0.001,0.0001,0.00005,0.00001
TAU = jnp.float32((1-1/jnp.e)*TOT_EPOCHS) # 0.01
optimizer = optax.adamw(learning_rate=UPDATE,weight_decay=WD) #optax.adam(learning_rate=UPDATE)#

# assemble loop_params pytree
loop_params = {
        'TOT_EPOCHS': TOT_EPOCHS,
        'EPOCHS':   EPOCHS,
        'LOOPS' :   LOOPS,
        'VMAPS' :   VMAPS,
        'IT'    :   IT,
        'TESTS' :   TESTS,
    	'UPDATE':   jnp.float32(UPDATE),
        'WD'    :   jnp.float32(WD)
	}

## KEY_INIT = rnd.PRNGKey(0) # 0
ki = rnd.split(rnd.PRNGKey(0),num=30)
# Wr_z0 = (INIT/(G*N))*rnd.normal(ki[1],(G,N),dtype=jnp.float32)
# Wg_z0 = (INIT/(G*N))*rnd.normal(ki[1],(G,N),dtype=jnp.float32)
# Wb_z0 = (INIT/(G*N))*rnd.normal(ki[1],(G,N),dtype=jnp.float32)
# U_z0 = (INIT/(G*G))*rnd.normal(ki[2],(G,G),dtype=jnp.float32)
# b_z0 = (INIT/G)*rnd.normal(ki[3],(G,),dtype=jnp.float32)
# Wr_r0 = (INIT/(G*N))*rnd.normal(ki[4],(G,N),dtype=jnp.float32)
# Wg_r0 = (INIT/(G*N))*rnd.normal(ki[4],(G,N),dtype=jnp.float32)
# Wb_r0 = (INIT/(G*N))*rnd.normal(ki[4],(G,N),dtype=jnp.float32)
# U_r0 = (INIT/(G*G))*rnd.normal(ki[5],(G,G),dtype=jnp.float32)
# b_r0 = (INIT/G)*rnd.normal(ki[6],(G,),dtype=jnp.float32)
# Wr_h0 = (INIT/(G*N))*rnd.normal(ki[7],(G,N),dtype=jnp.float32)
# Wg_h0 = (INIT/(G*N))*rnd.normal(ki[7],(G,N),dtype=jnp.float32)
# Wb_h0 = (INIT/(G*N))*rnd.normal(ki[7],(G,N),dtype=jnp.float32)
# U_h0 = (INIT/(G*G))*rnd.normal(ki[8],(G,G),dtype=jnp.float32)
# b_h0 = (INIT/G)*rnd.normal(ki[9],(G,),dtype=jnp.float32)
# W_r_z0 = (INIT/G)*rnd.normal(ki[10],(G,),dtype=jnp.float32)
# W_r_f0 = (INIT/G)*rnd.normal(ki[11],(G,),dtype=jnp.float32)
# W_r_hh0 = (INIT/G)*rnd.normal(ki[12],(G,),dtype=jnp.float32)
# C0 = (INIT/(2*G))*rnd.normal(ki[13],(2,G),dtype=jnp.float32)
# E0 = (INIT/(4*G))*rnd.normal(ki[14],(4,G),dtype=jnp.float32)
# D0 = (INIT/(4*G))*rnd.normal(ki[15],(4,G),dtype=jnp.float32)
# S0 = (INIT/(N_DOTS*G))*rnd.normal(ki[16],(N_DOTS,G),dtype=jnp.float32) # (check produce logits)
DOTS = gen_dots(ki[17],N_DOTS,VMAPS,EPOCHS) # [EPOCHS,N_DOTS,2,VMAPS]
EPS = gen_eps(ki[18],EPOCHS,IT,VMAPS)
SELECT = gen_select(ki[19],N_DOTS,EPOCHS,VMAPS) #jnp.eye(N_DOTS)[rnd.choice(ki[17],N_DOTS,(EPOCHS,VMAPS))]
THETA_I = gen_neurons(NEURONS,APERTURE)
THETA_J = gen_neurons(NEURONS,APERTURE)
H0 = gen_h0(ki[18],G,VMAPS,EPOCHS)

# assemble theta pytree
theta_0 = { "GRU" : {
    	# "h0"     : h0,
    	# "Wr_z"   : Wr_z0,
		# "Wg_z"   : Wg_z0,
		# "Wb_z"   : Wb_z0,
    	# "U_z"    : U_z0,
    	# "b_z"    : b_z0,
    	# "Wr_r"   : Wr_r0,
		# "Wg_r"   : Wg_r0,
		# "Wb_r"   : Wb_r0,
    	# "U_r"    : U_r0,
    	# "b_r"    : b_r0,
    	# "Wr_h"   : Wr_h0,
		# "Wg_h"   : Wg_h0,
		# "Wb_h"   : Wb_h0,
    	# "U_h"    : U_h0,
    	# "b_h"    : b_h0,
		# "W_r_z"  : W_r_z0,
        # "W_r_f"  : W_r_f0,
        # "W_r_hh" : W_r_hh0,
    	# "C"	     : C0,
        # "E"      : E0,
        # "D"      : D0,
        # "S"      : S0
	},
        	"ENV" : {
        "H0"        : H0,
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
        "IT"        : IT,
        "VMAPS"     : VMAPS,
        "EPOCHS"    : EPOCHS,
        "LAMBDA_N"  : LAMBDA_N,
        "LAMBDA_E"  : LAMBDA_E,
        "LAMBDA_D"  : LAMBDA_D,
        "LAMBDA_S"  : LAMBDA_S,
        "ALPHA"     : ALPHA
	}
        	}

theta_0_ = load_('v9_theta_test_trained_24_04-1433.pkl')
theta_0["GRU"] = theta_0_["GRU"]
theta_0["ENV"] = jax.lax.stop_gradient(theta_0["ENV"])

###
startTime = datetime.now()
(vals_train,vals_test,theta_test) = full_loop(loop_params,theta_0) #,vals_test
time_elapsed = datetime.now() - startTime
print(f'Completed in: {time_elapsed}, {time_elapsed/TOT_EPOCHS} s/epoch')

# plot training
(R_tot,R_obj,R_env,R_dot,R_sel),(sd_tot,sd_obj,sd_env,sd_dot,sd_sel) = vals_train
plt.figure()
title__ = f'v9 training, tot epochs={TOT_EPOCHS}, it={IT}, vmaps={VMAPS}, init={INIT:.2f}, update={UPDATE:.5f}, SIGMA_A={SIGMA_A:.1f}, SIGMA_R0={SIGMA_R0:.1f}, SIGMA_RINF={SIGMA_RINF:.1f}, \n SIGMA_N={SIGMA_N:.1f}, STEP={STEP:.3f} WD={WD:.5f}, LAMBDA_D={LAMBDA_D:.4f}, LAMBDA_E={LAMBDA_E:.4f}, LAMBDA_S={LAMBDA_S:.4f}, NEURONS={NEURONS}, p=R_norm/{7}' # \n colors={jnp.array_str(COLORS[0][:]) + jnp.array_str(COLORS[1][:]) + jnp.array_str(COLORS[2][:])}' #  + jnp.array_str(COLORS[3][:]) + jnp.array_str(COLORS[4][:])}'
fig,ax = plt.subplots(2,3,figsize=(16,9))
plt.suptitle(title__,fontsize=14)
plt.subplot(2,3,1)
plt.errorbar(jnp.arange(len(R_tot)),R_tot,yerr=sd_tot/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.ylabel(r'$R_{tot}$',fontsize=15)
plt.xlabel(r'Iteration',fontsize=12)
plt.subplot(2,3,2)
plt.errorbar(jnp.arange(len(R_obj)),R_obj,yerr=sd_obj/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.ylabel(r'$R_{obj}$',fontsize=15)
plt.xlabel(r'Iteration',fontsize=12)
plt.subplot(2,3,3)
plt.errorbar(jnp.arange(len(R_env)),R_env,yerr=sd_env/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.ylabel(r'$R_{env}$',fontsize=15)
plt.xlabel(r'Iteration',fontsize=12)
plt.subplot(2,3,4)
plt.errorbar(jnp.arange(len(R_dot)),R_dot,yerr=sd_dot/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.ylabel(r'$R_{dot}$',fontsize=15)
plt.xlabel(r'Iteration',fontsize=12)
plt.subplot(2,3,5)
plt.errorbar(jnp.arange(len(R_sel)),R_sel,yerr=sd_sel/2,ecolor="black",elinewidth=0.5,capsize=1.5)
plt.ylabel(r'$R_{sel}$',fontsize=15)
plt.xlabel(r'Iteration',fontsize=12)
plt.tight_layout()
# plt.show()

path_ = str(Path(__file__).resolve().parents[1]) + '/figs/task9/'
dt = datetime.now().strftime("%d_%m-%H%M")
plt.savefig(path_ + 'train_' + dt + '.png')

#analyse testing...
pos_arr,switch_arr,(R_test,R_r,R_g,R_b) = vals_test # dis_arr, R_obj_t,R_env_t,R_dot_t,R_sel_t,dis_t,pos_t = vals_test

geod_dist_vmap = jax.vmap(geod_dist,in_axes=(None,0),out_axes=0)# (e_t,pos_t)
colors_ = np.float32([[255,0,0],[0,255,0],[0,0,255]])/255 #theta_0["ENV"]["COLORS"]/255
colormap = cm.seismic(np.linspace(0,1,IT+1), alpha=1)

# plot testing
plt.figure()
title__ = f'v9 testing, tot epochs={TOT_EPOCHS}, it={IT}, vmaps={VMAPS}, init={INIT:.2f}, update={UPDATE:.5f}, SIGMA_A={SIGMA_A:.1f}, SIGMA_R0={SIGMA_R0:.1f}, SIGMA_RINF={SIGMA_RINF:.1f}, \n SIGMA_N={SIGMA_N:.1f}, STEP={STEP:.3f} WD={WD:.5f}, LAMBDA_D={LAMBDA_D:.4f}, LAMBDA_E={LAMBDA_E:.4f}, LAMBDA_S={LAMBDA_S:.4f}, NEURONS={NEURONS}, p=R_norm/{7}'
fig,axis = plt.subplots(2*TESTS,4,figsize=(15,5*TESTS+2))#(4,5)
plt.suptitle(title__,fontsize=14)
for i in range(loop_params["TESTS"]):
    k = rnd.randint(ki[18+i],(),0,loop_params["VMAPS"]) # rnd.choice(ki[18+j],loop_params["VMAPS"],replace=False)
    ax0 = plt.subplot2grid((2*TESTS,4),(2*i,2),colspan=2)
    ax0.set_ylabel(r'$dis$',fontsize=16)
    # dis_old = dis_arr[i,k,:,:] # [TESTS,VMAPS,IT,3]
    ax0.tick_params(axis='both', which='major', labelsize=14)
    ax0.set_ylim(-0.2,jnp.sqrt(2)*jnp.pi+0.2) # 2*jnp.pi*jnp.sqrt(2)

    ax1 = plt.subplot2grid((2*TESTS,4),(2*i,0),rowspan=2,colspan=2)
    pos_ = pos_arr[i,k,:,:] # [TESTS,VMAPS,IT,2]
    ax1.scatter(pos_[:,0],pos_[:,1],s=25,color=colormap,marker='.')#30 axis[i,1]

    sel_ = theta_test["ENV"]["SELECT"][i,k,:] # [EPOCHS,VMAPS,3]
    dots_ = theta_test["ENV"]["DOTS"][i,:,:,k] # [EPOCHS,3,2,VMAPS]
    dis_ = geod_dist_vmap(dots_,pos_)
    ax2 = plt.subplot2grid((2*TESTS,4),(2*i+1,2),colspan=2)
    ax2.plot(R_test[i,k,1:],color='k',linewidth=2) #axis[i,2]
    ax2.set_ylabel(r'$R_{tot}$',fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylim(-1.1,0.1) # -1.1,0 -jnp.sqrt(2*jnp.pi)*
    for j in range(theta_test["ENV"]["N_DOTS"]):
        ax0.plot(dis_[1:,j],color=(colors_[j,:]),linewidth=2)#3,axis[i,0], tuple
        ax0.plot(dis_[1:,j],color=(colors_[j,:]),alpha=0.5,linewidth=1)
        ax1.scatter(dots_[j,0],dots_[j,1],s=70,marker='x',color=(colors_[j,:]))#60,

    ax1.set_xlim(-jnp.pi,jnp.pi)
    ax1.set_ylim(-jnp.pi,jnp.pi)
    ax1.set_xticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
    ax1.set_xticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=14)
    ax1.set_yticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
    ax1.set_yticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=14)
    ax1.set_aspect('equal')
    ax1.set_title(f'sel={sel_}',fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94) 
    
plt.savefig(path_ + 'test_' + dt + '.png')

save_pkl((vals_train,vals_test,theta_test),'v9_all')
# save_pkl(theta_test,'v9_theta_test_trained')