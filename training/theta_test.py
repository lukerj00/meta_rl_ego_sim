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

# def gen_neurons(NEURONS,APERTURE):
# 	return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype="float32")
# gen_neurons = jit(gen_neurons,static_argnums=(0,1))  

# def create_dots(N_DOTS,KEY_DOT,VMAPS,EPOCHS):
# 	return rnd.uniform(KEY_DOT,shape=[EPOCHS,N_DOTS,2,VMAPS],minval=-jnp.pi,maxval=jnp.pi,dtype="float32")
# create_dots = jit(create_dots,static_argnums=(0,2,3))

# SIGMA_A = jnp.float32(1.2) # 0.9
# SIGMA_R = jnp.float32(0.5) # 0.3
# SIGMA_N = jnp.float32(1.8) # 1.6
# STEP = jnp.float32(0.005) # play around with! 0.005
# APERTURE = jnp.pi/3
# COLORS = jnp.float32([[255,100,50],[50,255,100],[100,50,255]]) # ,[100,100,100],[200,200,200]]) # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[50,0,200]]) # [[255,0,0],[0,200,200],[100,100,100]]
# N_DOTS = COLORS.shape[0]
# NEURONS = 11

# # GRU parameters
# N = NEURONS**2
# G = 80 # size of mGRU (total size = G+N_DOTS)
# KEY_INIT = rnd.PRNGKey(0) # 0
# INIT = jnp.float32(0.1) # 0.1

# # loop params
# EPOCHS = 5001
# IT = 20
# VMAPS = 500
# UPDATE = jnp.float32(0.009) # 0.001
# R_arr = jnp.empty(EPOCHS)*jnp.nan
# std_arr = jnp.empty(EPOCHS)*jnp.nan
# optimizer = optax.adam(learning_rate=UPDATE)

# # assemble loop_params pytree
# loop_params = {
#     	'UPDATE':   jnp.float32(UPDATE),
#     	'R_arr':	jnp.zeros(EPOCHS)*jnp.nan,
#     	'std_arr':  jnp.zeros(EPOCHS)*jnp.nan
# 	}

# # generate initial values
# ki = rnd.split(KEY_INIT,num=20)
# h0 = rnd.normal(ki[0],(G,),dtype="float32")
# Wr_f0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype="float32")
# Wg_f0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype="float32")
# Wb_f0 = (INIT/G*N)*rnd.normal(ki[1],(G,N),dtype="float32")
# U_f0 = (INIT/G*G)*rnd.normal(ki[2],(G,G),dtype="float32")
# b_f0 = (INIT/G)*rnd.normal(ki[3],(G,),dtype="float32")
# Wr_h0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype="float32")
# Wg_h0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype="float32")
# Wb_h0 = (INIT/G*N)*rnd.normal(ki[4],(G,N),dtype="float32")
# W_s = (INIT)*rnd.normal(ki[5],(G,N_DOTS),dtype="float32")
# U_h0 = (INIT/G*G)*rnd.normal(ki[6],(G,G),dtype="float32")
# b_h0 = (INIT/G)*rnd.normal(ki[7],(G,),dtype="float32")
# C0 = (INIT/2*G)*rnd.normal(ki[8],(2,G),dtype="float32")
# THETA_I = gen_neurons(NEURONS,APERTURE)
# THETA_J = gen_neurons(NEURONS,APERTURE)
# DOTS = create_dots(N_DOTS,ki[9],VMAPS,EPOCHS)
# EPS = rnd.normal(ki[10],shape=[EPOCHS,IT,2,VMAPS],dtype="float32")
# SELECT = jnp.eye(N_DOTS)[rnd.choice(ki[11],N_DOTS,(EPOCHS,VMAPS))]

# # assemble theta pytree
# theta = { "GRU" : {
#     	"h0"   : h0, # ?
#     	"Wr_f" : Wr_f0,
#     	"Wg_f" : Wg_f0,
#     	"Wb_f" : Wb_f0,       	 
#     	"U_f"  : U_f0,
#     	"b_f"  : b_f0,
#     	"Wr_h" : Wr_h0,
#     	"Wg_h" : Wg_h0,
#     	"Wb_h" : Wb_h0,
#     	"W_s"  : W_s,
#     	"U_h"  : U_h0,
#     	"b_h"  : b_h0,
#     	"C"	: C0
# 	},
#         	"ENV" : {
#     	"THETA_I"  	: THETA_I,
#     	"THETA_J"  	: THETA_J,
#     	"COLORS"   	: COLORS,
#     	"SIGMA_N"  	: SIGMA_N,
#     	"SIGMA_A"  	: SIGMA_A,
#     	"SIGMA_R"  	: SIGMA_R,
#     	"STEP"     	: STEP,
#     	"DOTS"     	: DOTS,
#     	"EPS"      	: EPS,
#     	"SELECT"   	: SELECT
# 	}
#         	}

# gru_ = jax.tree_util.tree_map(lambda g: jnp.mean(g,axis=0), theta["GRU"])
# val_flat,val_tree = jax.tree_util.tree_flatten(theta["GRU"])
# # print(len(val_flat),val_flat,val_tree)
# # print(len(transformed))
# # print(type(theta["GRU"]),len(theta["GRU"]),type(gru_),len(gru_))
# # print(list(theta["GRU"].keys()),list(theta["GRU"].values()))
# print(val_flat,val_tree)
# transformed = jax.tree_util.tree_unflatten(val_tree,val_flat)
# print(transformed)
# # print(gru_.keys())
# # print(list(theta["GRU"].keys())[:])
# # g_k = list(theta["GRU"].keys())[1:]
# # g_v = list(theta["GRU"].values())[1:]
# # dict_new = {}
# # for i,k in enumerate(g_k):
# # 	dict_new[k] = g_v[i]
# # print(dict_new.keys())
# a = np.arange(24).reshape(2,3,4)
# b = np.arange(12).reshape(3,4)
# a[:,:,:] = b
# print(b)
# print('a',a[0],a[1])

# c = np.random.random((2,3,4))
# print('c',c)
# shape_ = c.shape
# c = jnp.resize(c[:,:,0],(shape_[2],shape_[0],shape_[1])).transpose(1,2,0)
# print('c',c)

# d = np.random.random((3,4))
# print('d',d)
# d = d[0,:]
# print('d',d)
# c = d
# print('e',c)

# a = 1
# b = 2
# c = 10

# print('*******',a/b*c)
# print('-------',(a/b*c))

# @jit
# def abs_dist(e_t,pos):
# 	# e_t_ = (e_t + jnp.pi)%(2*jnp.pi)-jnp.pi
#     pos_ = (pos + jnp.pi)%(2*jnp.pi)-jnp.pi
#     dis_rel = e_t-pos_
#     r = jnp.sqrt(jnp.square(dis_rel[:,0])+jnp.square(dis_rel[:,1]))
#     return (e_t,pos,pos_,dis_rel,r)

# e_t = 10*jnp.ones((3,2))
# pos = jnp.arange(8,10).reshape(2,)#10*
# (e,p,p_,d,r) = (abs_dist(e_t,pos))

# print(e,p,p_,d,r)

# e = 10000
# b = 1000
# c = e/b
# d = e//b
# print(c,type(c),d,type(d))
# def true_fnc(dpa):
#     dot,pos_t,ALPHA = dpa
#     print('true')
#     return (jnp.linalg.norm((dot-pos_t),ord=2) + 5)

# def false_fnc(dpa):
#     dot,pos_t,ALPHA = dpa
#     print('false')
#     return jnp.linalg.norm((dot-pos_t),ord=2)

# dot = jnp.array([1.0,1.0])
# pos_t = jnp.array([1.1,1.1])
# ALPHA = 0.2
# # print(jnp.linalg.norm((dot-pos_t),ord=2))
# dpa = (dot,pos_t,ALPHA)
# ret = jax.lax.cond(jnp.linalg.norm((dot-pos_t))<=ALPHA,true_fnc,false_fnc,dpa)
# print(ret)

a = jnp.array([0,1,2,3])
b = jnp.array([1,0.5,0,4])
c = jnp.minimum(a,b)
print('c=',c)