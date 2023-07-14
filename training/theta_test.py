import jax
import jax.numpy as jnp
from jax import jit
import jax.random as rnd
# from jax.experimental.host_callback import id_print
# from jax.experimental.host_callback import call
import optax
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('sAgg')
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

# def gen_vectors(MODULES,APERTURE):
#     m = MODULES
#     M = m**2
#     A = APERTURE
#     x = jnp.linspace(-A,A,m) # x = jnp.linspace(-(A-A/m),(A-A/m),m) # CHECK
#     x_ = jnp.tile(x,(m,1))
#     x__ = x_.reshape((M,)) # .transpose((1,0))
#     y = jnp.linspace(-A,A,m) # y = jnp.linspace(-(A-A/m),(A-A/m),m) # CHECK
#     y_ = jnp.tile(jnp.flip(y).reshape(m,1),(1,m))
#     y__ = y_.reshape((M,))
#     v = jnp.vstack([x__,y__])
#     return x_,y_,v

# A = jnp.pi
# M = 2
# E=3
# V=4

# POS_0 = rnd.choice(rnd.PRNGKey(0),jnp.linspace(-A,A,M),(E,V,2))

# a = jnp.array([0,1,2,3])
# b = jnp.array([1,0.5,0,4])
# c = jnp.minimum(a,b)
# print('c=',c)

# x,y,v = gen_vectors(4,jnp.pi)
# print(x,x.shape,y,y.shape,v,v.shape)
# print(POS_0,POS_0.shape)

# arr = jnp.array([1,2,3,4,5,6,7,8,9,10]).reshape(2,5)
# for i in (arr.T):
#     print('i=',i)


# NEURONS = 10
# APERTURE = jnp.pi

# neuron_locs = gen_vectors(NEURONS,APERTURE)

# for ind,n in enumerate(neuron_locs.T):
#     print('n=',n,'ind=',ind)

# arr = jnp.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,5)

##
# def neuron_act(params,dots,pos): #e_t_1,th_j,th_i,SIGMA_A,COLORS
#     COLORS = params["COLORS"]
#     th_x = params["THETA_X"]
#     th_y = params["THETA_Y"]
#     SIGMA_A = params["SIGMA_A"]
#     D_ = COLORS.shape[0]
#     N_ = th_x.size
#     th_x = jnp.tile(th_x,(N_,1)).reshape((N_**2,))
#     th_y = jnp.tile(jnp.flip(th_y).reshape(N_,1),(1,N_)).reshape((N_**2,))
#     G_0 = jnp.vstack([th_x,th_y])
#     # print('G_0=',G_0)
#     # G_0 = jnp.vstack((jnp.tile(th_x,N_),jnp.tile(th_y,N_)))
#     G1 = jnp.tile(G_0.reshape(2,N_**2,1),(1,1,D_))
#     G2 = jnp.moveaxis(G1,(0,1,2),(2,0,1)) #jnp.tile(G_0.reshape(N_**2,1,2),(1,D_,1)) # 
#     print("G1={}",G1[:,:,0],G1.shape)
#     print("G2={}",G2[:,0,:],G2.shape)
#     # jax.debug.print("dots={}",dots)
#     E1 = G1.transpose((1,0,2)) - ((dots-pos)).T #.reshape((2,1))
#     E2 = G2 - (dots-pos) #
#     print("E1={}",E1.shape,E1[:,:,0])
#     print("E2={}",E2.shape,E2[:,0,:])
#     print("E1[:,0,:]=",E1[:,0,:],E1[:,0,:].shape)
#     print("E1[:,1,:]=",E1[:,1,:],E1[:,1,:].shape)
#     print("EXP_E_TOT1=",jnp.exp(jnp.cos(E1[:,0,:])+jnp.cos(E1[:,1,:])-2))
#     # print("E_TOT2=",E2[:,:,0] + E2[:,:,1])
#     act1 = jnp.exp((jnp.cos(E1[:,0,:]) + jnp.cos(E1[:,1,:]) - 2)/SIGMA_A**2).T
#     act1R = jnp.exp((jnp.cos(E1[:,0,:]) + jnp.cos(E1[:,1,:]) - 2)/SIGMA_A**2).reshape((D_,N_**2))
#     act2 = jnp.exp((jnp.cos(E2[:,:,0]) + jnp.cos(E2[:,:,1]) - 2)/SIGMA_A**2).T
#     print("act1={}",act1,act1.shape)
#     print("act1(RESHAPED)={}",act1R,act1R.shape)
#     print("act2={}",act2,act2.shape)
#     C = (COLORS/255).transpose((1,0))
#     act_r,act_g,act_b = jnp.matmul(C,act2) #.reshape((3*N_**2,))
#     act_rgb = jnp.concatenate((act_r,act_g,act_b))
#     return act_rgb

# def gen_vectors(MODULES,APERTURE):
#     m = MODULES
#     M = m**2
#     A = APERTURE
#     x = jnp.linspace(-A,A,m) # x = jnp.linspace(-(A-A/m),(A-A/m),m) # CHECK
#     x_ = jnp.tile(x,(m,1))
#     x_ = x_.reshape((M,)) # .transpose((1,0))
#     y = jnp.linspace(-A,A,m) # y = jnp.linspace(-(A-A/m),(A-A/m),m) # CHECK
#     y_ = jnp.tile(jnp.flip(y).reshape(m,1),(1,m))
#     y_ = y_.reshape((M,))
#     v = jnp.vstack([x_,y_]) # [2,M]
#     return v

# def gen_neurons(NEURONS,APERTURE):
# 	return jnp.linspace(-APERTURE,APERTURE,NEURONS,dtype=jnp.float32)

# def mod_(val):
#     return (val+jnp.pi)%(2*jnp.pi)-jnp.pi

# NEURONS = 10
# COLORS = jnp.array([[255,0,0],[0,255,0],[0,0,255]])
# APERTURE = jnp.pi
# th_j = gen_neurons(NEURONS,APERTURE)
# th_i = gen_neurons(NEURONS,APERTURE)

# params = {
#     "THETA_X": th_j,
#     "THETA_Y": th_i,
#     "SIGMA_A": 0.3,
#     "COLORS": COLORS
# }

# pos = jnp.array([3,3])
# dots = jnp.array([[-1,-1],[0,0],[2,2]])
# N_DOTS = dots.shape[0]
# act = neuron_act(params,dots,pos)
# act = act.reshape((3,NEURONS**2))
# print('act=',act.shape,act)

# neuron_locs = gen_vectors(NEURONS,APERTURE)

# plt.figure()
# for ind,n in enumerate(neuron_locs.T):
#     # print('n=',n,'ind=',ind)
#     plt.scatter(n[0],n[1],c=np.float32(act[:,ind]),s=np.sum(act[:,ind])*100)
# for j in range(N_DOTS):
#     plt.scatter(mod_(dots[j,0]-pos[0]),mod_(dots[j,1]-pos[1]),c=COLORS[j,:]/255,s=200,marker='x')
#     # plt.scatter(dots,dots,c='r',s=100,marker='x')
# plt.axis('equal')
# plt.show()
##
# APERTURE = jnp.pi
# MODULES=3
N_DOTS=3
EPOCHS=5
VMAPS=2
# POS_0 = rnd.choice(rnd.PRNGKey(0),jnp.linspace(-APERTURE,APERTURE,MODULES),(EPOCHS,VMAPS,2)) #rnd.uniform(ke[3],shape=(EPOCHS,VMAPS,2),minval=-APERTURE,maxval=APERTURE) ### FILL IN; rand array [E,V,2]
# print(POS_0,POS_0.shape)

# def gen_dots(key,EPOCHS,VMAPS,N_DOTS,APERTURE):
#     keys = rnd.split(key,N_DOTS)
#     dots_0 = rnd.uniform(keys[0],shape=(EPOCHS,VMAPS,1,2),minval=jnp.array([0,0]),maxval=jnp.array([APERTURE,APERTURE]))
#     dots_1 = rnd.uniform(keys[1],shape=(EPOCHS,VMAPS,1,2),minval=jnp.array([0,-APERTURE]),maxval=jnp.array([APERTURE,0]))
#     dots_2 = rnd.uniform(keys[2],shape=(EPOCHS,VMAPS,1,2),minval=jnp.array([-APERTURE,-APERTURE]),maxval=jnp.array([0,APERTURE]))
#     dots_tot = jnp.concatenate((dots_0,dots_1,dots_2),axis=2)
#     # DOTS = rnd.uniform(key,shape=(EPOCHS,VMAPS,N_DOTS,2),minval=-APERTURE,maxval=APERTURE)
#     return dots_tot

# dots_tot = gen_dots(rnd.PRNGKey(0),10,5,3,jnp.pi)
# print(dots_tot,dots_tot.shape)

def adder(x,y):
    return x+y

z=0
for i in range(5):
    z = adder(i,z)
    print(z)

# x = 200
# y = 100

# print(x//y,range(x//y),range(x//x))
# for i in range(x//y):
#     print(i)
# for j in range(x//x):
#     print(j)

# print(jnp.linspace(-(2-2/5),(2-2/5),5))

# a = jnp.arange(9).reshape((3,3))
# b = a[0]
# c=a[0,:]

# print(a,b,c)

# SELECT = jnp.eye(N_DOTS)[rnd.choice(rnd.PRNGKey(0),N_DOTS,(EPOCHS,VMAPS))]
# print(SELECT,SELECT.shape)

def function_a(x):
    return x + 5

def function_b(x):
    return x ** 2

def dynamic_while_loop(condition_fn):
    def loop_body(carry):
        i, x = carry
        x = x + 3  # Code block that runs every iteration

        # fn = lambda x: function_a(x)
        fn = jax.lax.cond(x % 3 == 0, function_a, function_b, operand=x)

        new_x = fn(x)
        return (i + 1, new_x)  # Return the updated carry as a tuple

    initial_carry = (0, 1)  # ((i, x),)
    _, final_value = jax.lax.while_loop(condition_fn, loop_body, initial_carry)
    return final_value

def condition_fn(carry):
    i,x = carry
    return i > 15  # Example condition for terminating the loop

final_value = dynamic_while_loop(condition_fn)
print(final_value)
print('bjhvkjhv')