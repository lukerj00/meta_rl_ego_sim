import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import pathlib
from pathlib import Path
import os
import sys

# # for e in range(EPOCHS):
#     # use e
# EPOCHS = 100
# #TRR = (theta,R_arr,R_std)
# def body_fnc(e,TRR): # returns theta
#     # stuff with e
#     (theta,R_arr,R_std) = TRR
#     h0 = theta["GRU"]["h0"]
#     e0 = theta["ENV"]["DOTS"][e,:,:,:] 
#     sel = theta["ENV"]["SELECT"][e,:,:]
#     eps = theta["ENV"]["EPS"][e,:,:,:]

    
    
#     # each iteration effects next LTRR (L{R_arr,std_arr},T{GRU})
#     # vmap tot_reward over dots (e0), eps (EPS) and sel (SELECT)); find avg r_tot, grad
#     val_grad_vmap = jax.vmap(jax.value_and_grad(tot_reward,argnums=2,allow_int=True),in_axes=(2,None,None,0,2,None),out_axes=(0,0))
#     R_tot,grads = val_grad_vmap(e0,h0,theta,sel,eps,e)
#     grads_ = jax.tree_util.tree_map(lambda g: jnp.mean(g,axis=0), grads["GRU"])
#     R_arr = R_arr.at[e].set(jnp.mean(R_tot))
#     std_arr = std_arr.at[e].set(jnp.std(R_tot))
    
#     # update theta
#     update, opt_state = optimizer.update(grads_, opt_state, theta["GRU"])
#     theta["GRU"] = optax.apply_updates(theta["GRU"], update)

#     return TRR # becomes val
# final = jax.lax.fori_loop(0, EPOCHS, body_fnc, 0)
# print(final)
# print(os.path.basename(__file__).split('.')[0])

# colors, select
# colors_path = 'pkl_colors_.pkl'
select_path = 'pkl_select_3.pkl'
path_ = str(Path(__file__).resolve().parents[1])
path_pkl = path_ + "\\pkl\\"
# colors_ = open(path_pkl + colors_path,'rb') # (remember to open as binary)
select_ = open(path_pkl + select_path,'rb')
# COLORS = pickle.load(colors_) # [r,g,b*5]
SELECT = pickle.load(select_) # [EPOCHS=1000,VMAPS=200,N_DOTS=5]
# SEL_ = SELECT[0::100,:,:]
# SEL_999 = SELECT[999,:,:]

print(SELECT.shape)
# print(COLORS,COLORS.shape)
# sel_ = np.array(SELECT[0,0,:],dtype=bool)
# print(sel_) # [EPOCHS,VMAPS,N_DOTS] SELECT[EPOCHS_PLOT[0],np.random.randint(VMAPS),:]
# print(COLORS[0,:]) # [N_DOTS,(R,G,B)]
# clr = tuple(COLORS[sel_,:]) # tuple(COLORS[SELECT[EPOCHS_PLOT[0],np.random.randint(VMAPS),:],:])
# print(clr,type(clr))

# t = [1,2,3]
# print(len(t))