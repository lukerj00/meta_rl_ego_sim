import jax.numpy as jnp
import jax.random as rnd
import csv
from pathlib import Path
import os
from os.path import dirname, abspath
from datetime import datetime

dt = datetime.now().strftime("%d/%m_%H:%M")
print(dt)

t = jnp.arange(10).reshape((5,2))
print(t)
print(jnp.sum(t**2,axis=1))

def abs_dist(e_t):
    e_t_ = (e_t + jnp.pi)%(2*jnp.pi)-jnp.pi
    return jnp.sqrt(e_t_[:,0]**2+e_t_[:,1]**2)

print(t, abs_dist(t))

print(0%50)

# obj = -jnp.exp(-jnp.sum((t)**2,axis=1)/1**2)
# print(obj, obj.shape)
# print(os.getcwd()) # os.path.dirname(

# EPOCHS = 1000
# N_DOTS = 5
# VMAPS = 100
# KEY_INIT = rnd.PRNGKey(1)
# ki = rnd.split(KEY_INIT,num=50)

# print('***',str(Path(__file__).resolve().parents[1]) + '\\')

# for e in range(1000):
#     if (e % 100 == 0)|(e==999):
#         print(e, 'is')

# SELECT = jnp.eye(N_DOTS)[rnd.choice(ki[8],N_DOTS,(EPOCHS,VMAPS,))] # 1000 x 5 N_DOTS,EPOCHS,VMAPS->1000,100,5 , N_DOTS,VMAPS,EPOCHS->100,1000,5 want 5,1000,100
# print(SELECT.shape)

# def csv_write(data,file): # use 'csv_test_multi.csv'
#     # data = data.ravel()
#     path = str(Path(__file__).parent.absolute()) + '\\' # "C:\\Users\\lukej\\Documents\\MPhil\\meta_rl_ego_sim\\csv_plotter\\"
#     with open(path + file,'a',newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(data)

# csv_write(jnp.arange(EPOCHS),'test_delete.csv')

# test = jnp.arange(120).reshape((2,3,4,5))
# test_e = test[0,:,:,:]
# print(test_e.shape)


