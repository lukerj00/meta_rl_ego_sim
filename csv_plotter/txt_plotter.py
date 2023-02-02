import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax
# import jax._src.device_array
# import jax._src.abstract_arrays
import jax.random as rnd
import numpy as np
from numpy import genfromtxt
import pickle
import pathlib
from pathlib import Path
import os
import sys
from datetime import datetime

path_ = str(Path(__file__).resolve().parents[1])
stdout_ = path_ + '\\stdout\\'
dump_ = open(stdout_ + 'dump10.txt','r').read()
### specify colors; select; epochs to plot
colors_file = 'pkl_colors_3_.pkl' # CHANGE
# select_file = 'pkl_select_3_.pkl' # CHANGE
# print(dump_[0:50])

path_pkl = path_ + "\\pkl\\"
colors_ = open(path_pkl + colors_file,'rb')
colors = np.float32([[255,100,50],[50,255,100],[100,50,255]])/255 # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[# colors = pickle.load(colors_) # [r,g,b*5]

# txt to arrays (regex)
### sel
epoch3 = ['10000','11000']
epoch_reg = r'(?<=epoch \= '+re.escape(epoch3[0])+')'+r'(.*)(?=epoch \= '+re.escape(epoch3[1])+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
# epoch_reg = r'(?<=epoch \= 10000)(.*)(?=epoch \= 11000)' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
dumpstr_ = repr(dump_)
epoch_ = re.findall(epoch_reg,dumpstr_,re.DOTALL)
epoch_ = ''.join(epoch_)
epoch_ = re.sub(r'\s+','',epoch_)

# epoch_ = re.sub(r'\\n',' ',epoch_)
sel_reg = r'(?<=sel\=\[)(.*)(?=\.\]\\n)'
sel_ = re.findall(sel_reg,epoch_)
sel_ = ''.join(sel_)
sel_ = re.findall(r'\d+',sel_)
sel_ = ''.join(sel_)
# print('\n sel:',sel_)
sel_list = [sel_[i:i+3] for i in range(0, len(sel_), 3)]# sel_ = sel_.lstrip(r'\[')
sel = np.asarray(sel_list)
print(sel[0],type(sel))

### dis 3
epoch_reg = r'(?<=epoch \= '+re.escape(epoch3[0])+')'+r'(.*)(?=epoch \= '+re.escape(epoch3[1])+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
dumpstr_ = repr(dump_)
epoch_ = re.findall(epoch_reg,dumpstr_,re.DOTALL)
epoch_ = ' '.join(epoch_)
epoch_ = re.sub(r'\s+',' ',epoch_)

dis_reg = r'(?<=dis\=\[)(.*)(?=R_tot)'
dis_ = re.findall(dis_reg,epoch_)
# print('\n','dis:',type(dis_))
dis_ = ''.join(dis_)
dis_ = re.sub('dis',' ',dis_)
dis_ = re.sub(r'(\[|\]|\\n|=)','',dis_)
dis_ = re.sub('dis','',dis_)

# print('\n','dis:',dis_,len(dis_))
dis_ = dis_.split(' ')
dis_ = list(filter(None, dis_))
# print('\n','dis:',dis_,len(dis_),len(dis_)/3)
# l = [dis_[i:i+3] for i in range(0, len(dis_), 3)]
# print(dis_,len(dis_))
dis_ = dis_[:(3*25*12)]
dis = np.asarray(dis_).reshape([-1,3]).astype(np.float32)
print(dis.shape,type(dis[0]))

# print('\n','dis:',dis_)
# print(len(dis_))
# dis_ = dis_.split(']]\n')
# print(dis_)
# clr_ = clr_.rstrip(r',')
# clr_ = re.sub(r'(\[|\])','',clr_)
# clr_ = clr_.split(",")
# print(clr_)
# COLORS = np.array(clr_,dtype=np.float32).reshape((-1,3))/255 # print(COLORS,COLORS.shape,type(COLORS),tuple(COLORS[0,:]))

# # csv to array (regex)
# path_csv = path_ + "\\csv_plotter\\"
# df = pd.read_csv(path_csv + "csv_test_3_new.csv",header=None) # df_red = df.iloc[0:2,0:2]
# EPOCHS = df.shape[0]
# # print('**',EPOCHS)
# VMAPS = 200 # 100 for csv_test_100, 200 otherwise
# IT_TOT = df.shape[1]
# IT = 25
# N_DOTS = IT_TOT//IT
# arr = np.empty((EPOCHS,IT_TOT,VMAPS)) # 
# csv_reg = r'(?<=val \= DeviceArray\()(.*)(?=dtype\=float32\))' # r'(?<=val \= Array\()(.*)(?=dtype\=float32\))'
# for i, row in df.iterrows():
#     for j in range(IT_TOT):
#         vals = re.findall(csv_reg,row[j],re.DOTALL)
#         vals_ = vals[0]
#         vals_ = re.sub(r'\s+','',vals_)
#         vals_ = vals_.rstrip(r',') # vals_ = re.sub(r',\\r\\n',',',vals_)
#         vals_ = re.sub(r'(\[|\])','',vals_) # re.sub(r'/[^a-zA-Z ]/g','',test_grp)
#         vals_ = vals_.split(",")
#         arr[i,j] = np.array(vals_,dtype=np.float32) # print(i, j, type(vals_), vals_, type(arr[i,j])) # ,b[0],'\n','&&&&&&&&&&&&&&&&&&&&',b[1])

# # plot
plt.figure()
vmaps = [0,1,2]
IT = 25
fig, axis = plt.subplots(3,3) # len(EPOCHS_PLOT),EX)
for d in range(3):
    for v in range(3):
        sel_ = sel[vmaps[v]] # [v]]
        axis[2,v].plot(dis[((v-1)*IT):(v*IT-1),d],color=tuple(colors[d,:]))
        axis[2,v].set_title(f'epoch 10000, select = {sel_}',fontsize=10)
fig.tight_layout(pad=1.5)
plt.show()



# EPOCHS_PLOT = np.int32([0, np.floor(0.3*EPOCHS), EPOCHS-1]) # (0,0.3,1) of total range
# # SELECT_ = 50*EPOCHS_PLOT # SELECT_[2] = 999
# KEY_INIT = rnd.PRNGKey(1)
# EX = 3
# rand_int = rnd.randint(KEY_INIT,(len(EPOCHS_PLOT),EX),0,VMAPS) # [epochs,examples] randints
# fig, axis = plt.subplots(len(EPOCHS_PLOT),EX)
# for d in range(N_DOTS):
#     for ep in range(len(EPOCHS_PLOT)):
#         for ex in range(EX):
#             # select color that corresponds # col = tuple(COLORS[np.array(SELECT[50*(EPOCHS_PLOT[ep]+1),rand_int[ep,ex],:],dtype=bool),:][0]) # CHANGE; FUNCTION OF d
#             sel = np.array(SELECT[50*(EPOCHS_PLOT[ep]),rand_int[ep,ex],:],dtype=np.int32) #SELECT=[EPOCHS,VMAPS,N_DOTS],arr=[EPOCHS,IT_TOT,VMAPS]
#             axis[ep,ex].plot(arr[EPOCHS_PLOT[ep],d::N_DOTS,rand_int[ep,ex]],color=tuple(COLORS[d,:])) # CHECK VMAP INDEX IS CORRECT (AS TUPLE BASED ON RANDINT(EP,EX)?) check correct; that d'th section of csv corresponds to d'th color #CHANGE; NEED CORRECT col FOR EACH d
#             axis[ep,ex].set_title(f'Epoch {str(50*(EPOCHS_PLOT[ep]))}, select={sel}',fontsize=10)
# fig.tight_layout(pad=1.5)
# plt.show()