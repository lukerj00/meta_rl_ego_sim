import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax
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
path_pkl = path_ + "\\pkl\\"
# colors_ = open(path_pkl + colors_file,'rb')
colors = np.float32([[255,100,50],[50,255,100],[100,50,255]])/255 # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[# colors = pickle.load(colors_) # [r,g,b*5]

# txt to arrays (regex)
### sel
epoch0 = ['0','1000']
epoch_reg0 = r'(?<=epoch \= '+re.escape(epoch0[0])+')'+r'(.*)(?=epoch \= '+re.escape(epoch0[1])+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
dumpstr_ = repr(dump_)
epoch_ = re.findall(epoch_reg0,dumpstr_,re.DOTALL)
epoch_ = ''.join(epoch_)
epoch_ = re.sub(r'\s+','',epoch_)
sel_reg = r'(?<=sel\=\[)(.*)(?=\.\]\\n)'
sel_ = re.findall(sel_reg,epoch_)
sel_ = ''.join(sel_)
sel_ = re.findall(r'\d+',sel_)
sel_ = ''.join(sel_)
sel_list = [sel_[i:i+3] for i in range(0, len(sel_), 3)]# sel_ = sel_.lstrip(r'\[')
sel0 = np.asarray(sel_list)

epoch1 = ['1000','2000']
epoch_reg1 = r'(?<=epoch \= '+re.escape(epoch1[0])+')'+r'(.*)(?=epoch \= '+re.escape(epoch1[1])+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
dumpstr_ = repr(dump_)
epoch_ = re.findall(epoch_reg1,dumpstr_,re.DOTALL)
epoch_ = ''.join(epoch_)
epoch_ = re.sub(r'\s+','',epoch_)
sel_reg = r'(?<=sel\=\[)(.*)(?=\.\]\\n)'
sel_ = re.findall(sel_reg,epoch_)
sel_ = ''.join(sel_)
sel_ = re.findall(r'\d+',sel_)
sel_ = ''.join(sel_)
sel_list = [sel_[i:i+3] for i in range(0, len(sel_), 3)]# sel_ = sel_.lstrip(r'\[')
sel1 = np.asarray(sel_list)

epoch2= ['10000','11000']
epoch_reg2 = r'(?<=epoch \= '+re.escape(epoch2[0])+')'+r'(.*)(?=epoch \= '+re.escape(epoch2[1])+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
dumpstr_ = repr(dump_)
epoch_ = re.findall(epoch_reg2,dumpstr_,re.DOTALL)
epoch_ = ''.join(epoch_)
epoch_ = re.sub(r'\s+','',epoch_)
sel_reg = r'(?<=sel\=\[)(.*)(?=\.\]\\n)'
sel_ = re.findall(sel_reg,epoch_)
sel_ = ''.join(sel_)
sel_ = re.findall(r'\d+',sel_)
sel_ = ''.join(sel_)
sel_list = [sel_[i:i+3] for i in range(0, len(sel_), 3)]# sel_ = sel_.lstrip(r'\[')
sel2 = np.asarray(sel_list)

### dis
epoch_reg0 = r'(?<=epoch \= '+re.escape(epoch0[0])+')'+r'(.*)(?=epoch \= '+re.escape(epoch0[1])+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
dumpstr_ = repr(dump_)
epoch_ = re.findall(epoch_reg0,dumpstr_,re.DOTALL)
epoch_ = ' '.join(epoch_)
epoch_ = re.sub(r'\s+',' ',epoch_)
dis_reg = r'(?<=dis\=\[)(.*)(?=R_tot)'
dis_ = re.findall(dis_reg,epoch_)
dis_ = ''.join(dis_)
dis_ = re.sub('dis',' ',dis_)
dis_ = re.sub(r'(\[|\]|\\n|=)','',dis_)
dis_ = re.sub('dis','',dis_)
dis_ = dis_.split(' ')
dis_ = list(filter(None, dis_))
dis_ = dis_[:(3*25*12)]
dis0 = np.asarray(dis_).reshape([-1,3]).astype(np.float32)

epoch_reg1 = r'(?<=epoch \= '+re.escape(epoch1[0])+')'+r'(.*)(?=epoch \= '+re.escape(epoch1[1])+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
dumpstr_ = repr(dump_)
epoch_ = re.findall(epoch_reg1,dumpstr_,re.DOTALL)
epoch_ = ' '.join(epoch_)
epoch_ = re.sub(r'\s+',' ',epoch_)
dis_reg = r'(?<=dis\=\[)(.*)(?=R_tot)'
dis_ = re.findall(dis_reg,epoch_)
dis_ = ''.join(dis_)
dis_ = re.sub('dis',' ',dis_)
dis_ = re.sub(r'(\[|\]|\\n|=)','',dis_)
dis_ = re.sub('dis','',dis_)
dis_ = dis_.split(' ')
dis_ = list(filter(None, dis_))
dis_ = dis_[:(3*25*12)]
dis1 = np.asarray(dis_).reshape([-1,3]).astype(np.float32)

epoch_reg2 = r'(?<=epoch \= '+re.escape(epoch2[0])+')'+r'(.*)(?=epoch \= '+re.escape(epoch2[1])+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
dumpstr_ = repr(dump_)
epoch_ = re.findall(epoch_reg2,dumpstr_,re.DOTALL)
epoch_ = ' '.join(epoch_)
epoch_ = re.sub(r'\s+',' ',epoch_)
dis_reg = r'(?<=dis\=\[)(.*)(?=R_tot)'
dis_ = re.findall(dis_reg,epoch_)
dis_ = ''.join(dis_)
dis_ = re.sub('dis',' ',dis_)
dis_ = re.sub(r'(\[|\]|\\n|=)','',dis_)
dis_ = re.sub('dis','',dis_)
dis_ = dis_.split(' ')
dis_ = list(filter(None, dis_))
dis_ = dis_[:(3*25*12)]
dis2 = np.asarray(dis_).reshape([-1,3]).astype(np.float32)
# print(dis.shape,type(dis[0]))

# # plot
# plt.figure()
vmaps = [0,1,2]
IT = 25
fig, axis = plt.subplots(3,3) # len(EPOCHS_PLOT),EX)
for d in range(3):
    for v in range(3):
        sel_ = sel0[vmaps[v]] # [v]]
        axis[0,v].plot(dis0[((v-1)*IT):(v*IT-1),d],color=tuple(colors[d,:]))
        axis[0,v].set_title(f'epoch {epoch0[0]}, select = {sel_}',fontsize=10)
        sel_ = sel1[vmaps[v]] # [v]]
        axis[1,v].plot(dis1[((v-1)*IT):(v*IT-1),d],color=tuple(colors[d,:]))
        axis[1,v].set_title(f'epoch {epoch1[0]}, select = {sel_}',fontsize=10)
        sel_ = sel2[vmaps[v]] # [v]]
        axis[2,v].plot(dis2[((v-1)*IT):(v*IT-1),d],color=tuple(colors[d,:]))
        axis[2,v].set_title(f'epoch {epoch2[0]}, select = {sel_}',fontsize=10)
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