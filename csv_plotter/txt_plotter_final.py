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
print('path_',path_)
stdout_ = path_ + '/stdout/' # \\ on windows
dump_ = open(stdout_ + 'dump.txt','r').read() # 41(e,s,d),30(e,s,r,d)
# path_pkl = path_ + "\\pkl\\"
# colors_ = open(path_pkl + colors_file,'rb')
colors = np.float32([[255,0,0],[0,255,0],[0,0,255]])/255 # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[# colors = pickle.load(colors_) # [r,g,b*5]
DOTS = colors.shape[0]
EPOCHS_ = [0,1000,2000,3000,4000,5000,6000]
VMAPS = 1 # 3
IT = 50 # 25

### txt to arrays (regex)
### sel
sel = []
for e in EPOCHS_:
    epoch_reg0 = r'(?<=epoch \= '+re.escape(str(e))+')'+r'(.*)(?=epoch \= '+re.escape(str(e+1000))+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
    dumpstr_ = repr(dump_)
    epoch_ = re.findall(epoch_reg0,dumpstr_,re.DOTALL)
    epoch_ = ' '.join(epoch_) # only one match
    epoch_ = re.sub(r'(\\|\=|\_)',' ',epoch_)
    sel_reg = r'(?<=nsel)(.*)(?=ndis)' # (tot) buggy end condition.. try /i
    sel_ = re.findall(sel_reg,epoch_)
    # print('_sel_',sel_)
    sel_ = ''.join(sel_) # only one match
    sel_ = re.findall(r'\d+',sel_)
    sel_ = ''.join(sel_)
    # print('_sel_',sel_)
    sel_ = sel_[:DOTS*VMAPS]
    sel_ = [sel_[i:i+DOTS] for i in range(0, DOTS*VMAPS, DOTS)] # sel_ = sel_.lstrip(r'\[')
    sel += sel_
sel_arr = np.asarray(sel)
# print('sel',sel_arr.shape) # [EPOCHS_,]

### dis
dis = []
for e in EPOCHS_:
    epoch_reg0 = r'(?<=epoch \= '+re.escape(str(e))+')'+r'(.*)(?=epoch \= '+re.escape(str(e+1000))+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
    dumpstr_ = repr(dump_)
    epoch_ = re.findall(epoch_reg0,dumpstr_,re.DOTALL) # returns set of matching strings. DOTALL matches across lines
    # print('epoch_',epoch_)
    epoch_ = ' '.join(epoch_) # joins all matches with space
    epoch_ = re.sub(r'\s+',' ',epoch_) # replaces one or more whitespace with single whitespace
    dis_reg = r'(?<=dis\=\[)(.*)(?=\]\])' # \]\]\\nsel # finds dis = [ ... ]]\nsel
    dis_ = re.findall(dis_reg,epoch_)
    # print('dis_',dis_)
    dis_ = ''.join(dis_) # (only matching once)
    dis_ = re.sub('dis',' ',dis_)
    dis_ = re.sub(r'(\[|\]|\\n|\=|dis)','',dis_) # removes [ ] \n =
    dis_ = dis_.split(' ') # split by space
    dis_ = list(filter(None, dis_))
    dis_ = dis_[:(DOTS*IT*VMAPS)] # VMAP=1 (prev max VMAP=4)
    dis += dis_
dis_arr = np.asarray(dis).reshape([-1,DOTS]).astype(np.float32) # [EPOCHS*(VMAPS*IT),DOTS]
print('dis_arr',dis_arr.shape) # [EPOCHS*(VMAPS*IT),DOTS]

# ### dis
# dis = []
# for e in EPOCHS_:
#     epoch_reg0 = r'(?<=epoch \= '+re.escape(str(e))+')'+r'(.*)(?=epoch \= '+re.escape(str(e+1000))+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
#     dumpstr_ = repr(dump_)
#     epoch_ = re.findall(epoch_reg0,dumpstr_,re.DOTALL) # returns set of matching strings. DOTALL matches across lines
#     epoch_ = ' '.join(epoch_) # joins all matches with space
#     print('epoch_',epoch_)
#     epoch_ = re.sub(r'\s+',' ',epoch_) # replaces one or more whitespace with single whitespace
#     dis_reg = r'(?<=dis\=\[)(.*)(?=R_tot)' # finds dis = [ ... R_tot
#     dis_ = re.findall(dis_reg,epoch_)
#     dis_ = ''.join(dis_)
#     dis_ = re.sub('dis',' ',dis_)
#     dis_ = re.sub(r'(\[|\]|\\n|=)','',dis_)
#     dis_ = re.sub('dis','',dis_)
#     dis_ = dis_.split(' ')
#     dis_ = list(filter(None, dis_))
#     dis_ = dis_[:(DOTS*IT*VMAPS)]
#     dis += dis_
#     # print(e,type(dis_),len(dis_),len(dis))
# dis_arr = np.asarray(dis).reshape([-1,VMAPS]).astype(np.float32) # [DOTS*IT,VMAPS]
# print('dis_arr',dis_arr.shape)

### plot
fig, axis = plt.subplots(len(EPOCHS_),VMAPS)
axis = axis.reshape(-1,VMAPS)
for e in range(len(EPOCHS_)):
    for v in range(VMAPS):
            for d in range(DOTS):
                sel_ = sel_arr[VMAPS*e+v] # [EPOCHS*VMAPS,] # [1,0,0]
                # print('e',e,'v',v,'d',d,'sel',sel_) # ,'indices:',(e*(DOTS*IT)+d*DOTS),(e*(DOTS*IT)+(d+1)*DOTS-1),v)
                axis[e,v].plot(dis_arr[(e*(VMAPS*IT)+v*IT):(e*(VMAPS*IT)+(v+1)*IT-1),d],color=tuple(colors[d,:]))
                axis[e,v].set_title(f'epoch {EPOCHS_[e]}, select = {sel_}',fontsize=8)
                axis[e,v].tick_params(axis='both', labelsize=8)
fig.tight_layout(pad=0.8)
plt.show()