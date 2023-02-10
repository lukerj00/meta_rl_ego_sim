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
dump_ = open(stdout_ + 'dump30.txt','r').read()
# path_pkl = path_ + "\\pkl\\"
# colors_ = open(path_pkl + colors_file,'rb')
colors = np.float32([[255,100,50],[50,255,100],[100,50,255]])/255 # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[# colors = pickle.load(colors_) # [r,g,b*5]
DOTS = colors.shape[0]
EPOCHS_ = [0,2000,4000]
VMAPS = 4
IT = 25

### txt to arrays (regex)

### sel
sel = []
for e in EPOCHS_:
    epoch_reg0 = r'(?<=epoch \= '+re.escape(str(e))+')'+r'(.*)(?=epoch \= '+re.escape(str(e+1000))+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
    dumpstr_ = repr(dump_)
    epoch_ = re.findall(epoch_reg0,dumpstr_,re.DOTALL)
    epoch_ = ''.join(epoch_)
    epoch_ = re.sub(r'\s+','',epoch_)
    sel_reg = r'(?<=sel\=\[)(.*)(?=\.\]\\n)'
    sel_ = re.findall(sel_reg,epoch_)
    sel_ = ''.join(sel_)
    sel_ = re.findall(r'\d+',sel_)
    sel_ = ''.join(sel_)
    sel_ = [sel_[i:i+DOTS] for i in range(0, VMAPS*DOTS, DOTS)]# sel_ = sel_.lstrip(r'\[')
    # print(e,len(sel_))
    sel += sel_
sel_arr = np.asarray(sel)
print('sel',sel_arr.shape)

### dis
dis = []
for e in EPOCHS_:
    epoch_reg0 = r'(?<=epoch \= '+re.escape(str(e))+')'+r'(.*)(?=epoch \= '+re.escape(str(e+1000))+')' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
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
    dis_ = dis_[:(DOTS*IT*VMAPS)]
    dis += dis_
    # print(e,type(dis_),len(dis_),len(dis))
dis_arr = np.asarray(dis).reshape([-1,VMAPS]).astype(np.float32) # [DOTS*IT,VMAPS]
print('dis_arr',dis_arr.shape)

### plot
fig, axis = plt.subplots(len(EPOCHS_),VMAPS)
for e in range(len(EPOCHS_)):
    for v in range(VMAPS):
            for d in range(DOTS):
                sel_ = sel_arr[DOTS*v+d]
                # axis[e,v].plot(dis_arr[e*IT+((v)*IT):((v+1)*IT-1),d],color=tuple(colors[d,:]))
                # print('e',e,'v',v,'d',d,'indices:',(e*(DOTS*IT)+d*DOTS),(e*(DOTS*IT)+(d+1)*DOTS-1),v)
                axis[e,v].plot(dis_arr[(e*(DOTS*IT)+d*IT):(e*(DOTS*IT)+(d+1)*IT-1),v],color=tuple(colors[d,:]))
                axis[e,v].set_title(f'epoch {EPOCHS_[e]}, select = {sel_}',fontsize=10)
fig.tight_layout(pad=1.5)
plt.show()