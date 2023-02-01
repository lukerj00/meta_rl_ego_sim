import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as np
import pickle
from pathlib import Path

### specify colors; select; epochs to plot
colors_file = 'pkl_colors_3_.pkl' # CHANGE
select_file = 'pkl_select_3_.pkl' # CHANGE

path_ = str(Path(__file__).resolve().parents[1])
path_pkl = path_ + "\\pkl\\"
colors_ = open(path_pkl + colors_file,'rb')
select_ = open(path_pkl + select_file,'rb')
colors = pickle.load(colors_) # [r,g,b*5]
SELECT = pickle.load(select_) # [EPOCHS=1000,VMAPS=200,N_DOTS=5]
# print(SELECT.shape) # print(COLORS,type(COLORS), tuple(SELECT[0,0,:])) # jax.debug.print(COLORS) # new; not in this version

# color_str to array (regex)
clr_reg = r'(?<=DeviceArray\()(.*)(?=dtype)' # r'(?<=DeviceArray\(\[\[]])(.*)(?=\]\])'
color_str = repr(colors)
clr_ = re.findall(clr_reg,color_str,re.DOTALL)
clr_ = ''.join(clr_)
clr_ = re.sub(r'\s+','',clr_)
clr_ = clr_.rstrip(r',')
clr_ = re.sub(r'(\[|\]|\.)','',clr_)
clr_ = clr_.split(",")
COLORS = np.array(clr_,dtype=np.float32).reshape((-1,3))/255 # print(COLORS,COLORS.shape,type(COLORS),tuple(COLORS[0,:]))

# csv to array (regex)
path_csv = path_ + "\\csv_plotter\\"
df = pd.read_csv(path_csv + "csv_test_3_new.csv",header=None) # df_red = df.iloc[0:2,0:2]
EPOCHS = df.shape[0]
# print('**',EPOCHS)
VMAPS = 200 # 100 for csv_test_100, 200 otherwise
IT_TOT = df.shape[1]
IT = 25
N_DOTS = IT_TOT//IT
arr = np.empty((EPOCHS,IT_TOT,VMAPS)) # 
csv_reg = r'(?<=val \= DeviceArray\()(.*)(?=dtype\=float32\))' # r'(?<=val \= Array\()(.*)(?=dtype\=float32\))'
for i, row in df.iterrows():
    for j in range(IT_TOT):
        vals = re.findall(csv_reg,row[j],re.DOTALL)
        vals_ = vals[0]
        vals_ = re.sub(r'\s+','',vals_)
        vals_ = vals_.rstrip(r',') # vals_ = re.sub(r',\\r\\n',',',vals_)
        vals_ = re.sub(r'(\[|\])','',vals_) # re.sub(r'/[^a-zA-Z ]/g','',test_grp)
        vals_ = vals_.split(",")
        arr[i,j] = np.array(vals_,dtype=np.float32) # print(i, j, type(vals_), vals_, type(arr[i,j])) # ,b[0],'\n','&&&&&&&&&&&&&&&&&&&&',b[1])

# plot
EPOCHS_PLOT = np.int32([0, np.floor(0.3*EPOCHS), EPOCHS-1]) # (0,0.3,1) of total range
# SELECT_ = 50*EPOCHS_PLOT # SELECT_[2] = 999
KEY_INIT = rnd.PRNGKey(1)
EX = 3
rand_int = rnd.randint(KEY_INIT,(len(EPOCHS_PLOT),EX),0,VMAPS) # [epochs,examples] randints
fig, axis = plt.subplots(len(EPOCHS_PLOT),EX)
for d in range(N_DOTS):
    for ep in range(len(EPOCHS_PLOT)):
        for ex in range(EX):
            # select color that corresponds # col = tuple(COLORS[np.array(SELECT[50*(EPOCHS_PLOT[ep]+1),rand_int[ep,ex],:],dtype=bool),:][0]) # CHANGE; FUNCTION OF d
            sel = np.array(SELECT[50*(EPOCHS_PLOT[ep]),rand_int[ep,ex],:],dtype=np.int32) #SELECT=[EPOCHS,VMAPS,N_DOTS],arr=[EPOCHS,IT_TOT,VMAPS]
            axis[ep,ex].plot(arr[EPOCHS_PLOT[ep],d::N_DOTS,rand_int[ep,ex]],color=tuple(COLORS[d,:])) # CHECK VMAP INDEX IS CORRECT (AS TUPLE BASED ON RANDINT(EP,EX)?) check correct; that d'th section of csv corresponds to d'th color #CHANGE; NEED CORRECT col FOR EACH d
            axis[ep,ex].set_title(f'Epoch {str(50*(EPOCHS_PLOT[ep]))}, select={sel}',fontsize=10)
fig.tight_layout(pad=1.5)
plt.show()

### extra regex
#  r'(?<=val \= Array\()(.*)(?=\,      dtype\=float32\))' # r'(.*)' 