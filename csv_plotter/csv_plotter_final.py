import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax.numpy as jnp
import jax.random as rnd
import numpy as np
import pickle
from pathlib import Path

# # colors, select
# colors_path = 'pkl_colors_.pkl'
# select_path = 'pkl_select_.pkl'
path_ = str(Path(__file__).resolve().parents[1])
# path_pkl = path_ + "\\pkl\\"
# colors_ = open(path_pkl + colors_path)
# select_ = open(path_pkl + select_path)
# COLORS = pickle.load(colors_) # [r,g,b*5]
# SELECT = pickle.load(select_) # [EPOCHS=1000,VMAPS=200,N_DOTS=5]
# SEL_ = SELECT[0::100,:,:]
# SEL_999 = SELECT[999,:,:]

# csv to array (regex)
path_csv = path_ + "\\csv_plotter\\"
df = pd.read_csv(path_csv + "csv_test_multi_.csv",header=None) # df_red = df.iloc[0:2,0:2]
EPOCHS = df.shape[0]
VMAPS = 200 # 100 for csv_test_100, 200 otherwise
IT_TOT = df.shape[1]
IT = 25
N_DOTS = IT_TOT//IT
arr = np.empty((EPOCHS,IT_TOT,VMAPS)) # 
reg = r'(?<=val \= DeviceArray\()(.*)(?=dtype\=float32\))' # r'(?<=val \= Array\()(.*)(?=dtype\=float32\))'
for i, row in df.iterrows():
    for j in range(IT_TOT):
        vals = re.findall(reg,row[j],re.DOTALL)
        vals_ = vals[0]
        vals_ = re.sub(r'\s+','',vals_)
        vals_ = vals_.rstrip(r',') # vals_ = re.sub(r',\\r\\n',',',vals_)
        vals_ = re.sub(r'(\[|\])','',vals_) # re.sub(r'/[^a-zA-Z ]/g','',test_grp)
        vals_ = vals_.split(",")
        arr[i,j] = np.array(vals_,dtype=np.float32) # print(i, j, type(vals_), vals_, type(arr[i,j])) # ,b[0],'\n','&&&&&&&&&&&&&&&&&&&&',b[1])

# plot
# rnd gen
KEY_INIT = rnd.PRNGKey(0)
ki = rnd.split(KEY_INIT,num=10)
ran = rnd.randint(ki[0],(0,),0,VMAPS) # 2x2
EPOCHS_PLOT = [0, 1, 8, 9] # , 1, 3, 9]
fig, axis = plt.subplots(2,2)
for i in range(N_DOTS):# np.random.randint(0,VMAPS,5):
    axis[0,0].plot(arr[EPOCHS_PLOT[0],i::N_DOTS,np.random.randint(VMAPS)])
    axis[0,0].set_title("EPOCH:" + str(EPOCHS_PLOT[0]))
    axis[0,1].plot(arr[EPOCHS_PLOT[1],i::N_DOTS,np.random.randint(VMAPS)])
    axis[0,1].set_title("EPOCH:" + str(EPOCHS_PLOT[1]))
    axis[1,0].plot(arr[EPOCHS_PLOT[2],i::N_DOTS,np.random.randint(VMAPS)])
    axis[1,0].set_title("EPOCH:" + str(EPOCHS_PLOT[2]))
    axis[1,1].plot(arr[EPOCHS_PLOT[3],i::N_DOTS,np.random.randint(VMAPS)])
    axis[1,1].set_title("EPOCH:" + str(EPOCHS_PLOT[3]))
fig.tight_layout(pad=1.5)
plt.show()

###
#  r'(?<=val \= Array\()(.*)(?=\,      dtype\=float32\))' # r'(.*)' 