import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax.numpy as jnp
import numpy as np

df = pd.read_csv("csv_test_100.csv",header=None) # df_red = df.iloc[0:2,0:2]
VMAPS = 100
IT = df.shape[1]
EPOCHS = df.shape[0]
arr = np.empty((EPOCHS,IT,VMAPS))
reg = r'(?<=val \= Array\()(.*)(?=dtype\=float32\))' # r'(?<=val \= Array\()(.*)(?=\,      dtype\=float32\))' # r'(.*)' 
for i, row in df.iterrows():
    for j in range(IT):
        vals = re.findall(reg,row[j],re.DOTALL)
        vals_ = vals[0]
        vals_ = re.sub(r'\s+','',vals_)
        vals_ = vals_.rstrip(r',') # vals_ = re.sub(r',\\r\\n',',',vals_)
        vals_ = re.sub(r'(\[|\])','',vals_) # re.sub(r'/[^a-zA-Z ]/g','',test_grp)
        vals_ = vals_.split(",")
        arr[i,j] = np.array(vals_,dtype=np.float32) # print(i, j, type(vals_), vals_, type(arr[i,j])) # ,b[0],'\n','&&&&&&&&&&&&&&&&&&&&',b[1])

EPOCHS_ = [0, 10, 20, 99]
fig, axis = plt.subplots(2,2)
for i in np.random.randint(0,VMAPS,5):
    axis[0,0].plot(arr[EPOCHS_[0],:,i])
    axis[0,0].set_title("EPOCH:" + str(EPOCHS_[0]))
    axis[0,1].plot(arr[EPOCHS_[1],:,i])
    axis[0,1].set_title("EPOCH:" + str(EPOCHS_[1]))
    axis[1,0].plot(arr[EPOCHS_[2],:,i])
    axis[1,0].set_title("EPOCH:" + str(EPOCHS_[2]))
    axis[1,1].plot(arr[EPOCHS_[3],:,i])
    axis[1,1].set_title("EPOCH:" + str(EPOCHS_[3]))
fig.tight_layout(pad=1.5)
plt.show()