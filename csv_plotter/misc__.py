import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path

# colors, select
colors_path = 'pkl_colors_.pkl'
select_path = 'pkl_select_.pkl'
path_ = str(Path(__file__).resolve().parents[1])
path_pkl = path_ + "\\pkl\\"
colors_ = open(path_pkl + colors_path,'rb') # (remember to open as binary)
select_ = open(path_pkl + select_path,'rb')
COLORS = pickle.load(colors_) # [r,g,b*5]
SELECT = pickle.load(select_) # [EPOCHS=1000,VMAPS=200,N_DOTS=5]
SEL_ = SELECT[0::100,:,:]
SEL_999 = SELECT[999,:,:]

print(COLORS,COLORS.shape)
sel_ = np.array(SELECT[0,0,:],dtype=bool)
print(sel_) # [EPOCHS,VMAPS,N_DOTS] SELECT[EPOCHS_PLOT[0],np.random.randint(VMAPS),:]
print(COLORS[0,:]) # [N_DOTS,(R,G,B)]
clr = tuple(COLORS[sel_,:]) # tuple(COLORS[SELECT[EPOCHS_PLOT[0],np.random.randint(VMAPS),:],:])
print(clr,type(clr))

t = [1,2,3]
print(len(t))