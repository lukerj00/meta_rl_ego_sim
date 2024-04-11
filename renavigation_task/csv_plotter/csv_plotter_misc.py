import csv
import pandas as pd
import matplotlib
import re
import jax.numpy as jnp
import numpy as np
import pprint as pp

df = pd.read_csv("csv_test_100.csv",header=None)
df_red = df.iloc[0:2,0:2]
# print(df_red.shape)
VMAPS = 100
IT = df.shape[1]
EPOCHS = df.shape[0]
arr = np.empty((EPOCHS,IT,VMAPS))
arr_val = np.empty((EPOCHS,IT,VMAPS))
reg = r'(?<=val \= Array\()(.*)(?=dtype\=float32\))' # r'(?<=val \= Array\()(.*)(?=\,      dtype\=float32\))' # r'(.*)' 
expr = re.compile(reg)
for index, row in df_red.iterrows():
    for j in range(2):
        e = row[j]
        # print(type(e),e)
        # a = expr.search(row[j],re.MULTILINE)
        # print(a.group())
        b = re.findall(reg,row[j],re.DOTALL)
        f = b[0]
        vals_ = re.sub(r'\s+','',f)
        vals_ = vals_.rstrip(r',')
        # vals_ = re.sub(r',\\r\\n',',',vals_)
        vals_ = re.sub(r'(\[|\])','',vals_) # re.sub(r'/[^a-zA-Z ]/g','',test_grp)
        vals_ = vals_.split(",")
        arr_val[index,j] = np.array(vals_,dtype=np.float32)
        print(index, j, type(vals_), vals_, type(arr_val[index,j])) # ,b[0],'\n','&&&&&&&&&&&&&&&&&&&&',b[1])
        # print('******************')
        # for l in b:
        #     print('newrow',l)
        # c = ''.join(b)
        # print('.........................\n..........',type(c),c)
        # c = b.replace(r"\r', '', '","")
        # print(c)
# df___ = df.iloc[:,:]
# df1 = df.iloc[0,0]
# print(df1)
# df_np = df.to_numpy()
# print(type(df_np))
# df_row = df_np[0,:]
# df_0 = df_np[0,0]
# print(type(df_0))
# print(df_0)
# # df_str = df_0.to_string()

# vals = expr.search(df_0)
# print(vals)
# vals_ = vals.group()
# vals_ = re.sub(r'\s+','',vals_)
# vals_ = vals_.rstrip(r',')
# vals_ = re.sub(r',\\r\\n',',',vals_)
# vals_ = re.sub(r'(\[|\])','',vals_) # re.sub(r'/[^a-zA-Z ]/g','',test_grp)
# vals_ = vals_.split(",")
# print(vals_)
# # print('**************************',df_row.str.contains(r'(?<=val \= Array\()(.*)(?=dtype\=float32\))')) # r'Traced\<ShapedArray\(float32\[\]'
# # df_row.str.extract(r'([a])') # r'(?<=Traced)(.*)(?=Array)' r'/[^a-zA-Z ]/g' # 