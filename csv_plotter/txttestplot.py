import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import jax
# import jax._src.device_array
# import jax._src.abstract_arrays
# import jax
import numpy as np
from numpy import genfromtxt
import pickle
import pathlib
from pathlib import Path
import os
import sys
from datetime import datetime

def obj(e_t_1,SIGMA_R): # R_t
    obj = -np.exp(-((e_t_1)**2)/SIGMA_R**2)
    return obj# np.dot(obj,sel)

R_tot="""-3.1680513e-18 -7.6529654e-19 -1.8546512e-19 -4.7373525e-20
 -5.5059561e-21 -2.3092051e-21 -1.3321949e-21 -1.3258322e-21
 -3.4394327e-21 -2.5482601e-20 -2.1368332e-19 -8.0571905e-19
 -7.0182294e-19 -5.4964973e-19 -3.1505622e-19 -8.3564745e-20
 -2.2976077e-20 -2.8860358e-21 -6.6252017e-22 -1.2019588e-22
 -5.3863505e-23 -1.8129567e-23 -1.5066966e-23 -4.5915199e-24
 -1.1786422e-24"""
R_tot = [float(i) for i in R_tot.split()]
R_tot = np.asarray(R_tot,dtype=np.float32)
# print(R_tot, type(R_tot), len(R_tot))

dis = '''[[3.206364   3.1463733  2.6728477 ],
[3.1765432  3.1765308  2.63243   ],
[3.1535351  3.125804   2.5877569 ],
[3.106156   3.0790122  2.5508544 ],
[3.0979488  3.0423496  2.5183737 ],
[3.1254401  2.9626107  2.441477  ],
[3.1897323  2.8556526  2.3344846 ],
[3.2871766  2.7499828  2.2232127 ],
[3.358433   2.5531194  2.0119615 ],
[3.2783303  2.2885172  1.724553  ],
[3.2273269  2.0162604  1.4248794 ],
[3.2326694  1.7603731  1.1436884 ],
[3.242106   1.6050338  0.97272056],
[3.2634926  1.5135174  0.87731886],
[3.3139348  1.4624078  0.8262644 ],
[3.3622847  1.4462049  0.81226575],
[3.438546   1.3927766  0.7630723 ],
[3.4916325  1.3433148  0.715883  ],
[3.5522146  1.2953517  0.67291284],
[3.5803485  1.2924303  0.67704993],
[3.6181657  1.2673392  0.6578686 ],
[3.624553   1.2627239  0.65424645],
[3.6653044  1.2139652  0.607372  ],
[3.7113903  1.213666   0.6248942 ],
[3.761756   1.1856257  0.6110338 ]]'''
dis = re.sub(r'\s+', ' ', dis)
dis = re.sub(r'(\[|\]|\,)','', dis)
dis = np.asarray(dis.split(),dtype=np.float32).reshape((-1,3))
print(dis, type(dis), len(dis))

sel = np.array([1,0,0],dtype=np.float32)
SIGMA_R = 0.5

axis,fig = plt.subplots()
plt.plot(abs(obj(dis[:,0],SIGMA_R)),'r') #,R_tot,'o')
plt.plot(abs(obj(dis[:,1],SIGMA_R)),'g') #,R_tot,'o')
plt.plot(abs(obj(dis[:,2],SIGMA_R)),'b') #,R_tot,'o')
plt.plot(abs(R_tot),'.',color='black')#,'dotted')
plt.yscale('log')
plt.grid()
plt.show()

