# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:30:44 2022

@author: lukej
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as rnd

def init_mlp_params(layer_widths):
  params = []
  for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
    print('n,n=',n_in,n_out)
    params.append(
        dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
             biases=np.ones(shape=(n_out,))
            )
    )
  return params

params = init_mlp_params([1, 128, 128, 1])
#print(params)

update = 0.01
test = { "GRU_params" : [
        2.2,
        3.3
        ]}
grad = { "GRU_params" : [
        1,
        1
        ]}


update = 1
test = { "GRU_params" : {
        "a": 2.2,
        "b": 3.3
        }}
grad = { "GRU_params" : {
        "a": 1,
        "b": 1
        }}


#jax.tree_map(lambda t, g: t - update * g, test, grad)
#print(test["GRU_params"])
test = jax.tree_map(lambda x,y: x + update * y, test, grad)

a = 6 - jnp.float32(2)
print(a)
for a in range(1):
    print(a)
    
R_arr = jnp.arange(5)
print((R_arr.shape))
for e in range(5):
    print(e)
    R_tot = e + 3
    print(R_tot)
    R_arr = R_arr.at[e].set(R_tot)
    print('R_tot =', R_arr)
    
def print_p(x):
    return print(x)

print_p(5)
R_arr = jnp.zeros((5))
print(R_arr.shape)

dot = jnp.array([1,2],shape=[2,1])
n_j = jnp.array()

def f(dot,n_j,n_i,SIGMA_T,NEURONS):
    del_j = dot[0] - jnp.array(n_j[0,:]).reshape([1,NEURONS])
    del_i = dot[1] - jnp.array(n_i[:,0]).reshape([NEURONS,1])
    f_ = jnp.exp((jnp.cos(jnp.tile(del_j,(NEURONS,1))) + jnp.cos(jnp.tile(del_i,(1,NEURONS))) - 2)/SIGMA_T**2)
    return f_.ravel().reshape([f_.size,1])

def f2(dot,n_j,n_i,SIGMA_T,NEURONS):
    del_j = dot[0] - n_j
    del_i = dot[1] - n_i
    f_ = jnp.exp(((jnp.cos(del_j) + jnp.cos(del_i)) - 2)/SIGMA_T**2)
    return f_.ravel().reshape([f_.size,1])    

k = rnd.PRNGKey(5)
def create_dots(N_DOTS,k): # d_ji
    return rnd.uniform(k,shape=[N_DOTS,2],minval=-jnp.pi,maxval=jnp.pi,dtype="float32")

a = create_dots(3,k)
print('a = ',a)

for i,v in enumerate(a):
    print('i=',i,'v=', v)

v = jnp.array([0.6,0.5])
print(v.shape)
e = jnp.array([[1,1], [2,2], [3,3]])
print(e.shape)
t = e - v.T
print(t)

l = [[1,1],[2,2]]
nl = jnp.array(l).flatten()
nl2 = jnp.array(l).ravel()
print(nl2)
a = jnp.array([1,2,3,4])
la = nl2*a
print(la)

amat = np.random.random([3,3,4])
bvec1 = jnp.ones([4,]).reshape([1,1,4])
bvec2 = jnp.array([[0],[5],[10],[15]])
bvec3 = bvec2.reshape([1,1,4])
print(amat,amat.shape)
print(bvec3,bvec3.shape)
print(amat+bvec3)

N_COLORS = jnp.float32([[255,0,0],[50,100,50],[0,0,255]])
N_COLORS = jnp.array(N_COLORS/255).ravel()
testt = N_COLORS.repeat(2)
test = N_COLORS.tile(3)
print(N_COLORS)
rss = N_COLORS.reshape([-1,3])
#print(rss)
rs = jnp.sum(N_COLORS.reshape([-1,3]),axis=0)
#print(rs)
#print(N_COLORS.shape)

NEURONS = 5
n_j = jnp.arange(NEURONS).reshape([1,NEURONS])
n_i = jnp.arange(NEURONS).reshape([NEURONS,1])
#e_t_1 = jnp.array([[[10,20],[30,40],[50,60]],[[10,20],[30,40],[50,60]]])
e_t_1 = jnp.array([[10,11],[50,45],[100,101]])
#print(e_t_1)
#print(e_t_1.ravel('F'))
#summ = e_t_1.sum(axis=(2))
#print(summ)
N_DOTS = jnp.size(e_t_1,0)
#sl = e_t_1[:,1]
#sl = sl.reshape(1,1,N_DOTS)
n_j_tot = jnp.tile(jnp.array(n_j).reshape([1,NEURONS,1]),[NEURONS,1,3*N_DOTS]) # [neurons,neurons,n_dots]
print(n_j_tot)
print(n_j_tot.ravel('F'))
#del_j = jnp.tile((n_j_tot - e_t_1[:,0].reshape([1,1,N_DOTS])),[1,1,3]) # [n_dots,1]->[neurons,neurons,n_dots] , [neurons,neurons,n_dots]
del_j = n_j_tot - e_t_1[:,0].repeat(3)
print(del_j)
print(del_j.ravel('F'))
dc = del_j*N_COLORS
print(dc)
print(dc.ravel('F'))
dc_s = dc.reshape([-1,NEURONS,NEURONS,3]).sum(axis=0)
print(dc_s)
print(dc_s.ravel('F'))

n_i_tot = jnp.tile(jnp.array(n_i).reshape([NEURONS,1,1]),[1,NEURONS,N_DOTS]) # [neurons,neurons,n_dots]
del_i = n_i_tot - e_t_1[:,1].reshape([1,1,N_DOTS])
print(del_i)

randmat = np.arange(36).reshape([2,3,6])
print(randmat)
rr = randmat[:,:,0::3].sum(axis=2).ravel('F')
print(rr)
rg = randmat[:,:,1::3].sum(axis=2).ravel('F')
rb = randmat[:,:,2::3].sum(axis=2).ravel('F')
print(rg)
print(rb)
randmatf = randmat.reshape([-1,2,3,2])
#print(randmatf)
randmatf2 = randmatf.sum(axis=0)
print(randmatf2)

e_t_1 = jnp.array([[0,50],[200,250],[3000,3050]])
N_COLORS = jnp.float32([[255,0,0],[50,100,50],[0,0,255]])
NEURONS = 5
n_i = jnp.arange(NEURONS).reshape([NEURONS,1])
n_j = jnp.arange(NEURONS).reshape([1,NEURONS])
SIGMA_T = 1

def f(dot,th_j,th_i,SIGMA_T,NEURONS):
    n_j = jnp.tile(jnp.array(th_j),[NEURONS,1])
    n_i = jnp.tile(jnp.array(th_i),[1,NEURONS])
    del_j = dot[0] - n_j
    del_i = dot[1] - n_i
    f_ = jnp.exp(((jnp.cos(del_j) + jnp.cos(del_i)) - 2)/SIGMA_T**2)
    return f_.ravel().reshape([f_.size,1]) 
# ^ v can combine these 2 fncs to something simpler  
def neuron_act(e_t_1,n_j,n_i,SIGMA_T,NEURONS,N_COLORS): # s_t
    act_r = act_g = act_b = jnp.zeros([NEURONS**2,1],dtype="float32")
    for i,dot in enumerate(e_t_1):
        r,g,b = N_COLORS[i]
        f_ = f(dot,n_j,n_i,SIGMA_T,NEURONS)
        act_r += r/255 * f_.reshape([NEURONS**2,1])
        act_g += g/255 * f_.reshape([NEURONS**2,1])
        act_b += b/255 * f_.reshape([NEURONS**2,1])
    print(act_r.shape)
    return jnp.concatenate((act_r,act_g,act_b),axis=0)

def neuron_act2(e_t_1,n_j,n_i,SIGMA_T,NEURONS,N_COLORS):
    N_DOTS = jnp.size(e_t_1,0)
    n_j_tot = jnp.tile(jnp.array(n_j).reshape([1,NEURONS,1]),[NEURONS,1,N_DOTS])
    n_i_tot = jnp.tile(jnp.array(n_i).reshape([NEURONS,1,1]),[1,NEURONS,N_DOTS])
    del_j = e_t_1[:,0] - n_j_tot
    #print(del_j.shape)
    del_i = e_t_1[:,1] - n_i_tot
    #print(del_i.shape)
    act = jnp.repeat(jnp.exp((jnp.cos(del_j) + jnp.cos(del_i) - 2)/SIGMA_T**2),3,axis=2)
    #print(act.shape)
    act_rgb = act*jnp.array(N_COLORS/255).ravel()
    act_r = act_rgb[:,:,0::3].sum(axis=2).ravel('F')
    print(act_r.shape)
    act_g = act_rgb[:,:,1::3].sum(axis=2).ravel('F')
    act_b = act_rgb[:,:,2::3].sum(axis=2).ravel('F')
    return jnp.concatenate((act_r,act_g,act_b),axis=0)

test1 = neuron_act(e_t_1,n_j,n_i,SIGMA_T,NEURONS,N_COLORS)
# print(test1)
test2 = neuron_act2(e_t_1,n_j,n_i,SIGMA_T,NEURONS,N_COLORS)
# print(test2)

a = jnp.arange(4).reshape([2,2])
print(a[0])
print(a.ravel('F'))

l = jnp.arange(10).reshape(2,5)
for i in l:
    print(i)

e1 = jnp.arange(6)
e2 = jnp.arange(6).reshape([6,1])
e3 = e1+e2
print(e3)

a = jnp.arange(6).reshape(3,2)
v = jnp.array([1,2])
print(v.shape)
print(a-v)

arr1 = jnp.array([[1,2,3],[4,5,6],[7,8,100]])
dict1 = {'d':arr1}
print('d1',dict1)
m = jnp.mean(arr1,axis=0)
print(m)
dict1 = jax.tree_util.tree_map(lambda d: jnp.mean(d,axis=0), dict1) # jnp.mean(d,axis=0)
print('d2',dict1)

a = 1
b = 2
c = 3
a,b,c = jnp.multiply(6,(a,b,c))
print(a,b,c)

l = [[255,0,0],[0,255,0],[0,0,255]]
print(jnp.size(l))

e = np.eye(3)
print(e)
c = e[np.random.choice(3,5)]
print(c)

SELECT = np.tile(np.eye(3)[np.random.choice(3,7)],3)
s = np.eye(3)[np.random.choice(3,7)].tile(3)
print(s[5])

a = jnp.arange(6).reshape(3,2)
b = jnp.array([5,10])
# print(b.shape)
print(a)
print(a-b)