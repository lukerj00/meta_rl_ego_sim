import jax
from jax import jit
import jax.numpy as jnp
import jax.random as rnd
from pprint import pprint
import matplotlib.pyplot as plt

# # @jit
# def fnc(x,xs):
#     x=x+xs
#     return x,x

# def fnc_2(W,x):
#     y = jnp.matmul(W,x)
#     return y

# vmap_fnc = jax.vmap(fnc_2,in_axes=(None,0),out_axes=(0,0))

# # x_0 = jnp.arange(24,dtype=jnp.float32).reshape((2,3,4))
# # xs = jnp.ones((6,4),dtype=jnp.float32)
# # x_final,x_scan = jax.lax.scan(fnc,x_0,xs)

# # print('x_0',x_0,x_0.shape)
# # print('x_final',x_final,x_final.shape,'x_scan',x_scan,x_scan.shape)

# # for i in xs:
# #     x=x_0+i
# #     print(i.shape,x_0.shape,xs.shape)

# W = jnp.ones([3,3])
# k = jnp.array([1,2,3])
# k3 = 3*jnp.arange(9).reshape((3,3))
# # # 1 1 1   0 3 6
# # # 1 1 1   9 12 15
# # # 1 1 1   18 21 24
# y = fnc_2(W,k)
# y_vmap = vmap_fnc(W,k3)
# print('k3',k3,'y',y,'y_vmap',y_vmap)
# print(k3[0,:])

#sel 3,
#e_t 3,2
#dot 2,

# e_t = jnp.arange(3).reshape([3,])
# print(jnp.linalg.norm(e_t,2))
# sel = jnp.array([0,1,0])
# dot_ = jnp.dot(sel,e_t)
# print(dot_)

# a=b=c=0

# a+=5

# print(a,b,c)

e_test = jnp.linspace(-jnp.pi,jnp.pi,6).reshape([3,2])
pos = 10*jnp.ones(2,)
poss = 0.2*jnp.arange(30,).reshape([15,2])
# @jit
# def abs_geod_old(e_t,pos):###CHECK; calculate correct geodesic
#     pos_ = (pos+jnp.pi)%(2*jnp.pi)-jnp.pi
#     del_x = jnp.abs(e_t[:,0]-pos_[0]) #[3,]
#     del_y = jnp.abs(e_t[:,1]-pos_[1]) #[3,]
#     hav = jnp.sin(del_y/2)**2 + jnp.cos(pos_[1])*jnp.multiply(jnp.cos(e_t[:,1]),jnp.sin(del_x/2)**2)
#     dis = 2*jnp.arctan2(jnp.sqrt(hav),jnp.sqrt(1-hav))
#     dis__ = 2*jnp.arcsin(jnp.sqrt(hav))
#     e_t_ = (e_t + jnp.pi)%(2*jnp.pi)-jnp.pi
#     dis_rel = e_t_-pos_
#     dis_rel_ = (dis_rel+jnp.pi)%(2*jnp.pi)-jnp.pi
#     return dis,dis__,jnp.sqrt(dis_rel_[:,0]**2+dis_rel_[:,1]**2)

# dis1,dis2,dis3 = abs_dist(e_test,pos)
# print(dis1,dis2,dis3)

def abs_geod(dots,pos_):
    # print(dot,pos)
    # pos_ = (pos+jnp.pi)%(2*jnp.pi)-jnp.pi
    # print(dot,pos_)
    del_x = dots[:,0]-pos_[0] #[3,]
    del_y = jnp.abs(dots[:,1]-pos_[1]) #[3,]
    hav = jnp.sin(del_y/2)**2 + jnp.cos(pos_[1])*jnp.cos(dots[:,1])*(jnp.sin(del_x/2))**2
    dis = 2*jnp.arctan2(jnp.sqrt(hav),jnp.sqrt(1-hav))
    dis_ = 2*jnp.arcsin(jnp.sqrt(hav))

    th1 = jnp.pi/2 - pos_[1]
    th2 = jnp.pi/2 - dots[:,1]
    ld1 = pos_[0]
    ld2 = dots[:,0]
    s = jnp.arccos(jnp.cos(th1)*jnp.cos(th2)+jnp.sin(th1)*jnp.multiply(jnp.sin(th2),jnp.cos(ld1-ld2)))
    
    return dis,dis_,s

def abs_geod2(dots,pos):
    ld1 = pos[0]
    ld2 = dots[:,0]
    phi1 = pos[1]
    phi2 = dots[:,1]
    ang = jnp.cos(phi1)*jnp.multiply(jnp.cos(phi2),(jnp.cos(ld1-ld2))) + jnp.sin(phi2)*jnp.sin(phi1)
    geod = jnp.arccos(ang)
    e_rel = dots-pos
    obj_orig = -jnp.exp((jnp.cos(e_rel[:,0]) + jnp.cos(e_rel[:,1]) - 2))
    obj = -jnp.exp(-(geod**2)/2)
    # obj_ = jnp.exp(jnp.cos(geod)-1)
    obj__ = -jnp.exp(ang-1)
    return geod,obj_orig,obj,obj__

# dis1,dis2,dis3 = abs_geod(e_test,pos)
# print('***normal',dis1,dis2,dis1.shape,dis2.shape,dis3,dis3.shape)

# abs_geod_vmap = jax.vmap(abs_geod,in_axes=(None,0),out_axes=(0))
# dis1,dis2,dis3 = abs_geod_vmap(e_test,poss)
# print('***vmapped',dis1,dis2,dis1.shape,dis2.shape,dis3,dis3.shape)

# plt.plot(dis3[:,0])
# plt.plot(dis3[:,1])
# plt.plot(dis3[:,2])
# plt.show()
# print('***nsdfvcfdlnormal',e_test**2,(e_test**2).shape)

# print('&&&&&&&&&&&&&&',abs_geod2(e_test,pos))
# pos0 = jnp.array([0,0])
pos1 = rnd.uniform(rnd.PRNGKey(0),shape=(5,2),minval=-10,maxval=10)
e_test2 = jnp.array([[0,0],[0,-jnp.pi/2],[0,jnp.pi/2],[jnp.pi,jnp.pi/2],[jnp.pi,-jnp.pi/2],[-jnp.pi,jnp.pi/2],[-jnp.pi,-jnp.pi/2],[1,1]])
# # e_test3 = jnp.array([[0,0],[0
# # print('pos1',pos1)
for r in range(5):
    d = pos1[r,:] # jnp.array([0,0])#
    print('dots=',e_test2,'pos=',d,'\n')
    r1,o1,o2,o3 = abs_geod2(e_test2,d)
    r2 = abs_geod(e_test2,d)
    print('new',r1,'\n')
    print('exp old',o1,'\n','exp geod',o2,'\n','vm_approx',o3,'\n')
    # print('old',r2,'\n')

# def gen_test(a,b,c):
#     i = rnd.uniform(rnd.PRNGKey(0),shape=(a,b,c,1),minval=-10,maxval=10)
#     j = rnd.uniform(rnd.PRNGKey(0),shape=(a,b,c,1),minval=-10,maxval=10)
#     t = jnp.concatenate((i,j),axis=3)
#     return t

# print(gen_test(2,3,4).shape)