import jax
from jax import jit
import jax.numpy as jnp
from pprint import pprint

# # @jit
# def fnc(x,xs):
#     x=x+xs
#     return x,x

# def fnc_2(W,x):
#     y = jnp.matmul(W,x)
#     return y

# vmap_fnc = jax.vmap(fnc_2,in_axes=(None,0),out_axes=1)

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
# # 1 1 1   0 3 6
# # 1 1 1   9 12 15
# # 1 1 1   18 21 24
# y = fnc_2(W,k)
# y_vmap = vmap_fnc(W,k3)
# print('k3',k3,'y',y,'y_vmap',y_vmap)
# print(k3[0,:])

#sel 3,
#e_t 3,2
#dot 2,

# e_t = jnp.arange(6).reshape([3,2])
# sel = jnp.array([0,1,0])
# dot_ = jnp.dot(sel,e_t)
# print(dot_)

a=b=c=0

a+=5

print(a,b,c)
