import jax.numpy as jnp

COLORS = jnp.float32([[255,100,50],[50,255,100]]) # ,[100,50,255]]) # ,[100,100,100],[200,200,200]]) # [[255,100,50],[50,255,100],[100,50,255],[200,0,50]]) # ,[50,0,200]]) # [[255,0,0],[0,200,200],[100,100,100]]
N_DOTS = COLORS.shape[0]
print('*********** \n ***********',COLORS.shape[0],COLORS.shape[1])