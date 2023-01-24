# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:05:20 2022

@author: lukej
"""

import pytorch as torch
import numpy as np

# e_t = state of env; h_t = hidden state of GRU (v_x,v_y); theta = [psi,(env params)]

def single_step(h_t,e_t,theta): 
    
    # extract data from stores
    
    # s_t = f(e_t)
    
    # R_t = g(s_t)
    
    # GRU equations; generate h_t+1
    
    # v_t+1 = C*h_t+1 + eps (eps = torch.rand())
    
    # e_t+1 = y(e_t,h_t,v_t+1,theta)
    
    return # e_t1_h_t1

# use jax.lax.scan(single_step,...)
# jax treewrap (?)
# G = grad(tot_reward,argnum=2)
