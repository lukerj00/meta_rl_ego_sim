import jax.numpy as jnp
import jax.random as rnd
import jax
import numpy as np
import matplotlib.pyplot as plt

### code structure for sc project

## outer loop

# full_loop(weights,params) # hyperparams, env params, etc
# sc_0 = gen_sc(params) # create sc model of functional units
# loss,trained_weights,data = full_train(sc,weights,params) # trained params of outer rnn
# data = full_test(sc,trained_weights,params) # test data
# return loss,trained_params,data # plot stats

# full_train(sc,weights,params)
# loop around:
# loss,data = .vmap(single_trial(sc_0,weights,params))
# grad = .policygrad(loss) # grad of rnn params (which dictate policy as a fnc of sc)
# weights = .update(weights,grad)
# end loop
# trained_weights = weights
# return loss,trained_weights,data

# single_trial(sc_0,weights,params)
# env_0 = gen_env(params)
# loss,data = .scan(trial_step(carry_0,weights,params)) # scan through trial
# return loss,data

# trial_step(carry,weights,params)
# (hs_t_1,hp_t_1,hv_t_1,pos_t_1,v_t_1,r_t_1,sc,plan,*_) = carry # unpack from previous timestep -- need to clarify e_t; refer to pos_t/dots/visual activations?
# s_weights,p_weights,r_weights = weights
# policy = .softmaargs(sc,s_weights) # softmaargs layer outputs policy; weights needed to determine action/plan
# module,action (move/plan) = .sample(policy) # sample movement/plan based on policy (rnn params) (OR choose action based on weighted sum)
# if action==0
# pos_t,v_t = module.move(self,pos_t_1,v_t_1,r_t_1,params,p_weights) # use value model
# hv_t,r_t = module.value(self,hv_t_1,v_t,r_t_1,params,r_weights) # use value model
# if action==1
# hp_0 = gen_hp()
# loop around: # (for pre-determined planning horizon)
# hp_t,^pos_t,^v_t = module.plan(self,hv_t_1,pos_t_1,v_t_1,r_t_1,params,p_weights) # use world model,value model (return sequence of movement vectors/dreamt value)
# hv_t,^r_t = module.value(self,hv_t_1,^v_t,r_t_1,params,r_weights) # use value model
# end loop
# sc,hs_t = RNN_map_update(sc,hs_t_1,pos_t,v_t,r_t,s_weights) # RNN computations to update sc_map
# return (hs_t,hp_t,hv_t,pos_t,v_t,r_t,sc,plan,*_),(data) # (carry,stacked output)

# class sc_module()
# __init__(self,id,vector,params)
# self.id = id # number within array, used for 1-hot vector
# self.1hvec = [0,...1,...,0] # (1 at id'th position)
# self.vector = vector # [v_args,v_y]
# self.value = rnd.normal(params["sigma_value"]) # scalar value of module
#
# move(pos_t_1,v_t_1,r_t_1,weights,params) # ,(needs movement vector/current visual input/value at this input)
# pos_t = pos_t_1 + self.vector
# v_t = neuron_act(pos_t,params["dots"]) # find v_t from neuron activations
# return pos_t,v_t
#
# plan(hp_t_1,pos_t_1,v_t_1,r_t_1,weights,params) # predicting pos/value without calculating losses (no knowledge of dots)
# hp_t,^pos_t,^v_t = RNN_forward(hp_t_1,pos_t_1,v_t_1,p_weights) # find ^v_t as a predictive scalar array
# return hp_t,^pos_t,^v_t # also return ^dots (estimate of dot locations)?
#
# value(hv_t_1,v_t,r_t_1,weights,params) # get value of current visual input
# hv_t,r_t = RNN_value(hv_t_1,v_t,r_t_1,r_weights) # just v_t->r_t (not recurrent)
# module.value = r_t
# return hv_t,r_t

# def gen_sc(params)
## need number of modules (id range), vectors (parameterised; arranged in space) + noise, values (uniformly initialised)
# sc = jnp.empty(MODULES)
# for i in range(MODULES):
# args,y = i%(jnp.sqrt(MODULES))
# vector = [args,y] + rnd.normal(params["sigma_vec"])
# sc[i] = Module(ind,vector,params)

## forward model training (self-supervised - 'motor babbling')
#-want to predict v_t,pos_t as a fnc of 1hvec,v_t_1,pos_t_1,h_t_1
#-given v_t_1,pos_t_1,1hvec attempt to predict neargst v_t,pos_t with RNN; compare with true v_t to get loss:

# forward_model_loop(sc,weights,params)
# loop around:
# hp_0,pos_0,v_0 = gen_pv(params)
# loss = body_fnc(sc,hp_0,pos_0,v_0,weights,params)
# grad = .grad(loss)
# weights = .update(weights,grad)
# end loop
# trained_weights = weights
# return loss,trained_weights

# body_fnc(sc,hp_t_1,pos_t_1,v_t_1,weights,params)
# loop around:
# module = .sample(sc_modules)
# hp_t,^pos_t,^v_t = module.plan(self,hp_t_1,pos_t_1,v_t_1,p_weights) # find ^v_t as a predictive scalar array
# pos_t,v_t = module.move(self,hv_t_1,pos_t_1,v_t_1,r_t_1,params) # use value model
# end loop
# loss = (^v_arr-v_arr)**2 # (include pos_t?)
# return loss

# gen_pv(params)
# pos_0 = rnd.uniform(params)
# v_0 = neuron_act(pos_0,params) # uses params["dots"]
# return pos_0,v_0

## value model training (supervised) ***
#-want to predict r_t as a fnc of v_t,r_t_1,h_t_1
#-given 'dataset' of r_t_1,v_t_1 attempt to predict r_t

# value_model_loop(weights,params)
# loop around:
# loss_1 = train_phase_1(weights,params)
# grad = .grad(loss_1)
# weights_1 = .update(weights,grad)
# end loop
# loop around:
# loss_2 = train_phase_2(weights,params)
# grad = .grad(loss_2)
# weights_2 = .update(weights,grad)
# end loop
# trained_weights = weights
# return loss_1,loss_2,trained_weights

# train_phase_1(weights,params)
# v_arr_1,r_arr_1 = gen_vr_arr_1(params) # dataset of v_t,r_t values
# policy = params["policy_0"]
# hv_0 = params["hv_0"] # or gen_hv()...
# loop for t=t_train_phase_1 timesteps:
# module = .sample(policy)
# hv_t,r_t_pred = module.value(self,hv_t_1,v_arr_1[i+1],r_arr_1[i],params,r_weights) # use value model
# loss_train = (r_t_pred-r_arr_1)**2 # across all timeseries
# end
# return loss_train

# train_phase_2(weights,params)
# v_arr_2,r_arr_2 = gen_vr_arr_2(params) # dataset of v_t,r_t values (with additional v_t values and no corresponding r_t)
# policy = params["policy_0"]
# hv_0 = params["hv_0"] # or gen_hv()...
# loop for t=t_train_phase_2 timesteps:
# module = .sample(policy)
# hv_t,r_t_pred = module.value(self,hv_t_1,v_arr_2[i+1],r_arr_2[i],params,r_weights) # use value model
# loss_train = (r_t_pred-r_arr_2)**2
# end
# return loss_train

# eargsisting:
# full_loop
# train_loop
# -grad of tot_reward
# -optimizer update
# tot_reward
# -scan single_step
# single_step
# -env loop; gru computations

# def function_a(carry_tot):
#     (ind,params,args,lists) = carry_tot
#     args[1] = args[1] + 1
#     jnp.append(lists[0],args[1])
#     jnp.append(lists[1],args[2])
#     jax.debug.print('f_a={}',args[1])
#     return args

# def function_b(carry_tot):
#     (ind,params,args,lists) = carry_tot
#     args[1] = args[1] + 5
#     jnp.append(lists[0],args[1])
#     jnp.append(lists[1],args[2])
#     jax.debug.print('f_b={}',args[1])
#     return args

# def dynamic_while_loop(init_carry_tot):
#     def loop_body(carry_tot):
#         ind,params,args,lists = carry_tot
#         ### INITIAL LOGIC ###
#         args[2] = args[2] + 10 
#         jnp.append(lists[0],0)
#         jnp.append(lists[1],1)
#         #
#         jax.debug.print('lb_args=={}',args)
#         ### CONDITIONAL LOGIC ###
#         new_args = jax.lax.cond(args[0] == 1, function_a, function_b, (ind,params,args,lists)) 
#         #
#         return (ind + 1,params,new_args,lists)  # Return the updated carry as a tuple
    
#     (*_,final_args,final_lists) = jax.lax.while_loop(while_cnd, loop_body, init_carry_tot)
#     return final_args,final_lists

# def while_cnd(carry_tot):
#     ind,params,args,lists = carry_tot
#     jax.debug.print('wc_args=={}',args)
#     return ind < params[0]  # Example condition for terminating the loop

# # def switch_cnd(carry_):
# #     args, cnd = carry_
# #     return args > cnd

# cnd = 100
# cnd_ = 1000
# params = (cnd,cnd_)
# init_ind = 0
# x,y,z = 0,1,2
# init_args = [x,y,z]
# init_lists = (jnp.array([]),jnp.array([]))
# init_carry_tot = (init_ind,params,init_args,init_lists)
# result = dynamic_while_loop(init_carry_tot)
# print(result)
# # print('jkhgvk')

# def interpolate(indices,values,N):
#     result = jnp.zeros(N)
#     result = result.at[indices].set(values)
#     nonzero_indices = jnp.nonzero(result)[0]
#     zero_indices = jnp.where(result == 0)[0]
#     if zero_indices.size > 0:
#         left_indices = jnp.searchsorted(nonzero_indices, zero_indices, side='left')
#         right_indices = jnp.searchsorted(nonzero_indices, zero_indices, side='right')
#         left_values = result[nonzero_indices[left_indices]]
#         right_values = result[nonzero_indices[right_indices]]
#         result = result.at[zero_indices].set((left_values + right_values) / 2)
#     return result

# ind = jnp.array([0,1,2,3,8,5,6,16])
# val = jnp.array([6,5,3,1,0,0,1,8])

# res = interpolate(ind,val,20)

# @jit
# def create_reward_array(R, T):
#     R_new = jnp.zeros(jnp.max(T) + 1)
#     indices = jnp.searchsorted(T, jnp.arange(R_new.shape[0]), side='left')
#     indices = jnp.clip(indices, 0, R.shape[0] - 1)
#     R_new = jnp.take(R, indices)
#     return R_new



# R = jnp.array([5,3,1])
# T = jnp.array([1,2,6])
# R_new = create_reward_array(R,T)

# print(R_new)  # prints: [2 2 3 5 5 5 5]

# def nancumsum_reverse(x):
#     y = jnp.flip(jnp.nancumsum(jnp.flip(x, axis=-1), axis=-1), axis=-1)
#     return jnp.where(jnp.isnan(x), jnp.nan, y)
    
# # Test
# r_arr_ = jnp.array([[2,3,5,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,jnp.nan]])
# t_arr_ = jnp.array([[0,2,4,jnp.nan,jnp.nan,jnp.nan], [0,1,2,3,4,jnp.nan]])
# Q_t = nancumsum_reverse(r_arr_)
# b_t = jnp.nanmean(Q_t, axis=0)
# A_t = Q_t - b_t

# def r_arr_indexed_row(R, T):
#     R_no_nan = jnp.nan_to_num(R, nan=0.0)
#     T_no_nan = jnp.nan_to_num(T, nan=jnp.iinfo(jnp.int32).max)
#     R_new = jnp.zeros(jnp.max(T_no_nan) + 1)
#     indices = jnp.searchsorted(T_no_nan, jnp.arange(R_new.shape[0]), side='left')
#     indices = jnp.clip(indices, 0, R_no_nan.shape[0] - 1)
#     R_new = jnp.take(R_no_nan, indices)
#     # Replace zeros with nan where T was nan
#     T_is_nan = jnp.isnan(T)
#     R_new = jnp.where(T_is_nan[indices], jnp.nan, R_new)
#     return R_new

# print(Q_t,b_t,A_t)

# def create_reward_array_row(R, T, T_max):
#     R_no_nan = jnp.nan_to_num(R, nan=0.0)
#     # Create new array with length equal to T_max + 1
#     R_new = jnp.zeros(T_max + 1)
#     # Exclude nan values for searchsorted
#     T_no_nan = T[~jnp.isnan(T)]
#     indices = jnp.searchsorted(T_no_nan, jnp.arange(R_new.shape[0]), side='left')
#     indices = jnp.clip(indices, 0, R_no_nan.shape[0] - 1)
#     R_new = jnp.take(R_no_nan, indices)
#     # If indices corresponding to nan values in T were used, set R_new to nan at these indices
#     T_is_nan = jnp.isnan(T)
#     R_new = jnp.where(T_is_nan[indices], jnp.nan, R_new)
#     return R_new

# def create_reward_array(R, T):
#     return jax.vmap(create_reward_array_row,in_axes=(0,0,None),out_axes=(0))(R, T, T_max)

# from jax.lax import fori_loop

# def masked_searchsorted(a, v, side='left'):
#     return jnp.min(jnp.where(a < v if side == 'left' else a <= v, len(a), jnp.arange(len(a))))

# def masked_take(a, indices):
#     mask = (indices >= 0) & (indices < len(a))
#     safe_indices = jnp.where(mask, indices, 0)
#     return jnp.where(mask, a[safe_indices], jnp.nan)

# def create_reward_array_row(R, T, T_max):
#     R_no_nan = jnp.nan_to_num(R, nan=0.0)
#     T_no_nan = jnp.nan_to_num(T, nan=T_max+1)
#     R_new = jnp.zeros(T_max + 1)

#     def body_fun(i, R_new):
#         index = masked_searchsorted(T_no_nan, i)
#         index = jnp.clip(index, 0, R_no_nan.shape[0] - 1)
#         value = masked_take(R_no_nan, index)
#         R_new = R_new.at[i].set(value)
#         return R_new

#     R_new = fori_loop(0, T_max + 1, body_fun, R_new)

#     return R_new

# def create_reward_array(R, T, T_max):
#     return jax.vmap(create_reward_array_row, in_axes=(0, 0, None))(R, T, T_max)

# def compute_rewards_to_go(r_arr_):
#     rewards_to_go = nancumsum_reverse(r_arr_)
#     baseline = jnp.nanmean(rewards_to_go, axis=0)
#     return rewards_to_go - baseline

# # Create array with rewards at their corresponding times
# R = jnp.array([[2,3,5,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan]])
# T = jnp.array([[1,2,6,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,7,jnp.nan,jnp.nan,jnp.nan]])
# T_max = T.shape[1] - 1
# R_new = create_reward_array(R, T, T_max)
# print('r_new=',R_new)  # prints: [2 2 3 5 5 5 5]

# # Compute 'reward to go' for each row
# rewards_to_go = compute_rewards_to_go(R_new)
# print('rtg=',rewards_to_go)

# # Subtract the baseline (column mean)
# baseline = jnp.nanmean(rewards_to_go, axis=0)
# print('b=',baseline)
# A = rewards_to_go - baseline
# print(A)

### WAY 1

# def masked_searchsorted(a, v, side='left'):
#     return jnp.min(jnp.where(a < v if side == 'left' else a <= v, len(a), jnp.arange(len(a))))

# def masked_take(a, indices):
#     mask = (indices >= 0) & (indices < len(a))
#     safe_indices = jnp.where(mask, indices, 0)
#     return jnp.where(mask, a[safe_indices], jnp.nan)

# def create_reward_array_row(R, T, T_max):
#     R_no_nan = jnp.nan_to_num(R, nan=0.0)
#     T_no_nan = jnp.nan_to_num(T, nan=T_max+1)
#     R_new = jnp.zeros(T_max + 1)

#     def body_fun(i, R_new):
#         index = masked_searchsorted(T_no_nan, i)
#         index = jnp.clip(index, 0, R_no_nan.shape[0] - 1)
#         value = masked_take(R_no_nan, index)
#         R_new = R_new.at[i].set(value)
#         return R_new

#     R_new = jax.lax.fori_loop(0, T_max + 1, body_fun, R_new)

#     return R_new

# def create_reward_array(R, T, T_max):
#     return jax.vmap(create_reward_array_row, in_axes=(0, 0, None))(R, T, T_max)
# from datetime import datetime

# def nancumsum_reverse(x):
#     y = jnp.flip(jnp.nancumsum(jnp.flip(x, axis=-1), axis=-1), axis=-1)
#     return jnp.where(jnp.isnan(x), jnp.nan, y)

# def create_reward_array(R, T, T_len):
#     R_new = jnp.zeros(T_len)
#     indices = jnp.searchsorted(T, jnp.arange(R_new.shape[0]), side='left')
#     indices = jnp.clip(indices, 0, R.shape[0] - 1)
#     R_new = jnp.take(R, indices)
#     return R_new

# def compute_A_1(R, T, T_len):
#     print('r=',R)
#     # 1. Create an array of 'reward to go' values
#     R_new = nancumsum_reverse(R) #jax.vmap(create_reward_array_row, in_axes=(0, 0, None))(R, T, T_len)
#     print('r_new=',R_new)
#     # 2. Place 'reward to go' values at correct points in new array
#     R_to_go = jax.vmap(create_reward_array,in_axes=(0,0,None),out_axes=(0))(R_new, T, T_len)
#     print('rtg=',R_to_go)
#     # 3. Compute the 'baseline'
#     baseline = jnp.nanmean(R_to_go, axis=0)  # Mean over rows (time steps)
#     print('b=',baseline)
#     # 4. Subtract the baseline from R_to_go
#     A = R_to_go - baseline
#     print('A=',A)
#     return A

# R = jnp.array([[2,3,5,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [2,2,3,4,5,6,7,jnp.nan], [3,2,3,4,5,6,7,jnp.nan], [4,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [5,3,5,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [6,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [7,2,3,4,5,6,7,jnp.nan], [8,2,3,4,5,6,7,jnp.nan], [9,2,3,4,5,jnp.nan,jnp.nan,jnp.nan],[2,3,5,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [2,2,3,4,5,6,7,jnp.nan], [3,2,3,4,5,6,7,jnp.nan], [4,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [5,3,5,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [6,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [7,2,3,4,5,6,7,jnp.nan], [8,2,3,4,5,6,7,jnp.nan], [9,2,3,4,5,jnp.nan,jnp.nan,jnp.nan],[2,3,5,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [2,2,3,4,5,6,7,jnp.nan], [3,2,3,4,5,6,7,jnp.nan], [4,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [5,3,5,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [6,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [7,2,3,4,5,6,7,jnp.nan], [8,2,3,4,5,6,7,jnp.nan], [9,2,3,4,5,jnp.nan,jnp.nan,jnp.nan],[2,3,5,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [2,2,3,4,5,6,7,jnp.nan], [3,2,3,4,5,6,7,jnp.nan], [4,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [5,3,5,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [6,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [7,2,3,4,5,6,7,jnp.nan], [8,2,3,4,5,6,7,jnp.nan], [9,2,3,4,5,jnp.nan,jnp.nan,jnp.nan]])
# LP = -R
# T = jnp.array([[1,2,6,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,7,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [1,2,6,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,7,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan],[1,2,6,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,7,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [1,2,6,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,7,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan],[1,2,6,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,7,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [1,2,6,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,7,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan],[1,2,6,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,7,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan], [1,2,6,jnp.nan,jnp.nan,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,7,jnp.nan,jnp.nan,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,6,7,jnp.nan], [1,2,3,4,5,jnp.nan,jnp.nan,jnp.nan]])
# # create R and T with many more rows and columns but still with many nans
# # R = jax.random.randint(jax.random.PRNGKey(0),  shape=(1000, 1000),minval=0,maxval=10)
# # T = jnp.repeat(jnp.arange(1000).reshape(1,1000),1000,axis=0) # jax.random.randint(jax.random.PRNGKey(0), shape=(1000, 1000),minval=0,maxval=1000)
# # R = jnp.where(R < 5, R, jnp.nan) 
# # T = jnp.where(T < 500, T, jnp.nan)

# T_len = T.shape[1]
# startTime = datetime.now()
# A1 = compute_A_1(R, T, T_len)
# print(datetime.now() - startTime)
# print(A1)

# ### WAY 2

# def create_reward_array_row(R,LP,T,T_len):
#     R_new = jnp.zeros(T_len)
#     LP_new = jnp.zeros(T_len)
#     def set_val(args):
#         i,R_new,LP_new = args
#         R_new = R_new.at[T[i].astype(int)].set(R[i])
#         LP_new = LP_new.at[T[i].astype(int)].set(LP[i])
#         return (R_new,LP_new)
#     def set_nan(args):
#         (i,R_new,LP_new) = args
#         R_new = R_new.at[T_len:].set(jnp.nan)
#         LP_new = LP_new.at[T_len:].set(jnp.nan)
#         return (R_new,LP_new)
#     for i in range(R.shape[0]):
#         (R_new,LP_new) = jax.lax.cond(~jnp.isnan(T[i]),set_val,set_nan,((i,R_new,LP_new)))
#     return R_new,LP_new

# def reward_to_go(R):
#     R_filled = jnp.nan_to_num(R,nan=0.0)
#     R_cumsum_reverse = jnp.cumsum(R_filled[:,::-1], axis=1)[:,::-1]
#     R_to_go = jnp.where(jnp.isnan(R),jnp.nan,R_cumsum_reverse)  # Retain nans
#     R_to_go = jnp.where(R_to_go == 0,jnp.nan,R_to_go)
#     return R_to_go

# def compute_A_2(R,LP, T, T_len):
#     print('r=',R)
#     # 1. Create array of rewards at their corresponding times, with 0's inbetween
#     R_new,LP_new = jax.vmap(create_reward_array_row, in_axes=(0, 0, 0,None))(R, LP,T, T_len)
#     print('r_new=',R_new)
#     # 2. Create an array of 'reward to go' values
#     R_to_go = reward_to_go(R_new)
#     print('rtg=',R_to_go)
#     # 3. Compute the 'baseline'
#     baseline = jnp.nanmean(R_to_go, axis=0)  # Mean over rows (time steps)
#     print('b=',baseline)
#     # 4. Subtract the baseline from R_to_go
#     A = R_to_go - baseline
#     print('A=',A)
#     LP_nonan = jnp.where(jnp.isnan(LP_new),0,LP_new)
#     A_nonan = jnp.where(jnp.isnan(A),0,A)
#     obj = -jnp.mean(jnp.nansum(jnp.multiply(LP_nonan,A_nonan),axis=1))
#     return obj

# # R = jax.random.randint(jax.random.PRNGKey(0),  shape=(1000, 1000),minval=0,maxval=10)
# # T = jnp.repeat(jnp.arange(1000),1000,axis=0) # jax.random.randint(jax.random.PRNGKey(0), shape=(1000, 1000),minval=0,maxval=1000)
# # R = jnp.where(R < 5, R, jnp.nan)
# # T = jnp.where(T < 500, T, jnp.nan)
# # T_len = T.shape[1]

# startTime = datetime.now()
# obj = compute_A_2(R, LP,T, T_len)
# print(datetime.now() - startTime)
# print(obj)

# print(jnp.array_equiv(A1,A2))

# def fnc(x,y,s):
#     return jnp.exp((jnp.cos(x) + jnp.cos(y) - 2)/s**2)

# x_ = jnp.linspace(-jnp.pi, jnp.pi, 500)
# y_ = jnp.linspace(-jnp.pi, jnp.pi, 500)

# integral=0

# for xi in x_:
#     for yi in y_:
#         integral += fnc(xi,yi,1)

# mean_val = integral/(len(x_)*len(y_))
# # print(mean_val)
# PLOTS = 3
# INIT_LENGTH = 0
# TEST_LENGTH = 5
# N_DOTS = 3
# # r_init_arr = np.random.uniform(0,1,size=(PLOTS,INIT_LENGTH))
# # r_test_arr = np.random.uniform(0,1,size=(PLOTS,TEST_LENGTH))
# # vecs_arr_0 = np.linspace(0,2*np.pi,10 )# np.random.uniform(0,1,size=(PLOTS,INIT_LENGTH,2))
# # vecs_arr_1 = np.linspace(0,2*np.pi,10 )# np.random.uniform(0,1,size=(PLOTS,INIT_LENGTH,2))
# vecs_arr = np.array([[[0,0],[0,1],[1,0],[1,1],[0.5,0.5]],[[0,0],[0,1],[1,0],[1,1],[0.5,0.5]],[[0,0],[0,1],[1,0],[1,1],[0.5,0.5]]]) #.reshape(PLOTS,N_DOTS,2)
# dots = np.array([[[0,0],[0,1],[1,0]], [[0,0],[0,1],[1,0]], [[0,0],[0,1],[1,0]]])
# # sel = np.random.randint(0,INIT_LENGTH,size=(PLOTS,N_DOTS))

# colors_ = np.float32([[255,0,0],[0,255,0],[0,0,255]])/255 # ,[255,0,0],[0,255,0],[0,0,255],[100,100,100]])/255

# # plot timeseries of true/planned reward
# fig,axis = plt.subplots(2*PLOTS,4,figsize=(15,6*PLOTS)) #9,4x plt.figure(figsize=(12,6))
# # title__ = f'EPOCHS={TOT_EPOCHS}, VMAPS={VMAPS}, PLAN_ITS={PLAN_ITS}, INIT_STEPS={INIT_STEPS}, TRIAL_LENGTH={TRIAL_LENGTH} \n SIGMA_R={SIGMA_R:.1f}, NEURONS={NEURONS**2}, MODULES={M}, H_P={H_P}, H_R={H_R}'
# plt.suptitle('outer_loop_pg_v4_test_, ',fontsize=10)
# for i in range(PLOTS):
#     ax1 = plt.subplot2grid((2*PLOTS,4),(2*i,2),colspan=2,rowspan=2)
#     for t in range(TEST_LENGTH):
#         # if sample_arr[i,t] == 1:
#         #     ax1.scatter(x=vecs_arr[i,t,0],y=vecs_arr[i,t,1],color='red',alpha=0.4,s=60,marker='o')
#         # else:
#         ax1.scatter(x=vecs_arr[i,t,0],y=vecs_arr[i,t,1],color='black',alpha=0.2,s=60,marker='o')
#     for d in range(N_DOTS):
#         ax1.scatter(x=dots[i,d,0],y=dots[i,d,1],color=colors_[d,:],s=120,marker='x')
#     ax1.set_xlim(-jnp.pi,jnp.pi)
#     ax1.set_ylim(-jnp.pi,jnp.pi)
#     ax1.set_xticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
#     ax1.set_xticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=14)
#     ax1.set_yticks([-jnp.pi,-jnp.pi/2,0,jnp.pi/2,jnp.pi])
#     ax1.set_yticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'],fontsize=14)
#     ax1.set_aspect('equal')
#     # ax1.set_title(f'vector heatmap, sel={sel[i,:]}',fontsize=14)

#     ax2 = plt.subplot2grid((2*PLOTS,4),(2*i+1,0),colspan=2,rowspan=1)
#     ax2.axis('off')
# plt.tight_layout()
# plt.subplots_adjust(top=0.94)
# plt.show()

a = jnp.array([2,3,4])
b = jnp.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
print('a=',a,a.shape)
print('b=',b,b.shape)
c = a.reshape((3,1)) + b
print('c=',c,c.shape)