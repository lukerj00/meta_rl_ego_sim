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

# a = jnp.array([2,3,4])
# b = jnp.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
# print('a=',a,a.shape)
# print('b=',b,b.shape)
# c = a.reshape((3,1)) + b
# print('c=',c,c.shape)

# import numpy as np

# def gen_arrays(N,keys):
#     # Total size of the grid
#     M = (2 * N + 1)**2

#     # Generate the plan vector array (A)
#     x = jnp.tile(jnp.arange(-2 * N, 2 * N + 1), 2 * N + 1)
#     y = jnp.repeat(jnp.arange(-2 * N, 2 * N + 1)[::-1], 2 * N + 1)
#     A = jnp.stack([x, y], axis=-1)
#     print('A,shape=',A,A.shape)

#     # Define indices corresponding to a centered N x N grid (inner area)
#     inner_indices = jnp.array([i * (2 * N + 1) + j for i in range(N, 3 * N) for j in range(N, 3 * N)])
#     print('inner_indices=',inner_indices)

#     # Shuffle all indices
#     shuffled_indices = rnd.permutation(keys[0], jnp.arange((2 * N + 1)**2), independent=False)

#     # Rearrange inner_indices to the start of the permutation
#     shuffled_indices = jnp.concatenate([shuffled_indices[jnp.isin(shuffled_indices, inner_indices)], 
#                                         shuffled_indices[~jnp.isin(shuffled_indices, inner_indices)]])

#     # Use shuffled indices to permute A and create V
#     A = A[shuffled_indices]
#     V = jnp.eye((2 * N + 1)**2)[shuffled_indices]

#     return A, V

# k = rnd.PRNGKey(0)
# keys = rnd.split(k,2)

# # # Example usage:
# # N = 1
# # A, V = gen_arrays(N,keys)
# # print("A :", A, A.shape)  # [4N**2, 2]
# # print("V :", V, V.shape)  # [4N**2, 4N**2]

# def gen_arrays(N, keys):
#     # Create the full grid of vectors
#     x = jnp.arange(-2 * N, 2 * N + 1)
#     y = jnp.arange(-2 * N, 2 * N + 1)
#     xv, yv = jnp.meshgrid(x, y)
#     A_full = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

#     # Determine the indices that are part of the inner grid
#     inner_mask = (jnp.abs(xv) <= N) & (jnp.abs(yv) <= N)
#     print('inner_mask=',inner_mask.flatten(),inner_mask.shape)
#     tot_ind = jnp.arange((2 * (2*N) + 1)**2)
#     A_inner_ind = tot_ind[inner_mask.flatten()]
#     A_outer_ind = tot_ind[~inner_mask.flatten()]

#     # Split into inner and outer vectors
#     A_inner = A_full[A_inner_ind]
#     A_outer = A_full[A_outer_ind]

#     print('A_inner=',A_inner,A_inner.shape)
#     print('A_outer=',A_outer,A_outer.shape)

# def gen_sc(keys,MODULES,ACTION_FRAC):
#     M = (MODULES-1)//(1/ACTION_FRAC)
#     vec_range = jnp.arange(MODULES**2)
#     x = jnp.arange(-M,M+1)
#     y = jnp.arange(-M,M+1)[::-1]
#     xv,yv = jnp.meshgrid(x,y)
#     A_full = jnp.vstack([xv.flatten(),yv.flatten()])

#     inner_mask = (jnp.abs(xv) <= M//2) & (jnp.abs(yv) <= M//2) # (all above just to create boolean mask, to select 'inner' indices)
#     print('inner_mask=',inner_mask.flatten())
#     A_inner_ind = vec_range[inner_mask.flatten()]
#     A_outer_ind = vec_range[~inner_mask.flatten()]
#     A_inner_perm = rnd.permutation(keys[0],A_inner_ind)
#     A_outer_perm = rnd.permutation(keys[1],A_outer_ind)
#     ID_ARR = jnp.concatenate((A_inner_perm,A_outer_perm),axis=0)

#     VEC_ARR = A_full[:,ID_ARR]
#     H1VEC_ARR = jnp.eye(MODULES**2) # [:,ID_ARR]
#     SC = (ID_ARR,VEC_ARR,H1VEC_ARR)
#     return SC

# (ID_ARR,VEC_ARR,H1VEC_ARR) = gen_sc(keys,5,0.5)
# print('ID_ARR=',ID_ARR,ID_ARR.shape)
# print('VEC_ARR=',VEC_ARR,VEC_ARR.shape)
# print('H1VEC_ARR=',H1VEC_ARR,H1VEC_ARR.shape)

# MODULES = 17 # (4*N+1)
# M = MODULES**2
# APERTURE = jnp.pi/2 ###
# ACTION_FRAC = 1/2
# ACTION_SPACE = ACTION_FRAC*APERTURE # 'AGENT_SPEED'
# DOT_SPEED = 2/3
# NEURONS_AP = 10
# NEURONS_FULL = jnp.int32(NEURONS_AP*(jnp.pi//APERTURE))
# print('N_F=',NEURONS_FULL)
# N = (NEURONS_AP**2)
# SIGMA_A = 0.5 # 0.5,1,0.3,1,0.5,1,0.1
# SIGMA_D = 0.5

# THETA_AP = jnp.linspace(-(APERTURE-APERTURE/NEURONS_AP),(APERTURE-APERTURE/NEURONS_AP),NEURONS_AP)
# THETA_FULL = jnp.linspace(-(jnp.pi-jnp.pi/NEURONS_FULL),(jnp.pi-jnp.pi/NEURONS_FULL),NEURONS_FULL)

# print('t_a=',THETA_AP,'t_f=',THETA_FULL)

# def neuron_act1(THETA, SIGMA_A, dot, pos):
#     N_ = THETA.size
#     dot = dot.reshape((1, 2))
#     th_x = jnp.tile(THETA, (N_, 1)).reshape((N_**2,))
#     th_y = jnp.tile(jnp.flip(THETA).reshape(N_, 1), (1, N_)).reshape((N_**2,))
#     G_0 = jnp.vstack([th_x, th_y])
#     print('G_0=',G_0)
#     E = G_0.T - (dot - pos)
#     act = jnp.exp((jnp.cos(E[:, 0]) + jnp.cos(E[:, 1]) - 2) / SIGMA_A**2)
#     return act

# def neuron_act2(THETA, SIGMA_A, dot, pos):
#     N_ = THETA.size
#     dot = dot.reshape((1, 2))
#     # th_x = jnp.tile(THETA, (N_, 1)).reshape((N_**2,))
#     # th_y = jnp.tile(jnp.flip(THETA).reshape(N_, 1), (1, N_)).reshape((N_**2,))
#     # G_0 = jnp.vstack([th_x, th_y])
#     xv,yv = jnp.meshgrid(THETA,THETA[::-1])
#     G_0 = jnp.vstack([xv.flatten(),yv.flatten()])
#     print('G_0=',G_0)
#     E = G_0.T - (dot - pos)
#     act = jnp.exp((jnp.cos(E[:, 0]) + jnp.cos(E[:, 1]) - 2) / SIGMA_A**2)
#     return act

# act1 = neuron_act1(THETA_FULL,SIGMA_A,jnp.array([1,2]),jnp.array([0,0]))
# print('a1=',act1)
# act2 = neuron_act2(THETA_FULL,SIGMA_A,jnp.array([1,2]),jnp.array([0,0]))
# print('a2=',act2)

# eq = jnp.array_equal(act1,act2)
# print(eq)

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse

# theta_x = np.random.uniform(-np.pi,np.pi,size=(100,))
# theta_y = np.random.normal(-np.pi,np.pi,size=(100,))
# weights = np.random.rand(100) # Random weights between 0 and 1

# # Map to [-pi, pi] space
# def circ_mean_var(v_t,x,y):
#     # Convert to complex numbers, weighted
#     z_x = weights*(np.cos(theta_x) + 1j*np.sin(theta_x))
#     z_y = weights*(np.cos(theta_y) + 1j*np.sin(theta_y))

#     mean_x = np.angle(np.sum(z_x)/np.sum(weights))
#     mean_y = np.angle(np.sum(z_y)/np.sum(weights))

#     # Compute weighted circular variance for each angle
#     circular_var_x = 1 - np.abs(np.sum(z_x) / np.sum(weights))
#     circular_var_y = 1 - np.abs(np.sum(z_y) / np.sum(weights))
#     circular_cov_matrix = np.diag([circular_var_x, circular_var_y])

#     # Draw the ellipse representing the circular covariance in blue
#     eigvals,eigvecs = np.linalg.eigh(circular_cov_matrix)
#     sigma_x,sigma_y = np.sqrt(eigvals)  # no Scale factor for visualization
#     return mean_x,mean_y,sigma_x,sigma_y

# mean_x,mean_y,sigma_x,sigma_y = circ_mean_var(weights,theta_x,theta_y)

# Scatter plot of the generated points, colored by weights
# plt.scatter(theta_x, theta_y, c=weights, alpha=0.5, cmap='viridis')

# # Draw the circular mean as a blue cross
# plt.scatter(mean_x, mean_y, c='blue', marker='x')

# ell_circular = Ellipse(xy=(mean_x, mean_y), width=sigma_x, height=sigma_y, edgecolor='blue', facecolor='none')
# plt.gca().add_patch(ell_circular)

# plt.xlim([-np.pi, np.pi])
# plt.ylim([-np.pi, np.pi])
# plt.xlabel('Theta X')
# plt.ylabel('Theta Y')
# plt.title('Weighted Circular Covariance')
# # plt.colorbar(label='Weights')
# plt.show()

# def gen_sc__(keys,MODULES,ACTION_SPACE,PLAN_SPACE):
#     index_range = jnp.arange(MODULES**2)
#     x = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)
#     y = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)[::-1]
#     xv,yv = jnp.meshgrid(x,y)
#     print('xv=',xv,'yv=',yv,'xv.shape=',xv.shape,'yv.shape=',yv.shape)
#     A_full = jnp.vstack([xv.flatten(),yv.flatten()])

#     inner_mask = (jnp.abs(xv) <= ACTION_SPACE) & (jnp.abs(yv) <= ACTION_SPACE)
#     print('s,f=',inner_mask.shape,inner_mask.flatten().shape,inner_mask)
#     A_inner_ind = index_range[inner_mask.flatten()]
#     A_outer_ind = index_range[~inner_mask.flatten()]
#     A_inner_perm = rnd.permutation(keys[0],A_inner_ind)
#     A_outer_perm = rnd.permutation(keys[1],A_outer_ind)
#     ID_ARR = jnp.concatenate((A_inner_perm,A_outer_perm),axis=0)

#     VEC_ARR = A_full[:,ID_ARR]
#     H1VEC_ARR = jnp.eye(MODULES**2) # [:,ID_ARR]
#     SC = (ID_ARR,VEC_ARR,H1VEC_ARR)
#     return SC

# sc = gen_sc__(keys,7,(1/4)*(jnp.sqrt(2)/2)*jnp.pi,(3/8)*(jnp.sqrt(2)/2)*jnp.pi)
# print(sc[0].shape,sc[1].shape,sc[2].shape)
# print('vec_arr=',sc[1],'h1vec_arr=',sc[2])

# APERTURE = (jnp.sqrt(2)/2)*jnp.pi ###
# NEURONS_FULL = 12 # jnp.int32(NEURONS_AP*(jnp.pi//APERTURE))
# N_F = (NEURONS_FULL**2)
# NEURONS_AP = jnp.int32(jnp.floor(NEURONS_FULL*(APERTURE/jnp.pi))) # 6 # 10
# N_A = (NEURONS_AP**2)
# THETA_FULL = jnp.linspace(-(jnp.pi-jnp.pi/NEURONS_FULL),(jnp.pi-jnp.pi/NEURONS_FULL),NEURONS_FULL)
# THETA_AP = THETA_FULL[NEURONS_FULL//2 - NEURONS_AP//2 : NEURONS_FULL//2 + NEURONS_AP//2]

# print('AP=',APERTURE,'THETA_FULL=',THETA_FULL,'THETA_AP=',THETA_AP)

def gen_sc(keys,MODULES,ACTION_SPACE,PLAN_SPACE):
    index_range = jnp.arange(MODULES**2)
    x = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)
    y = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)[::-1]
    xv,yv = jnp.meshgrid(x,y)
    A_full = jnp.vstack([xv.flatten(),yv.flatten()])

    inner_mask = (jnp.abs(xv) <= ACTION_SPACE) & (jnp.abs(yv) <= ACTION_SPACE)
    A_inner_ind = index_range[inner_mask.flatten()]
    A_outer_ind = index_range[~inner_mask.flatten()]
    A_inner_perm = rnd.permutation(keys[0],A_inner_ind)
    A_outer_perm = rnd.permutation(keys[1],A_outer_ind)
    ID_ARR = jnp.concatenate((A_inner_perm,A_outer_perm),axis=0)

    VEC_ARR = A_full[:,ID_ARR]
    H1VEC_ARR = jnp.eye(MODULES**2) # [:,ID_ARR]
    SC = (ID_ARR,VEC_ARR,H1VEC_ARR)
    return SC

# def gen_timeseries(SC,pos_0,dot_0,dot_vec,samples,step_array):
#     ID_ARR,VEC_ARR,H1VEC_ARR = SC
#     h1vec_arr = H1VEC_ARR[:,samples]
#     vec_arr = VEC_ARR[:,samples]
#     cum_vecs = jnp.cumsum(vec_arr,axis=1)
#     pos_1_end = (pos_0+cum_vecs.T).T
#     dot_1_end = (dot_0+jnp.outer(dot_vec,step_array).T).T
#     pos_arr = jnp.concatenate((pos_0.reshape(2,1),pos_1_end),axis=1)
#     dot_arr = jnp.concatenate((dot_0.reshape(2,1),dot_1_end),axis=1)
#     return pos_arr,dot_arr,h1vec_arr,vec_arr

# SC = gen_sc(keys,7,(1/4)*(jnp.sqrt(2)/2)*jnp.pi,(3/8)*(jnp.sqrt(2)/2)*jnp.pi)
# pa,da,h1v,va = gen_timeseries(SC,jnp.array([0.,0.]),jnp.array([1.,1.]),jnp.array([0.5,0.5]),rnd.randint(keys[0],shape=(19,),minval=0,maxval=9),jnp.arange(1,20))
# print('pa=',pa,'da=',da,'h1v=',h1v,'va=',va)

# def get_inner_activation_indices(N, x):
#     inner_N = jnp.int32(N * x) # Length of one side of the central region
#     start_idx = (N - inner_N) // 2 # Starting index
#     end_idx = start_idx + inner_N # Ending index
#     row_indices, col_indices = jnp.meshgrid(jnp.arange(start_idx, end_idx), jnp.arange(start_idx, end_idx))
#     flat_indices = jnp.ravel_multi_index((row_indices.flatten(), col_indices.flatten()), (N, N))
#     return flat_indices
# flat_indices = get_inner_activation_indices(4,0.5)
# ind_take = jnp.take(jnp.arange(32),flat_indices)
# print('INNER=',flat_indices,ind_take,ind_take.shape)

import jax
import jax.numpy as jnp
from jax import random as rnd

# def gen_binary_timeseries(keys, N, switch_prob, K): # new weird
#     ts_init = jnp.zeros(N)  # start with 0s
    
#     def body_fn(i, carry):
#         ts, count_ones, keys = carry
        
#         # Function to execute if the current element is 0.
#         true_fn = lambda _: (jnp.int32(rnd.bernoulli(keys[i], p=switch_prob)), 0)
        
#         # Function to execute if the current element is 1.
#         false_fn = lambda _: (jnp.int32(rnd.bernoulli(keys[i], p=1-switch_prob)), count_ones + 1)
        
#         # Determine the value and count to set for the next element.
#         next_val, count_ones_next = jax.lax.cond(ts[i] == 0, true_fn, false_fn, None)
            
#         def set_k_zeros_true_fn(_):
#             return ts.at[i + 1:i + K + 1].set(0), i + K
        
#         def set_k_zeros_false_fn(_):
#             def set_one_true_fn(_):
#                 return ts.at[i + 1].set(next_val), i + 1
#             def set_one_false_fn(_):
#                 return ts, i  # do nothing, return ts and i as is.
#             return jax.lax.cond(i + 1 < N, set_one_true_fn, set_one_false_fn, None)
        
#         ts, i = jax.lax.cond((next_val == 0) & (i + K < N), set_k_zeros_true_fn, set_k_zeros_false_fn, None)
        
#         return (ts, count_ones_next, keys)

    
#     # Starting from the second element, loop until the end of the array.
#     result, _, _ = jax.lax.fori_loop(1, N, body_fn, (ts_init, 0, keys))
    
#     return result

def gen_sc(keys,MODULES,ACTION_SPACE,PLAN_SPACE):
    index_range = jnp.arange(MODULES**2)
    x = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)
    y = jnp.linspace(-PLAN_SPACE,PLAN_SPACE,MODULES)[::-1]
    xv,yv = jnp.meshgrid(x,y)
    A_full = jnp.vstack([xv.flatten(),yv.flatten()])

    inner_mask = (jnp.abs(xv) <= ACTION_SPACE) & (jnp.abs(yv) <= ACTION_SPACE)
    A_inner_ind = index_range[inner_mask.flatten()]
    A_outer_ind = index_range[~inner_mask.flatten()]
    A_inner_perm = rnd.permutation(keys[0],A_inner_ind)
    A_outer_perm = rnd.permutation(keys[1],A_outer_ind)
    ID_ARR = jnp.concatenate((A_inner_perm,A_outer_perm),axis=0)

    VEC_ARR = A_full[:,ID_ARR]
    H1VEC_ARR = jnp.eye(MODULES**2) # [:,ID_ARR]
    SC = (ID_ARR,VEC_ARR,H1VEC_ARR)
    return SC

def gen_binary_timeseries(keys, N, switch_prob, K):
    ts_list = [0]  # Start with a zero
    current_val = 0  # Initial value
    i = 0  # Initialize index
    
    while i < N - 1:  # Continue until the end of the array
        if rnd.uniform(keys[i]) < switch_prob:  # Decide whether to switch the value
            current_val = 1 - current_val  # Switch the value
        
        if current_val == 0:  # If current value is 0 and there is enough space
            ts_list.extend([0] * (K))  # Insert K zeros
            i += K   # Move index K steps forward
        else:  # If there is space for one more element
            ts_list.append(current_val)  # Insert current value
            i += 1  # Move index 1 step forward
        # else:
        #     break  # Exit the loop if the end of the array is reached
            
    # Convert list to a numpy array, ensuring the length is exactly N
    ts_array = jnp.array(ts_list[:N], dtype=jnp.int32)
    return ts_array

# def gen_binary_timeseries(keys, N, switch_prob, max_plan_length): # old
#     ts_init = jnp.zeros(N)  # start with 0s
#     if max_plan_length == 0:
#         return ts_init
#     def body_fn(i, carry):
#         ts, count_ones, keys = carry
#         true_fn = lambda _: (jnp.int32(rnd.bernoulli(keys[i], p=switch_prob)), 0)
#         def false_fn(_):
#             return jax.lax.cond(count_ones >= max_plan_length - 1,
#                                 lambda _: (0, 0), # ts, count_ones
#                                 lambda _: (jnp.int32(rnd.bernoulli(keys[i], p=1-switch_prob)), count_ones + 1),
#                                 None)
#         next_val, count_ones_next = jax.lax.cond(ts[i] == 0, true_fn, false_fn, None)
#         ts = ts.at[i+1].set(next_val)
#         i += 1
#         return (ts, count_ones_next, keys)
#     result, _, _ = jax.lax.fori_loop(1, N, body_fn, (ts_init, 0, keys))
#     return result

# def gen_binary_timeseries(keys, N, switch_prob, max_plan_length, K): # new
#     ts_init = jnp.zeros(N)  # start with 0s
#     if max_plan_length == 0 or K >= N:
#         return ts_init
    
#     def body_fn(i, carry):
#         ts, count_ones, keys = carry
#         end_idx = i + K  # End index for insertion of zeros
        
#         def insert_zeros(ts, start, end_idx):
#             end = jnp.minimum(end_idx, N)  # Ensure we don't exceed the array length
#             indices = jnp.arange(start, end)
#             return ts.at[indices].set(0)
        
#         true_fn = lambda _: (insert_zeros(ts, i, end_idx), 0)  # Insert K zeros
#         def false_fn(_):
#             return jax.lax.cond(
#                 count_ones >= max_plan_length - 1,
#                 lambda _: (insert_zeros(ts, i, i+1), 0),  # Insert a single 0
#                 lambda _: (ts.at[i].set(jnp.int32(rnd.bernoulli(keys[i], p=1-switch_prob))), count_ones + 1),
#                 None)
        
#         ts, count_ones_next = jax.lax.cond(ts[i] == 0, true_fn, false_fn, None)
#         i = jnp.minimum(i + K, N-1)  # Move the index by K steps, but don't exceed N-1
#         return (ts, count_ones_next, keys)
    
#     result, _, _ = jax.lax.fori_loop(1, N, body_fn, (ts_init, 0, keys))
#     return result

def gen_timeseries(key, SC, pos_0, dot_0, dot_vec, samples, switch_prob, N, PLAN_RATIO):
    ID_ARR, VEC_ARR, H1VEC_ARR = SC
    keys = rnd.split(key, N)
    binary_series = gen_binary_timeseries(keys, N, switch_prob, PLAN_RATIO)
    print('binary_series=',binary_series,binary_series.shape)
    binary_array = jnp.vstack([binary_series, 1 - binary_series]).T
    h1vec_arr = H1VEC_ARR[:, samples]
    vec_arr = VEC_ARR[:, samples]
    def time_step(carry, i):
        pos_plan, pos, dot = carry
        # dot_next = dot + dot_vec  # Assuming dot_vec is a globally available vector
        def true_fn(_):
            return pos + vec_arr[:, i], pos + vec_arr[:, i], dot + dot_vec
        def false_fn(_):
            return pos_plan + vec_arr[:, i], pos, dot + dot_vec
        pos_plan_next, pos_next, dot_next = jax.lax.cond(binary_series[i] == 0, true_fn, false_fn, None)
        return (pos_plan_next, pos_next, dot_next), (pos_plan_next, pos_next, dot_next)
    init_carry = (jnp.array(pos_0), jnp.array(pos_0), jnp.array(dot_0))
    _, (pos_plan_stacked, pos_stacked, dot_stacked) = jax.lax.scan(time_step, init_carry, jnp.arange(N-1))
    return binary_array,jnp.concatenate([jnp.reshape(pos_0,(-1,2)),pos_plan_stacked]),jnp.concatenate([jnp.reshape(pos_0,(-1,2)),pos_stacked]),jnp.concatenate([jnp.reshape(dot_0,(-1,2)),dot_stacked]),h1vec_arr,vec_arr # should be 2,N

APERTURE = (1/2)*jnp.pi ###
NEURONS_FULL = 12 # jnp.int32(NEURONS_AP*(jnp.pi//APERTURE))
N_F = (NEURONS_FULL**2)
NEURONS_AP = jnp.int32(jnp.floor(NEURONS_FULL*(APERTURE/jnp.pi))) # 6 # 10
N_A = (NEURONS_AP**2)
THETA_FULL = jnp.linspace(-(jnp.pi-jnp.pi/NEURONS_FULL),(jnp.pi-jnp.pi/NEURONS_FULL),NEURONS_FULL)
THETA_AP = THETA_FULL[NEURONS_FULL//2 - NEURONS_AP//2 : NEURONS_FULL//2 + NEURONS_AP//2]
N = 120
switch_prob = 0.2
max_plan_length = N
PLAN_RATIO = 10
MODULES = 9

k = rnd.PRNGKey(1)
keys = rnd.split(k,N)

SC = gen_sc(keys,MODULES,(1/2)*jnp.pi,(1/2)*jnp.pi)

result = gen_binary_timeseries(keys, N, switch_prob, PLAN_RATIO)
print(result,len(result))

binary_array,pos_plan_stacked,pos_stacked,dot_stacked,h1vec_arr,vec_arr = gen_timeseries(keys[0],SC,jnp.array([0.,0.]),jnp.array([1.,1.]),jnp.array([0.5,0.5]),rnd.randint(keys[0],shape=(N,),minval=0,maxval=9),switch_prob,N,PLAN_RATIO)

print('binary_array=',binary_array,binary_array.shape)
print('pos_plan_stacked=',pos_plan_stacked,pos_plan_stacked.shape)
print('pos_stacked=',pos_stacked,pos_stacked.shape)
print('dot_stacked=',dot_stacked,dot_stacked.shape)
print('h1vec_arr=',h1vec_arr,h1vec_arr.shape)
print('vec_arr=',vec_arr,vec_arr.shape)

norm_pos = jnp.linalg.norm(pos_stacked, axis=1)
norm_pos_plan = jnp.linalg.norm(pos_plan_stacked, axis=1)

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting norms
plt.plot(norm_pos[1:], label="Norm of pos_arr", color='blue')
plt.plot(norm_pos_plan[1:], label="Norm of pos_plan_arr", color='green')

# Plotting binary time series
plt.plot(binary_array[:,0] * norm_pos.max(), label="Binary time series", color='red', linestyle='--')

# Additional plot settings
plt.title(f"Norm of pos_arr and pos_plan_arr, switch_prob={'0.2'}, max_plan_length={''}", fontsize=14)
plt.xlabel("Time step")
plt.ylabel("Norm")
plt.legend()
plt.grid(True)

plt.show()
