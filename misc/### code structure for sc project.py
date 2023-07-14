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
# s_weights,m_weights,v_weights = weights
# (hs_t_1,hm_t_1,hv_t_1,pos_t_1,v_t_1,r_t_1,sc,*_) = carry # unpack from previous timestep -- need to clarify e_t; refer to pos_t/dots/visual activations?
# sc_map,sc_modules = sc # sc_map=scalar map which dictates policy; sc_modules=array of sc module objects
# policy = .softmax(sc_map,s_weights) # softmax layer outputs policy; weights needed to determine action/plan
# module,action (move/plan) = .sample(policy) # sample movement/plan based on policy (rnn params) (OR choose action based on weighted sum)
# pos_t,v_t,r_t = module.move(self,hv_t_1,pos_t_1,v_t_1,r_t_1,params,weights) # use value model
# OR
# ^pos_t,^v_t,^r_t = module.plan(self,hv_t_1,pos_t_1,v_t_1,r_t_1,weights) # use world model,value model (return sequence of movement vectors/dreamt reward)
# sc_map,hs_t = RNN_map_update(sc_map,hs_t_1,pos_t,v_t,r_t,module,weights) # RNN computations to update sc_map
# return (hs_t,hm_t,hv_t,pos_t,v_t,r_t,sc,*_),(data) # (carry,stacked output)

# class sc_module()
# __init__(self,id,vector)
# self.id = id # number within array, used for 1-hot vector
# self.1hvec = [0,...1,...,0] # (1 at id'th position)
# self.vector = vector # [v_x,v_y]
#
# move(module,hv_t_1,pos_t_1,v_t_1,r_t_1,params,weights) # ,(needs movement vector/current visual input/reward at this input)
# plan(module,hv_t_1,pos_t_1,v_t_1,r_t_1,weights) # predicting pos/value without calculating losses (no knowledge of dots)

# move(self,hv_t_1,pos_t_1,v_t_1,r_t_1,params,weights)
# pos_t,v_t = env_interact(self,pos_t_1,params.dots)
# hv_t->r_t = value_model(hv_t_1,v_t,r_t_1,v_weights) # just v_t->r_t (not recurrent)
# return pos_t,v_t,r_t

# plan(self,weights,hv_t_1,pos_t_1,v_t_1,r_t_1,weights)
# s_weights,m_weights,v_weights = weights
# hm_0 = params.hm_0
# loop around: # (for pre-determined planning horizon)
# hm_t->^pos_t,^v_t = forward_model(self,hm_t_1,pos_t_1,v_t_1,m_weights)
# hv_t->^r_t = value_model(hv_t_1,^v_t,r_t_1,v_weights)
# end loop
# return ^pos_t,^v_t,^r_t # also return ^dots (estimate of dot locations)?

# env_interact(self,pos_t_1,params)
# pos_t = pos_t_1 + self.vector
# v_t = neuron_act(pos_t,params.dots) # find v_t from neuron activations
# return pos_t,v_t

# forward_model(self,hm_t_1,pos_t_1,v_t_1,m_weights)
# loop aroun: (for pre-determined prediction length)
# hm_t->^pos_t,^v_t = RNN_forward(self.1hvec,hm_t_1,pos_t_1,v_t_1,m_weights) # find ^v_t as a predictive scalar array
# end loop
# return hm_t->^pos_t,^v_t

# value_model(hv_t_1,v_t,r_t_1,v_weights)
# hv_t->r_t = RNN_value(hv_t_1,v_t,r_t_1,v_weights)
# return hv_t->r_t

## forward model training (self-supervised - 'motor babbling')
#-want to predict v_t,pos_t as a fnc of 1hvec,v_t_1,pos_t_1,h_t_1
#-given v_t_1,pos_t_1,1hvec attempt to predict next v_t,pos_t with RNN; compare with true v_t to get loss:

# forward_model_loop(sc,weights,params)
# loop around:
# hm_0,pos_0,v_0 = gen_pv(params)
# loss = body_fnc(sc,h_0,pos_0,v_0,params,weights)
# grad = .grad(loss)
# weights = .update(weights,grad)
# end loop
# trained_weights = weights
# return loss,trained_weights

# body_fnc(sc,hm_t_1,pos_t_1,v_t_1,params,weights)
# sc_map,sc_modules = sc
# loop around:
# module = .sample(sc_modules)
# hm_t->^pos_t,^v_t = forward_model(module,hm_t_1,pos_t_1,v_t_1,m_weights)
# pos_t,v_t = env_interact(module,pos_t_1,params.dots)
# end loop
# loss = (^v_arr-v_arr)**2 # (include pos_t?)
# return loss

# gen_pv(params,dots)
# pos_0 = rnd.uniform(params)
# v_0 = neuron_act(pos_0,params.dots)
# return pos_0,v_0

## value model training (supervised)
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
# hv_0 = params.hv_0
# loop for t=t_train_phase_1 timesteps:
# hv_t->r_t_pred = value_model(hv_t_1,v_t,r_t_1,v_weights)
# loss_train = (r_t_pred-r_t)**2 # across all timeseries
# end
# return loss_train

# train_phase_2(weights,params)
# v_arr_2,r_arr_2 = gen_vr_arr_2(params) # dataset of v_t,r_t values (with additional v_t values and no corresponding r_t)
# hv_0 = params.hv_0
# loop for t=t_train_phase_2 timesteps:
# hv_t->r_t_pred = value_model(hv_t_1,v_t,r_t_1,v_weights)
# loss_train = (r_t_pred-r_t)**2
# end
# return loss_train

# gen_vr_arr_1(params)
# (gen pos/dot arrays)
# calculate v_arr_1
# calculate r_arr_1
# return v,r

# gen_vr_arr_2(params)
# (gen pos/dot arrays)
# calculate v_arr_2
# calculate r_arr_2
# return v,r

# existing:
# full_loop
# train_loop
# -grad of tot_reward
# -optimizer update
# tot_reward
# -scan single_step
# single_step
# -env loop; gru computations

