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
# policy = .softmax(sc,s_weights) # softmax layer outputs policy; weights needed to determine action/plan
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
# self.vector = vector # [v_x,v_y]
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
# x,y = i%(jnp.sqrt(MODULES))
# vector = [x,y] + rnd.normal(params["sigma_vec"])
# sc[i] = Module(i,vector,params)

## forward model training (self-supervised - 'motor babbling')
#-want to predict v_t,pos_t as a fnc of 1hvec,v_t_1,pos_t_1,h_t_1
#-given v_t_1,pos_t_1,1hvec attempt to predict next v_t,pos_t with RNN; compare with true v_t to get loss:

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

# existing:
# full_loop
# train_loop
# -grad of tot_reward
# -optimizer update
# tot_reward
# -scan single_step
# single_step
# -env loop; gru computations

