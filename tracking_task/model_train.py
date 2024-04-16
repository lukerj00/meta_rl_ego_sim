from functools import partial
import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit
import optax
import copy

def gen_sc(keys, modules, action_space, plan_space):
    index_range = jnp.arange(modules**2)
    x = jnp.linspace(-plan_space, plan_space, modules)
    y = jnp.linspace(-plan_space, plan_space, modules)[::-1]
    xv, yv = jnp.meshgrid(x, y)
    A_full = jnp.vstack([xv.flatten(), yv.flatten()])

    inner_mask = (jnp.abs(xv) <= action_space) & (jnp.abs(yv) <= action_space)
    A_inner_ind = index_range[inner_mask.flatten()]
    A_outer_ind = index_range[~inner_mask.flatten()]
    A_inner_perm = rnd.permutation(keys[0], A_inner_ind)
    A_outer_perm = rnd.permutation(keys[1], A_outer_ind)
    ID_ARR = jnp.concatenate((A_inner_perm, A_outer_perm), axis=0)

    VEC_ARR = A_full[:, ID_ARR]
    vec_norm = jnp.linalg.norm(VEC_ARR, axis=0)
    prior_vec = jnp.exp(-vec_norm) / jnp.sum(jnp.exp(-vec_norm))
    H1VEC_ARR = jnp.eye(modules**2)
    zero_vec_index = jnp.where(jnp.all(VEC_ARR == jnp.array([0,0])[:, None], axis=0))[0][0]
    SC = (ID_ARR, VEC_ARR, H1VEC_ARR)
    return SC, prior_vec, zero_vec_index

def new_params(params, epoch):
    VMAPS = params["VMAPS"]
    INIT_S = params["INIT_S"]
    INIT_P = params["INIT_P"]
    H_S = params["H_S"]
    H_P = params["H_P"]
    APERTURE = params["APERTURE"]
    MODULES = params["MODULES"]
    T = params["TRIAL_LENGTH"]
    ACTION_SPACE = params["ACTION_SPACE"]

    ki = rnd.split(rnd.PRNGKey(epoch), num=6)
    params["HS_0"] = jnp.sqrt(INIT_S / H_S) * rnd.normal(ki[0], (VMAPS, H_S))
    params["HP_0"] = jnp.sqrt(INIT_P / H_P) * rnd.normal(ki[1], (VMAPS, H_P))
    params["POS_0"] = rnd.uniform(ki[2], shape=(VMAPS, 2), minval=-jnp.pi, maxval=jnp.pi)
    DOT_0_VEC = rnd.uniform(ki[3], shape=(VMAPS, 2), minval=-ACTION_SPACE, maxval=ACTION_SPACE)
    params["DOT_0"] = params["POS_0"] + DOT_0_VEC
    params["DOT_VEC"] = (-DOT_0_VEC + rnd.uniform(ki[4], shape=(VMAPS, 2), minval=-ACTION_SPACE, maxval=ACTION_SPACE)) * (params["MAX_DOT_SPEED"] / (2 * params["PLAN_RATIO"]))
    params["IND"] = rnd.randint(ki[5], (VMAPS, T), minval=0, maxval=params["M"], dtype=jnp.int32)

# def get_initial_scan_args(scan_params):
#     h0, dots, select, epsilon_n = scan_params
#     activations_0 = neuron_act(None, dots, params)
#     h_0 = GRU_step(activations_0, h0, weights)
#     carry = (None, h_0, dots, select, None, params)
#     return carry, epsilon_n

@jit
def neuron_act(dot,pos,params):
    THETA = params["THETA"]
    SIGMA_A = params["SIGMA_A"]
    SIGMA_N = params["SIGMA_N"]
    val = params["VAL"]
    key = rnd.PRNGKey(val)
    N_ = THETA.size
    dot = dot.reshape((1, 2))
    xv,yv = jnp.meshgrid(THETA,THETA[::-1])
    G_0 = jnp.vstack([xv.flatten(),yv.flatten()])
    E = G_0.T - (dot - pos)
    act = jnp.exp((jnp.cos(E[:, 0]) + jnp.cos(E[:, 1]) - 2) / SIGMA_A**2)
    noise = (SIGMA_N*act)*rnd.normal(key,shape=(N_**2,))
    return act + noise

def old_trajectory_step(carry_args, x):
    (pos_tm1, h_tm1, dots, select, weights, params) = carry_args
    activations_t = neuron_act(pos_tm1, dots, params)
    h_t = GRU_step(activations_t, h_tm1, weights)
    policy_t = jax.nn.softmax(weights['D'] @ h_t)
    pos_ind_t = rnd.choice(rnd.PRNGKey(0), policy_t.size, p=policy_t)
    pos_t = dots[pos_ind_t]
    carry = (pos_t, h_t, dots, select, weights, params)
    old_trajectory = {
        "log_probs": jax.lax.stop_gradient(jnp.log(policy_t[pos_ind_t])),
        "decisions": jnp.eye(params['N_DECISIONS'])[pos_ind_t],
        "mask_array": jnp.ones(params["TOT_STEPS"]),
    }
    old_debug = (pos_ind_t, pos_t, dots)
    return carry, (old_trajectory, old_debug)

def new_trajectory_step(carry_args, old_trajectory_t):
    old_pos_ind_t, old_pos_t, old_dots = old_trajectory_t
    (pos_tm1, h_tm1, dots, select, weights, params) = carry_args
    activations_t = neuron_act(pos_tm1, dots, params)
    h_t = GRU_step(activations_t, h_tm1, weights)
    policy_t = jax.nn.softmax(weights['D'] @ h_t)
    new_trajectory = {
        "log_probs": jnp.log(policy_t[old_pos_ind_t]),
        "rewards": get_reward(old_pos_t, dots, select, params),
        "values": critic_forward(h_t, weights),
        "kl_vectors": optax.kl_divergence(jax.lax.stop_gradient(policy_t), policy_t),
        "kl_decisions": 0.0,
    }
    carry_args = (old_pos_t, h_t, dots, select, weights, params)
    return carry_args, (new_trajectory, None)

def get_reward(pos, dots, select, params, epoch):
    SIGMA_R0 = params["SIGMA_R0"]
    SIGMA_RINF = params["SIGMA_RINF"]
    TAU = params["TAU"]

    dot = jnp.matmul(select.reshape((1,select.size)), dots)
    theta_d_0 = jnp.arctan2(dot[0,0], dot[0,1])
    theta_d_1 = jnp.arctan2(pos[0], pos[1])
    sigma_e = SIGMA_RINF * (1 - jnp.exp(-epoch/TAU)) + SIGMA_R0 * jnp.exp(-epoch/TAU)

    reward = jnp.exp((jnp.cos(theta_d_0 - theta_d_1) - 1)/sigma_e**2)
    return reward

def GRU_step(activations, h, weights):
    z_t = jax.nn.sigmoid(jnp.matmul(weights["Wr_z"], activations[0]) + 
                         jnp.matmul(weights["Wg_z"], activations[1]) + 
                         jnp.matmul(weights["Wb_z"], activations[2]) + 
                         jnp.matmul(weights["U_z"], h) + 
                         weights["b_z"])
    f_t = jax.nn.sigmoid(jnp.matmul(weights["Wr_r"], activations[0]) + 
                         jnp.matmul(weights["Wg_r"], activations[1]) + 
                         jnp.matmul(weights["Wb_r"], activations[2]) + 
                         jnp.matmul(weights["U_r"], h) + 
                         weights["b_r"])
    hhat_t = jnp.tanh(jnp.matmul(weights["Wr_h"], activations[0]) + 
                      jnp.matmul(weights["Wg_h"], activations[1]) + 
                      jnp.matmul(weights["Wb_h"], activations[2]) + 
                      jnp.matmul(weights["U_h"], jnp.multiply(f_t, h)) + 
                      weights["b_h"])
    h_t = jnp.multiply(z_t, h) + jnp.multiply((1 - z_t), hhat_t)
    return h_t

def critic_forward(h, weights, params):
    W_full = weights["W_full"]
    W_ap = weights["W_ap"]
    v_pred_full = jnp.matmul(W_full, h)
    v_pred_ap = jnp.take(v_pred_full, params["INDICES"])
    return v_pred_ap

def loss_obj(dot, pos, params, epoch):
    sigma_k = sigma_fnc(params["VMAPS"], params["SIGMA_RINF"], params["SIGMA_R0"], params["TAU"], epoch)
    dis = dot - pos
    obj = (params["SIGMA_R0"] / sigma_k) * jnp.exp((jnp.cos(dis[0]) + jnp.cos(dis[1]) - 2) / sigma_k**2)
    return obj

def sigma_fnc(VMAPS, SIGMA_RINF, SIGMA_R0, TAU, epoch):
    k = epoch
    sigma_k = SIGMA_RINF * (1 - jnp.exp(-k / TAU)) + SIGMA_R0 * jnp.exp(-k / TAU)
    return sigma_k

def get_policy(args, weights_s, params):
    (hs_t_1,hv_t_1,*_,act_t,v_t,r_t,rp_t,move_counter,epoch) = args
    r_arr = jnp.array([r_t,rp_t])

    z_t = jax.nn.sigmoid(jnp.matmul(weights_s["Ws_vt_z"],v_t) + 
                         jnp.matmul(weights_s["Ws_rt_z"],r_arr) + 
                         jnp.matmul(weights_s["Ws_at_1z"],act_t) + 
                         jnp.matmul(weights_s["Ws_ht_1z"],hv_t_1) + 
                         jnp.matmul(weights_s["Us_z"],hs_t_1) + 
                         weights_s["bs_z"])
    f_t = jax.nn.sigmoid(jnp.matmul(weights_s["Ws_vt_f"],v_t) + 
                         jnp.matmul(weights_s["Ws_rt_f"],r_arr) + 
                         jnp.matmul(weights_s["Ws_at_1f"],act_t) + 
                         jnp.matmul(weights_s["Ws_ht_1f"],hv_t_1) + 
                         jnp.matmul(weights_s["Us_f"],hs_t_1) + 
                         weights_s["bs_f"])
    hhat_t = jax.nn.tanh(jnp.matmul(weights_s["Ws_vt_h"],v_t) + 
                         jnp.matmul(weights_s["Ws_rt_h"],r_arr) + 
                         jnp.matmul(weights_s["Ws_at_1h"],act_t) + 
                         jnp.matmul(weights_s["Ws_ht_1h"],hv_t_1) + 
                         jnp.matmul(weights_s["Us_h"],jnp.multiply(f_t,hs_t_1)) + 
                         weights_s["bs_h"])
    hs_t = jnp.multiply(1-z_t,hs_t_1) + jnp.multiply(z_t,hhat_t)
    vec_logits = jnp.matmul(weights_s["Ws_vec"],hs_t)
    act_logits = jnp.matmul(weights_s["Ws_act"],hs_t)
    val_inter_t = jax.nn.relu(jnp.matmul(weights_s["Ws_val_inter"],hs_t))
    val_t = jnp.matmul(weights_s["Ws_val_r"],val_inter_t)
    vectors_t = jax.nn.softmax((vec_logits/params["TEMP_VS"]) - jnp.max(vec_logits/params["TEMP_VS"]) + 1e-8)
    actions_t = jax.nn.softmax((act_logits/params["TEMP_AS"]) - jnp.max(act_logits/params["TEMP_AS"]) + 1e-8)
    return (vectors_t,actions_t),val_t,hs_t

@jit
def v_predict(h1vec, v_0, hv_t_1, act_t, weights_v, params, NONE_PLAN):
    def loop_body(carry,i):
        return RNN_forward(carry)
    def RNN_forward(carry):
        weights_v,h1vec,v_0,act_t,hv_t_1 = carry
        plan_t = act_t[0]
        move_t = act_t[1]
        W_h1_f = weights_v["W_h1_f"]
        W_h1_hh = weights_v["W_h1_hh"]
        W_p_f = weights_v["W_p_f"]
        W_p_hh = weights_v["W_p_hh"]
        W_m_f = weights_v["W_m_f"]
        W_m_hh = weights_v["W_m_hh"]
        W_v_f = weights_v["W_v_f"]
        W_v_hh = weights_v["W_v_hh"]
        U_f = weights_v["U_f"]
        U_hh = weights_v["U_hh"]
        b_f = weights_v["b_f"]
        b_hh = weights_v["b_hh"]
        f_t = jax.nn.sigmoid(W_p_f*plan_t + W_m_f*move_t + jnp.matmul(W_h1_f,h1vec) + jnp.matmul(W_v_f,v_0) + jnp.matmul(U_f,hv_t_1) + b_f)
        hhat_t = jax.nn.tanh(W_p_hh*plan_t + W_m_hh*move_t + jnp.matmul(W_h1_hh,h1vec) + jnp.matmul(W_v_hh,v_0) + jnp.matmul(U_hh,jnp.multiply(f_t,hv_t_1)) + b_hh)
        hv_t = jnp.multiply((1-f_t),hv_t_1) + jnp.multiply(f_t,hhat_t)
        return (weights_v,h1vec,v_0,act_t,hv_t),None
    W_full = weights_v["W_full"]
    W_ap = weights_v["W_ap"]
    carry_0 = (weights_v,h1vec,v_0,act_t,hv_t_1)
    (*_,hv_t),_ = jax.lax.scan(loop_body,carry_0,NONE_PLAN)
    v_pred_full = jnp.matmul(W_full,hv_t)
    v_pred_ap = jnp.take(v_pred_full, params["INDICES"])
    return v_pred_ap,v_pred_full,hv_t

def circular_mean(v_t, NEURON_GRID_AP):
    x_data = NEURON_GRID_AP[0,:]
    y_data = NEURON_GRID_AP[1,:]
    theta = jnp.arctan2(y_data, x_data)
    mean_sin = jnp.sum(jnp.sin(theta) * v_t) / jnp.sum(v_t)
    mean_cos = jnp.sum(jnp.cos(theta) * v_t) / jnp.sum(v_t)
    mean_x = jnp.arctan2(mean_sin, mean_cos) # cartesian
    mean_y = jnp.sqrt(mean_sin**2 + mean_cos**2)
    return jnp.array([mean_x,mean_y])

@jit
def r_predict(v_t, dot_t, MODULES, APERTURE, NEURON_GRID_AP, params, e):
    mean = circular_mean(v_t,NEURON_GRID_AP)
    r_pred_t = loss_obj(mean,dot_t,params,e)
    return r_pred_t

def sample_policy(policy, SC, ind):
    vectors, actions = policy
    ID_ARR, VEC_ARR, H1VEC_ARR = SC
    keys = rnd.split(rnd.PRNGKey(ind), num=2)
    vec_ind = rnd.choice(keys[0], a=jnp.arange(len(vectors)), p=vectors)
    h1vec = H1VEC_ARR[:,vec_ind]
    vec = VEC_ARR[:,vec_ind]
    act_ind = rnd.choice(keys[1], a=jnp.arange(len(actions)), p=actions)

    act = jnp.eye(len(actions))[act_ind]
    logit_t = jnp.log(vectors[vec_ind] + 1e-8) + jnp.log(actions[act_ind] + 1e-8)
    return h1vec, vec, vec_ind, act_ind, act, logit_t

def kl_loss(policy, params):
    len_ = len(policy[0])
    val_ = (1 - params["PRIOR_STAT"]) / (len_ - 1)
    vec_prior = jnp.full(len_, 1/len_)
    vec_prior = vec_prior / jnp.sum(vec_prior)
    vec_kl = optax.kl_divergence(jnp.log(policy[0]), vec_prior)
    act_prior = jnp.array([params["PRIOR_PLAN"], 1 - params["PRIOR_PLAN"]])
    act_kl = optax.kl_divergence(jnp.log(policy[1]), act_prior)
    return vec_kl, act_kl

def plan(h1vec, vec, v_t_1, hv_t_1, r_t_1, pos_plan_t_1, pos_t_1, dot_t_1, dot_vec, act_t, val, weights_v, params, epoch):
    dot_t = dot_t_1 + dot_vec
    pos_plan_t = pos_plan_t_1 + vec
    pos_t = pos_t_1
    v_pred_t, v_full_t, hv_t = v_predict(h1vec, v_t_1, hv_t_1, act_t, weights_v, params["NONE_PLAN"])
    rp_t = r_predict(v_full_t, dot_t, params, epoch)  
    return rp_t, v_pred_t, hv_t, pos_plan_t, pos_t, dot_t

def move(h1vec, vec, v_t_1, hv_t_1, r_t_1, pos_plan_t_1, pos_t_1, dot_t_1, dot_vec, act_t, val, weights_v, params, epoch):
    dot_t = dot_t_1 + dot_vec
    pos_plan_t = pos_t_1 + vec

    pos_t = pos_t_1 + vec
    v_t_, _, hv_t = v_predict(h1vec, v_t_1, hv_t_1, act_t, weights_v, params["NONE_PLAN"])
    v_t = neuron_act(val, params["THETA_AP"], params["SIGMA_A"], params["SIGMA_N"], dot_t, pos_t)
    r_t = loss_obj(dot_t, pos_t, params, epoch)
    return r_t, v_t, hv_t, pos_plan_t, pos_t, dot_t

def sample_action(move_counter, plan_ratio):
    return jnp.array([0, 1], dtype=jnp.float32) if move_counter % plan_ratio == 0 else jnp.array([1, 0], dtype=jnp.float32)

@partial(jax.jit, static_argnums=(4,))
def scan_body(carry_tm1, x):
    (t, args_tm1, theta) = carry_tm1
    (hs_tm1, hv_tm1, val_tm1, pos_plan_tm1, pos_tm1, dot_tm1, dot_vec, ind, act_tm1, v_tm1, r_tm1, rp_tm1, move_counter, epoch) = args_tm1
    (SC, weights_v, weights_s, params) = theta
    policy_t, val_t, hs_t = get_policy(args_tm1, weights_s, params)  
    vec_kl, act_kl = kl_loss(policy_t, params)
    h1vec_t, vec_t, vec_ind, act_ind, act_t, logit_t = sample_policy(policy_t, SC, ind[t])

    args_t = (hs_t, hv_tm1, val_t, pos_plan_tm1, pos_tm1, dot_tm1, dot_vec, ind, act_t, v_tm1, r_tm1, rp_tm1, move_counter, epoch)
    carry_args = (t, args_t, logit_t, vec_kl, act_kl, theta, h1vec_t, vec_t, act_t, policy_t)

    if move_counter % params["PLAN_RATIO"] == 0:
        rp_t, v_t, hv_t, pos_plan_t, pos_t, dot_t = plan(h1vec_t, vec_t, v_tm1, hv_tm1, r_tm1, pos_plan_tm1, pos_tm1, dot_tm1, dot_vec, act_t, ind[t], weights_v, params, epoch)
        r_t = 0.0
        move_counter = 0
    else:
        r_t, v_t, hv_t, pos_plan_t, pos_t, dot_t = move(h1vec_t, vec_t, v_tm1, hv_tm1, r_tm1, pos_plan_tm1, pos_tm1, dot_tm1, dot_vec, act_t, ind[t], weights_v, params, epoch)
        rp_t = 0.0
        move_counter += 1

    carry_t = (t+1, (hs_t, hv_t, val_t, pos_plan_t, pos_t, dot_t, dot_vec, ind, sample_action(move_counter, params["PLAN_RATIO"]), v_t, r_t, rp_t, move_counter, epoch), theta)
    arrs_t = (rp_t, r_t, pos_plan_t, pos_t, dot_t, logit_t, val_t, act_t[0], move_counter > 0, vec_kl, act_kl, policy_t, hs_t, hv_t)
    return carry_t, (vec_ind, act_ind, arrs_t)

def pg_obj(SC, hs_0, hp_0, pos_0, dot_0, dot_vec, ind, weights, weights_s, params, epoch):
    v_0 = neuron_act(ind[-1], params["THETA_AP"], params["SIGMA_A"], params["SIGMA_N"], dot_0, pos_0)
    r_0 = loss_obj(dot_0, pos_0, params, epoch)
    val_0 = 0.0
    act_0 = jnp.array([0,1], dtype=jnp.float32)
    rp_0 = 0
    pos_plan_0 = pos_0
    move_counter = 0
    args_0 = (hs_0, hp_0, val_0, pos_plan_0, pos_0, dot_0, dot_vec, ind, act_0, v_0, r_0, rp_0, move_counter, epoch)
    theta = (SC, weights["v"], weights_s, params)
    (_, args_final, _), arrs_stack = jax.lax.scan(scan_body, (0, args_0, theta), None, params["TEST_LENGTH"])
    vec_ind_arr, act_ind_arr, (rp_arr, r_arr, pos_plan_arr, pos_arr, dot_arr, lp_arr, val_arr, sample_arr, mask_arr, vec_kl_arr, act_kl_arr, policy_arr, hs_arr, hv_arr) = arrs_stack

    pos_plan_arr = jnp.concatenate([jnp.array([pos_0]), pos_plan_arr])
    pos_arr = jnp.concatenate([jnp.array([pos_0]), pos_arr])
    dot_arr = jnp.concatenate([jnp.array([dot_0]), dot_arr])

    r_to_go = jnp.cumsum(r_arr.T[:,::-1], axis=1)[:,::-1]
    adv_arr = r_to_go - val_arr.T
    adv_norm = (adv_arr - jnp.mean(adv_arr, axis=None)) / jnp.std(adv_arr, axis=None)  
    adv_norm_masked = jnp.multiply(adv_norm, mask_arr.T)

    actor_loss_arr = -(jnp.multiply(lp_arr.T, adv_norm_masked))
    actor_loss = jnp.mean(actor_loss_arr)
    sem_actor = jnp.std(actor_loss_arr)/params["VMAPS"]**0.5

    vec_kl_masked = jnp.multiply(vec_kl_arr, mask_arr)
    vec_kl_loss = jnp.mean(vec_kl_masked, axis=None)
    sem_vec_kl = jnp.std(vec_kl_masked, axis=None)/params["VMAPS"]**0.5

    act_kl_masked = jnp.multiply(act_kl_arr, mask_arr)
    act_kl_loss = jnp.mean(act_kl_masked, axis=None)
    sem_act_kl = jnp.std(act_kl_masked, axis=None)/params["VMAPS"]**0.5

    actor_losses = actor_loss + params["LAMBDA_VEC_KL"]*vec_kl_loss + params["LAMBDA_ACT_KL"]*act_kl_loss
    std_actor_loss = (params["VMAPS"]*sem_actor**2+(params["VMAPS"]*params["LAMBDA_VEC_KL"]**2)*(sem_vec_kl**2))**0.5 
    sem_loss = std_actor_loss/params["VMAPS"]**0.5

    critic_loss = jnp.mean(jnp.square(adv_arr), axis=None) 
    sem_critic = jnp.std(jnp.square(adv_arr), axis=None)/params["VMAPS"]**0.5

    tot_loss = actor_losses + critic_loss

    r_tot = jnp.mean(r_to_go[:,0])
    sem_r = jnp.std(r_to_go[:,0])/params["VMAPS"]**0.5
    plan_rate = jnp.mean(jnp.sum(jnp.multiply(sample_arr,mask_arr), axis=1), axis=None)
    sem_plan_rate = jnp.std(jnp.sum(jnp.multiply(sample_arr,mask_arr), axis=1), axis=None)/params["VMAPS"]**0.5

    losses = (tot_loss, actor_loss, critic_loss, vec_kl_loss, act_kl_loss, r_tot, plan_rate)  
    stds = (sem_loss, sem_actor, sem_critic, sem_act_kl, sem_vec_kl, sem_r, sem_plan_rate)
    other = (r_arr.T, rp_arr.T, sample_arr.T, mask_arr.T, pos_plan_arr.transpose([1,0,2]), pos_arr.transpose([1,0,2]), dot_arr.transpose([1,0,2]), policy_arr, hs_arr, hv_arr, vec_ind_arr.T, act_ind_arr.T)
    return (actor_losses, critic_loss), (losses, stds, other)

def get_entropy(policy, params):
    mean_entropy = jnp.mean(jnp.sum(-jnp.multiply(policy, jnp.log(policy)), axis=2), axis=None)
    sem_entropy = jnp.std(jnp.sum(-jnp.multiply(policy, jnp.log(policy)), axis=2), axis=None) / params["VMAPS"]**0.5
    return mean_entropy, sem_entropy

def full_loop(SC, weights, params, actor_opt_state, critic_opt_state):
    loss_arr = jnp.zeros((params["TOT_EPOCHS"],))
    sem_loss_arr = jnp.zeros((params["TOT_EPOCHS"],))
    actor_loss_arr = jnp.zeros((params["TOT_EPOCHS"],))
    sem_actor_arr = jnp.zeros((params["TOT_EPOCHS"],))
    critic_loss_arr = jnp.zeros((params["TOT_EPOCHS"],))
    sem_critic_arr = jnp.zeros((params["TOT_EPOCHS"],))
    vec_kl_arr = jnp.zeros((params["TOT_EPOCHS"],))
    sem_vec_kl_arr = jnp.zeros((params["TOT_EPOCHS"],))
    act_kl_arr = jnp.zeros((params["TOT_EPOCHS"],))
    sem_act_kl_arr = jnp.zeros((params["TOT_EPOCHS"],))
    r_tot_arr = jnp.zeros((params["TOT_EPOCHS"],))
    sem_r_arr = jnp.zeros((params["TOT_EPOCHS"],))
    plan_rate_arr = jnp.zeros((params["TOT_EPOCHS"],))
    sem_plan_rate_arr = jnp.zeros((params["TOT_EPOCHS"],))
    policy_entropy_arr = jnp.zeros((params["TOT_EPOCHS"],))
    sem_policy_entropy_arr = jnp.zeros((params["TOT_EPOCHS"],))

    weights_s = weights["s"]
    max_r_tot = 0.0
    max_rx_plan_rate = 0.0
    rx_plan_rate = 0.0

    actor_optimizer = optax.chain(
    optax.clip_by_global_norm(params["ACTOR_GC"]),
    optax.adam(learning_rate=params["ACTOR_LR"], eps=1e-7)
    )
    critic_optimizer = optax.chain(
    optax.clip_by_global_norm(params["CRITIC_GC"]),
    optax.adam(learning_rate=params["CRITIC_LR"], eps=1e-7)
    )
    plan_info = {}

    for epoch in range(params["TOT_EPOCHS"]):
        new_params(params, epoch)
        hs_0 = params["HS_0"]
        hp_0 = params["HP_0"]
        pos_0 = params["POS_0"]
        dot_0 = params["DOT_0"]
        dot_vec = params["DOT_VEC"]
        ind = params["IND"]

        for _ in range(params["CRITIC_UPDATES"]):
            isolated_pg_obj = lambda weights_s: pg_obj(SC, hs_0, hp_0, pos_0, dot_0, dot_vec, ind, weights, weights_s, params, epoch)
            (actor_losses, critic_loss), pg_obj_vjp, (losses, stds, other) = jax.vjp(isolated_pg_obj, weights_s, has_aux=True)
            critic_grad, = pg_obj_vjp((0.0, 1.0))

            critic_global_norm = optax.global_norm(critic_grad)
            print(f"Critic Gradient Norm: {critic_global_norm}")

            critic_update, critic_opt_state = critic_optimizer.update(critic_grad, critic_opt_state, weights_s)
            weights_s = optax.apply_updates(weights_s, critic_update)

        actor_grad, = pg_obj_vjp((1.0, 0.0))
        (tot_loss, actor_loss, critic_loss, vec_kl_loss, act_kl_loss, r_tot, plan_rate) = losses
        print('actor_loss=', actor_loss, 'vec_kl_loss=', vec_kl_loss, 'act_kl_loss=', act_kl_loss)

        actor_global_norm = optax.global_norm(actor_grad)
        print(f"Actor Gradient Norm: {actor_global_norm}")

        actor_update, actor_opt_state = actor_optimizer.update(actor_grad, actor_opt_state)
        weights_s = optax.apply_updates(weights_s, actor_update)

        (tot_loss, actor_loss, critic_loss, vec_kl_loss, act_kl_loss, r_tot, plan_rate) = losses
        (sem_loss, sem_actor, sem_critic, sem_act_kl, sem_vec_kl, sem_r, sem_plan_rate) = stds
        (r_arr, rp_arr, sample_arr, mask_arr, pos_plan_arr, pos_arr, dot_arr, policy_arr, hs_arr, hv_arr, vec_ind_arr, act_ind_arr) = other

        if r_tot > max_r_tot:
            max_r_tot = r_tot
            selected_other = (other, copy.deepcopy(params))

        if (r_tot * plan_rate) > max_rx_plan_rate:
            max_rx_plan_rate = r_tot * plan_rate
            rx_plan_rate = plan_rate
            plan_info = {
                'other_': (other, copy.deepcopy(params)),
                'actor_opt_state': actor_opt_state,
                'critic_opt_state': critic_opt_state,
                'weights_s': weights_s,
                'rx_plan_rate': plan_rate,
                'max_r_tot': max_r_tot,
            }

        mean_policy_entropy, sem_policy_entropy = get_entropy(policy_arr[0])
        policy_entropy_arr = policy_entropy_arr.at[epoch].set(mean_policy_entropy)
        sem_policy_entropy_arr = sem_policy_entropy_arr.at[epoch].set(sem_policy_entropy)
        loss_arr = loss_arr.at[epoch].set(tot_loss)
        sem_loss_arr = sem_loss_arr.at[epoch].set(sem_loss)
        actor_loss_arr = actor_loss_arr.at[epoch].set(actor_loss)
        sem_actor_arr = sem_actor_arr.at[epoch].set(sem_actor)
        critic_loss_arr = critic_loss_arr.at[epoch].set(critic_loss)
        sem_critic_arr = sem_critic_arr.at[epoch].set(sem_critic)
        vec_kl_arr = vec_kl_arr.at[epoch].set(vec_kl_loss)
        sem_vec_kl_arr = sem_vec_kl_arr.at[epoch].set(sem_vec_kl)
        act_kl_arr = act_kl_arr.at[epoch].set(act_kl_loss)
        sem_act_kl_arr = sem_act_kl_arr.at[epoch].set(sem_act_kl)
        r_tot_arr = r_tot_arr.at[epoch].set(r_tot)
        sem_r_arr = sem_r_arr.at[epoch].set(sem_r)
        plan_rate_arr = plan_rate_arr.at[epoch].set(plan_rate)
        sem_plan_rate_arr = sem_plan_rate_arr.at[epoch].set(sem_plan_rate)

        print("epoch=", epoch, "r=", r_tot, "r_sem=", sem_r)
        print("actor_loss=", actor_loss, "sem_actor=", sem_actor, "critic_loss=", critic_loss, "sem_critic=", sem_critic)
        print("vec_kl=", vec_kl_loss, "sem_vec=", sem_vec_kl, "act_kl=", act_kl_loss, "sem_act=", sem_act_kl, "plan=", plan_rate, 'entropy=', mean_policy_entropy, 'sem_entropy=', sem_policy_entropy, '\n')
        print('rx_plan_rate=', rx_plan_rate, 'max_r_tot=', max_r_tot)

        if epoch > 0 and (epoch % 50) == 0:
            print("*clearing cache*")
            jax.clear_caches()

    loss_arrs = (loss_arr, actor_loss_arr, critic_loss_arr, vec_kl_arr, act_kl_arr, r_tot_arr, plan_rate_arr, policy_entropy_arr)
    sem_arrs = (sem_loss_arr, sem_actor_arr, sem_critic_arr, sem_act_kl_arr, sem_vec_kl_arr, sem_r_arr, sem_plan_rate_arr, sem_policy_entropy_arr)
    return loss_arrs, sem_arrs, selected_other, actor_opt_state, critic_opt_state, weights_s, plan_info