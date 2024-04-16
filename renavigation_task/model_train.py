from functools import partial
import jax
import jax.numpy as jnp
import jax.random as rnd
import optax
from pathlib import Path
from params import params

def load_(str_):
    path_ = str(Path(__file__).resolve().parents[1])
    with open(path_+str_,'rb') as file_:
        param_ = jnp.load(file_,allow_pickle=True)
    return param_

def gen_neurons(neurons, aperture):
    return jnp.linspace(-aperture, aperture, neurons, dtype=jnp.float32)

def new_scan_params(epoch):
    scan_params = {
        "H0": rnd.normal(rnd.PRNGKey(epoch), (params["BATCH_SIZE"], params["GRU_SIZE"]), dtype=jnp.float32),
        "DOTS": rnd.uniform(rnd.PRNGKey(epoch+1), (params["BATCH_SIZE"], params["N_DOTS"], 2), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32),
        "SELECT": jax.nn.one_hot(rnd.randint(rnd.PRNGKey(epoch+2), (params["BATCH_SIZE"],), 0, params["N_DOTS"]), params["N_DOTS"]),
        "EPSILON_N": rnd.normal(rnd.PRNGKey(epoch+3), (params["BATCH_SIZE"], params["TOT_ITERS"], 2), dtype=jnp.float32)
    }
    return scan_params
    
def neuron_act(pos, dots, params):
    D = params["N_DOTS"]
    N = params["NEURONS"]**2
    THETA_J = gen_neurons(params["NEURONS"], params["APERTURE"])  
    THETA_I = gen_neurons(params["NEURONS"], params["APERTURE"])

    G_0 = jnp.vstack((jnp.tile(THETA_J,N), jnp.tile(THETA_I,N)))
    G = jnp.tile(G_0.reshape(2,N**2,1),(1,1,D))
    C = (params["COLORS"]/255).transpose((1,0))
    E = G.transpose((1,0,2)) - (dots-pos).T
    act = jnp.exp((jnp.cos(E[:,0,:]) + jnp.cos(E[:,1,:]) - 2)/params["SIGMA_A"]**2).reshape((D,N**2)) 
    act_C = jnp.matmul(C, act)
    return act_C

def GRU_step(activations, h, weights):
    z_t = jax.nn.sigmoid(jnp.matmul(weights["Wr_z"],activations[0]) + jnp.matmul(weights["Wg_z"],activations[1]) 
                         + jnp.matmul(weights["Wb_z"],activations[2]) + jnp.matmul(weights["U_z"],h) + weights["b_z"])
    f_t = jax.nn.sigmoid(jnp.matmul(weights["Wr_r"],activations[0]) + jnp.matmul(weights["Wg_r"],activations[1]) 
                         + jnp.matmul(weights["Wb_r"],activations[2]) + jnp.matmul(weights["U_r"],h) + weights["b_r"]) 
    hhat_t = jnp.tanh(jnp.matmul(weights["Wr_h"],activations[0]) + jnp.matmul(weights["Wg_h"],activations[1]) 
                      + jnp.matmul(weights["Wb_h"],activations[2]) + jnp.matmul(weights["U_h"],(jnp.multiply(f_t,h))) + weights["b_h"])
    h_t = jnp.multiply(z_t,h) + jnp.multiply((1-z_t),hhat_t)
    return h_t

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

def get_loss_predict(rdot_hat, pos, dots, params, epoch):
    SIGMA_R0 = params["SIGMA_R0"]
    SIGMA_RINF = params["SIGMA_RINF"]
    TAU = params["TAU"]
    sigma_e = SIGMA_RINF * (1 - jnp.exp(-epoch/TAU)) + SIGMA_R0 * jnp.exp(-epoch/TAU)

    theta_d_0 = jnp.arctan2(rdot_hat[0], rdot_hat[1])
    theta_d_1 = jnp.arctan2(pos[0], pos[1])  
    loss_predict = jnp.exp((jnp.cos(theta_d_0 - theta_d_1) - 1)/sigma_e**2)
    return loss_predict

def get_loss_cross_entropy(sel_hat, select, params): 
    loss_CE = optax.softmax_cross_entropy(logits=sel_hat, labels=select)
    return loss_CE

@partial(jax.jit, static_argnums=(5,))  
def single_step(args, epsilon):
    pos_t, h_t, dots, select, weights, params = args
    
    activations_t1 = neuron_act(pos_t, dots, params) 

    h_t1 = GRU_step(activations_t1, h_t, weights)

    v_t1 = params["ALPHA"] * (weights["C"] @ h_t1 + params["SIGMA_NOISE"] * epsilon) 
    pos_t1 = pos_t + v_t1
    
    rdot_hat_t1 = weights["D"] @ h_t1
    sel_hat_t1 = weights["S"] @ h_t1

    reward_t1 = get_reward(pos_t1, dots, select, params)
    loss_predict_t1 = get_loss_predict(rdot_hat_t1, pos_t1, dots, params)
    loss_cross_entropy_t1 = get_loss_cross_entropy(sel_hat_t1, select, params)

    args = (pos_t1, h_t1, dots, select, weights, params)
    losses = (reward_t1, loss_predict_t1, loss_cross_entropy_t1)
    debug = (activations_t1, pos_t1, rdot_hat_t1, sel_hat_t1)

    return args, (losses, debug)

@partial(jax.jit, static_argnums=(5,))
def get_trajectory(h0, dots, select, epsilon_n, weights, params):
    pos_t = jnp.array([0,0], dtype=jnp.float32)
    args_0 = (pos_t, h0, dots, select, weights, params)

    args_final, (losses, debug) = jax.lax.scan(single_step, args_0, epsilon_n)
    rewards, loss_predict, loss_CE = losses
    loss_total = - jnp.sum(rewards) - params["LAMBDA_PREDICT"] * jnp.sum(loss_predict) + params["LAMBDA_CE"] * jnp.sum(loss_CE)

    return loss_total, (losses, debug)

get_loss_and_grad = jax.value_and_grad(get_trajectory, argnums=4, allow_int=True, has_aux=True)
get_losses_and_grads_vmap = jax.vmap(get_loss_and_grad, in_axes=(0, 0, 0, 0, None, None), out_axes=(0, 0))

def full_loop(params, weights):
    loss_keys = ['loss_tot', 'loss_predict', 'loss_CE']
    sem_keys = ['sem_tot', 'sem_predict', 'sem_CE']
    loss_arrays = {key: jnp.zeros((params["TOT_EPOCHS"],)) for key in loss_keys}
    sem_arrays = {key: jnp.zeros((params["TOT_EPOCHS"],)) for key in sem_keys}

    optimiser = optax.adamw(learning_rate=params["LEARNING_RATE"], weight_decay=params["WEIGHT_DECAY"])
    opt_state = optimiser.init(weights)

    for epoch in range(params["TOT_EPOCHS"]):

        scan_params = new_scan_params(epoch)

        (loss_total, aux), grads_ = get_losses_and_grads_vmap(scan_params["H0"], scan_params["DOTS"], scan_params["SELECT"], scan_params["EPSILON_N"], weights, params)
        grad_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0), grads_)
        opt_update, opt_state = optimiser.update(grad_mean, opt_state, weights)
        weights = optax.apply_updates(weights, opt_update)

        loss_total_mean = jnp.mean(loss_total)
        sem_total = jnp.std(loss_total) / jnp.sqrt(loss_total.shape[0])

        losses, _ = aux
        losses_mean = (jnp.mean(jnp.sum(x, axis=1)) for x in losses)
        sems = (jnp.std(jnp.sum(x, axis=1)) / jnp.sqrt(x.shape[0]) for x in losses)

        loss_arrays = {key: loss_arrays[key].at[epoch].set(losses_mean[key]) for key in loss_arrays.keys()}
        sem_arrays = {key: sem_arrays[key].at[epoch].set(sems[key]) for key in sem_arrays.keys()}

        losses = jax.tree_map(lambda x: x.block_until_ready(), losses)
        sems = jax.tree_map(lambda x: x.block_until_ready(), sems)

        (mean_reward, loss_predict, loss_CE) = losses_mean
        (sem_rewards, sem_predict, sem_CE) = sems

        print(f"epoch={epoch} reward={mean_reward:.3f} loss_total={loss_total_mean:.3f} loss_predict={loss_predict:.3f} loss_CE={loss_CE:.3f}")
        print(f"sem_rewards={sem_rewards:.3f} sem_total={sem_total:.3f} sem_predict={sem_predict:.3f} sem_CE={sem_CE:.3f}")

    return loss_arrays, sem_arrays, opt_state, weights