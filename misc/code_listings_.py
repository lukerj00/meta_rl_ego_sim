import jax
import jax.numpy as jnp
from jax import jit
import jax.random as rnd
# from jax.experimental.host_callback import id_print
# from jax.experimental.host_callback import call
import optax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# matplotlib.use('Agg')
from drawnow import drawnow
import numpy as np
import csv
import pickle
from pathlib import Path
from datetime import datetime
import re
import os
import sys
# from os.path import dirname, abspath
import scipy
# jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')

### outer training loop and other primary functions ###

def full_loop(params, init_weights_p ,weights_v):
    """ full PPO training loop: outer loop over epochs, inner loop over mini batch updates.

    Args:   params: training hyperparameters and environment parameters, pytree of scalars.
            init_weights_p: initial policy network weights, pytree of arrays.
            weights_v: (static) pre-trained visual model network weights, pytree of arrays.

    Returns:    loss_arrays: arrays to store losses, pytree of arrays.
                sem_arrays: arrays to store standard errors of the mean, pytree of arrays.
                actor_opt_state: actor optimiser, optimiser state.
                critic_opt_state: critic optimiser, optimiser state.
                new_weights_p: trained policy network weights, pytree of arrays.
    """
    loss_keys = ['loss_tot', 'loss_actor', 'loss_critic', 'loss_kl_vectors', 'loss_kl_decisions', 'mean_reward', 'mean_plan_rate']
    sem_keys = ['sem_tot', 'sem_actor', 'sem_critic', 'sem_kl_vectors', 'sem_kl_decisions', 'sem_rewards', 'sem_plan_rate']
    TOT_EPOCHS = params["EPOCHS"] * params["VMAPS"] // params["BATCH_SIZE"]
    loss_arrays = {key: jnp.zeros((TOT_EPOCHS,)) for key in loss_keys} # initialise loss arrays
    sem_arrays = {key: jnp.zeros((TOT_EPOCHS,)) for key in sem_keys} # initialise standard error arrays

    old_weights_p = copy.deepcopy(init_weights_p) # initialise old policy
    new_weights_p = copy.deepcopy(init_weights_p) # initialise new policy

    actor_optimiser = optax.chain(
        optax.clip_by_global_norm(params["GRAD_CLIP"]),
        optax.adam(learning_rate=params["ACTOR_LR"])
        )
    actor_opt_state = actor_optimiser.init(new_weights_p) # initialise actor optimiser

    critic_optimiser = optax.chain(
        optax.clip_by_global_norm(params["GRAD_CLIP"]),
        optax.adamw(learning_rate=params["CRITIC_LR"], weight_decay=params["CRITIC_WD"])
        )
    critic_opt_state = critic_optimiser.init(new_weights_p) # initialise critic optimiser

    for epoch in range(params["EPOCHS"]): # outer optimisation loop

        scan_params = new_scan_params(epoch) # generate new batch of random environments
        
        old_trajectories, _ = get_old_trajectories_vmap(scan_params, params, old_weights_p, weights_v) # get total batch of old trajectories

        for b, batch in enumerate(range(0, params["VMAPS"], params["BATCH_SIZE"])): # inner mini batch update loop

            old_trajectories_batch = jax.tree_map(lambda x: x[batch: batch + params["BATCH_SIZE"]], old_trajectories) # select mini batch of old trajectories
            scan_params_batch = jax.tree_map(lambda x: x[batch: batch + params["BATCH_SIZE"]], scan_params)
            
            for _ in range(params["CRITIC_UPDATES"]): # multiple critic updates for each mini-batch of old/new trajectories
                
                isolated_batch_loss = lambda new_weights_p: get_batch_losses(old_trajectories_batch, scan_params_batch, params, new_weights_p, weights_v) # get_batch_losses as a univariate function of new_weights_p
                (loss_actor, loss_critic), get_actor_critic_grads, (losses, sems) = jax.vjp(isolated_batch_loss, new_weights_p, has_aux=True) # obtain jacobian-returning function

                critic_grad, = get_actor_critic_grads((0.0, 1.0)) # critic grads using vjp
                critic_update, critic_opt_state = critic_optimiser.update(critic_grad, critic_opt_state, new_weights_p)
                new_weights_p = optax.apply_updates(new_weights_p, critic_update) # critic update

            actor_grad, = get_actor_critic_grads((1.0, 0.0)) # actor grads using vjp
            actor_update, actor_opt_state = actor_optimiser.update(actor_grad, actor_opt_state)
            new_weights_p = optax.apply_updates(new_weights_p, actor_update) # actor update

            ind = epoch * (params["VMAPS"] // params["BATCH_SIZE"]) + b
            loss_arrays = {key: loss_arrays[key].at[ind].set(losses[key]) for key in loss_arrays.keys()} # populate loss arrays
            sem_arrays = {key: sem_arrays[key].at[ind].set(sems[key]) for key in sem_arrays.keys()} # populate sem arrays

            losses = jax.tree_map(lambda x: x.block_until_ready(), losses) # ensure computations complete before printing; saving memory profile
            sems = jax.tree_map(lambda x: x.block_until_ready(), sems)
            
            (loss_tot, loss_actor, loss_critic, loss_kl_vectors, loss_kl_decisions, mean_reward, mean_plan_rate) = losses
            (sem_tot, sem_actor, sem_critic, sem_kl_vectors, sem_kl_decisions, sem_reward, sem_plan_rate) = sems
            
            print(f"epoch={epoch}.{b} reward={mean_reward:.3f} sem_reward={sem_reward:.3f} loss_tot={loss_tot:.3f} sem_tot={sem_tot:.3f}")
            print(f"loss_actor={loss_actor:.3f} sem_actor={sem_actor:.3f} loss_critic={loss_critic:.3f} sem_critic={sem_critic:.3f}")
            print(f"loss_kl_vec={loss_kl_vectors:.3f} sem_kl_vec={sem_kl_vectors:.3f} loss_kl_dec={loss_kl_decisions:.3f} sem_kl_dec={sem_kl_decisions:.3f} plan={mean_plan_rate:.3f} sem_plan={sem_plan_rate:.3f}") # print information for debugging (optional)
            
            jax.profiler.save_device_memory_profile(f"memory_profile_epoch{epoch}.prof") # save memory profile for debugging (optional)

            if ind > 0 and (ind % 100) == 0: 
                print("*clearing cache*")
                jax.clear_caches() # clear compile cache every 100 iterations

        old_weights_p = copy.deepcopy(new_weights_p) # set old weights to new weights

    return loss_arrays, sem_arrays, actor_opt_state, critic_opt_state, new_weights_p

def get_batch_losses(old_trajectories_batch, scan_params_batch, params, new_weights_p, weights_v):
    """ obtain actor and critic losses by comparing new and old trajectories.

    Args:   old_trajectories_batch: batch of old trajectories, pytree of arrays [B,T].
            scan_params_batch: parameters to batch over, pytree of arrays [B,T].
            params: training hyperparameters and environment parameters, pytree of scalars.
            new_weights_p: new policy network weights, pytree of arrays.
            weights_v: (static) pre-trained visual model network weights, pytree of arrays.

    Returns:    (loss_actor, loss_critic): actor and critic losses, tuple of scalars.
                (losses, sems): auxiliary debugging information, tuple of pytrees.
    """
    new_trajectories, _ = get_new_trajectories_vmap(old_trajectories_batch, scan_params_batch, params, new_weights_p, weights_v) # get new trajectories
    log_probs_new, rewards, values, kl_vectors, kl_decisions = new_trajectories["log_probs"], new_trajectories["rewards"], new_trajectories["values"], new_trajectories["kl_vectors"], new_trajectories["kl_decisions"]
    
    log_probs_old, mask_array, decisions = old_trajectories_batch["log_probs"], old_trajectories_batch["mask_array"], old_trajectories_batch["decisions"]

    advantages = compute_gae(rewards, values, params["GAMMA"], params["LAMBDA_"]) # compute advantages
    loss_ppo, sem_ppo = compute_ppo_loss(advantages, log_probs_old, log_probs_new, mask_array, params["EPSILON"]) # compute ppo loss

    kl_vectors_masked = kl_vectors * mask_array # mask out refactory periods from KL losses
    loss_kl_vectors = jnp.sum(kl_vectors_masked) / jnp.sum(mask_array)
    sem_kl_vectors = jnp.std(kl_vectors_masked[kl_vectors_masked != 0]) / jnp.sum(mask_array) ** 0.5
    kl_decisions_masked = kl_decisions * mask_array
    loss_kl_decisions = jnp.sum(kl_decisions_masked) / jnp.sum(mask_array)
    sem_kl_decisions = jnp.std(kl_decisions_masked[kl_decisions_masked != 0]) / jnp.sum(mask_array) ** 0.5

    loss_actor = - loss_ppo - params["LAMBDA_KL_VECTORS"] * loss_kl_vectors - params["LAMBDA_KL_DECISIONS"] * loss_kl_decisions # losses negated as optimiser minimises
    sem_actor = jnp.sqrt((params["LAMBDA_KL_VECTORS"] * sem_kl_vectors) ** 2 + 
                          (params["LAMBDA_KL_DECISIONS"] * sem_kl_decisions) ** 2 + 
                          sem_ppo ** 2)

    critic_array_masked = (advantages * mask_array) ** 2 # mask out refactory periods from critic losses
    loss_critic = jnp.sum(critic_array_masked) / jnp.sum(mask_array)
    sem_critic = jnp.std(critic_array_masked[critic_array_masked != 0]) / jnp.sum(mask_array) ** 0.5

    loss_tot = loss_actor + params["LAMBDA_CRITIC"] * loss_critic # total loss
    sem_tot = (sem_actor ** 2 + (params["LAMBDA_CRITIC"] * sem_critic) ** 2) ** 0.5

    tot_reward = jnp.mean(jnp.sum(rewards, axis=1)) # mean total reward
    sem_reward = jnp.std(jnp.sum(rewards, axis=1)) / rewards.shape[0] ** 0.5
    plan_rate_masked = decisions * mask_array # mean plan rate via multiplication of decision and mask binary arrays
    mean_plan_rate = jnp.sum(plan_rate_masked) / jnp.sum(mask_array)
    sem_plan_rate = jnp.std(plan_rate_masked[plan_rate_masked != 0]) / jnp.sum(mask_array) ** 0.5

    losses = (loss_tot, loss_actor, loss_critic, loss_kl_vectors, loss_kl_decisions, tot_reward, mean_plan_rate)
    sems = (sem_tot, sem_actor, sem_critic, sem_kl_vectors, sem_kl_decisions, sem_reward, sem_plan_rate)
    
    return (loss_actor, loss_critic), (losses, sems)

@partial(jax.jit, static_argnums=(1,)) # params contains non-array values so is marked static
def get_old_trajectory(scan_params_i, params, old_weights_p, weights_v):
    """ obtain single trajectory under old policy.

    Args:   scan_params_i: i'th scan parameters, pytree of arrays [T,].
            params: training hyperparameters and environment parameters, pytree of scalars.
            old_weights_p: old policy network weights, pytree of arrays.
            weights_v: (static) pre-trained visual model network weights, pytree of arrays.

    Returns:    old_trajectory: single old trajectory, pytree of arrays [T,].
                old_debug: additional debugging information, pytree of arrays [T,].
    """
    scan_args_0 = get_initial_scan_args(scan_params_i) # initialise tuple of scanning arguments (initial observation/reward, etc.)
    _, (old_trajectory, old_debug) = jax.lax.scan(old_trajectory_step, (scan_args_0, params, old_weights_p, weights_v), None, params["TRIAL_LENGTH"])

    return old_trajectory, old_debug

get_old_trajectories_vmap = jax.vmap(get_old_trajectory, in_axes=(0, None, None, None), out_axes=0) # scan_params input is a pytree of arrays [N,T]

@partial(jax.jit, static_argnums=(2,))
def get_new_trajectory(old_trajectory_j, scan_params_j, params, new_weights_p, weights_v):
    """ obtain single trajectory under new policy using equivalent old trajectory.

    Args:   old_trajectory_j: j'th old trajectory, pytree of arrays [T,].
            scan_params_j: j'th scan parameters from old trajectory, pytree of arrays [T,].
            params: training hyperparameters and environment parameters, pytree of scalars.
            new_weights_p: new policy network weights, pytree of arrays.
            weights_v: (static) pre-trained visual model network weights, pytree of arrays.

    Returns:    new_trajectory: single new trajectory, pytree of arrays [T,].
                new_debug: additional debugging information, pytree of arrays [T,].
    """
    scan_args_0 = get_initial_scan_args(scan_params_j) # initialise tuple of scanning arguments (initial observation/reward, etc.)
    _, (new_trajectory, new_debug) = jax.lax.scan(new_trajectory_step, (scan_args_0, params, new_weights_p, weights_v), old_trajectory_j, params["TRIAL_LENGTH"])

    return new_trajectory, new_debug

get_new_trajectories_vmap = jax.vmap(get_new_trajectory, in_axes=(0, 0, None, None, None), out_axes=0) # old_trajectories_batch and scan_params_batch inputs are pytrees of arrays [B,T]

### main routine ###

params = {
    # initialise simulation parameters including training hyperparameters (number of epochs, learning rates etc.) and environment parameters (aperture size, activation function scale etc.)
}

init_weights_p = {
    # initialise policy GRU network weights with normal glorot initialisation
}

_,(*_,weights_v) = load_('/path/to/visual_model_weights.pkl') # load pre-trained visual model

faulthandler.enable() # detailed traceback for debugging memory errors
start_time = datetime.now()
loss_arrays, sem_arrays, actor_opt_state, critic_opt_state, final_weights_p = full_loop(params, init_weights_p ,weights_v)
print(f"Sim time: {datetime.now() - start_time} s/epoch= {((datetime.now() - start_time) / params['TOT_EPOCHS']).total_seconds()}")

save_checkpoint((actor_opt_state, critic_opt_state, final_weights_p), 'tracking_task_checkpoint_') # checkpoint training; saving optimiser state is key
save_outputs((loss_arrays, sem_arrays), 'tracking_task_outputs_') # save other outputs to separate file

########################################################

### outer training loop and other primary functions ###

def full_loop(params, weights):
    """ full training loop for (fully differentiable) navigation task objective.

    Args:   params: training parameters (hyperparameters and environment parameters), pytree of scalars.
            weights: initial agent network weights, pytree of arrays.

    Returns:    loss_arrays: arrays to store losses, pytree of arrays.
                sem_arrays: arrays to store standard errors of the mean, pytree of arrays.
                opt_state: network optimiser, optimiser state.
                weights: trained agent network weights, pytree of arrays.
    """
    loss_keys = ['loss_tot', 'loss_predict', 'loss_CE']
    sem_keys = ['sem_tot', 'sem_predict', 'sem_CE']
    loss_arrays = {key: jnp.zeros((params["TOT_EPOCHS"],)) for key in loss_keys} # initialise loss arrays
    sem_arrays = {key: jnp.zeros((params["TOT_EPOCHS"],)) for key in sem_keys} # initialise standard error arrays

    optimiser = optax.adamw(learning_rate=params["LEARNING_RATE"], weight_decay=params["WEIGHT_DECAY"])
    opt_state = optimiser.init(weights) # initialise optimiser

    for epoch in range(params["TOT_EPOCHS"]):

        scan_params = new_scan_params(epoch) # generate new random environments

        (loss_total, aux), grads_ = get_losses_and_grads_vmap(scan_params["H0"], scan_params["DOTS"], scan_params["SELECT"], scan_params["EPSILON_N"], weights, params)
        grad_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0), grads_) # find mean gradient over batch
        opt_update, opt_state = optimiser.update(grad_mean, opt_state, weights)
        weights = optax.apply_updates(weights, opt_update)

        loss_total_mean = jnp.mean(loss_total)
        sem_total = jnp.std(loss_total) / jnp.sqrt(loss_total.shape[0])

        losses, _ = aux
        losses_mean = (jnp.mean(jnp.sum(x, axis=1)) for x in losses)
        sems = (jnp.std(jnp.sum(x, axis=1)) / jnp.sqrt(x.shape[0]) for x in losses)

        loss_arrays = {key: loss_arrays[key].at[epoch].set(losses_mean[key]) for key in loss_arrays.keys()} # populate loss arrays
        sem_arrays = {key: sem_arrays[key].at[epoch].set(sems[key]) for key in sem_arrays.keys()} # populate standard error arrays

        losses = jax.tree_map(lambda x: x.block_until_ready(), losses) # ensure computations complete before printing; saving memory profile
        sems = jax.tree_map(lambda x: x.block_until_ready(), sems)

        (mean_reward, loss_predict, loss_CE) = losses_mean
        (sem_rewards, sem_predict, sem_CE) = sems

        print(f"epoch={epoch} reward={mean_reward:.3f} loss_total={loss_total_mean:.3f} loss_predict={loss_predict:.3f} loss_CE={loss_CE:.3f}") # print information for debugging (optional)
        print(f"sem_rewards={sem_rewards:.3f} sem_total={sem_total:.3f} sem_predict={sem_predict:.3f} sem_CE={sem_CE:.3f}")

    return loss_arrays, sem_arrays, opt_state, weights

@partial(jax.jit, static_argnums=(5,))
def get_trajectory(h0, dots, select, epsilon_n, weights, params):
    """ calculate the total loss for a single trajectory.

    Args:   h0: initial hidden state for the agent network, array [H,].
            dots: locations of the 3 RGB objects, array [3,2].
            select: one-hot vector denoting the rewarded object in that trajectory, array [3,].
            epsilon_n: pre-generated motor noise for each step of the trajectory, array [T,2].
            weights: current agent network weights, pytree of arrays.
            params: training parameters (hyperparameters and environment parameters), pytree of scalars.

    Returns:    loss_total: total loss summed over a single trajectory, scalar.
                (losses, debug): individual losses and additional debugging information, tuple of tuples.
    """
    pos_t = jnp.array([0,0], dtype=jnp.float32) # initial position
    args_0 = (pos_t, h0, dots, select, weights, params)

    args_final, (losses, debug) = jax.lax.scan(single_step, args_0, epsilon_n)
    rewards, loss_predict, loss_CE = losses
    loss_total = - jnp.sum(rewards) - params["LAMBDA_PREDICT"] * jnp.sum(loss_predict) + params["LAMBDA_CE"] * jnp.sum(loss_CE) # assemble total (negative) loss

    return loss_total, (losses, debug)

get_loss_and_grad = jax.value_and_grad(get_trajectory, argnums=4, allow_int=True, has_aux=True)
get_losses_and_grads_vmap = jax.vmap(get_loss_and_grad, in_axes=(0, 0, 0, 0, None, None), out_axes=(0, 0)) # the first 4 arguments in get_trajectory() are vmap'd across their leading axis, length N

def single_step(args, epsilon):
    """ calculate the total loss for a single trajectory.

    Args:   args: current trajectory step carry, tuple of multiple types.
            epsilon: pre-generated motor noise for current step, array [2,].

    Returns:    args: next trajectory step carry, tuple of multiple types.
                (losses, debug): individual losses and additional debugging information, tuple of tuples.
    """
    pos_t, h_t, dots, select, weights, params = args
    
    activations_t1 = neuron_act(pos_t, dots, params) # neuron activations

    h_t1 = GRU_step(activations_t1, h_t, weights) # GRU equations

    v_t1 = params["ALPHA"] * (weights["V"] @ h_t1 + params["SIGMA_NOISE"] * epsilon) # velocity readout and 'motor noise'
    pos_t1 = pos_t + v_t1 # update position; new state
    
    rdot_hat_t1 = weights["D"] @ h_t1 # predict rewarded object location
    sel_hat_t1 = weights["S"] @ h_t1 # predict rewarded object index

    reward_t1 = get_reward(pos_t1, dots, select, params) # reward
    loss_predict_t1 = get_loss_predict(rdot_hat_t1, pos_t1, dots, params) # aux prediction loss
    loss_cross_entropy_t1 = get_loss_cross_entropy(sel_hat_t1, select, params) # aux decision loss

    args = (pos_t1, h_t1, dots, select, weights, params)
    losses = (reward_t1, loss_predict_t1, loss_cross_entropy_t1)
    debug = (activations_t1, pos_t1, rdot_hat_t1, sel_hat_t1)

    return args, (losses, debug)

### main routine ###

params = {
    # initialise simulation parameters including training hyperparameters (number of epochs, learning rates etc.) and environment parameters (noise magnitude, activation function scale etc.)
}

init_weights = {
    # initialise agent GRU network weights with normal glorot initialisation
}

start_time = datetime.now()
loss_arrays, sem_arrays, opt_state, final_weights = full_loop(params, init_weights)
print(f"Sim time: {datetime.now() - start_time} s/epoch= {((datetime.now() - start_time) / params['TOT_EPOCHS']).total_seconds()}")

save_checkpoint((opt_state, final_weights), 'navigation_task_checkpoint_') # checkpoint training
save_outputs((loss_arrays, sem_arrays), 'navigation_task_outputs_') # save other outputs to separate file




# scan_params_N = (params["HP_0"], params["HV_0"], params["TARGET_0"], params["TARGET_VEC"], params["RND_KEY"]) # scan parameters for vmap: initial hidden states for both models; target position and vector; random numbers for PRNGKey generation
