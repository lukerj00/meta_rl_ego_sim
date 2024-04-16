from datetime import datetime
from functools import partial
import jax
import jax.numpy as jnp
import optax
import copy

from params import init_params, init_weights
from model_train import full_loop, load_, gen_sc

def main():
    params = init_params()
    key = jax.random.PRNGKey(0)
    weights = init_weights(key)

    # Load weights from file
    _, _, actor_opt_state, critic_opt_state, weights_s = load_('/sc_project/pkl_sc/outer_loop_pg_new_v4f_06_01-035522.pkl')
    weights["s"] = weights_s
    weights["v"] = load_('/sc_project/test_data/forward_new_v10_81M_144N_22_10-021849.pkl')[1][1] 

    ke = jax.random.split(key, 10) 
    SC, PRIOR_VEC, ZERO_VEC_IND = gen_sc(ke, params["MODULES"], params["ACTION_SPACE"], params["PLAN_SPACE"])

    start_time = datetime.now()
    loss_arrs, sem_arrs, selected_other, actor_opt_state, critic_opt_state, weights_s, plan_info = full_loop(SC, weights, params, actor_opt_state, critic_opt_state)
    print("Simulation time: ", datetime.now() - start_time, "s/epoch= ", ((datetime.now() - start_time) / params["TOT_EPOCHS"]).total_seconds())

    (loss_arr, actor_loss_arr, critic_loss_arr, vec_kl_arr, act_kl_arr, r_tot_arr, plan_rate_arr, policy_entropy_arr) = loss_arrs
    (sem_loss_arr, sem_actor_arr, sem_critic_arr, sem_act_kl_arr, sem_vec_kl_arr, sem_r_arr, sem_plan_rate_arr, sem_policy_entropy_arr) = sem_arrs
    ((r_arr, rp_arr, sample_arr, mask_arr, pos_plan_arr, pos_arr, dot_arr, policy_arr, hs_arr, hv_arr, vec_ind_arr, act_ind_arr), params) = selected_other

if __name__ == "__main__":
    main()