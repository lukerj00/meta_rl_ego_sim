import jax.numpy as jnp
import jax.random as rnd

params = {
    "TOT_EPOCHS": 2000,
    "ACTOR_LR": 0.0003,
    "CRITIC_LR": 0.0012,
    "CRITIC_WD": 0,
    "ACTOR_GC": 0.4,
    "CRITIC_GC": 0.9,
    "VMAPS": 1000,
    "BATCH_SIZE": 1000,
    "H_S": 100,
    "H_INTER": 32,
    "H_P": 300,
    "N_DOTS": 1,
    "MODULES": 9,
    "M": 81,
    "PLAN_ITS": 10,
    "NONE_PLAN": jnp.zeros((10,)),
    "TRIAL_LENGTH": 60,
    "TEST_LENGTH": 60,
    "INIT_S": 2,
    "INIT_P": 2,
    "LAMBDA_VEC_KL": 0.01,
    "LAMBDA_ACT_KL": 0.05,
    "TEMP_VS": 1,
    "TEMP_AS": 0.5,
    "APERTURE": (1/2)*jnp.pi,
    "SIGMA_A": 0.3,
    "SIGMA_N": 0.2,
    "SIGMA_R0": 0.3,
    "SIGMA_RINF": 0.3,
    "MAX_DOT_SPEED": 0.5497787143782138,
    "PLAN_RATIO": 5,
    "PRIOR_PLAN_FRAC": 0.2,
    "PRIOR_STAT": 0.2,
    "ACTION_SPACE": 1.5707963267948966,
}

key = rnd.PRNGKey(0)
G = 80
N = 81
H_S = params["H_S"]
H_P = params["H_P"]
H_INTER = params["H_INTER"] 
INIT = 100

keys = rnd.split(key, num=22)
weights = {
    "s": {
        "Ws_vt_z": jnp.sqrt(INIT/(H_S+N))*rnd.normal(keys[0],(H_S,N)),
        "Ws_vt_f": jnp.sqrt(INIT/(H_S+N))*rnd.normal(keys[1],(H_S,N)),
        "Ws_vt_h": jnp.sqrt(INIT/(H_S+N))*rnd.normal(keys[2],(H_S,N)),
        "Ws_rt_z": jnp.sqrt(INIT/(H_S+2))*rnd.normal(keys[3],(H_S,2)), 
        "Ws_rt_f": jnp.sqrt(INIT/(H_S+2))*rnd.normal(keys[4],(H_S,2)),
        "Ws_rt_h": jnp.sqrt(INIT/(H_S+2))*rnd.normal(keys[5],(H_S,2)),
        "Ws_at_1z": jnp.sqrt(INIT/(H_S+2))*rnd.normal(keys[6],(H_S,2)),
        "Ws_at_1f": jnp.sqrt(INIT/(H_S+2))*rnd.normal(keys[7],(H_S,2)),
        "Ws_at_1h": jnp.sqrt(INIT/(H_S+2))*rnd.normal(keys[8],(H_S,2)),
        "Ws_ht_1z": jnp.sqrt(INIT/(H_S+2))*rnd.normal(keys[9],(H_S,H_P)),
        "Ws_ht_1f": jnp.sqrt(INIT/(H_S+2))*rnd.normal(keys[10],(H_S,H_P)),
        "Ws_ht_1h": jnp.sqrt(INIT/(H_S+2))*rnd.normal(keys[11],(H_S,H_P)),
        "Us_z": jnp.linalg.qr(jnp.sqrt(INIT/(H_S+H_S))*rnd.normal(keys[12],(H_S,H_S)))[0],
        "Us_f": jnp.linalg.qr(jnp.sqrt(INIT/(H_S+H_S))*rnd.normal(keys[13],(H_S,H_S)))[0], 
        "Us_h": jnp.linalg.qr(jnp.sqrt(INIT/(H_S+H_S))*rnd.normal(keys[14],(H_S,H_S)))[0],
        "bs_z": jnp.sqrt(INIT/(H_S))*rnd.normal(keys[15],(H_S,)),
        "bs_f": jnp.sqrt(INIT/(H_S))*rnd.normal(keys[16],(H_S,)), 
        "bs_h": jnp.sqrt(INIT/(H_S))*rnd.normal(keys[17],(H_S,)),
        "Ws_vec": jnp.sqrt(INIT/(N+H_S))*rnd.normal(keys[18],(N,H_S)),
        "Ws_act": jnp.sqrt(INIT/(2+H_S))*rnd.normal(keys[19],(2,H_S)),
        "Ws_val_inter": jnp.sqrt(INIT/(H_INTER+H_S))*rnd.normal(keys[20],(H_INTER,H_S)),
        "Ws_val_r": jnp.sqrt(INIT/(H_INTER))*rnd.normal(keys[21],(H_INTER,)),          
    },
    "v": {
        # Load from file
    },
}