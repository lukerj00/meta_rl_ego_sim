import jax.numpy as jnp
import jax.random as rnd

params = {
    "TOT_EPOCHS": 501,
    "LEARNING_RATE": 0.0001,
    "WEIGHT_DECAY": 0,
    "BATCH_SIZE": 400,
    "APERTURE": jnp.pi/3,
    "NEURONS": 11,
    "N_DOTS": 3,
    "COLORS": jnp.float32([[255,0,0], [0,255,0], [0,0,255]]),
    "SIGMA_A": 1.0,
    "SIGMA_NOISE": 1.8, 
    "SIGMA_R0": 1.0,
    "SIGMA_RINF": 0.3,
    "TAU": 0.01,
    "LAMBDA_PREDICT": 0.001,
    "LAMBDA_CE": 0.001,
    "ALPHA": 0.005
}

G = 80 
N = params["NEURONS"]**2
INIT = 100
key = rnd.PRNGKey(0)

keys = rnd.split(key, num=14)
weights = {
    "h0": rnd.normal(keys[0], (G,), dtype=jnp.float32),
    "Wr_z": (INIT/G*N)*rnd.normal(keys[1], (G,N), dtype=jnp.float32),
    "Wg_z": (INIT/G*N)*rnd.normal(keys[1], (G,N), dtype=jnp.float32), 
    "Wb_z": (INIT/G*N)*rnd.normal(keys[1], (G,N), dtype=jnp.float32),
    "U_z": (INIT/G*G)*rnd.normal(keys[2], (G,G), dtype=jnp.float32),
    "b_z": (INIT/G)*rnd.normal(keys[3], (G,), dtype=jnp.float32),
    "Wr_r": (INIT/G*N)*rnd.normal(keys[4], (G,N), dtype=jnp.float32), 
    "Wg_r": (INIT/G*N)*rnd.normal(keys[4], (G,N), dtype=jnp.float32),
    "Wb_r": (INIT/G*N)*rnd.normal(keys[4], (G,N), dtype=jnp.float32),
    "U_r": (INIT/G*G)*rnd.normal(keys[5], (G,G), dtype=jnp.float32),
    "b_r": (INIT/G)*rnd.normal(keys[6], (G,), dtype=jnp.float32),
    "Wr_h": (INIT/G*N)*rnd.normal(keys[7], (G,N), dtype=jnp.float32),
    "Wg_h": (INIT/G*N)*rnd.normal(keys[7], (G,N), dtype=jnp.float32),
    "Wb_h": (INIT/G*N)*rnd.normal(keys[7], (G,N), dtype=jnp.float32), 
    "U_h": (INIT/G*G)*rnd.normal(keys[8], (G,G), dtype=jnp.float32),
    "b_h": (INIT/G)*rnd.normal(keys[9], (G,), dtype=jnp.float32),
    "W_r": (INIT/G)*rnd.normal(keys[10], (G,), dtype=jnp.float32),
    "C": (INIT/2*G)*rnd.normal(keys[11], (2,G), dtype=jnp.float32),
    "E": (INIT/4*G)*rnd.normal(keys[12], (4,G), dtype=jnp.float32),
    "D": (INIT/4*G)*rnd.normal(keys[13], (4,G), dtype=jnp.float32),
    "S": (INIT/params["N_DOTS"]*G)*rnd.normal(keys[14], (3,G), dtype=jnp.float32)
}