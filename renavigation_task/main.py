from datetime import datetime
from functools import partial
import jax
import jax.numpy as jnp
import optax

from params import init_params, init_weights
from model_train import full_loop

def main():
    params = init_params()
    key = jax.random.PRNGKey(0)
    weights = init_weights(key)

    start_time = datetime.now()
    loss_arrays, sem_arrays, opt_state, final_weights = full_loop(params, weights)
    print(f"Simulation time: {datetime.now() - start_time}")
    print(f"Time per epoch: {((datetime.now() - start_time) / params['TOT_EPOCHS']).total_seconds()} seconds")

if __name__ == "__main__":
    main()
