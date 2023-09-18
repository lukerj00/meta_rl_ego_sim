import jax
import jax.numpy as jnp
import jax.random as rnd
# import numpy as np
# import matplotlib.pyplot as plt

# def generate_binary_timeseries(length, switch_prob=0.2, start_bit=0):
#     # Generate an array of random values between 0 and 1
#     rand_vals = np.random.rand(length - 1)
    
#     # Determine where switches occur based on the switch probability
#     switches = (rand_vals < switch_prob).astype(int)
    
#     # Calculate cumulative sum to determine the state of the bit over time
#     cum_sum = np.cumsum(switches) % 2
    
#     # Insert the starting bit at the beginning of the array
#     series = np.insert(cum_sum, 0, start_bit)
    
#     return series

# # Generate time series for each probability
# length = 30
# probs = [0.2, 0.25, 0.3]
# time_series_list = [generate_binary_timeseries(length, p) for p in probs]

# # Plot each time series
# for p, ts in zip(probs, time_series_list):
#     plt.figure(figsize=(10, 2))
#     plt.step(range(len(ts)), ts, where='mid')
#     plt.ylim(-0.1, 1.1)
#     plt.title(f"Time Series with p={p}")
#     plt.xlabel("Time step")
#     plt.ylabel("Bit value")
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.show()

import numpy as np
import jax.numpy as jnp
import jax.random as rnd

# def gen_timeseries(keys_,SC, pos_0, dot_0, dot_vec, samples, switch_prob, N, max_plan_length):
#     ID_ARR, VEC_ARR, H1VEC_ARR = SC
#     binary_series = gen_binary_timeseries(keys_,N+1, switch_prob, max_plan_length)
#     binary_array = jnp.vstack([binary_series,1-binary_series])
#     h1vec_arr = H1VEC_ARR[:, samples]
#     vec_arr = VEC_ARR[:, samples]
#     pos_plan_arr = np.zeros((2, N+1))
#     pos_arr = np.zeros((2, N+1))
#     dot_arr = np.zeros((2, N+1))
#     pos_plan_arr[:, 0] = pos_0
#     pos_arr[:, 0] = pos_0
#     dot_arr[:, 0] = dot_0
#     for i in range(1, N+1):
#         dot_arr[:, i] = dot_arr[:, i-1] + dot_vec
#         if binary_series[i] == 0:
#             if binary_series[i-1] == 1:  # Transition from 1 to 0
#                 pos_plan_arr[:, i] = pos_arr[:, i-1]
#             pos_arr[:, i] = pos_arr[:, i-1] + vec_arr[:, i-1]
#             pos_plan_arr[:, i] = pos_arr[:, i]
#         else:
#             pos_arr[:, i] = pos_arr[:, i-1]
#             pos_plan_arr[:, i] = pos_plan_arr[:, i-1] + vec_arr[:, i-1]
#     return binary_array,jnp.array(pos_plan_arr),jnp.array(pos_arr),jnp.array(dot_arr),jnp.array(h1vec_arr),jnp.array(vec_arr)

# For the binary time series generation
# def gen_binary_timeseries(keys,N,switch_prob,max_plan_length):
#     ts = [0]  # start with a 0
#     count_ones = 0
#     for i in range(1, N):
#         if ts[i-1] == 0:
#             ts.append(rnd.bernoulli(keys[i],p=switch_prob))
#             if ts[i] == 1:
#                 count_ones = 1
#         else:
#             if count_ones > max_plan_length - 1:  # subtract 1 to account for 0-based index
#                 ts.append(0)
#                 count_ones = 0
#             else:
#                 ts.append(rnd.bernoulli(keys[2*i+1],p=1-switch_prob))
#                 if ts[i] == 1:
#                     count_ones += 1
#                 else:
#                     count_ones = 0
#     return jnp.array(ts)

def gen_binary_timeseries(keys, N, switch_prob, max_plan_length):
    ts_init = jnp.zeros(N)  # start with 0s
    if max_plan_length == 0:
        return ts_init
    def body_fn(i, carry):
        ts, count_ones, keys = carry
        true_fn = lambda _: (jnp.int32(rnd.bernoulli(keys[i], p=switch_prob)), 0)
        def false_fn(_):
            return jax.lax.cond(count_ones >= max_plan_length - 1,
                                lambda _: (0, 0), # ts, count_ones
                                lambda _: (jnp.int32(rnd.bernoulli(keys[i], p=1-switch_prob)), count_ones + 1),
                                None)
        next_val, count_ones_next = jax.lax.cond(ts[i] == 0, true_fn, false_fn, None)
        ts = ts.at[i+1].set(next_val)
        i += 1
        return (ts, count_ones_next, keys)
    result, _, _ = jax.lax.fori_loop(1, N, body_fn, (ts_init, 0, keys))
    return result

def gen_timeseries(keys_, SC, pos_0, dot_0, dot_vec, samples, switch_prob, N, max_plan_length):
    ID_ARR, VEC_ARR, H1VEC_ARR = SC
    binary_series = gen_binary_timeseries(keys_, N, switch_prob, max_plan_length)
    binary_array = jnp.vstack([binary_series, 1 - binary_series])
    h1vec_arr = H1VEC_ARR[:, samples]
    vec_arr = VEC_ARR[:, samples]
    def time_step(carry, i):
        pos_plan, pos, dot = carry
        dot_next = dot + dot_vec  # Assuming dot_vec is a globally available vector
        def true_fn(_):
            return pos + vec_arr[:, i-1], pos + vec_arr[:, i-1]
        def false_fn(_):
            return pos_plan + vec_arr[:, i-1], pos
        pos_plan_next, pos_next = jax.lax.cond(binary_series[i] == 0, true_fn, false_fn, None)
        return (pos_plan_next, pos_next, dot_next), (pos_plan_next, pos_next, dot_next)
    init_carry = (jnp.array(pos_0), jnp.array(pos_0), jnp.array(dot_0))
    _, (pos_plan_stacked, pos_stacked, dot_stacked) = jax.lax.scan(time_step, init_carry, jnp.arange(1,N))
    return binary_array,jnp.concatenate([jnp.reshape(pos_0,(1,2)),pos_plan_stacked]),jnp.concatenate([jnp.reshape(pos_0,(1,2)),pos_stacked]),jnp.concatenate([jnp.reshape(dot_0,(1,2)),dot_stacked]),h1vec_arr,vec_arr

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

# Sample usage
# Assuming SC, pos_0, dot_0, dot_vec, samples, step_array have been defined elsewhere
N = 50
pos_0 = jnp.array([0.0, 0.0])
dot_0 = jnp.array([0.5, 0.5])
dot_vec = jnp.array([0.1, 0.1])
samples = rnd.choice(rnd.PRNGKey(0), jnp.arange(N), shape=(N,))
# jnp.arange(0,50)
step_array = jnp.arange(0,50)
MODULES = 9
APERTURE = (1/2)*jnp.pi # (3/5)*jnp.pi # (jnp.sqrt(2)/2)*jnp.pi ### unconstrained
ACTION_FRAC = 1/2 # unconstrained
ACTION_SPACE = ACTION_FRAC*APERTURE # 'AGENT_SPEED'
PLAN_FRAC_REL = 2 # 3/2
PLAN_SPACE = PLAN_FRAC_REL*ACTION_SPACE
keys = rnd.split(rnd.PRNGKey(0),2)
switch_prob = 0.2
max_plan_length = 4
SC = gen_sc(keys,MODULES,ACTION_SPACE,PLAN_SPACE)
keys_ = rnd.split(keys[0],2*N+1)
binary_array,pos_plan_arr,pos_arr,dot_arr,h1vec_arr,vec_arr = gen_timeseries(keys_,SC, pos_0, dot_0, dot_vec, samples, switch_prob, N, max_plan_length)
print('pos_plan_stack.tyoe=',type(pos_plan_arr),pos_plan_arr.shape)

print('bin_series=',binary_array,'pos_arr=',pos_arr,'\n','pos_plan_arr=',pos_plan_arr) #,'\n','dot_arr=',dot_arr,'\n','h1vec_arr=',h1vec_arr,'\n','vec_arr=',vec_arr,'\n')

import matplotlib.pyplot as plt
# import numpy as np

# Assuming you have generated pos_arr, pos_plan_arr, and binary_series using the provided functions

# Calculate the norms
norm_pos_arr = jnp.linalg.norm(pos_arr, axis=1)
norm_pos_plan_arr = jnp.linalg.norm(pos_plan_arr, axis=1)

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting norms
plt.plot(norm_pos_arr, label="Norm of pos_arr", color='blue')
plt.plot(norm_pos_plan_arr, label="Norm of pos_plan_arr", color='green')

# Plotting binary time series
plt.plot(binary_array[0,:] * norm_pos_arr.max(), label="Binary time series (scaled)", color='red', linestyle='--')

# Additional plot settings
plt.title(f"Norm of pos_arr and pos_plan_arr, switch_prob={switch_prob}, max_plan_length={max_plan_length}", fontsize=14)
plt.xlabel("Time step")
plt.ylabel("Norm")
plt.legend()
plt.grid(True)

plt.show()
