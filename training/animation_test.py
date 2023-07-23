import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend to prevent segmentation fault
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import matplotlib.ticker as ticker

current_backend = matplotlib.get_backend()
print(current_backend)

def mod_(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

# Generating timeseries data
time_points = np.linspace(0, 10, 100)
agent_x = np.sin(time_points) * np.pi/2  
agent_y = np.cos(time_points) * np.pi/2

# Setting up the random initial position and velocity for the dot
dot_speed = 10  # Speed of the dot
initial_angle = 2 * np.pi * np.random.rand((2)) - np.pi  # Random value between [-π, π]
angular_velocity = (2 * np.pi * np.random.rand((2)) - np.pi) / 10  # Random angular velocity

# Compute angular trajectory of the dot
dot_xy = mod_(initial_angle + dot_speed*np.outer(angular_velocity, time_points).T).T

fig, ax = plt.subplots(figsize=(6, 6))  # Making sure the figure is square
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.set_aspect('equal', 'box')  # Ensuring the aspect ratio is equal

# Set x and y ticks at intervals of pi/2
ticks = np.arange(-np.pi, np.pi+1, np.pi/2)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

# Use LaTeX formatted labels for ticks
labels = ["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"]
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Plot agent and dot with initial data
agent, = ax.plot(agent_x[0], agent_y[0], 'k+', markersize=12, label='Agent')
dot, = ax.plot(dot_xy[0,0], dot_xy[1,0], 'rx', markersize=8, label='Dot')
box_size = np.pi
agent_box = plt.Rectangle((agent_x[0]-box_size/2, agent_y[0]-box_size/2), box_size, box_size, fill=False, color='lightgrey')
ax.add_patch(agent_box)

def animate(i):
    agent.set_data(agent_x[i], agent_y[i])
    dot.set_data(dot_xy[0,i], dot_xy[1,i])
    agent_box.set_xy((agent_x[i]-box_size/2, agent_y[i]-box_size/2))
    return agent, dot, agent_box,

ani = animation.FuncAnimation(
    fig, animate, frames=len(time_points), interval=50, repeat=True
)

ax.legend()
plt.show()
# plt.ion()
