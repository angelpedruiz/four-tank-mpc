'''
Simulation of the open-loop system for the SDE model.
'''
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from models.sde_model import FourTankSDE
from utils import create_piecewise_cnst_seq

#===============================
# PARAMETERS
#===============================

# System parameters for the four tank system
params = np.array([
    # Pipe cross-sectional areas a [cm^2]
    0.071, 0.057, 0.071, 0.057,
    # Tank cross-sectional areas A [cm^2]
    28, 32, 28, 32,
    # Flow distribution ratios gamma [-]
    0.7, 0.6,
    # Gravity g [cm/s^2]
    981,
    # Density rho [g/cm^3]
    1.0
])


#===============================
# SIMULATION INITIALIZATION
#===============================

# Time
T = 20  # total simulation time [s]
dt = 0.01  # time step [s]
time = np.arange(0, T, dt)

# Initial conditions
x0 = np.array([500, 500, 500, 500])  # initial mass in tanks [g]
d_const = np.array([5, 5])  # disturbance flows [cm^3/s]

# Noise model
measurement_noise_std = 3.0  # standard deviation of measurement noise

# Initialize model
model = FourTankSDE(params, measurement_noise_std=measurement_noise_std, x0=x0)

# Results storage
x_history = []
u_history = []
d_history = []

#===============================
# SIMULATION
#===============================

# Piecewise constant input sequence (random steps)
u_seq = create_piecewise_cnst_seq(num_inputs=2, total_time=T, dt=dt, min_val=0, max_val=10, step_resolution=0.5, number_of_steps=1)

# Initialize current state
x_current = x0.copy()

for k in range(len(time)):
    # Store results
    x_history.append(x_current.copy())
    u_history.append(u_seq[:, k])
    d_history.append(d_const.copy())
    
    # Simulate one step
    dxdt = model.dynamics(time[k], x_current, u_seq[:, k], d_const)
    x_next = x_current + dxdt * dt
    
    # Update state
    x_current = x_next

#===============================
# PLOTTING
#===============================

# Convert lists to arrays for easier plotting
x_history = np.array(x_history)
u_history = np.array(u_history)
d_history = np.array(d_history)

# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Four Tank System - SDE Model Open Loop Simulation', fontsize=16)

# Plot tank masses
for i in range(4):
    row = i // 2
    col = i % 2
    axes[row, col].plot(time, x_history[:, i], 'b-', linewidth=2)
    axes[row, col].set_title(f'Tank {i+1} Mass')
    axes[row, col].set_ylabel('Mass [g]')
    axes[row, col].grid(True)
    if row == 1:  # Bottom row
        axes[row, col].set_xlabel('Time [s]')

# Plot inputs
axes[2, 0].plot(time, u_history[:, 0], 'r-', linewidth=2, label='Pump 1')
axes[2, 0].plot(time, u_history[:, 1], 'g-', linewidth=2, label='Pump 2')
axes[2, 0].set_title('Control Inputs')
axes[2, 0].set_xlabel('Time [s]')
axes[2, 0].set_ylabel('Flow Rate [cm³/s]')
axes[2, 0].legend()
axes[2, 0].grid(True)

# Plot disturbances
axes[2, 1].plot(time, d_history[:, 0], 'm-', linewidth=2, label='Disturbance 1')
axes[2, 1].plot(time, d_history[:, 1], 'c-', linewidth=2, label='Disturbance 2')
axes[2, 1].set_title('Disturbances')
axes[2, 1].set_xlabel('Time [s]')
axes[2, 1].set_ylabel('Flow Rate [cm³/s]')
axes[2, 1].legend()
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()

# Print final states
print(f"\nSDE Model Simulation completed!")
print(f"Final tank masses: {x_current}")
print(f"Final tank heights: {model.measurement(x_current)}")