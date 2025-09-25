'''
Simulation of the open-loop system for the stochastic model.
'''
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from models.stochastic_model import FourTankStochastic
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

# Initialize model
measurement_noise_std = 10  # Standard deviation for measurement noise [cm]
model = FourTankStochastic(params, measurement_noise_std, x0)

# Results storage
x_history = []
u_history = []
d_history = []
y_history = []  # Noisy measurements: y(t) = g(x(t),p) + v(t)
z_history = []  # Deterministic outputs: z(t) = h(x(t),p)

#===============================
# SIMULATION
#===============================

# Piecewise constant disturbances F3 and F4: d(t) = dk for tk≤t<tk+1 
d_seq = create_piecewise_cnst_seq(num_inputs=2, total_time=T, dt=dt, min_val=200, max_val=400, step_resolution=10, number_of_steps=5)
                                    
# Piecewise constant input sequence (random steps)
u_seq = create_piecewise_cnst_seq(num_inputs=2, total_time=T, dt=dt, min_val=200, max_val=400, step_resolution=10, number_of_steps=1)

# Initialize current state
x_current = x0.copy()

for k in range(len(time)):
    # Store results
    x_history.append(x_current.copy())
    u_history.append(u_seq[:, k])
    d_history.append(d_seq[:, k])
    
    # Generate measurements and outputs
    y_k = model.measurement(x_current)  # y(t) = g(x(t),p) + v(t)
    z_k = model.output(x_current)       # z(t) = h(x(t),p)
    y_history.append(y_k)
    z_history.append(z_k)
    
    # Simulate one step: ẋ(t) = f(x(t),u(t),d(t),p)
    dxdt = model.dynamics(time[k], x_current, u_seq[:, k], d_seq[:, k])
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
y_history = np.array(y_history)  # Measurements with noise
z_history = np.array(z_history)  # Deterministic outputs

# Create subplots - 4 plots in 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Four Tank System - Stochastic Model Open Loop Simulation', fontsize=16)

# Plot states (tank heights from noisy measurements)
axes[0, 0].plot(time, y_history[:, 0], 'b-', linewidth=2, label='Tank 1')
axes[0, 0].plot(time, y_history[:, 1], 'r-', linewidth=2, label='Tank 2')
axes[0, 0].plot(time, y_history[:, 2], 'g-', linewidth=2, label='Tank 3')
axes[0, 0].plot(time, y_history[:, 3], 'm-', linewidth=2, label='Tank 4')
axes[0, 0].set_title('State Evolution (Noisy Measurements)')
axes[0, 0].set_ylabel('Height [cm]')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot actions (control inputs)
axes[0, 1].plot(time, u_history[:, 0], 'r-', linewidth=2, label='Pump 1')
axes[0, 1].plot(time, u_history[:, 1], 'g-', linewidth=2, label='Pump 2')
axes[0, 1].set_title('Action Evolution')
axes[0, 1].set_ylabel('Flow Rate [cm³/s]')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot measurements (same as states but with different title for clarity)
axes[1, 0].plot(time, y_history[:, 0], 'b-', linewidth=2, label='Tank 1')
axes[1, 0].plot(time, y_history[:, 1], 'r-', linewidth=2, label='Tank 2')
axes[1, 0].plot(time, y_history[:, 2], 'g-', linewidth=2, label='Tank 3')
axes[1, 0].plot(time, y_history[:, 3], 'm-', linewidth=2, label='Tank 4')
axes[1, 0].set_title('Measurement Evolution (with White Noise)')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Height [cm]')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot disturbances
axes[1, 1].plot(time, d_history[:, 0], 'm-', linewidth=2, label='Disturbance 1')
axes[1, 1].plot(time, d_history[:, 1], 'c-', linewidth=2, label='Disturbance 2')
axes[1, 1].set_title('Disturbance Evolution')
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].set_ylabel('Flow Rate [cm³/s]')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Print final states and demonstrate all model components
print(f"\nStochastic Nonlinear Model Simulation completed!")
print(f"Model components implemented:")
print(f"1. Deterministic dynamics: x_dot(t) = f(x(t),u(t),d(t),p)")
print(f"2. Measurement model: y(t) = g(x(t),p) + v(t) with v(t) ~ N(0,Rvv(p))")
print(f"3. Output model: z(t) = h(x(t),p)")
print(f"4. Piecewise constant disturbances: d(t) = dk for tk≤t<tk+1")
print(f"\nFinal results:")
print(f"Final tank masses [g]: {x_current}")
print(f"Final noisy measurements y(t) [cm]: {model.measurement(x_current)}")  
print(f"Final deterministic outputs z(t) [cm]: {model.output(x_current)}")
print(f"Final disturbances d(t) [cm³/s]: {d_seq[:, -1]}")