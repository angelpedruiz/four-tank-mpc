'''
Simulation of the open-loop system for the stochastic model.
'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from models.sde_model import FourTankSDE
from utils.piecewise import piecewise_constant


#===============================
# SIMULATION FUNCTION
#===============================

def simulate_sde(time, u_seq, d_seq, model, dt):
    """
    Run SDE open-loop simulation of the four-tank system.

    Parameters:
    -----------
    time : np.ndarray
        Time vector [s]
    u_seq : np.ndarray
        Control input sequence [cm³/s]
    d_seq : np.ndarray
        Disturbance sequence [cm³/s]
    model : FourTankSDE
        SDE model instance (contains params, x0, measurement_noise_std,
        process_noise_std, and correlation_time)
    dt : float
        Time step [s]

    Returns:
    --------
    dict : Dictionary containing simulation results with keys:
        - 'time': Time vector
        - 'x_history': State trajectory (mass in tanks)
        - 'u_history': Control input trajectory
        - 'd_history': Disturbance trajectory (deterministic + stochastic)
        - 'y_history': Noisy measurements (tank heights)
        - 'z_history': Deterministic outputs (tank heights)
        - 'q_history': Outflow rates from tanks
    """

    # Extract parameters and initial state from model
    params = model.params
    x0 = model.x0
    correlation_time = model.correlation_time

    # Results storage
    x_history = []
    u_history = []
    d_history = []
    y_history = []  # Noisy measurements: y(t) = g(x(t),p) + v(t)
    z_history = []  # Deterministic outputs: z(t) = h(x(t),p)
    q_history = []  # flowrates

    # Initialize current state
    x_current = x0.copy()

    for k in tqdm(range(len(time)), desc="Simulating", unit="time step"):
        # Store results
        x_history.append(x_current.copy())
        u_history.append(u_seq[k, :])

        # Generate measurements and outputs
        y_k = model.measurement(x_current)  # y(t) = g(x(t),p) + v(t)
        z_k = model.output(x_current)       # z(t) = h(x(t),p)
        y_history.append(y_k)
        z_history.append(z_k)

        # Simulate one step: dx(t) = f*dt + ρ*ε*√(τ*dt)*ξ
        dx, diffusion = model.dynamics(time[k], x_current, u_seq[k, :], d_seq[k, :], dt)
        x_next = x_current + dx

        # Update state
        x_current = x_next
        h = x_current / (params[4:8] * params[11])  # Heights
        q = params[0:4] * np.sqrt(2 * params[10] * h)  # Outflows

        q_history.append(q)

        # Extract stochastic flow disturbances for tanks 3 and 4
        # diffusion [g] = ρ [g/cm³] * ε [cm³/s] * √(τ*dt) [s]
        # To get equivalent flow rate: diffusion / (ρ * √(τ*dt))
        rho = params[11]
        tau = correlation_time
        stochastic_flow = diffusion[2:] / (rho * np.sqrt(tau * dt))  # [g] / ([g/cm³] * [s]) = [cm³/s]

        # Total disturbance: deterministic + stochastic
        d_history.append(d_seq[k, :] + stochastic_flow)

    # Convert lists to arrays
    x_history = np.array(x_history)
    u_history = np.array(u_history)
    d_history = np.array(d_history)
    y_history = np.array(y_history)
    z_history = np.array(z_history)
    q_history = np.array(q_history)

    # Return results as dictionary
    return {
        'time': time,
        'x_history': x_history,
        'u_history': u_history,
        'd_history': d_history,
        'y_history': y_history,
        'z_history': z_history,
        'q_history': q_history
    }


def main():
    """Main function for running SDE open-loop simulation with plotting."""

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
    T = 200  # total simulation time [s]
    dt = 0.01  # time step [s]
    time = np.arange(0, T, dt)

    # Initial conditions
    x0 = np.array([500, 500, 0, 0])  # initial mass in tanks [g]

    # Initialize model
    measurement_noise_std = 10  # Standard deviation for measurement noise [cm]
    disturbance_noise_std = 0  # Intensity (epsilon) for stochastic disturbances [cm^3/s]
    correlation_time = 1.0  # Correlation time (tau) for stochastic disturbances [s]
    model = FourTankSDE(params, measurement_noise_std, disturbance_noise_std, correlation_time, x0)

    # Piecewise constant input sequence (random steps)
    u_seq = piecewise_constant([[250, 0], [0, 250]], total_time=T, dt=dt)

    # Disturbance
    d_seq = piecewise_constant([[0, 0], [0, 0]], total_time=T, dt=dt)

    #===============================
    # SIMULATION
    #===============================

    # Run simulation
    results = simulate_sde(time, u_seq, d_seq, model, dt)

    # Extract results from dictionary
    x_history = results['x_history']
    u_history = results['u_history']
    d_history = results['d_history']
    y_history = results['y_history']
    z_history = results['z_history']
    q_history = results['q_history']

    #===============================
    # PLOTTING
    #===============================

    # Create subplots - 4 plots in 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Four Tank System - SDE Model Open Loop Simulation', fontsize=16)

    # Plot states (tank heights from noisy measurements)
    axes[0, 0].plot(time, x_history[:, 0], 'b-', linewidth=2, label='Tank 1')
    axes[0, 0].plot(time, x_history[:, 1], 'r-', linewidth=2, label='Tank 2')
    axes[0, 0].plot(time, x_history[:, 2], 'g-', linewidth=2, label='Tank 3')
    axes[0, 0].plot(time, x_history[:, 3], 'm-', linewidth=2, label='Tank 4')
    axes[0, 0].set_title('State Evolution')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Mass [g]')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot actions (control inputs)
    axes[0, 1].plot(time, u_history[:, 0], 'r-', linewidth=2, label='F1')
    axes[0, 1].plot(time, u_history[:, 1], 'g-', linewidth=2, label='F2')
    axes[0, 1].set_title('Action Evolution')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Flow Rate [cm³/s]')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot measurements (same as states but with different title for clarity)
    axes[1, 0].plot(time, y_history[:, 0], 'b-', linewidth=2, label='Tank 1')
    axes[1, 0].plot(time, y_history[:, 1], 'r-', linewidth=2, label='Tank 2')
    axes[1, 0].plot(time, y_history[:, 2], 'g-', linewidth=2, label='Tank 3')
    axes[1, 0].plot(time, y_history[:, 3], 'm-', linewidth=2, label='Tank 4')
    axes[1, 0].set_title('Measurement Evolution')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Height [cm]')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot disturbances
    axes[1, 1].plot(time, d_history[:, 0], 'm-', linewidth=2, label='F3')
    axes[1, 1].plot(time, d_history[:, 1], 'c-', linewidth=2, label='F4')
    axes[1, 1].set_title('Disturbance Evolution')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Flow Rate [cm³/s]')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    plt.plot(time, q_history)
    plt.title('Outflow Rates from Tanks')
    plt.xlabel('Time [s]')
    plt.ylabel('Outflow Rate [cm³/s]')
    plt.legend(['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'])
    plt.grid(True)
    plt.show()

    # Print final states and demonstrate all model components
    print("\nStochastic Nonlinear Model Simulation completed!")
    print("Model components implemented:")
    print("1. Deterministic dynamics: x_dot(t) = f(x(t),u(t),d(t),p)")
    print("2. Measurement model: y(t) = g(x(t),p) + v(t) ~ N(0,Rvv(p))")
    print("3. Output model: z(t) = h(x(t),p)")
    print("4. Piecewise constant disturbances: d(t) = dk for tk <= t < tk+1")
    print("\nFinal results:")


if __name__ == '__main__':
    main()