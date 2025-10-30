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

from models.nonlinear_model import FourTankNonlinear
from utils.piecewise import piecewise_constant, piecewise_random


#===============================
# SIMULATION FUNCTION
#===============================

def simulate_nonlinear(time, x0, u_seq, d_seq, params, model, dt):
    """
    Run nonlinear open-loop simulation of the four-tank system.

    Parameters:
    -----------
    time : np.ndarray
        Time vector [s]
    x0 : np.ndarray
        Initial state (mass in tanks) [g]
    u_seq : np.ndarray
        Control input sequence [cm³/s]
    d_seq : np.ndarray
        Disturbance sequence [cm³/s]
    params : np.ndarray
        System parameters
    model : FourTankNonlinear
        Nonlinear model instance
    dt : float
        Time step [s]

    Returns:
    --------
    dict : Dictionary containing simulation results with keys:
        - 'time': Time vector
        - 'x_history': State trajectory (mass in tanks)
        - 'u_history': Control input trajectory
        - 'd_history': Disturbance trajectory
        - 'y_history': Noisy measurements (tank heights)
        - 'z_history': Deterministic outputs (tank heights)
        - 'q_history': Outflow rates from tanks
    """

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
        d_history.append(d_seq[k, :])

        # Generate measurements and outputs
        y_k = model.measurement(x_current)  # y(t) = g(x(t),p) + v(t)
        z_k = model.output(x_current)       # z(t) = h(x(t),p)
        y_history.append(y_k)
        z_history.append(z_k)

        # Simulate one step: ẋ(t) = f(x(t),u(t),d(t),p)
        dxdt = model.dynamics(time[k], x_current, u_seq[k, :], d_seq[k, :])
        x_next = x_current + dxdt * dt

        # Update state
        x_current = x_next
        h = x_current / (params[4:8] * params[11])  # Heights
        q = params[0:4] * np.sqrt(2 * params[10] * h)  # Outflows
        q_history.append(q)

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
    """Main function for running nonlinear open-loop simulation with plotting."""

    #===============================
    # PARAMETERS
    #===============================

    # System parameters for the four tank system
    params = np.array([
        # Pipe cross-sectional areas a [cm^2]
        1.2, 1.2, 1.2, 1.2,
        # Tank cross-sectional areas A [cm^2]
        380, 380, 380, 380,
        # Flow distribution ratios gamma [-]
        0.58, 0.72,
        # Gravity g [cm/s^2]
        981.0,
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
    x0 = np.array([500, 500, 500, 500])  # initial mass in tanks [g]

    # Initialize model
    measurement_noise_std = 10  # Standard deviation for measurement noise [cm]
    model = FourTankNonlinear(params, x0)

    # Piecewise constant disturbances F3 and F4: d(t) = dk for tk <= t < tk+1
    d_seq = piecewise_constant([[100, 200], [200, 100]], total_time=T, dt=dt)

    # Piecewise constant input sequence
    u_seq = piecewise_constant([[250, 300], [300, 250]], total_time=T, dt=dt)

    #===============================
    # SIMULATION
    #===============================

    # Run simulation
    results = simulate_nonlinear(time, x0, u_seq, d_seq, params, model, dt)

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
    fig.suptitle('Four Tank System - Stochastic Model Open Loop Simulation', fontsize=16)

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
    print(f"\nStochastic Nonlinear Model Simulation completed!")
    print(f"Model components implemented:")
    print(f"1. Deterministic dynamics: x_dot(t) = f(x(t),u(t),d(t),p)")
    print(f"2. Measurement model: y(t) = g(x(t),p) + v(t) with v(t) ~ N(0,Rvv(p))")
    print(f"3. Output model: z(t) = h(x(t),p)")
    print(f"4. Piecewise constant disturbances: d(t) = dk for tk <= t < tk+1")
    print(f"\nFinal results:")


if __name__ == '__main__':
    main()