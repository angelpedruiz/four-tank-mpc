"""
Plotting utilities for the four-tank system simulation.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_nonlinear_results(results, title="Four Tank System - Nonlinear Model Open Loop Simulation"):
    """
    Plot simulation results from the nonlinear open-loop simulation.

    Parameters:
    -----------
    results : dict
        Dictionary containing simulation results with keys:
        - 'time': Time vector [s]
        - 'x_history': State trajectory (mass in tanks) [g]
        - 'u_history': Control input trajectory [cm�/s]
        - 'd_history': Disturbance trajectory [cm�/s]
        - 'y_history': Noisy measurements (tank heights) [cm]
        - 'z_history': Deterministic outputs (tank heights) [cm]
        - 'q_history': Outflow rates from tanks [cm�/s]
    title : str, optional
        Title for the main figure. Default: "Four Tank System - Nonlinear Model Open Loop Simulation"

    Returns:
    --------
    tuple : (fig, axes) - matplotlib figure and axes objects for further customization
    """

    # Extract data from results dictionary
    time = results['time']
    x_history = results['x_history']
    u_history = results['u_history']
    y_history = results['y_history']
    d_history = results['d_history']

    # Create subplots - 4 plots in 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)

    # Plot states (tank masses)
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
    axes[0, 1].set_ylabel('Flow Rate [cm�/s]')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot measurements (noisy outputs)
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
    axes[1, 1].set_ylabel('Flow Rate [cm�/s]')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    return fig, axes


def plot_outflow_rates(results):
    """
    Plot outflow rates from tanks.

    Parameters:
    -----------
    results : dict
        Dictionary containing simulation results with key 'q_history' and 'time'.

    Returns:
    --------
    tuple : (fig, ax) - matplotlib figure and axes objects for further customization
    """

    time = results['time']
    q_history = results['q_history']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, q_history)
    ax.set_title('Outflow Rates from Tanks')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Outflow Rate [cm�/s]')
    ax.legend(['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'])
    ax.grid(True)

    return fig, ax


def plot_normalized_steps(results_list, step_magnitudes_list, noise_label):
    """
    Plot normalized step responses for one noise level.

    Parameters
    ----------
    results_list : list
        List of simulation results, each containing time and output arrays.
        Each element expected to have {'time': t, 'y': y}, where y has shape (n_outputs, len(t)).
    step_magnitudes_list : list of float
        Percentages of step changes (e.g. [10, 25, 50]).
    noise_label : str
        Label of the current noise level (e.g., 'Low Noise').
    """

    num_inputs = 2
    num_outputs = 4
    num_steps = len(step_magnitudes_list)

    fig, axes = plt.subplots(num_inputs, num_outputs, figsize=(16, 6), sharex=True)
    fig.suptitle(f'Normalized Step Responses — {noise_label}', fontsize=14, fontweight='bold')

    # Ensure results correspond to [u1_10%, u1_25%, u1_50%, u2_10%, u2_25%, u2_50%]
    results_arr = np.array(results_list, dtype=object).reshape(num_inputs, num_steps)

    for i_in in range(num_inputs):           # Input index (u1, u2)
        for j_out in range(num_outputs):     # Output index (y1–y4)
            ax = axes[i_in, j_out]
            for k_step, p in enumerate(step_magnitudes_list):
                res = results_arr[i_in, k_step]
                t = res['time']
                y = res['y_history'][:, j_out]

                # Normalize: divide by max absolute change (step response)
                y_norm = (y - y[0]) / abs(y[-1] - y[0]) if abs(y[-1] - y[0]) > 1e-6 else y - y[0]
                ax.plot(t, y_norm, label=f'{p}% step')

            ax.set_title(f'Input {i_in+1} → Output {j_out+1}', fontsize=10)
            if i_in == num_inputs - 1:
                ax.set_xlabel('Time [s]')
            if j_out == 0:
                ax.set_ylabel('Normalized Response')

            ax.grid(True, alpha=0.3)

    # Legend only once
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout(rect=[0, 0, 0.97, 0.95])
    plt.show()

