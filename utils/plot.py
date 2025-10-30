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


def plot_normalized_steps(results_list, step_magnitudes_list, labels=None, title="Normalized Step Responses"):
    """
    Plot normalized step responses for multiple simulations.

    Args:
        results_list: list of results dictionaries (each like your simulation output)
        step_magnitudes_list: list of step magnitudes (Δu) corresponding to each results dict
                             Can be either:
                             - scalar value: same step magnitude applied to all inputs
                             - array of size (num_inputs,): step magnitude per input
        labels: optional list of labels for each step (e.g., ['10%', '25%', '50%'])
    """
    num_plots = results_list[0]['y_history'].shape[1]  # number of outputs

    for i in range(num_plots):
        plt.figure(figsize=(10, 5))
        for j, results in enumerate(results_list):
            time = results['time']
            y = np.array(results['y_history'])

            delta_u = step_magnitudes_list[j]

            # Convert to scalar if it's an array with identical elements
            if isinstance(delta_u, (list, np.ndarray)):
                delta_u_arr = np.array(delta_u)
                if delta_u_arr.ndim > 0:
                    # If all elements are the same, use the first one
                    if np.allclose(delta_u_arr, delta_u_arr[0]):
                        delta_u = delta_u_arr[0]
                    else:
                        # Multiple different step magnitudes - average them for overall normalization
                        delta_u = np.mean(delta_u_arr)

            y0 = y[0, :]
            y_norm = (y - y0) / delta_u  # normalized step response

            label = labels[j] if labels is not None else f'Step {j+1}'
            plt.plot(time, y_norm[:, i], label=label)

        plt.xlabel("Time [s]")
        plt.ylabel(f"Normalized Output y{i+1} / Δu")
        plt.title(f"Normalized Step Response of Output y{i+1}")
        plt.grid(True)
        plt.legend()
        plt.show()
