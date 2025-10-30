import numpy as np

import numpy as np

def piecewise_constant(values: list, total_time: float, dt: float) -> np.ndarray:
    """
    Create a deterministic piecewise constant input sequence where
    each input can have its own number of constant segments.

    Args:
        values: List of lists with shape (num_inputs, steps_per_input_i).
                Each sublist defines the constant values for that input.
                Example:
                    values = [
                        [100, 120],                 # input 1 (2 steps)
                        [300, 310, 320, 330, 340]   # input 2 (5 steps)
                    ]
        total_time: Total simulation time (float)
        dt: Time step (float)

    Returns:
        np.ndarray: Array of shape (time_steps, num_inputs),
                    where each column corresponds to one input.
    """
    # Convert to object array for variable-length lists
    if not isinstance(values, (list, tuple)) or not all(isinstance(v, (list, tuple)) for v in values):
        raise ValueError("values must be a list of lists, one sublist per input.")

    num_inputs = len(values)
    time = np.arange(0, total_time, dt)
    time_steps = len(time)

    # Prepare output array
    u = np.zeros((time_steps, num_inputs))

    # For each input, build its piecewise-constant sequence
    for i, input_values in enumerate(values):
        input_values = np.asarray(input_values, dtype=float)
        steps = len(input_values)
        if steps < 1:
            raise ValueError(f"Input {i} must have at least one step value.")

        # Step duration for this input
        step_duration = total_time / steps
        samples_per_step = int(step_duration / dt)

        # Generate step sequence
        step_start = 0
        for step_value in input_values:
            step_end = min(step_start + samples_per_step, time_steps)
            u[step_start:step_end, i] = step_value
            step_start = step_end

        # Fill any remaining samples (e.g., rounding issues)
        if step_start < time_steps:
            u[step_start:, i] = input_values[-1]

    return u



def piecewise_random(
    num_inputs: int,
    total_time: float,
    dt: float,
    value_range: tuple,
    step_resolution: float,
    number_of_steps: int
) -> np.ndarray:
    """
    Create a stochastic piecewise constant input sequence with random values.
    Values are randomly chosen from a discretized range.

    Args:
        num_inputs: Number of input channels
        total_time: Total simulation time
        dt: Time step
        value_range: Tuple (min_val, max_val) for random value generation
        step_resolution: Resolution for discretizing random values
        number_of_steps: Number of distinct steps in the sequence

    Returns:
        np.ndarray: Array of shape (time_steps, num_inputs)

    Example:
        u = piecewise_random(
            num_inputs=2,
            total_time=10.0,
            dt=0.1,
            value_range=(0.0, 5.0),
            step_resolution=0.5,
            number_of_steps=5
        )
    """
    min_val, max_val = value_range

    # Time vector
    time = np.arange(0, total_time, dt)
    time_steps = len(time)

    # Initialize the input sequence
    u = np.zeros((time_steps, num_inputs))

    # Calculate step duration and samples per step
    step_duration = total_time / number_of_steps
    samples_per_step = int(step_duration / dt)

    # Generate discretized values
    possible_values = np.arange(min_val, max_val + step_resolution, step_resolution)

    # Generate piecewise constant sequence
    for i in range(num_inputs):
        step_start = 0
        for step in range(number_of_steps):
            # Random value from discretized set
            value = np.random.choice(possible_values)

            # Calculate end of current step
            step_end = min(step_start + samples_per_step, time_steps)

            # Assign constant value to this step
            u[step_start:step_end, i] = value

            step_start = step_end

        # Fill any remaining samples with the last value
        if step_start < time_steps:
            u[step_start:, i] = u[step_start-1, i]

    return u


def create_piecewise_cnst_seq(
    num_inputs: int,
    total_time: float,
    dt: float,
    min_val: float,
    max_val: float,
    step_resolution: float,
    number_of_steps: int
) -> np.ndarray:
    """
    Create a deterministic piecewise constant input sequence with random values.
    This is an alias/wrapper for piecewise_random with a different parameter interface.

    Args:
        num_inputs: Number of input channels
        total_time: Total simulation time
        dt: Time step
        min_val: Minimum value for random value generation
        max_val: Maximum value for random value generation
        step_resolution: Resolution for discretizing random values
        number_of_steps: Number of distinct steps in the sequence

    Returns:
        np.ndarray: Array of shape (time_steps, num_inputs)
    """
    return piecewise_random(
        num_inputs=num_inputs,
        total_time=total_time,
        dt=dt,
        value_range=(min_val, max_val),
        step_resolution=step_resolution,
        number_of_steps=number_of_steps
    )
