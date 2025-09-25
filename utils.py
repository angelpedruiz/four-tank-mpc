import numpy as np

def create_piecewise_cnst_seq(num_inputs: int, total_time: float, dt: float, min_val: float, max_val: float, step_resolution: float, number_of_steps: int) -> np.ndarray:
    """
    Create a piecewise constant input sequence. The input changes value every `step_duration` seconds.
    The values are randomly chosen between `min_val` and `max_val`, discretized by step_resolution.
    
    Args:
        num_inputs: Number of input channels
        total_time: Total simulation time
        dt: Time step
        min_val: Minimum value for inputs
        max_val: Maximum value for inputs  
        step_resolution: Resolution for discretizing values
        number_of_steps: Number of distinct steps in the sequence
        
    Returns:
        np.ndarray: Array of shape (num_inputs, time_steps) where each row contains one input's time series
    """
    # Time vector
    time = np.arange(0, total_time, dt)
    time_steps = len(time)
    
    # Calculate step duration and samples per step
    step_duration = total_time / number_of_steps
    samples_per_step = int(step_duration / dt)
    
    # Initialize the input sequence
    u = np.zeros((num_inputs, time_steps))
    
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
            u[i, step_start:step_end] = value
            
            step_start = step_end
            
        # Fill any remaining samples with the last value
        if step_start < time_steps:
            u[i, step_start:] = u[i, step_start-1]
    
    return u


def test_create_piecewise_cnst_seq():
    """Test the create_piecewise_cnst_seq function"""
    
    # Test parameters
    num_inputs = 2
    total_time = 10.0
    dt = 0.1
    min_val = 0.0
    max_val = 5.0
    step_resolution = 0.5
    number_of_steps = 5
    
    # Generate sequence
    u = create_piecewise_cnst_seq(num_inputs, total_time, dt, min_val, max_val, step_resolution, number_of_steps)
    
    # Test shape
    expected_time_steps = int(total_time / dt)
    assert u.shape == (num_inputs, expected_time_steps), f"Expected shape ({num_inputs}, {expected_time_steps}), got {u.shape}"
    
    # Test value ranges
    assert np.all(u >= min_val), "Some values are below minimum"
    assert np.all(u <= max_val), "Some values are above maximum"
    
    # Test discretization (values should be multiples of step_resolution)
    discretized_values = np.round((u - min_val) / step_resolution) * step_resolution + min_val
    assert np.allclose(u, discretized_values), "Values are not properly discretized"
    
    # Test piecewise constant property
    step_duration = total_time / number_of_steps
    samples_per_step = int(step_duration / dt)
    
    for i in range(num_inputs):
        for step in range(number_of_steps - 1):  # Check all but the last step
            start_idx = step * samples_per_step
            end_idx = (step + 1) * samples_per_step
            if end_idx <= u.shape[1]:
                step_values = u[i, start_idx:end_idx]
                assert np.all(step_values == step_values[0]), f"Step {step} for input {i} is not constant"
    
    print("All tests passed!")
    return True


if __name__ == "__main__":
    # Run tests
    test_create_piecewise_cnst_seq()
    
    # Example usage with visualization
    import matplotlib.pyplot as plt
    
    # Generate example sequence
    u_example = create_piecewise_cnst_seq(
        num_inputs=2, 
        total_time=10.0, 
        dt=0.1, 
        min_val=0.0, 
        max_val=5.0, 
        step_resolution=0.5, 
        number_of_steps=5
    )
    
    # Create time vector for plotting
    time = np.arange(0, 10.0, 0.1)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, u_example[0, :], 'b-', linewidth=2, label='Input 1')
    plt.ylabel('Input 1 Value')
    plt.title('Piecewise Constant Input Sequences')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time, u_example[1, :], 'r-', linewidth=2, label='Input 2')
    plt.xlabel('Time [s]')
    plt.ylabel('Input 2 Value')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Generated sequence shape: {u_example.shape}")
    print(f"Input 1 unique values: {np.unique(u_example[0, :])}")
    print(f"Input 2 unique values: {np.unique(u_example[1, :])}")