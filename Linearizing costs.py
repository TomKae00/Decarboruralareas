import numpy as np
import matplotlib.pyplot as plt
from cost_functions import power_law_models

# Define the cost function
def cost_function(x):
    return x ** (-0.098) * 950.36


def correct_constant_cost_approx_with_dynamic_segments(cost_function, upper_limit, error_threshold):
    def calculate_average_error_for_intervals(num_intervals):
        log_intervals = np.linspace(np.log(5), np.log(upper_limit), num_intervals + 1)
        intervals = np.exp(log_intervals)

        constant_costs = []
        all_errors = []
        for i in range(len(intervals) - 1):
            start, end = intervals[i], intervals[i + 1]
            x_interval = np.linspace(start, end, 100)
            y_interval = cost_function(x_interval)
            average_cost = np.mean(y_interval)
            constant_costs.append(average_cost)
            errors = np.abs(y_interval - average_cost)
            all_errors.extend(errors)

        return np.mean(all_errors), intervals, constant_costs

    # Initial setup
    num_intervals = 5  # Start with a minimum number of intervals
    while True:
        average_error, intervals, constant_costs = calculate_average_error_for_intervals(num_intervals)
        if average_error <= error_threshold:
            break
        num_intervals += 1

    # Plotting
    x_range = np.linspace(1, upper_limit, 1000)
    y_range = cost_function(x_range)
    plt.figure(figsize=(12, 8))
    plt.plot(x_range, y_range, label='Original Function', color='blue', linewidth=2)

    for i in range(num_intervals):
        start = intervals[i]
        if i < num_intervals - 1:
            end = intervals[i + 1]
        else:
            end = upper_limit
        plt.hlines(constant_costs[i], start, end, colors='red', linestyles='dashed', linewidth=2,
                   label='Constant Cost Approximation' if i == 0 else "")

    plt.xlabel('Capacity (kW)')
    plt.ylabel('Specific Cost ($/kW)')
    plt.title(
        f'Cost Function vs. Constant Cost Approximation\n{num_intervals} Segments, Avg Error: {average_error:.2f}')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    return num_intervals, average_error, intervals, constant_costs


# Re-run the corrected function
num_intervals_final, average_error_final, intervals_final, constant_costs_final = correct_constant_cost_approx_with_dynamic_segments(
    cost_function, upper_limit=20000, error_threshold=5
)

# Display the final number of segments and the average error
print(num_intervals_final, average_error_final)
