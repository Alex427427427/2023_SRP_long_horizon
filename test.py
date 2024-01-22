import numpy as np
import matplotlib.pyplot as plt

def plot_average_with_uncertainty(*lists, confidence_interval=95):
    # Convert lists to numpy arrays
    arrays = [np.array(lst) for lst in lists]

    # Calculate the mean and standard deviation across lists
    mean_values = np.mean(arrays, axis=0)
    std_dev_values = np.std(arrays, axis=0)

    # Calculate confidence interval bounds
    lower_bound = np.percentile(arrays, (100 - confidence_interval) / 2, axis=0)
    upper_bound = np.percentile(arrays, 100 - (100 - confidence_interval) / 2, axis=0)

    # Plot the mean with error bars
    plt.errorbar(range(len(mean_values)), mean_values, yerr=std_dev_values, fmt='o', label='Mean with Std Dev')
    plt.fill_between(range(len(lower_bound)), lower_bound, upper_bound, alpha=0.3, label=f'{confidence_interval}% Confidence Interval')

    # Customize the plot as needed
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Average with Uncertainty')
    plt.legend()
    plt.show()

# Example usage with three lists
list1 = [1, 2, 3, 4, 5]
list2 = [2, 3, 4, 5, 6]
list3 = [3, 4, 5, 6, 7]

plot_average_with_uncertainty(list1, list2, list3)