# plot an increasing line with 100000 points, but with a large variance
# in the y direction
import matplotlib.pyplot as plt
import numpy as np
import random



import torch

def average_filter(input_list, window_size=3):
    # Create a tensor from the input list
    input_tensor = torch.tensor(input_list, dtype=torch.float32)

    # Use convolution to perform the average filter
    kernel = torch.ones(window_size) / window_size
    result_tensor = torch.nn.functional.conv1d(input_tensor.view(1, 1, -1), kernel.view(1, 1, -1), padding=window_size // 2)[0, 0, :]

    return result_tensor.numpy()



x = np.linspace(0, 10, 100000)
y = np.zeros(100000)
for i in range(100000):
    y[i] = random.random() * 1000 + i / 1000

y_filtered = average_filter(y, 1000)

plt.plot(x, y, "k-", alpha=0.1)
plt.show()

plt.plot(x, y_filtered[:-1], "ko", alpha=0.1)
plt.show()