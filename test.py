import torch

# positional encoding
feature_dimension = 10
max_spatial_period = 40
even_i = torch.arange(0, feature_dimension, 2).float()   # even indices starting at 0
odd_i = torch.arange(1, feature_dimension, 2).float()    # odd indices starting at 1
denominator = torch.pow(max_spatial_period, even_i / feature_dimension)
positions = torch.arange(max_spatial_period, dtype=torch.float).reshape(max_spatial_period, 1)
even_PE = torch.sin(positions / denominator)
odd_PE =  torch.cos(positions / denominator)
stacked = torch.stack([even_PE, odd_PE], dim=2)
final_PE = torch.flatten(stacked, start_dim=1, end_dim=2)

x = torch.randint(0, 40, (7, 2, 2))
print(x)

print(final_PE)

print(x[:,0,0])

print(final_PE[x[:, 0, 0]])

print(final_PE[x[:, 0, 0].flatten()])