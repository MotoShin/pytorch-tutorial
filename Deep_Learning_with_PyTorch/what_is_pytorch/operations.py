from __future__ import print_function
import torch

# Addition: syntax 1
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)

# Addition: syntax 2
print(torch.add(x, y))

# Addition: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition: in-place
y.add_(x)
print(y)

# You can use standard NumPy-like indexing with all bells and whistles!
print(x[:, 1])

# Resizing: If you want to resize/reshape tensor, you can use torch.view:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())
