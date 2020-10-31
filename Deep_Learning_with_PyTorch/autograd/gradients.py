import torch

"""
tonsor y の中の数字が１つのとき:
y.backward()
x.grad

tensor y の中がvectorのとき:
y.backward(v)
x.grad
"""

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
print("z: ", end="")
print(z)
out = z.mean()
print("out: ", end="")
print(out)

out.backward()

# Print gradients d(out)/dx
print(x.grad)

# Now let’s take a look at an example of vector-Jacobian product:
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# 一般的に、torch.autogradはベクトル・ヤコブ積を計算するためのエンジン
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# Or by using .detach() to get a new Tensor with the same content but that does not require gradients:
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
