import torch
import numpy as np

X = torch.rand(4, 4)

def my_softmax(X, dim=1):
    X -= torch.max(X, dim=dim, keepdim=True)[0]
    return torch.exp(X) / torch.sum(torch.exp(X), dim=dim, keepdim=True)

def online_softmax(X):
    value = torch.zeros_like(X)
    for row in range(X.shape[0]):
        row_max = 0.0
        normalizer_term = 0.0
        for col in range(X.shape[1]):
            val = X[row, col]
            old_row_max = row_max
            row_max = max(old_row_max, val)
            normalizer_term = normalizer_term * np.exp(old_row_max - row_max) + np.exp(val - row_max)
        value[row, :] = torch.exp(X[row, :] - row_max) / normalizer_term
    return value


print(torch.softmax(X, dim=1))
print(my_softmax(X, dim=1))
print(online_softmax(X))
assert torch.allclose(torch.softmax(X, dim=1), my_softmax(X, dim=1))
assert torch.allclose(torch.softmax(X, dim=1), online_softmax(X))