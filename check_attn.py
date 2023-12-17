import torch

import attention
import timeit

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def my_py_softmax(x, dim):
    e = torch.exp(x)
    s = torch.sum(e, dim=dim, keepdim=True)
    return e / s

def py_attention(q, k, v):
    return torch.softmax(q @ k.T, dim=1) @ v

def check_forward(q, k, v):
    baseline_values = py_attention(q, k, v)
    cpp_values = attention.forward(q, k, v)[-1]

    print("base o", baseline_values)
    print("cpp  o", cpp_values)
    print(torch.all(torch.isclose(baseline_values, cpp_values)))

def compare_time(q, k, v, loop=100):
    print("py", timeit.timeit(lambda: py_attention(q, k, v), number=loop))
    print("cpp", timeit.timeit(lambda: attention.forward(q, k, v), number=loop))

if __name__ == "__main__":
    m, n = 2, 4
    device = "cuda"
    q, k, v = torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device)
    # print("q", q)
    # print("k", k)
    # print("v", v)
    # print("="*20)
    check_forward(q, k, v)
    # q, k, v = torch.rand(size=(m, n)), torch.rand(size=(m, n)), torch.rand(size=(m, n))
    # compare_time(q, k, v)