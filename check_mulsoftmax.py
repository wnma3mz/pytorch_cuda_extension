import torch

import mulsoftmax
import timeit

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def my_py_softmax(x, dim):
    e = torch.exp(x)
    s = torch.sum(e, dim=dim, keepdim=True)
    return e / s

def py_mulsoft(q, k, v):
    # print(q@k.T)
    return torch.softmax(q @ k.T, dim=1)

def check_forward(q, k, v):
    baseline_values = py_mulsoft(q, k, v)
    cpp_values = mulsoftmax.forward(q, k, v)[0]
    print(torch.all(torch.isclose(baseline_values, cpp_values)))
    cpp_values = mulsoftmax.forward_vector(q, k, v)[0]
    print(torch.all(torch.isclose(baseline_values, cpp_values)))
    cpp_values = mulsoftmax.forward_vector_softmax(q, k, v)[0]
    print(torch.all(torch.isclose(baseline_values, cpp_values)))

    # print("base o", baseline_values)
    # print("cpp  o", cpp_values)
    # print(torch.all(torch.isclose(baseline_values, cpp_values)))

def compare_time(loop=100):
    q, k, v = torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device)
    print("py", timeit.timeit(lambda: py_mulsoft(q, k, v), number=loop))
    print("cpp", timeit.timeit(lambda: mulsoftmax.forward(q, k, v)[0], number=loop))
    print("cpp", timeit.timeit(lambda: mulsoftmax.forward_vector(q, k, v)[0], number=loop))
    print("cpp", timeit.timeit(lambda: mulsoftmax.forward_vector_softmax(q, k, v)[0], number=loop))

if __name__ == "__main__":
    m, n = 16, 40
    device = "cuda"
    q, k, v = torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device)

    # print("q", q)
    # print("k", k)
    # print("v", v)
    # print("="*20)
    # check_forward(q, k, v)
    # q, k, v = torch.rand(size=(m, n)), torch.rand(size=(m, n)), torch.rand(size=(m, n))
    compare_time(10000)