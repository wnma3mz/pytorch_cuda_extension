import torch

M, N, K = 4, 6, 8

A1 = torch.rand(size=(M, N))
A2 = torch.rand(size=(N, K))

block_size_M, block_size_N, block_size_K = 2, 3, 4
block_M, block_N, block_K = M // block_size_M, N // block_size_N, K // block_size_K

def block_matmul(sub_A1, sub_A2):
    output = torch.zeros(size=(sub_A1.shape[0], sub_A2.shape[1]))
    for i in range(sub_A1.shape[0]):
        for j in range(sub_A2.shape[1]):
            for k in range(sub_A2.shape[0]):
                output[i][j] += sub_A1[i][k] * sub_A2[k][j]
    return output

def matmul(A1, A2):
    output = torch.zeros(size=(A1.shape[0], A2.shape[1]))
    for i in range(0, A1.shape[0], block_M):
        start_i, end_i = i, i + block_M
        for j in range(0, A2.shape[1], block_N):
            start_j, end_j = j, j + block_N
            for k in range(0, A2.shape[0], block_K):
                start_k, end_k = k, k + block_K
                # 计算每个 block 的矩阵乘法
                sub_A1 = A1[start_i:end_i, start_k:end_k]
                sub_A2 = A2[start_k:end_k, start_j:end_j]
                # 把每个 block 的结果放到对应的位置
                output[start_i:end_i, start_j:end_j] += block_matmul(sub_A1, sub_A2)
    return output
print(matmul(A1, A2))
print(A1 @ A2)
assert torch.allclose(matmul(A1, A2), A1 @ A2)