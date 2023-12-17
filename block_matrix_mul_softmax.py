import torch
import numpy as np

M, N, K = 4, 6, 8
A1, A2 = torch.rand(size=(M, N)), torch.rand(size=(N, K))

def block_matmul(sub_A1, sub_A2):
    output = torch.zeros(size=(sub_A1.shape[0], sub_A2.shape[1]))
    for i in range(sub_A1.shape[0]):
        for j in range(sub_A2.shape[1]):
            for k in range(sub_A2.shape[0]):
                output[i][j] += sub_A1[i][k] * sub_A2[k][j]              
    return output


def tiled_matmul_softmax(A1, A2):
    block_size_M, block_size_N, block_size_K = 2, 3, 4
    block_M, block_N, block_K = M // block_size_M, N // block_size_N, K // block_size_K

    output = torch.zeros(size=(A1.shape[0], A2.shape[1]))
    for i in range(0, A1.shape[0], block_M):
        start_i, end_i = i, i + block_M
        row_max = torch.tensor([[0. for _ in range(block_N)] for _ in range(block_M)])
        old_row_max = torch.tensor([[0. for _ in range(block_N)] for _ in range(block_M)])
        normalizer_term = torch.tensor([[0. for _ in range(block_N)] for _ in range(block_M)])

        for j in range(0, A2.shape[1], block_N):
            start_j, end_j = j, j + block_N
            for k in range(0, A2.shape[0], block_K):
                start_k, end_k = k, k + block_K
                sub_A1 = A1[start_i:end_i, start_k:end_k]
                sub_A2 = A2[start_k:end_k, start_j:end_j]
                output[start_i:end_i, start_j:end_j] += block_matmul(sub_A1, sub_A2)

            # 这里算完了每个block的结果，所以需要将其拆分成每个block，然后再计算softmax
            for ii, row in enumerate(range(start_i, end_i)):              
                for jj, col in enumerate(range(start_j, end_j)):
                    val = output[row][col]
                    old_row_max[ii][jj] = row_max[ii][jj]
                    row_max[ii][jj] = max(old_row_max[ii][jj], val)
                    normalizer_term[ii][jj] = normalizer_term[ii][jj] * np.exp(old_row_max[ii][jj] - row_max[ii][jj]) + np.exp(val - row_max[ii][jj])

        for ii, row in enumerate(range(start_i, end_i)):
            row_max_v, _ = torch.max(row_max, dim=1)
            # 重算 sum, 代入公式 old_v*exp(old_max - new_max)
            sum_ = torch.sum(normalizer_term[ii] * torch.exp(row_max[ii] - row_max_v[ii]))
            output[row, :] = torch.exp(output[row, :] - row_max_v[ii]) / sum_
    return output

def matmul_softmax(A1, A2):
    output = torch.zeros(size=(A1.shape[0], A2.shape[1]))
    for i in range(A1.shape[0]):
        row_max = 0.0
        normalizer_term = 0.0    
        for j in range(A2.shape[1]):
            val = output[i, j] = sum(map(lambda x: x[0] * x[1], zip(A1[i], A2[:, j])))
            
            old_row_max = row_max
            row_max = max(old_row_max, val)
            normalizer_term = normalizer_term * np.exp(old_row_max - row_max) + np.exp(val - row_max)
        output[i, :] = torch.exp(output[i, :] - row_max) / normalizer_term

    return output

print(torch.softmax(A1 @ A2, dim=1))
print(matmul_softmax(A1, A2))
print(tiled_matmul_softmax(A1, A2))
# assert torch.allclose(torch.softmax(A1 @ A2, dim=1), matmul_softmax(A1, A2))
assert torch.allclose(torch.softmax(A1 @ A2, dim=1), tiled_matmul_softmax(A1, A2))
