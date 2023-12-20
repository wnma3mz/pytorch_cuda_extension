import torch

M = 4
N = 6

q = torch.rand((M, N))
k = torch.rand((M, N))
v = torch.rand((M, N))

def flash_attention(q, k, v):
    output = torch.zeros(q.shape)

    block_m = 2
    block_n = 2
    block_head = N
    for i in range(0, M, block_m):
        start_i, end_i = i, i + block_m

        old_row_max = torch.zeros([block_m]) - float("inf")
        denominator = torch.zeros([block_m]) # 用于存储分母
        acc = torch.zeros([block_m, block_head])
        q_sub = q[start_i:end_i, :]
        for j in range(0, M, block_n):
            start_j, end_j = j, j + block_n
            k_sub = k[start_j:end_j, :]
            v_sub = v[start_j:end_j, :]
            qk = q_sub @ k_sub.T
            # online softmax
            row_max = torch.max(
                torch.stack((torch.max(qk, dim=1).values, old_row_max), dim=0), dim=0
            ).values
            mod_denominator = denominator * torch.exp(old_row_max - row_max)    # 对之前的分母进行校准
            new_molecule = torch.exp(qk - row_max.reshape(-1, 1))            # 最新的分子
            cur_denominator = torch.sum(new_molecule, -1) + mod_denominator  # 对分母进行更新

            # 分子 / 分母，当前block的softmax
            new_softmax = new_molecule / torch.unsqueeze(cur_denominator, dim=1)
            # 校准系数
            acc *= torch.unsqueeze(mod_denominator / cur_denominator, dim=1)
            acc += new_softmax @ v_sub
            # 更新需要存的值
            old_row_max = row_max
            denominator = cur_denominator

        output[start_i:end_i, :] = acc

    return output


def naive_attention(q, k, v):
    return torch.softmax(q @ k.T, dim=1) @ v


if __name__ == "__main__":
    desired = naive_attention(q, k, v)
    actual = flash_attention(q, k, v)
    print(desired)
    print(actual)
    assert torch.allclose(desired, actual)
