#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void block_matmul_kernel(const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> sub_A1, 
const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> sub_A2, torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < sub_A1.size(0) && j < sub_A2.size(1)) {

        float val = 0.0;
        for (int k = 0; k < N; ++k) {
            val += sub_A1[i * N + k] * sub_A2[k * K + j];
        }
        output[i * K + j] = val;
    }
}

torch::Tensor block_matmul_cuda(torch::Tensor sub_A1, torch::Tensor sub_A2) {
    int M = sub_A1.size(0);
    int N = sub_A1.size(1);
    int K = sub_A2.size(1);

    auto options = torch::TensorOptions().device(sub_A1.device());
    auto output = torch::zeros({M, K}, options);

    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (K + threadsPerBlock.y - 1) / threadsPerBlock.y);

    AT_DISPATCH_FLOATING_TYPES(sub_A1.scalar_type(), "block_matmul_kernel", ([&] {
        block_matmul_kernel<<<blocks, threads>>>(
            sub_A1.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            sub_A2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
    }));
    return output;
}

template <typename scalar_t>
__global__ void tiled_matmul_softmax_kernel(const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> A1, const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> A2, torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Block size
    int block_size_M = 2;
    int block_size_N = 3;
    int block_size_K = 4;

    if (i < A1.size(0) && j < A2.size(1)) {
        // Block size
        int block_M = M / block_size_M;
        int block_N = N / block_size_N;
        int block_K = K / block_size_K;

        float block_output = 0.0;

        for (int k_block = 0; k_block < block_K; ++k_block) {
            int start_k = k_block * block_size_K;
            int end_k = start_k + block_size_K;

            float sub_A1[block_size_M][block_size_K];
            float sub_A2[block_size_K][block_size_N];

            // 计算每个 block 的结果
            for (int ii = 0; ii < block_size_M; ++ii) {
                for (int kk = 0; kk < block_size_K; ++kk) {
                    sub_A1[ii][kk] = A1[i * N + kk + start_k];
                }
            }

            for (int kk = 0; kk < block_size_K; ++kk) {
                for (int jj = 0; jj < block_size_N; ++jj) {
                    sub_A2[kk][jj] = A2[(kk + start_k) * K + j * block_size_N + jj];
                }
            }

            // 拆分每个 block 的结果，为后面的 softmax 计算做准备
            float block_matmul_output[block_size_M][block_size_N];
            for (int ii = 0; ii < block_size_M; ++ii) {
                for (int jj = 0; jj < block_size_N; ++jj) {
                    block_matmul_output[ii][jj] = 0.0;
                }
            }

            for (int ii = 0; ii < block_size_M; ++ii) {
                for (int jj = 0; jj < block_size_N; ++jj) {
                    for (int kk = 0; kk < block_size_K; ++kk) {
                        block_matmul_output[ii][jj] += sub_A1[ii][kk] * sub_A2[kk][jj];
                    }
                }
            }

            for (int ii = 0; ii < block_size_M; ++ii) {
                for (int jj = 0; jj < block_size_N; ++jj) {
                    block_output += block_matmul_output[ii][jj];
                }
            }
        }

        // Softmax calculation
        float row_max = 0.0;
        float normalizer_term = 0.0;

        for (int jj = 0; jj < block_size_N; ++jj) {
            float val = block_output + output[i * K + j];
            float old_row_max = row_max;
            row_max = fmaxf(old_row_max, val);
            normalizer_term = normalizer_term * expf(old_row_max - row_max) + expf(val - row_max);
        }

        output[i * K + j] = expf(block_output - row_max) / normalizer_term;
    }
}


torch::Tensor tiled_matmul_softmax_cuda(torch::Tensor A1, torch::Tensor A2) {
    int M = A1.size(0);
    int N = A1.size(1);
    int K = A2.size(1);


    auto options = torch::TensorOptions().device(input1.device());
    torch::Tensor output = torch::zeros({M, K}, options);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (K + threadsPerBlock.y - 1) / threadsPerBlock.y);

    AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "tiled_matmul_softmax_kernel", ([&] {
        tiled_matmul_softmax_kernel<<<blocks, threads>>>(
            A1.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            A2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
    }));
    return output;
}


std::vector<torch::Tensor> attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    torch::Tensor output = tiled_matmul_softmax_cuda(q, k);
    return {output, output};
    // torch::Tensor scores = matrix_multiply(q, k);
    // torch::Tensor attention = matrix_multiply(torch::softmax(scores, 1), v);
    // return {scores, attention};
}
