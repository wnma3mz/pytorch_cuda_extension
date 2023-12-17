#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_SIZE 16
#define BLOCK_SIZE_VECTOR 256


// Matrix multiply kernel
template <typename scalar_t>
__global__ void matrix_multiply_kernel(const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input1,
                                       const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input2,
                                       torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < input1.size(0) && col < input2.size(1)) {
        scalar_t value = 0.0;
        for (int k = 0; k < input1.size(1); ++k) {
            value += input1[row][k] * input2[k][col];
        }
        // 使用共享内存，计算每个 row 的最大值
        __shared__ scalar_t row_max[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ scalar_t normalizer_term[BLOCK_SIZE][BLOCK_SIZE];
        row_max[threadIdx.y][threadIdx.x] = value;
        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i /= 2) {
            if (threadIdx.x < i) {
                row_max[threadIdx.y][threadIdx.x] = max(row_max[threadIdx.y][threadIdx.x], row_max[threadIdx.y][threadIdx.x + i]);
            }
            __syncthreads();
        }
        // 计算每个 row 的 softmax 的分母
        normalizer_term[threadIdx.y][threadIdx.x] = exp(value - row_max[threadIdx.y][0]);

        __syncthreads();
        // 计算每个 row  normalizer_term之和
        for (int i = blockDim.x / 2; i > 0; i /= 2) {
            if (threadIdx.x < i) {
                normalizer_term[threadIdx.y][threadIdx.x] += normalizer_term[threadIdx.y][threadIdx.x + i];
            }
            __syncthreads();
        }

        // 计算每个 row 的 softmax
        output[row][col] = exp(value - row_max[threadIdx.y][0]) / normalizer_term[threadIdx.y][0];
    }
}

template <typename scalar_t>
__global__ void softmax_kernel(torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> output,
                                 const int M, const int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index / N;

    if (row < M) {
        float row_max = 0.0;
        float normalizer_term = 0.0;        
        float old_row_max = 0.0;

        for (int i = 0; i < N; ++i) {
            old_row_max = row_max;
            row_max = max(row_max, output[row * N + i]);
            normalizer_term = normalizer_term * exp(old_row_max - row_max) + exp(output[row * N + i] - row_max);
        }

        for (int i = 0; i < N; ++i) {
            output[row * N + i] = exp(output[row * N + i] - row_max) / normalizer_term;
        }
    }
}

template <typename scalar_t>
__global__ void matrix_multiply_vector_kernel(const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input1,
                                       const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input2,
                                       torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> output,
                                       const int M, const int N, const int K
                                       ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index / N;
    int col = index % N;

    if (row < M && col < N) {
        float value = 0.0;
        for (int k = 0; k < K; ++k) {
            value += input1[row * K + k] * input2[k * N + col];
        }
        output[row * N + col] = value;

        // 使用共享内存，计算每个 row 的最大值
        __shared__ scalar_t row_max[BLOCK_SIZE_VECTOR];
        __shared__ scalar_t normalizer_term[BLOCK_SIZE_VECTOR];
        // TODO

        // 计算每个 row 的 softmax
        // output[row * N + col] = exp(value - row_max[threadIdx_y + 0]) / normalizer_term[threadIdx_y + 0];
    }
}

template <typename scalar_t>
__global__ void matrix_multiply_vector_slow_kernel(const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input1,
                                       const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input2,
                                       torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> output,
                                       const int M, const int N, const int K
                                       ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index / N;
    int col = index % N;

    if (row < M && col < N) {
        float value = 0.0;
        for (int k = 0; k < K; ++k) {
            value += input1[row * K + k] * input2[k * N + col];
        }
        output[row * N + col] = value;
        if (col == N - 1) {
            float row_max = 0.0;
            float normalizer_term = 0.0;        
            float old_row_max = 0.0;
            for (int i = 0; i < N; ++i) {
                old_row_max = row_max;
                row_max = max(row_max, output[row * N + i]);
                normalizer_term = normalizer_term * exp(old_row_max - row_max) + exp(output[row * N + i] - row_max);
            }
            for (int i = 0; i < N; ++i) {
                output[row * N + i] = exp(output[row * N + i] - row_max) / normalizer_term;
            }
        }
    }
}

template <typename scalar_t>
__global__ void matrix_multiply_vector_softmax_kernel(const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input1,
                                       const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input2,
                                       torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> output,
                                       const int M, const int N, const int K
                                       ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index / N;
    int col = index % N;

    if (row < M && col < N) {
        float value = 0.0;
        for (int k = 0; k < K; ++k) {
            value += input1[row * K + k] * input2[k * N + col];
        }
        output[row * N + col] = value;
    }
}

torch::Tensor matrix_softmax(torch::Tensor input1, torch::Tensor input2) {
    int rows1 = input1.size(0);
    int cols1 = input1.size(1);
    int cols2 = input2.size(1);

    auto options = torch::TensorOptions().device(input1.device());
    torch::Tensor output = torch::zeros({rows1, cols2}, options);

    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((cols2 + threads.x - 1) / threads.x,
                      (rows1 + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "matrix_multiply_kernel", ([&] {
        matrix_multiply_kernel<<<blocks, threads>>>(
            input1.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            input2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
    }));

    return output;
}

torch::Tensor matrix_softmax_vector(torch::Tensor input1, torch::Tensor input2) {
    int M = input1.size(0);
    int K = input1.size(1);
    int N = input2.size(1);

    auto options = torch::TensorOptions().device(input1.device());
    
    const dim3 threads(BLOCK_SIZE_VECTOR);
    const dim3 blocks((M * N + threads.x - 1) / threads.x);

    // Reshape input tensors to vectors
    auto input1_vector = input1.reshape({-1});
    auto input2_vector = input2.reshape({-1});    
    torch::Tensor output_vector = torch::zeros({M * N}, options);

    AT_DISPATCH_FLOATING_TYPES(input1_vector.scalar_type(), "matrix_multiply_vector_slow_kernel", ([&] {
        matrix_multiply_vector_slow_kernel<<<blocks, threads>>>(
            input1_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            input2_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            output_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            M, N, K
        );
    }));
    return output_vector.reshape({M, N});
}

torch::Tensor matrix_softmax_vector_softmax(torch::Tensor input1, torch::Tensor input2) {
    int M = input1.size(0);
    int K = input1.size(1);
    int N = input2.size(1);

    auto options = torch::TensorOptions().device(input1.device());
    
    const dim3 threads(BLOCK_SIZE_VECTOR);
    const dim3 blocks((M * N + threads.x - 1) / threads.x);

    // Reshape input tensors to vectors
    auto input1_vector = input1.reshape({-1});
    auto input2_vector = input2.reshape({-1});    
    torch::Tensor output_vector = torch::zeros({M * N}, options);

    AT_DISPATCH_FLOATING_TYPES(input1_vector.scalar_type(), "matrix_multiply_vector_softmax_kernel", ([&] {
        matrix_multiply_vector_softmax_kernel<<<blocks, threads>>>(
            input1_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            input2_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            output_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            M, N, K
        );
    }));

    cudaDeviceSynchronize();

    AT_DISPATCH_FLOATING_TYPES(output_vector.scalar_type(), "softmax_kernel", ([&] {
        softmax_kernel<<<blocks, threads>>>(
            output_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            M, N
        );
    }));
    return output_vector.reshape({M, N});

}
std::vector<torch::Tensor> matrix_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    torch::Tensor scores = matrix_softmax(q, k);
    return {scores};
}

std::vector<torch::Tensor> matrix_cuda_forward_vector(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    torch::Tensor scores = matrix_softmax_vector(q, k);
    return {scores};
}

std::vector<torch::Tensor> matrix_cuda_forward_vector_softmax(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    torch::Tensor scores = matrix_softmax_vector_softmax(q, k);
    return {scores};
}
