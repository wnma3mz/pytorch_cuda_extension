#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

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
        output[row][col] = value;
    }
}

torch::Tensor matrix_multiply(torch::Tensor input1, torch::Tensor input2) {
    int rows1 = input1.size(0);
    int cols1 = input1.size(1);
    int cols2 = input2.size(1);

    auto options = torch::TensorOptions().device(input1.device());
    torch::Tensor output = torch::zeros({rows1, cols2}, options);

    const dim3 threads(16, 16);
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


std::vector<torch::Tensor> attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    torch::Tensor scores = matrix_multiply(q, k);
    torch::Tensor attention = matrix_multiply(torch::softmax(scores, 1), v);
    return {scores, attention};
}
