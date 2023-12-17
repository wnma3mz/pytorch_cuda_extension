#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> matrix_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);

std::vector<torch::Tensor> matrix_cuda_forward_vector(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);

std::vector<torch::Tensor> matrix_cuda_forward_vector_softmax(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);

torch::Tensor my_matmul(const torch::Tensor &a, const torch::Tensor &b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Input tensors must be 2-dimensional");
    TORCH_CHECK(a.size(1) == b.size(0), "Dimensions mismatch");

    auto m = a.size(0);
    auto n = b.size(1);
    auto p = a.size(1);

    torch::Tensor result = torch::zeros({m, n}, torch::dtype(torch::kFloat32));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0;
            for (int k = 0; k < p; k++) {
                sum += a[i][k].item<float>() * b[k][j].item<float>();
            }
            result[i][j] = sum;
        }
    }
    return result;
}

std::vector<torch::Tensor> matrix_cpu_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {
    torch::Tensor scores = my_matmul(q, k);
    torch::Tensor attention = my_matmul(torch::softmax(scores, 1), v);
    return {scores, attention};
}

// 参数：queries(Q)，keys(K)，values(V)
std::vector<torch::Tensor> matrix_forward(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v) {
    if (!(q.device().type() == k.device().type() && q.device().type() == v.device().type())) {
        throw std::runtime_error("Input tensors q, k, and v must be on the same device");
    }

    if (q.is_cuda()) {
        return matrix_cuda_forward(q, k.transpose(0, 1), v);
    } else {
        return matrix_cpu_forward(q, k.transpose(0, 1), v);
    }
}

std::vector<torch::Tensor> matrix_forward_vector(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v) {
    if (!(q.device().type() == k.device().type() && q.device().type() == v.device().type())) {
        throw std::runtime_error("Input tensors q, k, and v must be on the same device");
    }

    if (q.is_cuda()) {
        return matrix_cuda_forward_vector(q, k.transpose(0, 1), v);
    } else {
        return {torch::softmax(my_matmul(q, k.transpose(0, 1)), 1), my_matmul(torch::softmax(my_matmul(q, k.transpose(0, 1)), 1), v)};
    }
}

std::vector<torch::Tensor> matrix_forward_vector_softmax(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v) {
    if (!(q.device().type() == k.device().type() && q.device().type() == v.device().type())) {
        throw std::runtime_error("Input tensors q, k, and v must be on the same device");
    }

    if (q.is_cuda()) {
        return matrix_cuda_forward_vector_softmax(q, k.transpose(0, 1), v);
    } else {
        return {torch::softmax(my_matmul(q, k.transpose(0, 1)), 1), my_matmul(torch::softmax(my_matmul(q, k.transpose(0, 1)), 1), v)};
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matrix_forward, "Attention forward (CUDA)");
    m.def("forward_vector", &matrix_forward_vector, "Attention forward (CUDA)");
    m.def("forward_vector_softmax", &matrix_forward_vector_softmax, "Attention forward (CUDA)");
}