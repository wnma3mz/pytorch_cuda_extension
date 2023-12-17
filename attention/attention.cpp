#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> attention_cuda_forward(
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

std::vector<torch::Tensor> attention_cpu_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {
    torch::Tensor scores = my_matmul(q, k);
    torch::Tensor attention = my_matmul(torch::softmax(scores, 1), v);
    return {scores, attention};
}

// 参数：queries(Q)，keys(K)，values(V)
std::vector<torch::Tensor> attention_forward(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v) {
    if (!(q.device().type() == k.device().type() && q.device().type() == v.device().type())) {
        throw std::runtime_error("Input tensors q, k, and v must be on the same device");
    }

    if (q.is_cuda()) {
        return attention_cuda_forward(q, k.transpose(0, 1), v);
    } else {
        return attention_cpu_forward(q, k.transpose(0, 1), v);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Attention forward (CUDA)");
}