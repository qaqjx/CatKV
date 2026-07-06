#include "compression_runner.h"

#include <ATen/Parallel.h>

#include <mutex>
#include <stdexcept>

#include "tensor.h"

namespace disk_manager {

namespace {

std::once_flag kBackgroundThreadingOnce;

void configure_background_torch_threading() {
    std::call_once(kBackgroundThreadingOnce, []() {
        at::set_num_threads(1);
        try {
            at::set_num_interop_threads(1);
        } catch (...) {
        }
    });
}

DType torch_to_dtype(torch::ScalarType type) {
    switch (type) {
        case torch::kFloat32: return DType::FLOAT32;
        case torch::kFloat16: return DType::FLOAT16;
        case torch::kInt8: return DType::INT8;
        case torch::kUInt8: return DType::UINT8;
        case torch::kBFloat16: return DType::BFLOAT16;
        case torch::kInt32: return DType::INT32;
        case torch::kFloat64: return DType::FLOAT64;
        default:
            throw std::invalid_argument("Unsupported torch dtype supplied.");
    }
}

Tensor torch_to_cpp_tensor(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::invalid_argument("Expected a defined tensor.");
    }
    torch::Tensor normalized = tensor.detach().cpu().contiguous();
    const DType dtype = torch_to_dtype(normalized.scalar_type());

    std::vector<std::size_t> shape;
    shape.reserve(static_cast<std::size_t>(normalized.dim()));
    for (int64_t i = 0; i < normalized.dim(); ++i) {
        shape.push_back(static_cast<std::size_t>(normalized.size(i)));
    }

    auto owner = std::make_shared<torch::Tensor>(std::move(normalized));
    Tensor wrapped(dtype, std::move(shape), owner->data_ptr());
    wrapped.external_owner_ = owner;
    return wrapped;
}

}  // namespace

TensorDict run_single_compression(
    CatKVCompressor& compressor,
    const std::vector<torch::Tensor>& tensors) {
    configure_background_torch_threading();
    return compressor.compress(tensors);
}

std::vector<TensorDict> run_grouped_compression(
    CatKVCompressor& compressor,
    const std::vector<std::vector<torch::Tensor>>& tensor_groups,
    const std::vector<std::string>& uuids) {
    configure_background_torch_threading();
    if (tensor_groups.empty()) {
        return {};
    }
    if (!uuids.empty() && uuids.size() != tensor_groups.size()) {
        throw std::invalid_argument("uuids must match tensor_groups size when provided");
    }

    std::vector<torch::Tensor> key_chunks;
    std::vector<torch::Tensor> value_chunks;
    std::vector<std::pair<int64_t, int64_t>> indices;
    key_chunks.reserve(tensor_groups.size());
    value_chunks.reserve(tensor_groups.size());
    indices.reserve(tensor_groups.size());

    int64_t offset = 0;
    for (const auto& tensors : tensor_groups) {
        if (tensors.size() != 2) {
            throw std::runtime_error("grouped queue item must be [key, value]");
        }
        const int64_t chunk_len = tensors[0].size(1);
        indices.emplace_back(offset, offset + chunk_len);
        offset += chunk_len;
        key_chunks.push_back(tensors[0]);
        value_chunks.push_back(tensors[1]);
    }

    std::vector<torch::Tensor> merged{
        torch::cat(key_chunks, 1).contiguous(),
        torch::cat(value_chunks, 1).contiguous(),
    };

    auto compressed_group = compressor.compress_multi(merged, indices, uuids);
    if (compressed_group.size() != tensor_groups.size()) {
        throw std::runtime_error("compress_multi returned unexpected output size");
    }
    return compressed_group;
}

s3::S3Manager::TensorMap tensor_dict_to_s3_tensor_map(
    const TensorDict& tensor_dict) {
    s3::S3Manager::TensorMap tensor_map;
    tensor_map.reserve(tensor_dict.size());
    for (const auto& [key, tensor] : tensor_dict) {
        tensor_map.emplace(key, torch_to_cpp_tensor(tensor));
    }
    return tensor_map;
}

}  // namespace disk_manager
