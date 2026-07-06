#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "compressor.h"
#include "s3_manager.h"

namespace disk_manager {

using TensorDict = CatKVCompressor::TensorDict;

TensorDict run_single_compression(
    CatKVCompressor& compressor,
    const std::vector<torch::Tensor>& tensors);

std::vector<TensorDict> run_grouped_compression(
    CatKVCompressor& compressor,
    const std::vector<std::vector<torch::Tensor>>& tensor_groups,
    const std::vector<std::string>& uuids = {});

s3::S3Manager::TensorMap tensor_dict_to_s3_tensor_map(
    const TensorDict& tensor_dict);

}  // namespace disk_manager
