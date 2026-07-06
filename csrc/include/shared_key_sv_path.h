#pragma once

#include <string>

#include <torch/extension.h>

namespace catkv_ops {

torch::Tensor UuidStringToTensor(const std::string& uuid);
std::string UuidTensorToPathId(const torch::Tensor& uuid);
std::string SharedKeySvPath(
    const torch::Tensor& uuid,
    int64_t layer_idx,
    const std::string& base_path);
int64_t ParseLayerIdx(const std::string& path);

}  // namespace catkv_ops
