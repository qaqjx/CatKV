#pragma once

#include <string>
#include <unordered_map>

#include <torch/extension.h>

namespace catkv_ops {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

}  // namespace catkv_ops
