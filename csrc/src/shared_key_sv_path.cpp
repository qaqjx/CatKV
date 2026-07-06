#include "shared_key_sv_path.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace catkv_ops {

namespace {

int32_t ToSignedInt32(uint32_t value) {
  return static_cast<int32_t>(value);
}

}  // namespace

torch::Tensor UuidStringToTensor(const std::string& uuid) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char byte : uuid) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= 1099511628211ULL;
  }
  const auto high = ToSignedInt32(static_cast<uint32_t>((hash >> 32) & 0xFFFFFFFFULL));
  const auto low = ToSignedInt32(static_cast<uint32_t>(hash & 0xFFFFFFFFULL));
  return torch::tensor(
      {high, low},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
}

std::string UuidTensorToPathId(const torch::Tensor& uuid) {
  TORCH_CHECK(uuid.defined(), "uuid tensor must be defined");
  auto normalized = uuid.detach().cpu().to(torch::kInt32).contiguous().view(-1);
  std::ostringstream output;
  const auto accessor = normalized.accessor<int32_t, 1>();
  for (int64_t idx = 0; idx < normalized.numel(); ++idx) {
    if (idx > 0) {
      output << "_";
    }
    output << static_cast<int32_t>(accessor[idx]);
  }
  return output.str();
}

std::string SharedKeySvPath(
    const torch::Tensor& uuid,
    int64_t layer_idx,
    const std::string& base_path) {
  const auto slash_pos = base_path.find_last_of('/');
  const std::string directory =
      slash_pos == std::string::npos ? std::string{} : base_path.substr(0, slash_pos + 1);
  return directory + UuidTensorToPathId(uuid) + "_layer_" +
         std::to_string(layer_idx) + "_key_sv";
}

int64_t ParseLayerIdx(const std::string& path) {
  const std::string marker = "layer_";
  const auto marker_pos = path.rfind(marker);
  if (marker_pos == std::string::npos) {
    return -1;
  }
  const auto start = marker_pos + marker.size();
  auto end = start;
  while (end < path.size() && path[end] >= '0' && path[end] <= '9') {
    ++end;
  }
  if (end == start) {
    return -1;
  }
  return std::stoll(path.substr(start, end - start));
}

}  // namespace catkv_ops
