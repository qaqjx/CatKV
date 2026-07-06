#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/extension.h>

#include "tensor_map.h"

namespace catkv_ops {

class CPUMemoryStore {
 public:
  explicit CPUMemoryStore(
      bool pin_memory = true,
      size_t offload_workers = 1);
  ~CPUMemoryStore();

  CPUMemoryStore(const CPUMemoryStore&) = delete;
  CPUMemoryStore& operator=(const CPUMemoryStore&) = delete;

  void Offload(
      const std::string& path,
      const TensorMap& data,
      const std::string& uuid);
  TensorMap Load(const std::string& path, const std::string& device = "cpu") const;
  std::vector<TensorMap> LoadBatch(
      const std::vector<std::string>& paths,
      const std::string& device = "cpu") const;
  void EnableRemoteUpload(
      const std::string& config_path,
      double ratio = 0.2,
      torch::ScalarType dtype = torch::kBFloat16,
      size_t num_workers = 1,
      size_t max_queue_bytes = 0,
      bool skip_remote_save = false);
  TensorMap LoadRemote(const std::string& path, const std::string& device = "cpu") const;
  TensorMap WaitRemote(
      const std::string& path,
      const std::string& device = "cpu",
      double timeout_seconds = -1.0) const;
  void WaitRemoteReady(
      const std::string& path,
      double timeout_seconds = -1.0) const;
  bool RemoteContains(const std::string& path) const;
  void WaitRemoteAll(double timeout_seconds = -1.0) const;
  size_t RemotePendingCount() const;
  size_t RemoteCurrentQueueBytes() const;
  std::unordered_map<std::string, uint64_t> RemoteQueueStats() const;
  void ResetRemoteQueueStats() const;
  void Clear();
  size_t Size() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

torch::Tensor OffloadToCpu(const torch::Tensor& tensor, bool pin_memory = true);

}  // namespace catkv_ops
