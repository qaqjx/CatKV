#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <torch/extension.h>

#include "tensor_map.h"

namespace catkv_ops {

class RemotePipeline {
 public:
  RemotePipeline(
      const std::string& config_path,
      double ratio = 0.1,
      torch::ScalarType dtype = torch::kBFloat16,
      std::size_t compress_workers = 32,
      std::size_t save_workers = 32,
      size_t max_raw_queue_bytes = 0,
      size_t max_save_queue_bytes = 0,
      bool skip_remote_save = false);
  ~RemotePipeline();

  RemotePipeline(const RemotePipeline&) = delete;
  RemotePipeline& operator=(const RemotePipeline&) = delete;

  void Submit(
      const std::string& path,
      const TensorMap& data,
      const std::string& group_uuid = "",
      std::function<void()> on_success = {},
      std::function<void()> on_drop = {});

  TensorMap Load(const std::string& path, const std::string& device = "cpu") const;

  TensorMap Wait(
      const std::string& path,
      const std::string& device = "cpu",
      double timeout_seconds = -1.0) const;

  void WaitReady(
      const std::string& path,
      double timeout_seconds = -1.0) const;

  bool Contains(const std::string& path) const;

  void WaitAll(double timeout_seconds = -1.0) const;

  size_t PendingCount() const;

  size_t CurrentQueueBytes() const;

  std::unordered_map<std::string, uint64_t> QueueStats() const;

  void ResetQueueStats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace catkv_ops
