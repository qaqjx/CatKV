#pragma once

#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/extension.h>

#include "compressor.h"

namespace catkv_ops {

struct RawTask {
  std::string cache_key;
  std::string group_uuid;
  std::vector<torch::Tensor> tensors;
  uint64_t version = 0;
  size_t size_bytes = 0;
  std::function<void()> on_success;
};

struct CompressedPayload {
  bool is_split = false;
  disk_manager::CatKVCompressor::TensorDict full_payload;
  disk_manager::CatKVCompressor::TensorDict key_sv_payload;
  disk_manager::CatKVCompressor::TensorDict other_payload;
};

struct CompressedTask {
  std::string cache_key;
  std::string group_uuid;
  uint64_t version = 0;
  size_t raw_size_bytes = 0;
  size_t payload_size_bytes = 0;
  CompressedPayload payload;
  std::function<void()> on_success;
};

class CompressionManager {
 public:
  using EmitCompressedTask = std::function<void(CompressedTask)>;
  using ReportError = std::function<void(const RawTask&, const std::string&)>;
  using NotifyEvictedTask = std::function<void(const RawTask&)>;

  CompressionManager(
      double ratio,
      torch::ScalarType dtype,
      std::size_t worker_count,
      size_t max_queue_bytes,
      EmitCompressedTask emit_compressed_task,
      ReportError report_error,
      NotifyEvictedTask notify_evicted_task = {});
  ~CompressionManager();

  CompressionManager(const CompressionManager&) = delete;
  CompressionManager& operator=(const CompressionManager&) = delete;

  void Enqueue(RawTask task);
  void ReleaseBytes(size_t size_bytes);
  size_t CurrentQueueBytes() const;
  std::unordered_map<std::string, uint64_t> QueueStats() const;
  void ResetQueueStats();
  void RecordSaveDurationNs(uint64_t save_ns);
  void Shutdown();

 private:
  struct QueueStatsData {
    uint64_t total_dispatches = 0;
    uint64_t total_tasks = 0;
    uint64_t grouped_dispatches = 0;
    uint64_t standalone_dispatches = 0;
    uint64_t grouped_tasks = 0;
    uint64_t max_group_size = 0;
    uint64_t total_compress_ns = 0;
    uint64_t total_save_ns = 0;
  };

  bool HasDispatchableTaskLocked() const;
  std::vector<RawTask> TakeDispatchableTasksLocked(std::unique_lock<std::mutex>& lock);
  std::vector<RawTask> CollectGroupTasksWithWaitLocked(
      RawTask first_task,
      const std::string& group_key,
      std::unique_lock<std::mutex>& lock);
  bool EvictOneQueuedTaskLocked(RawTask* evicted_task);
  void RecordDispatchStats(const std::vector<RawTask>& tasks);
  void RecordCompressDurationNs(uint64_t compress_ns);
  void WorkerLoop();

  double ratio_;
  torch::ScalarType dtype_;
  std::size_t worker_count_;
  size_t max_queue_bytes_;
  EmitCompressedTask emit_compressed_task_;
  ReportError report_error_;
  NotifyEvictedTask notify_evicted_task_;

  mutable std::mutex mutex_;
  std::condition_variable task_cv_;
  std::condition_variable capacity_cv_;
  bool stopping_ = false;
  size_t current_queue_bytes_ = 0;
  std::deque<RawTask> task_queue_;
  std::unordered_set<std::string> active_group_keys_;
  std::vector<std::thread> workers_;

  mutable std::mutex stats_mutex_;
  QueueStatsData queue_stats_;
};

}  // namespace catkv_ops
