#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "compression_manager.h"
#include "tensor_map.h"

namespace catkv_ops {

class RemoteStorageManager {
 public:
  using MarkUploaded = std::function<void(const CompressedTask&)>;
  using ReportError = std::function<void(
      const CompressedTask&,
      const std::string&,
      uint64_t save_duration_ns)>;

  RemoteStorageManager(
      const std::string& config_path,
      std::size_t worker_count,
      size_t max_queue_bytes,
      bool skip_remote_save,
      MarkUploaded mark_uploaded,
      ReportError report_error);
  ~RemoteStorageManager();

  RemoteStorageManager(const RemoteStorageManager&) = delete;
  RemoteStorageManager& operator=(const RemoteStorageManager&) = delete;

  void Enqueue(CompressedTask task);
  TensorMap Load(const std::string& path, const std::string& device = "cpu") const;
  size_t CurrentQueueBytes() const;
  std::unordered_map<std::string, uint64_t> QueueStats() const;
  void ResetQueueStats();
  void Shutdown();

 private:
  struct QueueStatsData {
    uint64_t save_dispatches = 0;
    uint64_t save_tasks = 0;
    uint64_t total_save_ns = 0;
  };

  void WorkerLoop();

  MarkUploaded mark_uploaded_;
  ReportError report_error_;

  mutable std::mutex mutex_;
  std::condition_variable task_cv_;
  std::condition_variable capacity_cv_;
  bool stopping_ = false;
  size_t max_queue_bytes_ = 0;
  size_t current_queue_bytes_ = 0;
  std::deque<CompressedTask> task_queue_;
  std::vector<std::thread> workers_;

  mutable std::mutex stats_mutex_;
  QueueStatsData queue_stats_;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace catkv_ops
