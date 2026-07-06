#include "remote_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "compression_manager.h"
#include "remote_storage_manager.h"

namespace catkv_ops {

namespace {

size_t TensorsSizeBytes(const std::vector<torch::Tensor>& tensors) {
  size_t total = 0;
  for (const auto& tensor : tensors) {
    total += static_cast<size_t>(tensor.numel()) *
             static_cast<size_t>(tensor.element_size());
  }
  return total;
}

std::vector<torch::Tensor> NormalizeKV(const TensorMap& data) {
  TORCH_CHECK(!data.empty(), "data must not be empty");
  TORCH_CHECK(data.size() == 2, "remote upload expects exactly two tensors: k/v");

  auto lookup = [&data](const char* short_name, const char* long_name) -> torch::Tensor {
    auto it = data.find(short_name);
    if (it != data.end()) {
      return it->second;
    }
    it = data.find(long_name);
    if (it != data.end()) {
      return it->second;
    }
    TORCH_CHECK(false, "missing tensor entry: expected ", short_name, " or ", long_name);
  };

  auto key = lookup("k", "key");
  auto value = lookup("v", "value");
  return {std::move(key), std::move(value)};
}

std::vector<torch::Tensor> SnapshotCpuKV(const TensorMap& data) {
  auto ordered = NormalizeKV(data);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(ordered.size());
  for (auto& tensor : ordered) {
    TORCH_CHECK(tensor.defined(), "tensor must be defined");
    TORCH_CHECK(tensor.device().type() == c10::kCPU, "remote upload expects CPU tensors");
    auto snapshot = tensor.detach();
    if (!snapshot.is_contiguous()) {
      snapshot = snapshot.contiguous();
    }
    if (snapshot.dim() == 3) {
      snapshot = snapshot.unsqueeze(0).contiguous();
    }
    TORCH_CHECK(
        snapshot.dim() == 4,
        "remote upload expects 3D or 4D CPU tensors, got ",
        snapshot.dim(),
        "D");
    tensors.push_back(std::move(snapshot));
  }
  return tensors;
}

uint64_t ParseUint64Env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  try {
    return static_cast<uint64_t>(std::stoull(value));
  } catch (const std::exception&) {
    return 0;
  }
}

uint64_t ParseGiBEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  try {
    const double gib = std::stod(value);
    if (gib <= 0.0) {
      return 0;
    }
    return static_cast<uint64_t>(gib * 1024.0 * 1024.0 * 1024.0);
  } catch (const std::exception&) {
    return 0;
  }
}

uint64_t ResolveMaxProcessRssBytes() {
  const uint64_t bytes = ParseUint64Env("LMCACHE_CATKV_OPS_MAX_RSS_BYTES");
  if (bytes > 0) {
    return bytes;
  }
  return ParseGiBEnv("LMCACHE_CATKV_OPS_MAX_RSS_GIB");
}

bool QueueTraceStdoutEnabled() {
  const char* value = std::getenv("CATKV_OPS_QUEUE_TRACE_STDOUT");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::string(value) == "1";
}

std::mutex& QueueTraceLogMutex() {
  static std::mutex mutex;
  return mutex;
}

uint64_t CurrentProcessRssBytes() {
  std::ifstream statm("/proc/self/statm");
  uint64_t total_pages = 0;
  uint64_t resident_pages = 0;
  if (!(statm >> total_pages >> resident_pages)) {
    return 0;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return 0;
  }
  return resident_pages * static_cast<uint64_t>(page_size);
}

}  // namespace

struct RemotePipeline::Impl {
  Impl(
      const std::string& config_path,
      double ratio,
      torch::ScalarType dtype,
      std::size_t compress_workers,
      std::size_t save_workers,
      size_t max_raw_queue_bytes,
      size_t max_save_queue_bytes,
      bool skip_remote_save_)
      : skip_remote_save(skip_remote_save_),
        max_process_rss_bytes(ResolveMaxProcessRssBytes()),
        queue_trace_stdout(QueueTraceStdoutEnabled()) {
    if (compress_workers == 0) {
      throw std::invalid_argument("compress_workers must be positive");
    }
    if (save_workers == 0) {
      throw std::invalid_argument("save_workers must be positive");
    }
    storage_manager = std::make_unique<RemoteStorageManager>(
        config_path,
        save_workers,
        max_save_queue_bytes,
        skip_remote_save,
        [this](const CompressedTask& task) { MarkUploaded(task); },
        [this](const CompressedTask& task, const std::string& error, uint64_t) {
          HandleStorageError(task, error);
        });
    compression_manager = std::make_unique<CompressionManager>(
        ratio,
        dtype,
        compress_workers,
        max_raw_queue_bytes,
        [this](CompressedTask task) { EnqueueCompressedTask(std::move(task)); },
        [this](const RawTask& task, const std::string& error) {
          HandleCompressionError(task, error);
        },
        [this](const RawTask& task) { HandleEvictedTask(task); });
  }

  ~Impl() { Shutdown(); }

  void Submit(
      const std::string& path,
      const TensorMap& data,
      const std::string& group_uuid,
      std::function<void()> on_success,
      std::function<void()> on_drop) {
    RawTask task;
    task.cache_key = path;
    task.group_uuid = group_uuid;
    task.tensors = SnapshotCpuKV(data);
    task.size_bytes = TensorsSizeBytes(task.tensors);
    task.on_success = std::move(on_success);

    const uint64_t current_rss_bytes = CurrentProcessRssBytes();
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      max_observed_rss_bytes = std::max(max_observed_rss_bytes, current_rss_bytes);
    }

    {
      std::lock_guard<std::mutex> lock(state_mutex);
      const auto pending_it = pending_versions.find(path);
      const auto uploaded_it = uploaded_versions.find(path);
      const auto current_it = versions.find(path);
      const bool already_pending = pending_it != pending_versions.end();
      const bool already_uploaded =
          uploaded_it != uploaded_versions.end() &&
          current_it != versions.end() &&
          uploaded_it->second == current_it->second;
      if (already_pending || already_uploaded) {
        {
          std::lock_guard<std::mutex> stats_lock(stats_mutex);
          ++dedup_drop_count;
          dropped_duplicate_raw_bytes += task.size_bytes;
        }
        LogQueueTrace("submit_dropped_duplicate", path, task.size_bytes);
        result_cv.notify_all();
        if (on_drop) {
          on_drop();
        }
        return;
      }

      const uint64_t version = versions[path] + 1;
      versions[path] = version;
      uploaded_versions.erase(path);
      errors.erase(path);
      task.version = version;

      if (max_process_rss_bytes > 0 && current_rss_bytes >= max_process_rss_bytes) {
        errors[path] =
            "remote upload skipped: process RSS limit exceeded";
        {
          std::lock_guard<std::mutex> stats_lock(stats_mutex);
          ++rss_drop_count;
          dropped_raw_bytes += task.size_bytes;
        }
        LogQueueTrace("submit_dropped_rss", path, task.size_bytes);
        result_cv.notify_all();
        if (on_drop) {
          on_drop();
        }
        return;
      }

      pending_versions[path] = version;
    }

    compression_manager->Enqueue(std::move(task));
    LogQueueTrace("submit_enqueued", path, task.size_bytes);
  }

  TensorMap Load(const std::string& path, const std::string& device) {
    return storage_manager->Load(path, device);
  }

  TensorMap Wait(const std::string& path, const std::string& device, double timeout_seconds) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto pred = [this, &path]() {
      const bool known = versions.count(path) > 0;
      const bool uploaded = uploaded_versions.count(path) > 0;
      const bool pending = pending_versions.count(path) > 0;
      return errors.count(path) > 0 || stopping || (known && (uploaded || !pending));
    };
    if (timeout_seconds < 0.0) {
      result_cv.wait(lock, pred);
    } else if (!result_cv.wait_for(lock, std::chrono::duration<double>(timeout_seconds), pred)) {
      throw std::runtime_error("Timed out waiting for remote upload");
    }

    auto error_it = errors.find(path);
    if (error_it != errors.end()) {
      throw std::runtime_error(error_it->second);
    }
    auto uploaded_it = uploaded_versions.find(path);
    auto current_it = versions.find(path);
    const bool ready =
        uploaded_it != uploaded_versions.end() && current_it != versions.end() &&
        uploaded_it->second == current_it->second;
    TORCH_CHECK(ready, "remote path is not ready: ", path);
    lock.unlock();

    return Load(path, device);
  }

  void WaitReady(const std::string& path, double timeout_seconds) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto pred = [this, &path]() {
      const bool known = versions.count(path) > 0;
      const bool uploaded = uploaded_versions.count(path) > 0;
      const bool pending = pending_versions.count(path) > 0;
      return errors.count(path) > 0 || stopping || (known && (uploaded || !pending));
    };
    if (timeout_seconds < 0.0) {
      result_cv.wait(lock, pred);
    } else if (!result_cv.wait_for(lock, std::chrono::duration<double>(timeout_seconds), pred)) {
      throw std::runtime_error("Timed out waiting for remote upload readiness");
    }

    auto error_it = errors.find(path);
    if (error_it != errors.end()) {
      throw std::runtime_error(error_it->second);
    }
    auto uploaded_it = uploaded_versions.find(path);
    auto current_it = versions.find(path);
    const bool ready =
        uploaded_it != uploaded_versions.end() && current_it != versions.end() &&
        uploaded_it->second == current_it->second;
    TORCH_CHECK(ready, "remote path is not ready: ", path);
  }

  bool Contains(const std::string& path) const {
    std::lock_guard<std::mutex> lock(state_mutex);
    auto uploaded_it = uploaded_versions.find(path);
    auto current_it = versions.find(path);
    return uploaded_it != uploaded_versions.end() && current_it != versions.end() &&
           uploaded_it->second == current_it->second;
  }

  void WaitAll(double timeout_seconds) const {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto pred = [this]() { return pending_versions.empty() || stopping; };
    if (timeout_seconds < 0.0) {
      result_cv.wait(lock, pred);
    } else if (!result_cv.wait_for(lock, std::chrono::duration<double>(timeout_seconds), pred)) {
      throw std::runtime_error("Timed out waiting for all remote uploads");
    }
  }

  size_t PendingCount() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    return pending_versions.size();
  }

  size_t CurrentQueueBytes() const {
    return compression_manager->CurrentQueueBytes() + storage_manager->CurrentQueueBytes();
  }

  std::unordered_map<std::string, uint64_t> QueueStatsSnapshot() const {
    auto stats = compression_manager->QueueStats();
    for (const auto& [key, value] : storage_manager->QueueStats()) {
      stats[key] = value;
    }
    std::lock_guard<std::mutex> lock(stats_mutex);
    stats["rss_drop_count"] = rss_drop_count;
    stats["dropped_raw_bytes"] = dropped_raw_bytes;
    stats["max_observed_rss_bytes"] = max_observed_rss_bytes;
    stats["dedup_drop_count"] = dedup_drop_count;
    stats["dropped_duplicate_raw_bytes"] = dropped_duplicate_raw_bytes;
    return stats;
  }

  void ResetQueueStatsLocked() {
    compression_manager->ResetQueueStats();
    storage_manager->ResetQueueStats();
    std::lock_guard<std::mutex> lock(stats_mutex);
    rss_drop_count = 0;
    dropped_raw_bytes = 0;
    max_observed_rss_bytes = 0;
    dedup_drop_count = 0;
    dropped_duplicate_raw_bytes = 0;
  }

  void Shutdown() {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      if (stopping) {
        return;
      }
      stopping = true;
    }
    if (compression_manager) {
      compression_manager->Shutdown();
    }
    if (storage_manager) {
      storage_manager->Shutdown();
    }
    result_cv.notify_all();
  }

  void FinalizeSuccessLocked(const CompressedTask& task) {
    auto current_it = versions.find(task.cache_key);
    if (current_it != versions.end() && current_it->second == task.version) {
      uploaded_versions[task.cache_key] = task.version;
      pending_versions.erase(task.cache_key);
      errors.erase(task.cache_key);
    }
  }

  void FinalizeErrorLocked(
      const std::string& cache_key,
      uint64_t version,
      const std::string& error) {
    auto current_it = versions.find(cache_key);
    if (current_it != versions.end() && current_it->second == version) {
      errors[cache_key] = error;
      pending_versions.erase(cache_key);
    }
  }

  void EnqueueCompressedTask(CompressedTask task) {
    const std::string cache_key = task.cache_key;
    const size_t raw_size_bytes = task.raw_size_bytes;
    storage_manager->Enqueue(std::move(task));
    compression_manager->ReleaseBytes(raw_size_bytes);
    LogQueueTrace("compression_released", cache_key, raw_size_bytes);
  }

  void MarkUploaded(const CompressedTask& task) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      FinalizeSuccessLocked(task);
    }
    result_cv.notify_all();
    LogQueueTrace("upload_completed", task.cache_key, task.payload_size_bytes);
    if (task.on_success) {
      task.on_success();
    }
  }

  void HandleCompressionError(const RawTask& task, const std::string& error) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      FinalizeErrorLocked(task.cache_key, task.version, error);
    }
    compression_manager->ReleaseBytes(task.size_bytes);
    result_cv.notify_all();
    LogQueueTrace("compression_failed", task.cache_key, task.size_bytes);
  }

  void HandleStorageError(const CompressedTask& task, const std::string& error) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      FinalizeErrorLocked(task.cache_key, task.version, error);
    }
    result_cv.notify_all();
    LogQueueTrace("storage_failed", task.cache_key, task.payload_size_bytes);
  }

  void HandleEvictedTask(const RawTask& task) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      auto it = pending_versions.find(task.cache_key);
      if (it != pending_versions.end() && it->second == task.version) {
        pending_versions.erase(it);
      }
    }
    result_cv.notify_all();
    LogQueueTrace("queue_evicted", task.cache_key, task.size_bytes);
  }

  void LogQueueTrace(const char* event, const std::string& cache_key, size_t size_bytes) const {
    if (!queue_trace_stdout) {
      return;
    }
    uint64_t pending_count = 0;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      pending_count = pending_versions.size();
    }
    const uint64_t rss_bytes = CurrentProcessRssBytes();
    const uint64_t raw_queue_bytes = compression_manager->CurrentQueueBytes();
    const uint64_t save_queue_bytes = storage_manager->CurrentQueueBytes();
    std::lock_guard<std::mutex> log_lock(QueueTraceLogMutex());
    // std::cout << "[CATKV_OPS_PIPELINE] event=" << event
    //           << " key=" << cache_key
    //           << " size_bytes=" << size_bytes
    //           << " raw_queue_bytes=" << raw_queue_bytes
    //           << " save_queue_bytes=" << save_queue_bytes
    //           << " total_queue_bytes=" << (raw_queue_bytes + save_queue_bytes)
    //           << " pending=" << pending_count
    //           << " rss_bytes=" << rss_bytes
    //           << std::endl;
  }

  bool skip_remote_save = false;
  uint64_t max_process_rss_bytes = 0;
  bool queue_trace_stdout = false;
  std::unique_ptr<CompressionManager> compression_manager;
  std::unique_ptr<RemoteStorageManager> storage_manager;
  mutable std::mutex state_mutex;
  mutable std::condition_variable result_cv;
  mutable std::mutex stats_mutex;
  uint64_t rss_drop_count = 0;
  uint64_t dropped_raw_bytes = 0;
  uint64_t max_observed_rss_bytes = 0;
  uint64_t dedup_drop_count = 0;
  uint64_t dropped_duplicate_raw_bytes = 0;
  bool stopping = false;
  std::unordered_map<std::string, uint64_t> versions;
  std::unordered_map<std::string, uint64_t> pending_versions;
  std::unordered_map<std::string, uint64_t> uploaded_versions;
  std::unordered_map<std::string, std::string> errors;
};

RemotePipeline::RemotePipeline(
    const std::string& config_path,
    double ratio,
    torch::ScalarType dtype,
    std::size_t compress_workers,
    std::size_t save_workers,
    size_t max_raw_queue_bytes,
    size_t max_save_queue_bytes,
    bool skip_remote_save)
    : impl_([&]() {
        return std::make_unique<Impl>(
            config_path,
            ratio,
            dtype,
            compress_workers,
            save_workers,
            max_raw_queue_bytes,
            max_save_queue_bytes,
            skip_remote_save);
      }()) {}

RemotePipeline::~RemotePipeline() = default;

void RemotePipeline::Submit(
    const std::string& path,
    const TensorMap& data,
    const std::string& group_uuid,
    std::function<void()> on_success,
    std::function<void()> on_drop) {
  impl_->Submit(
      path,
      data,
      group_uuid,
      std::move(on_success),
      std::move(on_drop));
}

TensorMap RemotePipeline::Load(const std::string& path, const std::string& device) const {
  return impl_->Load(path, device);
}

TensorMap RemotePipeline::Wait(
    const std::string& path,
    const std::string& device,
    double timeout_seconds) const {
  return impl_->Wait(path, device, timeout_seconds);
}

void RemotePipeline::WaitReady(
    const std::string& path,
    double timeout_seconds) const {
  impl_->WaitReady(path, timeout_seconds);
}

bool RemotePipeline::Contains(const std::string& path) const {
  return impl_->Contains(path);
}

void RemotePipeline::WaitAll(double timeout_seconds) const {
  impl_->WaitAll(timeout_seconds);
}

size_t RemotePipeline::PendingCount() const {
  return impl_->PendingCount();
}

size_t RemotePipeline::CurrentQueueBytes() const {
  return impl_->CurrentQueueBytes();
}

std::unordered_map<std::string, uint64_t> RemotePipeline::QueueStats() const {
  return impl_->QueueStatsSnapshot();
}

void RemotePipeline::ResetQueueStats() const {
  impl_->ResetQueueStatsLocked();
}

}  // namespace catkv_ops
