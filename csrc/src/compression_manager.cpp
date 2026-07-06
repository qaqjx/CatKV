#include "compression_manager.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

#include "compression_runner.h"
#include "shared_key_sv_path.h"

namespace catkv_ops {

namespace {

using TensorDict = disk_manager::CatKVCompressor::TensorDict;

constexpr const char* kKeySvQuantized = "key_sv_quantized";
constexpr const char* kKeySvMeta = "key_sv_meta";
constexpr const char* kValueSvQuantized = "value_sv_quantized";
constexpr const char* kValueSvMeta = "value_sv_meta";

std::chrono::milliseconds GroupDispatchWaitDuration() {
  return std::chrono::milliseconds(1);
}

bool IsSplitCompressedTensorDict(const TensorDict& tensor_dict) {
  return tensor_dict.count(kKeySvQuantized) > 0 &&
         tensor_dict.count(kKeySvMeta) > 0 &&
         tensor_dict.count(kValueSvQuantized) > 0 &&
         tensor_dict.count(kValueSvMeta) > 0;
}

TensorDict ExtractTensorSubset(
    TensorDict& tensor_dict,
    std::initializer_list<const char*> keys) {
  TensorDict subset;
  subset.reserve(keys.size());
  for (const char* key : keys) {
    auto it = tensor_dict.find(key);
    if (it != tensor_dict.end()) {
      subset.emplace(it->first, std::move(it->second));
    }
  }
  return subset;
}

size_t TensorDictSizeBytes(const TensorDict& tensor_dict) {
  size_t total = 0;
  for (const auto& [_, tensor] : tensor_dict) {
    if (!tensor.defined()) {
      continue;
    }
    total += static_cast<size_t>(tensor.nbytes());
  }
  return total;
}

std::string DispatchGroupKey(const RawTask& task) {
  if (task.group_uuid.empty()) {
    return "";
  }
  return task.group_uuid + "|layer=" +
         std::to_string(ParseLayerIdx(task.cache_key));
}

CompressedPayload BuildCompressedPayload(TensorDict&& tensor_dict) {
  CompressedPayload payload;
  if (!IsSplitCompressedTensorDict(tensor_dict)) {
    payload.full_payload = std::move(tensor_dict);
    return payload;
  }

  payload.is_split = true;
  torch::Tensor uuid;
  auto uuid_it = tensor_dict.find("uuid");
  if (uuid_it != tensor_dict.end() && uuid_it->second.defined()) {
    uuid = uuid_it->second.detach().cpu().to(torch::kInt32).contiguous().view({-1});
  }
  payload.key_sv_payload = ExtractTensorSubset(
      tensor_dict,
      {kKeySvQuantized, kKeySvMeta, "key_residual_sv"});
  payload.other_payload = ExtractTensorSubset(
      tensor_dict,
      {"u_quantized",
       "u_meta",
       kValueSvQuantized,
       kValueSvMeta,
       "value_residual_sv"});
  if (uuid.defined()) {
    payload.key_sv_payload["uuid"] = uuid.clone();
    payload.other_payload["uuid"] = uuid.clone();
  }
  return payload;
}

size_t PayloadSizeBytes(const CompressedPayload& payload) {
  if (!payload.is_split) {
    return TensorDictSizeBytes(payload.full_payload);
  }
  return TensorDictSizeBytes(payload.key_sv_payload) +
         TensorDictSizeBytes(payload.other_payload);
}

CompressedTask BuildCompressedTask(const RawTask& raw_task, TensorDict&& tensor_dict) {
  CompressedTask compressed_task;
  compressed_task.cache_key = raw_task.cache_key;
  compressed_task.group_uuid = raw_task.group_uuid;
  compressed_task.version = raw_task.version;
  compressed_task.raw_size_bytes = raw_task.size_bytes;
  compressed_task.payload = BuildCompressedPayload(std::move(tensor_dict));
  compressed_task.payload_size_bytes = PayloadSizeBytes(compressed_task.payload);
  compressed_task.on_success = raw_task.on_success;
  return compressed_task;
}

std::vector<std::vector<torch::Tensor>> GatherTensorGroups(
    const std::vector<RawTask>& tasks) {
  std::vector<std::vector<torch::Tensor>> tensor_groups;
  tensor_groups.reserve(tasks.size());
  for (const auto& task : tasks) {
    tensor_groups.push_back(task.tensors);
  }
  return tensor_groups;
}

std::vector<std::string> GatherGroupUuids(const std::vector<RawTask>& tasks) {
  std::vector<std::string> uuids;
  uuids.reserve(tasks.size());
  for (const auto& task : tasks) {
    uuids.push_back(task.group_uuid);
  }
  return uuids;
}

}  // namespace

CompressionManager::CompressionManager(
    double ratio,
    torch::ScalarType dtype,
    std::size_t worker_count,
    size_t max_queue_bytes,
    EmitCompressedTask emit_compressed_task,
    ReportError report_error,
    NotifyEvictedTask notify_evicted_task)
    : ratio_(ratio),
      dtype_(dtype),
      worker_count_(worker_count),
      max_queue_bytes_(max_queue_bytes),
      emit_compressed_task_(std::move(emit_compressed_task)),
      report_error_(std::move(report_error)),
      notify_evicted_task_(std::move(notify_evicted_task)) {
  if (worker_count_ == 0) {
    throw std::invalid_argument("compression worker count must be positive");
  }
  workers_.reserve(worker_count_);
  for (std::size_t idx = 0; idx < worker_count_; ++idx) {
    workers_.emplace_back([this]() { WorkerLoop(); });
  }
}

CompressionManager::~CompressionManager() {
  Shutdown();
}

void CompressionManager::Enqueue(RawTask task) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (max_queue_bytes_ > 0 &&
         current_queue_bytes_ + task.size_bytes > max_queue_bytes_) {
    RawTask evicted_task;
    if (EvictOneQueuedTaskLocked(&evicted_task)) {
      lock.unlock();
      if (notify_evicted_task_) {
        notify_evicted_task_(evicted_task);
      }
      lock.lock();
      continue;
    }
    capacity_cv_.wait(lock, [this, &task]() {
      return stopping_ ||
             current_queue_bytes_ + task.size_bytes <= max_queue_bytes_;
    });
    if (stopping_) {
      throw std::runtime_error("compression manager is shutting down");
    }
  }

  if (stopping_) {
    throw std::runtime_error("compression manager is shutting down");
  }
  current_queue_bytes_ += task.size_bytes;
  task_queue_.push_back(std::move(task));
  lock.unlock();
  task_cv_.notify_one();
}

void CompressionManager::ReleaseBytes(size_t size_bytes) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_queue_bytes_ >= size_bytes) {
      current_queue_bytes_ -= size_bytes;
    } else {
      current_queue_bytes_ = 0;
    }
  }
  capacity_cv_.notify_all();
}

size_t CompressionManager::CurrentQueueBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return current_queue_bytes_;
}

std::unordered_map<std::string, uint64_t> CompressionManager::QueueStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return {
      {"total_dispatches", queue_stats_.total_dispatches},
      {"total_tasks", queue_stats_.total_tasks},
      {"grouped_dispatches", queue_stats_.grouped_dispatches},
      {"standalone_dispatches", queue_stats_.standalone_dispatches},
      {"grouped_tasks", queue_stats_.grouped_tasks},
      {"max_group_size", queue_stats_.max_group_size},
      {"total_compress_ns", queue_stats_.total_compress_ns},
      {"total_save_ns", queue_stats_.total_save_ns},
  };
}

void CompressionManager::ResetQueueStats() {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  queue_stats_ = QueueStatsData{};
}

void CompressionManager::RecordSaveDurationNs(uint64_t save_ns) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  queue_stats_.total_save_ns += save_ns;
}

void CompressionManager::Shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
  }
  task_cv_.notify_all();
  capacity_cv_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
}

bool CompressionManager::HasDispatchableTaskLocked() const {
  for (const auto& task : task_queue_) {
    const std::string group_key = DispatchGroupKey(task);
    if (group_key.empty() ||
        active_group_keys_.count(group_key) == 0) {
      return true;
    }
  }
  return false;
}

std::vector<RawTask> CompressionManager::TakeDispatchableTasksLocked(
    std::unique_lock<std::mutex>& lock) {
  for (auto it = task_queue_.begin(); it != task_queue_.end(); ++it) {
    const std::string group_key = DispatchGroupKey(*it);
    if (!group_key.empty() &&
        active_group_keys_.count(group_key) > 0) {
      continue;
    }

    RawTask task = std::move(*it);
    task_queue_.erase(it);
    if (group_key.empty()) {
      return {std::move(task)};
    }

    active_group_keys_.insert(group_key);
    return CollectGroupTasksWithWaitLocked(std::move(task), group_key, lock);
  }
  return {};
}

std::vector<RawTask> CompressionManager::CollectGroupTasksWithWaitLocked(
    RawTask first_task,
    const std::string& group_key,
    std::unique_lock<std::mutex>& lock) {
  std::vector<RawTask> tasks;
  tasks.push_back(std::move(first_task));
  if (group_key.empty()) {
    return tasks;
  }

  auto collect_ready = [&]() {
    for (auto it = task_queue_.begin(); it != task_queue_.end();) {
      if (DispatchGroupKey(*it) == group_key) {
        tasks.push_back(std::move(*it));
        it = task_queue_.erase(it);
      } else {
        ++it;
      }
    }
  };

  collect_ready();
  const auto wait_duration = GroupDispatchWaitDuration();
  if (wait_duration.count() <= 0) {
    return tasks;
  }

  const auto deadline = std::chrono::steady_clock::now() + wait_duration;
  while (std::chrono::steady_clock::now() < deadline) {
    const bool woke = task_cv_.wait_until(
        lock,
        deadline,
        [this, &group_key]() {
          return stopping_ ||
                 std::any_of(
                     task_queue_.begin(),
                     task_queue_.end(),
                     [&group_key](const RawTask& queued_task) {
                       return DispatchGroupKey(queued_task) == group_key;
                     });
        });
    collect_ready();
    if (!woke || stopping_) {
      break;
    }
  }

  return tasks;
}

bool CompressionManager::EvictOneQueuedTaskLocked(RawTask* evicted_task) {
  if (task_queue_.empty()) {
    return false;
  }
  *evicted_task = std::move(task_queue_.front());
  task_queue_.pop_front();
  if (current_queue_bytes_ >= evicted_task->size_bytes) {
    current_queue_bytes_ -= evicted_task->size_bytes;
  } else {
    current_queue_bytes_ = 0;
  }
  return true;
}

void CompressionManager::RecordDispatchStats(const std::vector<RawTask>& tasks) {
  if (tasks.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(stats_mutex_);
  ++queue_stats_.total_dispatches;
  queue_stats_.total_tasks += static_cast<uint64_t>(tasks.size());
  if (!tasks.front().group_uuid.empty()) {
    ++queue_stats_.grouped_dispatches;
    queue_stats_.grouped_tasks += static_cast<uint64_t>(tasks.size());
    queue_stats_.max_group_size = std::max<uint64_t>(
        queue_stats_.max_group_size,
        static_cast<uint64_t>(tasks.size()));
    return;
  }
  ++queue_stats_.standalone_dispatches;
}

void CompressionManager::RecordCompressDurationNs(uint64_t compress_ns) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  queue_stats_.total_compress_ns += compress_ns;
}

void CompressionManager::WorkerLoop() {
  disk_manager::CatKVCompressor compressor(ratio_, dtype_);
  while (true) {
    std::vector<RawTask> tasks;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      task_cv_.wait(lock, [this]() {
        return stopping_ || HasDispatchableTaskLocked();
      });
      if (stopping_ && task_queue_.empty()) {
        return;
      }
      tasks = TakeDispatchableTasksLocked(lock);
      if (tasks.empty()) {
        continue;
      }
    }

    RecordDispatchStats(tasks);
    const auto compress_started = std::chrono::steady_clock::now();
    try {
      if (tasks.front().group_uuid.empty()) {
        auto compressed =
            disk_manager::run_single_compression(compressor, tasks.front().tensors);
        const auto compress_finished = std::chrono::steady_clock::now();
        RecordCompressDurationNs(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                compress_finished - compress_started)
                .count()));
        emit_compressed_task_(BuildCompressedTask(tasks.front(), std::move(compressed)));
      } else {
        auto compressed_group =
            disk_manager::run_grouped_compression(
                compressor,
                GatherTensorGroups(tasks),
                GatherGroupUuids(tasks));
        const auto compress_finished = std::chrono::steady_clock::now();
        RecordCompressDurationNs(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                compress_finished - compress_started)
                .count()));
        for (std::size_t idx = 0; idx < tasks.size(); ++idx) {
          emit_compressed_task_(
              BuildCompressedTask(tasks[idx], std::move(compressed_group[idx])));
        }
      }
    } catch (const std::exception& exc) {
      for (const auto& task : tasks) {
        report_error_(task, exc.what());
      }
    }

    if (!tasks.empty() && !tasks.front().group_uuid.empty()) {
      std::lock_guard<std::mutex> lock(mutex_);
      active_group_keys_.erase(DispatchGroupKey(tasks.front()));
    }

    task_cv_.notify_all();
  }
}

}  // namespace catkv_ops
