#include "remote_storage_manager.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compression_runner.h"
#include "s3_manager.h"
#include "shared_key_sv_path.h"
#include "tensor.h"

namespace catkv_ops {

namespace {

constexpr const char* kKeySvSuffix = "_key_sv";
constexpr const char* kOtherSuffix = "_other";

std::string KeySvPath(const std::string& path) {
  return path + kKeySvSuffix;
}

std::string OtherPath(const std::string& path) {
  return path + kOtherSuffix;
}

torch::ScalarType DTypeToTorch(disk_manager::DType dtype) {
  switch (dtype) {
    case disk_manager::DType::FLOAT32:
      return torch::kFloat32;
    case disk_manager::DType::FLOAT16:
      return torch::kFloat16;
    case disk_manager::DType::INT8:
      return torch::kInt8;
    case disk_manager::DType::UINT8:
      return torch::kUInt8;
    case disk_manager::DType::BFLOAT16:
      return torch::kBFloat16;
    case disk_manager::DType::INT32:
      return torch::kInt32;
    case disk_manager::DType::FLOAT64:
      return torch::kFloat64;
  }
  throw std::runtime_error("Unsupported DType");
}

torch::Tensor CppToTensor(const disk_manager::Tensor& tensor) {
  const torch::ScalarType scalar_type = DTypeToTorch(tensor.dtype);
  std::vector<int64_t> sizes;
  sizes.reserve(tensor.shape.size());
  for (std::size_t dim : tensor.shape) {
    sizes.push_back(static_cast<int64_t>(dim));
  }
  auto options = torch::TensorOptions().dtype(scalar_type).device(torch::kCPU);
  if (tensor.data_size == 0 || !tensor.data_ptr()) {
    return torch::empty(sizes, options);
  }
  if (tensor.external_owner_) {
    std::shared_ptr<void> owner = tensor.external_owner_;
    return torch::from_blob(
        const_cast<void*>(static_cast<const void*>(tensor.data_ptr())),
        torch::IntArrayRef(sizes),
        [owner](void*) {},
        options);
  }
  auto view = torch::from_blob(
      const_cast<void*>(static_cast<const void*>(tensor.data_ptr())),
      torch::IntArrayRef(sizes),
      [](void*) {},
      options);
  return view.clone();
}

uint64_t ToNanoseconds(std::chrono::steady_clock::duration duration) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

void SaveCompressedPayload(
    s3::S3Manager& storage,
    const std::string& path,
    const CompressedPayload& payload,
    std::unordered_set<std::string>& saved_shared_key_sv_paths,
    std::mutex& saved_shared_key_sv_mutex) {
  if (!payload.is_split) {
    storage.save(path, disk_manager::tensor_dict_to_s3_tensor_map(payload.full_payload));
    return;
  }

  std::string key_sv_path = KeySvPath(path);
  bool use_shared_key_sv = false;
  const int64_t layer_idx = ParseLayerIdx(path);
  const auto uuid_it = payload.key_sv_payload.find("uuid");
  if (layer_idx > 2 && uuid_it != payload.key_sv_payload.end()) {
    key_sv_path = SharedKeySvPath(uuid_it->second, layer_idx, path);
    use_shared_key_sv = true;
  }

  bool should_save_key_sv = true;
  if (use_shared_key_sv) {
    std::lock_guard<std::mutex> lock(saved_shared_key_sv_mutex);
    should_save_key_sv = saved_shared_key_sv_paths.insert(key_sv_path).second;
  }

  if (should_save_key_sv) {
    try {
      storage.save(
          key_sv_path,
          disk_manager::tensor_dict_to_s3_tensor_map(payload.key_sv_payload));
    } catch (...) {
      if (use_shared_key_sv) {
        std::lock_guard<std::mutex> lock(saved_shared_key_sv_mutex);
        saved_shared_key_sv_paths.erase(key_sv_path);
      }
      throw;
    }
  }
  storage.save(
      OtherPath(path),
      disk_manager::tensor_dict_to_s3_tensor_map(payload.other_payload));
}

TensorMap LoadTensorMapFromS3(
    s3::S3Manager& storage,
    const std::string& path,
    const std::string& device) {
  auto loaded = storage.load(path);
  TORCH_CHECK(loaded.has_value(), "remote path not found: ", path);

  TensorMap result;
  result.reserve(loaded->size());
  for (const auto& [name, tensor] : *loaded) {
    auto torch_tensor = CppToTensor(tensor);
    if (device != "cpu") {
      torch_tensor = torch_tensor.to(torch::Device(device));
    }
    result.emplace(name, std::move(torch_tensor));
  }
  return result;
}

}  // namespace

struct RemoteStorageManager::Impl {
  explicit Impl(
      const std::string& config_path,
      std::size_t worker_count,
      bool skip_remote_save)
      : storage(
            config_path,
            worker_count == 0 ? 1 : worker_count,
            0,
            std::max<std::size_t>(worker_count, 16)),
        skip_remote_save(skip_remote_save) {}

  s3::S3Manager storage;
  bool skip_remote_save = false;
  std::unordered_set<std::string> saved_shared_key_sv_paths;
  std::mutex saved_shared_key_sv_mutex;
};

RemoteStorageManager::RemoteStorageManager(
    const std::string& config_path,
    std::size_t worker_count,
    size_t max_queue_bytes,
    bool skip_remote_save,
    MarkUploaded mark_uploaded,
    ReportError report_error)
    : mark_uploaded_(std::move(mark_uploaded)),
      report_error_(std::move(report_error)),
      max_queue_bytes_(max_queue_bytes),
      impl_(std::make_unique<Impl>(config_path, worker_count, skip_remote_save)) {
  if (worker_count == 0) {
    throw std::invalid_argument("save worker count must be positive");
  }
  workers_.reserve(worker_count);
  for (std::size_t idx = 0; idx < worker_count; ++idx) {
    workers_.emplace_back([this]() { WorkerLoop(); });
  }
}

RemoteStorageManager::~RemoteStorageManager() {
  Shutdown();
}

void RemoteStorageManager::Enqueue(CompressedTask task) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (max_queue_bytes_ > 0 &&
         current_queue_bytes_ + task.payload_size_bytes > max_queue_bytes_) {
    capacity_cv_.wait(lock, [this, &task]() {
      return stopping_ ||
             current_queue_bytes_ + task.payload_size_bytes <= max_queue_bytes_;
    });
    if (stopping_) {
      throw std::runtime_error("remote storage manager is shutting down");
    }
  }

  if (stopping_) {
    throw std::runtime_error("remote storage manager is shutting down");
  }

  current_queue_bytes_ += task.payload_size_bytes;
  task_queue_.push_back(std::move(task));
  lock.unlock();
  task_cv_.notify_one();
}

TensorMap RemoteStorageManager::Load(const std::string& path, const std::string& device) const {
  try {
    return LoadTensorMapFromS3(impl_->storage, path, device);
  } catch (const c10::Error&) {
  } catch (const std::runtime_error&) {
  }

  TensorMap other_payload = LoadTensorMapFromS3(impl_->storage, OtherPath(path), device);
  std::string key_sv_path = KeySvPath(path);
  const int64_t layer_idx = ParseLayerIdx(path);
  auto uuid_it = other_payload.find("uuid");
  if (layer_idx > 2 && uuid_it != other_payload.end()) {
    key_sv_path = SharedKeySvPath(uuid_it->second, layer_idx, path);
  }
  TensorMap key_sv_payload = LoadTensorMapFromS3(impl_->storage, key_sv_path, device);
  other_payload.insert(key_sv_payload.begin(), key_sv_payload.end());
  return other_payload;
}

size_t RemoteStorageManager::CurrentQueueBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return current_queue_bytes_;
}

std::unordered_map<std::string, uint64_t> RemoteStorageManager::QueueStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return {
      {"save_dispatches", queue_stats_.save_dispatches},
      {"save_tasks", queue_stats_.save_tasks},
      {"save_queue_bytes", static_cast<uint64_t>(CurrentQueueBytes())},
      {"total_save_ns", queue_stats_.total_save_ns},
  };
}

void RemoteStorageManager::ResetQueueStats() {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  queue_stats_ = QueueStatsData{};
}

void RemoteStorageManager::Shutdown() {
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

void RemoteStorageManager::WorkerLoop() {
  while (true) {
    CompressedTask task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      task_cv_.wait(lock, [this]() { return stopping_ || !task_queue_.empty(); });
      if (task_queue_.empty()) {
        if (stopping_) {
          return;
        }
        continue;
      }
      task = std::move(task_queue_.front());
      task_queue_.pop_front();
    }

    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      ++queue_stats_.save_dispatches;
      ++queue_stats_.save_tasks;
    }

    const auto save_started = std::chrono::steady_clock::now();
    try {
      if (!impl_->skip_remote_save) {
        SaveCompressedPayload(
            impl_->storage,
            task.cache_key,
            task.payload,
            impl_->saved_shared_key_sv_paths,
            impl_->saved_shared_key_sv_mutex);
      }
      const auto save_finished = std::chrono::steady_clock::now();
      {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        queue_stats_.total_save_ns += ToNanoseconds(save_finished - save_started);
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_queue_bytes_ >= task.payload_size_bytes) {
          current_queue_bytes_ -= task.payload_size_bytes;
        } else {
          current_queue_bytes_ = 0;
        }
      }
      capacity_cv_.notify_all();
      mark_uploaded_(task);
    } catch (const std::exception& exc) {
      const auto save_finished = std::chrono::steady_clock::now();
      const uint64_t save_duration_ns = ToNanoseconds(save_finished - save_started);
      {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        queue_stats_.total_save_ns += save_duration_ns;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_queue_bytes_ >= task.payload_size_bytes) {
          current_queue_bytes_ -= task.payload_size_bytes;
        } else {
          current_queue_bytes_ = 0;
        }
      }
      capacity_cv_.notify_all();
      report_error_(task, exc.what(), save_duration_ns);
    }
  }
}

}  // namespace catkv_ops
