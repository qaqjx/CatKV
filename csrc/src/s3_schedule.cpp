#include "s3_schedule.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include "compression_runner.h"
#include "compressor.h"

namespace s3 {
namespace {

template <typename FutureT>
bool FutureReady(FutureT& future) {
  return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

std::size_t NormalizeThreadCount(int num_threads) {
  if (num_threads <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(num_threads);
}

constexpr const char* kKeySvSuffix = "_key_sv";
constexpr const char* kOtherSuffix = "_other";

std::string KeySvPath(const std::string& path) {
  return path + kKeySvSuffix;
}

std::string OtherPath(const std::string& path) {
  return path + kOtherSuffix;
}

bool IsSplitCompressedTensorMap(const S3Manager::TensorMap& tensors) {
  return tensors.count("key_sv_quantized") > 0 &&
         tensors.count("key_sv_meta") > 0 &&
         tensors.count("value_sv_quantized") > 0 &&
         tensors.count("value_sv_meta") > 0;
}

S3Manager::TensorMap ExtractTensorSubset(
    const S3Manager::TensorMap& tensors,
    std::initializer_list<const char*> keys) {
  S3Manager::TensorMap subset;
  subset.reserve(keys.size());
  for (const char* key : keys) {
    auto it = tensors.find(key);
    if (it != tensors.end()) {
      subset.emplace(it->first, it->second);
    }
  }
  return subset;
}

void SavePossiblySplitTensorMap(
    S3Manager& manager,
    const std::string& key,
    const S3Manager::TensorMap& tensors) {
  if (!IsSplitCompressedTensorMap(tensors)) {
    manager.save(key, tensors);
    return;
  }

  auto key_sv_payload = ExtractTensorSubset(
      tensors,
      {"key_sv_quantized", "key_sv_meta", "key_residual_sv", "uuid"});
  auto other_payload = ExtractTensorSubset(
      tensors,
      {"u_quantized",
       "u_meta",
       "value_sv_quantized",
       "value_sv_meta",
       "value_residual_sv"});
  manager.save(KeySvPath(key), key_sv_payload);
  manager.save(OtherPath(key), other_payload);
}

}  // namespace

S3Schedule::S3Schedule(const std::string& config_path, int num_threads)
    : s3_manager_(config_path),
      thread_pool_(NormalizeThreadCount(num_threads)) {}

S3Schedule::S3Schedule(const S3Config& config, int num_threads)
    : s3_manager_(config),
      thread_pool_(NormalizeThreadCount(num_threads)) {}

int S3Schedule::submit_load(const std::string& key) {
  auto future = thread_pool_.submit([this, key]() { return s3_manager_.load(key); });
  const int task_id = get_task_id();
  {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    load_futures_.emplace(task_id, std::move(future));
  }
  return task_id;
}

int S3Schedule::submit_load_to_gpu(const std::string& key, int device_id) {
  auto future = thread_pool_.submit([this, key, device_id]() {
    return s3_manager_.batch_load(std::vector<std::string>{key}, device_id);
  });
  const int task_id = get_task_id();
  {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    load_futures_.emplace(task_id, std::move(future));
  }
  return task_id;
}

int S3Schedule::submit_save(const std::string& key, const S3Manager::TensorMap& tensors) {
  auto tensors_ptr = std::make_shared<S3Manager::TensorMap>(tensors);
  auto future = thread_pool_.submit(
      [this, key, tensors_ptr]() { SavePossiblySplitTensorMap(s3_manager_, key, *tensors_ptr); });
  const int task_id = get_task_id();
  {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    save_futures_.emplace(task_id, std::move(future));
  }
  return task_id;
}

int S3Schedule::submit_multi_save(
    const std::vector<std::string>& paths,
    const std::vector<std::pair<int64_t, int64_t>>& indices,
    const std::vector<torch::Tensor>& data,
    double ratio,
    const std::vector<std::string>& uuids) {
  if (paths.size() != indices.size()) {
    throw std::invalid_argument("paths and indices must have the same size");
  }
  if (!uuids.empty() && uuids.size() != indices.size()) {
    throw std::invalid_argument(
        "uuids and indices must have the same size when uuids are provided");
  }

  auto data_copy = std::make_shared<std::vector<torch::Tensor>>(data);
  auto future = thread_pool_.submit([this, paths, indices, data_copy, ratio, uuids]() {
    disk_manager::CatKVCompressor compressor(ratio);
    auto groups = compressor.compress_multi(*data_copy, indices, uuids);
    if (groups.size() != paths.size()) {
      throw std::runtime_error("compress_multi returned unexpected output size");
    }
    for (std::size_t i = 0; i < paths.size(); ++i) {
      SavePossiblySplitTensorMap(
          s3_manager_,
          paths[i],
          disk_manager::tensor_dict_to_s3_tensor_map(groups[i]));
    }
  });

  const int task_id = get_task_id();
  {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    save_futures_.emplace(task_id, std::move(future));
  }
  return task_id;
}

int S3Schedule::submit_batch_load(const std::vector<std::string>& keys) {
  auto future = thread_pool_.submit([this, keys]() { return s3_manager_.batch_load(keys); });
  const int task_id = get_task_id();
  {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    load_futures_.emplace(task_id, std::move(future));
  }
  return task_id;
}

int S3Schedule::submit_batch_load_to_gpu(const std::vector<std::string>& keys, int device_id) {
  auto future = thread_pool_.submit(
      [this, keys, device_id]() { return s3_manager_.batch_load(keys, device_id); });
  const int task_id = get_task_id();
  {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    load_futures_.emplace(task_id, std::move(future));
  }
  return task_id;
}

S3Manager::OptionalTensorMap S3Schedule::get_load_result(int task_id) {
  LoadFuture future;
  {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    auto it = load_futures_.find(task_id);
    if (it == load_futures_.end()) {
      throw std::runtime_error("Invalid S3 load task ID: " + std::to_string(task_id));
    }
    future = std::move(it->second);
    load_futures_.erase(it);
  }

  if (auto* single = std::get_if<std::future<S3Manager::OptionalTensorMap>>(&future)) {
    return single->get();
  }

  if (auto* batch = std::get_if<std::future<std::vector<S3Manager::OptionalTensorMap>>>(&future)) {
    (void)batch->get();
    throw std::runtime_error("Task is batch load; use get_batch_load_result.");
  }

  if (auto* gpu = std::get_if<std::future<S3Manager::BatchLoadResult>>(&future)) {
    auto result = gpu->get();
    if (!result.first.empty() && result.first.front().has_value()) {
      return std::move(result.first.front());
    }
    return std::nullopt;
  }

  throw std::runtime_error("Unknown future type for task: " + std::to_string(task_id));
}

std::vector<S3Manager::OptionalTensorMap> S3Schedule::get_batch_load_result(int task_id) {
  LoadFuture future;
  {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    auto it = load_futures_.find(task_id);
    if (it == load_futures_.end()) {
      throw std::runtime_error("Invalid S3 batch load task ID: " + std::to_string(task_id));
    }
    future = std::move(it->second);
    load_futures_.erase(it);
  }

  if (auto* batch = std::get_if<std::future<std::vector<S3Manager::OptionalTensorMap>>>(&future)) {
    return batch->get();
  }

  if (auto* single = std::get_if<std::future<S3Manager::OptionalTensorMap>>(&future)) {
    (void)single->get();
    throw std::runtime_error("Task is single load; use get_load_result.");
  }

  if (auto* gpu = std::get_if<std::future<S3Manager::BatchLoadResult>>(&future)) {
    auto result = gpu->get();
    return std::move(result.first);
  }

  throw std::runtime_error("Unknown future type for task: " + std::to_string(task_id));
}

S3Manager::BatchLoadResult S3Schedule::get_batch_load_to_gpu_result(int task_id) {
  LoadFuture future;
  {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    auto it = load_futures_.find(task_id);
    if (it == load_futures_.end()) {
      throw std::runtime_error("Invalid S3 GPU load task ID: " + std::to_string(task_id));
    }
    future = std::move(it->second);
    load_futures_.erase(it);
  }

  if (auto* gpu = std::get_if<std::future<S3Manager::BatchLoadResult>>(&future)) {
    return gpu->get();
  }

  if (auto* single = std::get_if<std::future<S3Manager::OptionalTensorMap>>(&future)) {
    (void)single->get();
    throw std::runtime_error("Task is single load; use get_load_result.");
  }

  if (auto* batch = std::get_if<std::future<std::vector<S3Manager::OptionalTensorMap>>>(&future)) {
    (void)batch->get();
    throw std::runtime_error("Task is CPU batch load; use get_batch_load_result.");
  }

  throw std::runtime_error("Unknown future type for task: " + std::to_string(task_id));
}

bool S3Schedule::is_ready(int task_id) {
  std::lock_guard<std::mutex> lock(futures_mutex_);
  auto load_it = load_futures_.find(task_id);
  if (load_it != load_futures_.end()) {
    return std::visit([](auto& future) { return FutureReady(future); }, load_it->second);
  }

  auto save_it = save_futures_.find(task_id);
  if (save_it != save_futures_.end()) {
    return FutureReady(save_it->second);
  }
  return false;
}

void S3Schedule::wait(int task_id) {
  std::lock_guard<std::mutex> lock(futures_mutex_);

  auto load_it = load_futures_.find(task_id);
  if (load_it != load_futures_.end()) {
    std::visit([](auto& future) { future.wait(); }, load_it->second);
    load_futures_.erase(load_it);
    return;
  }

  auto save_it = save_futures_.find(task_id);
  if (save_it != save_futures_.end()) {
    save_it->second.wait();
    save_futures_.erase(save_it);
    return;
  }

  throw std::runtime_error("Invalid S3 task ID for wait: " + std::to_string(task_id));
}

std::string S3Schedule::get_status(int task_id) {
  std::lock_guard<std::mutex> lock(futures_mutex_);
  auto load_it = load_futures_.find(task_id);
  if (load_it != load_futures_.end()) {
    const bool ready = std::visit(
        [](auto& future) { return FutureReady(future); },
        load_it->second);
    return ready ? "load-ready" : "load-running";
  }

  auto save_it = save_futures_.find(task_id);
  if (save_it != save_futures_.end()) {
    return FutureReady(save_it->second) ? "save-ready" : "save-running";
  }

  return "invalid";
}

}  // namespace s3
