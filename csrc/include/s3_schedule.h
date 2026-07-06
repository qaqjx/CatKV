#pragma once

#include <atomic>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <torch/torch.h>

#include "s3_manager.h"
#include "threadpool.h"

namespace s3 {

class S3Schedule {
 public:
  explicit S3Schedule(const std::string& config_path, int num_threads = 0);
  explicit S3Schedule(const S3Config& config, int num_threads = 0);

  S3Schedule(const S3Schedule&) = delete;
  S3Schedule& operator=(const S3Schedule&) = delete;

  int submit_load(const std::string& key);
  int submit_load_to_gpu(const std::string& key, int device_id = 0);
  int submit_save(const std::string& key, const S3Manager::TensorMap& tensors);
  int submit_multi_save(
      const std::vector<std::string>& paths,
      const std::vector<std::pair<int64_t, int64_t>>& indices,
      const std::vector<torch::Tensor>& data,
      double ratio,
      const std::vector<std::string>& uuids = {});
  int submit_batch_load(const std::vector<std::string>& keys);
  int submit_batch_load_to_gpu(const std::vector<std::string>& keys, int device_id = 0);

  S3Manager::OptionalTensorMap get_load_result(int task_id);
  std::vector<S3Manager::OptionalTensorMap> get_batch_load_result(int task_id);
  S3Manager::BatchLoadResult get_batch_load_to_gpu_result(int task_id);

  bool is_ready(int task_id);
  void wait(int task_id);
  std::string get_status(int task_id);

 private:
  int get_task_id() { return task_id_counter_.fetch_add(1); }

  using LoadFuture = std::variant<
      std::future<S3Manager::OptionalTensorMap>,
      std::future<std::vector<S3Manager::OptionalTensorMap>>,
      std::future<S3Manager::BatchLoadResult>>;

  S3Manager s3_manager_;
  disk_manager::ThreadPool thread_pool_;

  std::map<int, LoadFuture> load_futures_;
  std::map<int, std::future<void>> save_futures_;
  std::mutex futures_mutex_;
  std::atomic<int> task_id_counter_{0};
};

}  // namespace s3
