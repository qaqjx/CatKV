#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <curl/curl.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "s3_config.h"
#include "tensor.h"
#include "threadpool.h"

namespace s3 {

using disk_manager::ThreadPool;

std::vector<std::size_t> ComputeAlignedObjectOffsets(
    const std::vector<std::size_t>& sizes,
    std::size_t alignment);

/**
 * @brief S3/MinIO storage manager for tensor dictionaries.
 *
 * Optimized for high-throughput batch loading:
 * - Connection pool with libcurl multi-handle
 * - Parallel Range requests for large files
 * - Pre-allocated CPU/GPU arenas for zero-copy parsing
 * - Async transfers to GPU
 */
class S3Manager {
 public:
  using TensorMap = std::unordered_map<std::string, disk_manager::Tensor>;
  using OptionalTensorMap = std::optional<TensorMap>;
  using TorchTensorMap = std::unordered_map<std::string, torch::Tensor>;
  using BatchLoadResult = std::pair<std::vector<OptionalTensorMap>, std::vector<TorchTensorMap>>;

  static constexpr std::size_t kDefaultBufferSize = 10ULL * 1024 * 1024 * 1024;  // 10GB
  static constexpr std::size_t kDefaultConnectionPoolSize = 128;
  static constexpr std::size_t kChunkSize = 256 * 1024 * 1024;  // 256MB - disable chunked download

  /**
   * @brief Construct S3Manager with config file path.
   * @param config_path Path to INI config file.
   * @param thread_count Number of worker threads (0 = auto).
   * @param preallocate_buffer_size CPU arena size in bytes.
   * @param connection_pool_size Number of reusable CURL handles.
   */
  explicit S3Manager(const std::string& config_path,
                     std::size_t thread_count = 0,
                     std::size_t preallocate_buffer_size = kDefaultBufferSize,
                     std::size_t connection_pool_size = kDefaultConnectionPoolSize);

  /**
   * @brief Construct S3Manager with config struct.
   */
  explicit S3Manager(const S3Config& config,
                     std::size_t thread_count = 0,
                     std::size_t preallocate_buffer_size = kDefaultBufferSize,
                     std::size_t connection_pool_size = kDefaultConnectionPoolSize);

  ~S3Manager();

  S3Manager(const S3Manager&) = delete;
  S3Manager& operator=(const S3Manager&) = delete;

  /**
   * @brief Upload tensors to S3.
   * @param key Object key (path within bucket).
   * @param tensors Named tensors to store.
   */
  void save(const std::string& key, const TensorMap& tensors);

  /**
   * @brief Download and parse tensors from S3.
   * @param key Object key.
   * @return Tensor map reconstructed from the object, or nullopt if the object is missing.
   */
  OptionalTensorMap load(const std::string& key);

  /**
   * @brief Batch download multiple objects concurrently.
   * @param keys Object keys to load.
   * @return Vector of optional tensor maps ordered to match @p keys.
   */
  std::vector<OptionalTensorMap> batch_load(const std::vector<std::string>& keys);
  std::vector<OptionalTensorMap> batch_load_with_sizes(
      const std::vector<std::string>& keys,
      const std::vector<std::size_t>& sizes);

  /**
   * @brief Batch download and transfer to GPU.
   * @param keys Object keys to load.
   * @param device_id CUDA device ID.
   * @return Pair of (CPU tensor maps, GPU tensor maps).
   */
  BatchLoadResult batch_load(const std::vector<std::string>& keys, int device_id);
  BatchLoadResult batch_load(const std::vector<std::string>& keys,
                             const std::vector<std::size_t>& sizes,
                             int device_id);

  /**
   * @brief Get object size via HEAD request.
   * @param key Object key.
   * @return Object size in bytes, or 0 on error.
   */
  std::size_t get_object_size(const std::string& key);

  /**
   * @brief Check if object exists.
   * @param key Object key.
   * @return true if object exists.
   */
  bool exists(const std::string& key);

  /**
   * @brief Delete object from S3.
   * @param key Object key.
   */
  void remove(const std::string& key);

  /**
   * @brief Pre-establish TCP connections to S3 server.
   * @param num_connections Number of connections to warm up (default: pool size).
   *
   * Call this method before batch operations to eliminate TCP handshake latency.
   */
  void warmup_connections(std::size_t num_connections = 0);

 private:
  enum class ProbeStatus {
    kPresent,
    kMissing,
    kError,
  };

  struct ProbeResult {
    ProbeStatus status = ProbeStatus::kMissing;
    std::size_t size = 0;
    std::string error;
  };

  // Initialize libcurl and connection pool
  void init_curl();
  void cleanup_curl();

  // Get/release CURL handle from pool
  CURL* acquire_handle();
  void release_handle(CURL* handle);

  // S3 request helpers
  std::string build_url(const std::string& key) const;
  void sign_request(CURL* curl, const std::string& method, const std::string& key,
                    const std::string& content_sha256 = "UNSIGNED-PAYLOAD",
                    curl_slist** headers = nullptr);

  // Download helpers
  ProbeResult probe_object(const std::string& key);
  std::size_t download_range(const std::string& key, char* dest,
                             std::size_t offset, std::size_t size);
  void parallel_download(const std::string& key, char* dest, std::size_t total_size);
  BatchLoadResult batch_load_internal(const std::vector<std::string>& keys,
                                      const std::vector<std::size_t>& sizes,
                                      const std::vector<bool>& present_mask,
                                      int device_id);
  void batch_download_curl_multi(const std::vector<std::string>& keys,
                                 const std::vector<char*>& destinations,
                                 const std::vector<std::size_t>& sizes);

  // Upload helpers
  void upload_object(const std::string& key, const char* data, std::size_t size);

  // Arena allocation (same as DiskManager)
  char* allocate_from_arena(std::size_t required_size, std::shared_ptr<void>& out_owner);
  void* allocate_from_gpu_arena(std::size_t size, int device_id, std::shared_ptr<void>& out_owner);
  void* allocate_from_gpu_arena_unlocked(std::size_t size, int device_id,
                                          std::shared_ptr<void>& out_owner);

  // Configuration
  S3Config config_;

  // Connection pool
  std::vector<CURL*> curl_pool_;
  std::mutex curl_pool_mutex_;
  std::atomic<std::size_t> curl_pool_size_{0};

  // CPU Arena
  std::shared_ptr<void> arena_buffer_;
  std::size_t arena_capacity_ = 0;
  std::size_t arena_offset_ = 0;
  std::mutex buffer_mutex_;

  // GPU Arena
  void* gpu_arena_ = nullptr;
  std::size_t gpu_arena_capacity_ = 0;
  std::size_t gpu_arena_offset_ = 0;
  std::mutex gpu_arena_mutex_;
  int current_device_id_ = -1;

  // Thread pool
  ThreadPool thread_pool_;

  // Signing key cache (valid for same day)
  mutable std::mutex signing_key_mutex_;
  mutable std::string cached_date_;
  mutable std::vector<unsigned char> cached_signing_key_;
};

}  // namespace s3
