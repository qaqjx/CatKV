#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace disk_manager {

class CatKVCompressor {
 public:
  using TensorDict = std::unordered_map<std::string, torch::Tensor>;
  using SVDJobResult =
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

 explicit CatKVCompressor(double ratio = 0.2,
                          torch::ScalarType dtype = torch::kBFloat16);
  ~CatKVCompressor();

  CatKVCompressor(const CatKVCompressor&) = delete;
  CatKVCompressor& operator=(const CatKVCompressor&) = delete;
  CatKVCompressor(CatKVCompressor&&) = delete;
  CatKVCompressor& operator=(CatKVCompressor&&) = delete;

  TensorDict compress(const std::vector<torch::Tensor>& data);
  std::vector<TensorDict> compress_multi(
      const std::vector<torch::Tensor>& data,
      const std::vector<std::pair<int64_t, int64_t>>& indices,
      const std::vector<std::string>& uuids = {});

 private:
  struct EyeCacheKey {
    int64_t dim;
    int64_t scalar_type;

    bool operator==(const EyeCacheKey& other) const {
      return dim == other.dim && scalar_type == other.scalar_type;
    }
  };

  struct EyeCacheKeyHash {
    std::size_t operator()(const EyeCacheKey& key) const {
      return std::hash<int64_t>{}(key.dim) ^
             (std::hash<int64_t>{}(key.scalar_type) << 1);
    }
  };

  double ratio_;
  torch::ScalarType dtype_;
  mutable std::mutex cache_mutex_;
  mutable std::unordered_map<EyeCacheKey, torch::Tensor, EyeCacheKeyHash>
      eye_cache_;
  mutable std::unordered_map<int64_t, torch::Tensor> x_coords_cache_;

  int64_t calculate_rank(int64_t seq_len, int64_t hidden_dim) const;

  torch::Tensor get_cached_eye(int64_t dim,
                               const torch::TensorOptions& options) const;
  torch::Tensor get_cached_x_coords(int64_t size) const;

  static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  svd_lowrank(const torch::Tensor& A, int64_t q, int64_t niter = 2);

  static torch::Tensor get_approximate_basis(
      const torch::Tensor& A, int64_t q, int64_t niter = 2);

  torch::Tensor sigma_min_power_iter(
      const torch::Tensor& A, double eps = 1e-12,
      int64_t num_iters = 4);

  SVDJobResult compute_svd_with_sigma_min(
      const torch::Tensor& matrix, int64_t rank);

  int64_t calculate_elbow(
      const torch::Tensor& singular_values,
      int64_t max_x = -1, double min_s = -1.0);

  std::vector<TensorDict> compress_multi_impl(
      const std::vector<torch::Tensor>& data,
      const std::vector<std::pair<int64_t, int64_t>>& indices,
      const torch::Tensor& uuid = torch::Tensor());

  static TensorDict sq_compress(
      const torch::Tensor& tensor, const std::string& prefix);
};

}  // namespace disk_manager
