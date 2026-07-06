#include "compressor.h"
#include "shared_key_sv_path.h"

#include <ATen/Dispatch.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace disk_manager {

namespace py = pybind11;

namespace {

torch::Tensor reduced_qr(const torch::Tensor& input) {
  return std::get<0>(torch::linalg_qr(input));
}

torch::Tensor make_uuid_tensor() {
  static std::atomic<int64_t> counter{1};
  const auto value = counter.fetch_add(1, std::memory_order_relaxed);
  auto uuid = torch::empty(
      {2},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  uuid[0] = static_cast<int32_t>(value & 0x7fffffff);
  uuid[1] = static_cast<int32_t>((value >> 31) & 0x7fffffff);
  return uuid;
}

int64_t key_sv_rank(const CatKVCompressor::TensorDict& payload) {
  const auto it = payload.find("key_sv_quantized");
  if (it == payload.end() || !it->second.defined()) {
    return 0;
  }
  const auto& tensor = it->second;
  if (tensor.dim() >= 2) {
    return tensor.size(tensor.dim() - 2);
  }
  return tensor.numel();
}

void use_largest_rank_key_sv(
    std::vector<CatKVCompressor::TensorDict>& payloads) {
  if (payloads.empty()) {
    return;
  }

  std::size_t largest_idx = 0;
  int64_t largest_rank = key_sv_rank(payloads.front());
  for (std::size_t idx = 1; idx < payloads.size(); ++idx) {
    const int64_t rank = key_sv_rank(payloads[idx]);
    if (rank > largest_rank) {
      largest_rank = rank;
      largest_idx = idx;
    }
  }

  auto& source = payloads[largest_idx];
  const torch::Tensor key_sv_quantized = source.at("key_sv_quantized").clone();
  const torch::Tensor key_sv_meta = source.at("key_sv_meta").clone();
  const torch::Tensor key_residual_sv = source.at("key_residual_sv").clone();
  for (auto& payload : payloads) {
    payload["key_sv_quantized"] = key_sv_quantized.clone();
    payload["key_sv_meta"] = key_sv_meta.clone();
    payload["key_residual_sv"] = key_residual_sv.clone();
  }
}

using S = torch::indexing::Slice;

int64_t calculate_multi_rank(
    double ratio,
    int64_t num_layers,
    int64_t seq_len,
    int64_t hidden_dim,
    int64_t group_count,
    size_t element_size) {
  const int64_t token_num = seq_len / std::max<int64_t>(group_count, 1);
  const double total_original_mem_bytes =
      static_cast<double>(num_layers) * static_cast<double>(seq_len) *
      static_cast<double>(hidden_dim) * static_cast<double>(element_size) *
      ratio / static_cast<double>(std::max<int64_t>(group_count, 1));
  const double budget_per_rank =
      4.0 * 16.0 + static_cast<double>(token_num + hidden_dim) * 4.0;
  const int64_t all_rank = static_cast<int64_t>(
      std::floor(total_original_mem_bytes * 8.0 / budget_per_rank));

  int64_t max_rank = 512;
  if (token_num >= 8192) {
    max_rank = 256;
  } else if (token_num >= 4096) {
    max_rank = 384;
  }

  return std::max<int64_t>(
      0, std::min({all_rank, max_rank, static_cast<int64_t>(1024), token_num}));
}

std::pair<int64_t, int64_t> allocate_low_rank_multi(
    double total_budget_bytes,
    int64_t hidden_dim,
    int64_t token_num,
    int64_t high_rank) {
  const double budget_per_rank =
      4.0 * 16.0 + static_cast<double>(token_num + hidden_dim) * 4.0;
  int64_t low_rank = static_cast<int64_t>(std::floor(
      (total_budget_bytes * 8.0 -
       (static_cast<double>(high_rank) * token_num * 4.0 +
        static_cast<double>(high_rank) * 2.0 * 16.0 +
        static_cast<double>(high_rank) * hidden_dim * 16.0)) /
      budget_per_rank));

  if (low_rank < 0) {
    low_rank = 0;
    high_rank = static_cast<int64_t>(std::floor(
        total_budget_bytes * 8.0 /
        (static_cast<double>(token_num) * 4.0 + 2.0 * 16.0 +
         static_cast<double>(hidden_dim) * 16.0)));
  }

  return {low_rank, high_rank};
}

CatKVCompressor::TensorDict sq_compress_exact(
    const torch::Tensor& tensor, const std::string& prefix) {
  auto t = tensor;
  if (!t.is_contiguous()) {
    t = t.contiguous();
  }

  auto row_min = std::get<0>(torch::min(t, -1, true));
  auto row_max = std::get<0>(torch::max(t, -1, true));
  auto row_range = row_max - row_min + 1e-6;
  auto normalized = (t - row_min) / row_range;
  auto quantized =
      torch::clamp((normalized * 15.0).round(), 0, 15).to(torch::kUInt8);

  auto chunks = quantized.chunk(2, -1);
  auto chunk0 = chunks[0];
  auto chunk1_padded = torch::zeros_like(chunk0);
  if (chunks.size() > 1) {
    chunk1_padded.narrow(-1, 0, chunks[1].size(-1)).copy_(chunks[1]);
  }

  CatKVCompressor::TensorDict result;
  result[prefix + "_packed"] =
      (chunk0 * 16 + chunk1_padded).cpu().contiguous();
  result[prefix + "_scale"] =
      torch::cat({row_min, row_max}, -1).cpu().contiguous();
  return result;
}

}  // namespace

CatKVCompressor::CatKVCompressor(double ratio, torch::ScalarType dtype)
    : ratio_(ratio),
      dtype_(dtype) {
  if (ratio <= 0.0 || ratio >= 1.0) {
    throw std::invalid_argument("ratio must be between 0 and 1 (exclusive)");
  }
}

CatKVCompressor::~CatKVCompressor() = default;

torch::Tensor CatKVCompressor::get_cached_eye(
    int64_t dim, const torch::TensorOptions& options) const {
  EyeCacheKey key{dim, static_cast<int64_t>(options.dtype().toScalarType())};
  std::lock_guard<std::mutex> lock(cache_mutex_);
  auto it = eye_cache_.find(key);
  if (it != eye_cache_.end()) {
    return it->second;
  }
  auto eye = torch::eye(dim, options);
  eye_cache_.emplace(key, eye);
  return eye;
}

torch::Tensor CatKVCompressor::get_cached_x_coords(int64_t size) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  auto it = x_coords_cache_.find(size);
  if (it != x_coords_cache_.end()) {
    return it->second;
  }
  auto x_coords =
      torch::arange(size, torch::TensorOptions().dtype(torch::kFloat32))
          .unsqueeze(0);
  x_coords_cache_.emplace(size, x_coords);
  return x_coords;
}

// ---------------------------------------------------------------------------
// calculate_rank  (mirrors Python _calculate_rank)
// ---------------------------------------------------------------------------
int64_t CatKVCompressor::calculate_rank(int64_t seq_len,
                                        int64_t hidden_dim) const {
  double element_size = 2.0;
  double total_budget =
      static_cast<double>(seq_len) * hidden_dim * element_size * ratio_;
  double budget_per_rank =
      4.0 * 16.0 + (static_cast<double>(seq_len) + hidden_dim) * 4.0;
  int64_t rank = static_cast<int64_t>(total_budget * 8.0 / budget_per_rank);
  int64_t max_rank = 512;
  if (seq_len >= 8192) {
    max_rank = 256;
  } else if (seq_len >= 4096) {
    max_rank = 384;
  }
  rank = std::max(
      static_cast<int64_t>(256),
      std::min(rank, std::min({seq_len, hidden_dim, max_rank})));
  return rank;
}

// ---------------------------------------------------------------------------
// get_approximate_basis  (Algorithm 4.4, Halko et al. 2009)
// ---------------------------------------------------------------------------
torch::Tensor CatKVCompressor::get_approximate_basis(
    const torch::Tensor& A, int64_t q, int64_t niter) {
  auto n = A.size(-1);
  auto R = torch::randn({n, q}, A.options());
  auto X = torch::matmul(A, R);
  auto Q = reduced_qr(X);

  for (int64_t i = 0; i < niter; ++i) {
    X = torch::matmul(A.mH(), Q);
    Q = reduced_qr(X);
    X = torch::matmul(A, Q);
    Q = reduced_qr(X);
  }
  return Q;
}

// ---------------------------------------------------------------------------
// svd_lowrank  (Algorithm 5.1, Halko et al. 2009)
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CatKVCompressor::svd_lowrank(const torch::Tensor& A, int64_t q,
                             int64_t niter) {
  auto m = A.size(-2);
  auto n = A.size(-1);
  bool transposed = false;
  torch::Tensor working = A;

  if (m < n) {
    working = A.mH();
    transposed = true;
  }

  auto Q = get_approximate_basis(working, q, niter);
  auto B = torch::matmul(Q.mH(), working);
  auto [U_small, S, V] = at::svd(B, /*some=*/true, /*compute_uv=*/true);
  auto U = torch::matmul(Q, U_small);

  if (transposed) {
    std::swap(U, V);
  }
  return {U, S, V};
}

// ---------------------------------------------------------------------------
// sigma_min_power_iter  (inverse power iteration on Gram matrix)
// ---------------------------------------------------------------------------
torch::Tensor CatKVCompressor::sigma_min_power_iter(
    const torch::Tensor& A, double eps, int64_t num_iters) {
  torch::Tensor mat = A;
  if (mat.dim() == 3) {
    mat = mat.squeeze(0);
  }
  if (mat.scalar_type() != torch::kFloat32 &&
      mat.scalar_type() != torch::kFloat64) {
    mat = mat.to(torch::kFloat32);
  }

  auto m = mat.size(0);
  auto n = mat.size(1);
  torch::Tensor B;
  if (m >= n) {
    B = torch::matmul(mat.t(), mat);
  } else {
    B = torch::matmul(mat, mat.t());
  }
  auto dim = B.size(0);
  if (dim == 0) {
    return torch::zeros({}, B.options());
  }

  auto v = torch::randn({dim}, B.options());
  v = v / torch::norm(v);

  auto trace_val = torch::trace(B);
  auto shift = trace_val / static_cast<double>(dim) * 0.05;
  auto B_shifted = B.clone();
  B_shifted.diagonal().add_(shift);

  torch::Tensor lu, pivots;
  try {
    auto lu_result = torch::linalg_lu_factor(B_shifted);
    lu = std::get<0>(lu_result);
    pivots = std::get<1>(lu_result);
  } catch (...) {
    for (int64_t i = 0; i < num_iters; ++i) {
      v = torch::linalg_solve(B_shifted, v);
      auto v_norm = torch::norm(v);
      if (v_norm.item().toDouble() < 1e-10) break;
      v = v / v_norm;
    }
    auto Bv = torch::mv(B, v);
    auto lam = torch::dot(v, Bv);
    lam = torch::clamp(lam, /*min=*/0.0);
    return torch::sqrt(lam + eps);
  }

  for (int64_t i = 0; i < num_iters; ++i) {
    v = torch::linalg_lu_solve(lu, pivots, v.unsqueeze(-1)).squeeze(-1);
    auto v_norm = torch::norm(v);
    if (v_norm.item().toDouble() < 1e-10) break;
    v = v / v_norm;
  }

  auto Bv = torch::mv(B, v);
  auto lam = torch::dot(v, Bv);
  lam = torch::clamp(lam, /*min=*/0.0);
  return torch::sqrt(lam + eps);
}

CatKVCompressor::SVDJobResult CatKVCompressor::compute_svd_with_sigma_min(
    const torch::Tensor& matrix, int64_t rank) {
  auto [u, s, v] = svd_lowrank(matrix, rank, 2);
  auto s_min = sigma_min_power_iter(matrix);
  return std::make_tuple(std::move(u), std::move(s), std::move(v),
                         std::move(s_min));
}

// ---------------------------------------------------------------------------
// calculate_elbow  (distance-to-chord heuristic, mirrors Python version)
// ---------------------------------------------------------------------------
int64_t CatKVCompressor::calculate_elbow(
    const torch::Tensor& singular_values, int64_t max_x, double min_s) {
  torch::Tensor sv = singular_values;
  if (sv.dim() == 1) {
    sv = sv.unsqueeze(0);
  }
  sv = sv.to(torch::kFloat32).contiguous();

  const auto B = sv.size(0);
  const auto N = sv.size(1);
  if (N == 0) {
    return 0;
  }

  auto accessor = sv.accessor<float, 2>();
  double mean_elbow = 0.0;

  for (int64_t batch_idx = 0; batch_idx < B; ++batch_idx) {
    const double x0 = 0.0;
    const double y0 = static_cast<double>(accessor[batch_idx][0]);
    const double x1 =
        max_x >= 0 ? static_cast<double>(max_x) : static_cast<double>(N - 1);
    const double y1 =
        min_s >= 0.0 ? min_s : static_cast<double>(accessor[batch_idx][N - 1]);

    const double line_x = x1 - x0;
    const double line_y = y1 - y0;
    double line_norm = std::sqrt(line_x * line_x + line_y * line_y);
    if (line_norm < 1e-10) {
      line_norm = 1.0;
    }

    int64_t best_idx = 0;
    double best_distance = -1.0;
    for (int64_t idx = 0; idx < N; ++idx) {
      const double point_x = static_cast<double>(idx) - x0;
      const double point_y = static_cast<double>(accessor[batch_idx][idx]) - y0;
      const double cross =
          std::abs(line_x * point_y - line_y * point_x);
      const double distance = cross / line_norm;
      if (distance > best_distance) {
        best_distance = distance;
        best_idx = idx;
      }
    }
    mean_elbow += static_cast<double>(best_idx);
  }

  return static_cast<int64_t>(mean_elbow / static_cast<double>(B));
}

// ---------------------------------------------------------------------------
// sq_compress  (scalar quantization to packed uint4, mirrors Python exactly)
// ---------------------------------------------------------------------------
CatKVCompressor::TensorDict CatKVCompressor::sq_compress(
    const torch::Tensor& tensor, const std::string& prefix) {
  auto t = tensor;
  if (t.scalar_type() != torch::kFloat32 && t.scalar_type() != torch::kFloat16 &&
      t.scalar_type() != torch::kBFloat16) {
    t = t.to(torch::kFloat32);
  }
  if (!t.is_contiguous()) {
    t = t.contiguous();
  }
  auto original_sizes = t.sizes().vec();
  if (t.dim() == 1) {
    t = t.unsqueeze(0);
  }

  auto cols = t.size(-1);
  auto rows = t.numel() / std::max<int64_t>(cols, 1);
  auto half = (cols + 1) / 2;
  auto flat = t.reshape({rows, cols});

  auto packed_sizes = t.sizes().vec();
  packed_sizes.back() = half;
  auto scale_sizes = t.sizes().vec();
  scale_sizes.back() = 2;

  auto packed = torch::empty(
      {rows, half},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  auto scale = torch::empty(
      {rows, 2},
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));

  auto* packed_ptr = packed.data_ptr<uint8_t>();
  auto* scale_ptr = scale.data_ptr<at::BFloat16>();

  AT_DISPATCH_SWITCH(
      flat.scalar_type(), "sq_compress_cpu",
      AT_DISPATCH_CASE(
          at::ScalarType::Float, [&] {
            const auto* input_ptr = flat.data_ptr<float>();
            for (int64_t row = 0; row < rows; ++row) {
              const auto* row_ptr = input_ptr + row * cols;
              float row_min = row_ptr[0];
              float row_max = row_min;
              for (int64_t col = 1; col < cols; ++col) {
                const float value = row_ptr[col];
                row_min = std::min(row_min, value);
                row_max = std::max(row_max, value);
              }
              const float row_range = row_max - row_min + 1e-6f;
              scale_ptr[row * 2] = at::BFloat16(row_min);
              scale_ptr[row * 2 + 1] = at::BFloat16(row_max);

              auto* packed_row = packed_ptr + row * half;
              for (int64_t col = 0; col < half; ++col) {
                const float hi_val = row_ptr[col];
                const float hi_norm = (hi_val - row_min) / row_range;
                const auto hi_q = static_cast<int>(
                    std::nearbyint(std::clamp(hi_norm * 15.0f, 0.0f, 15.0f)));

                int lo_q = 0;
                const int64_t lo_col = col + half;
                if (lo_col < cols) {
                  const float lo_val = row_ptr[lo_col];
                  const float lo_norm = (lo_val - row_min) / row_range;
                  lo_q = static_cast<int>(std::nearbyint(
                      std::clamp(lo_norm * 15.0f, 0.0f, 15.0f)));
                }
                packed_row[col] = static_cast<uint8_t>((hi_q << 4) | lo_q);
              }
            }
          })
      AT_DISPATCH_CASE(
          at::ScalarType::Half, [&] {
            const auto* input_ptr = flat.data_ptr<c10::Half>();
            for (int64_t row = 0; row < rows; ++row) {
              const auto* row_ptr = input_ptr + row * cols;
              float row_min = static_cast<float>(row_ptr[0]);
              float row_max = row_min;
              for (int64_t col = 1; col < cols; ++col) {
                const float value = static_cast<float>(row_ptr[col]);
                row_min = std::min(row_min, value);
                row_max = std::max(row_max, value);
              }
              const float row_range = row_max - row_min + 1e-6f;
              scale_ptr[row * 2] = at::BFloat16(row_min);
              scale_ptr[row * 2 + 1] = at::BFloat16(row_max);

              auto* packed_row = packed_ptr + row * half;
              for (int64_t col = 0; col < half; ++col) {
                const float hi_val = static_cast<float>(row_ptr[col]);
                const float hi_norm = (hi_val - row_min) / row_range;
                const auto hi_q = static_cast<int>(
                    std::nearbyint(std::clamp(hi_norm * 15.0f, 0.0f, 15.0f)));

                int lo_q = 0;
                const int64_t lo_col = col + half;
                if (lo_col < cols) {
                  const float lo_val = static_cast<float>(row_ptr[lo_col]);
                  const float lo_norm = (lo_val - row_min) / row_range;
                  lo_q = static_cast<int>(std::nearbyint(
                      std::clamp(lo_norm * 15.0f, 0.0f, 15.0f)));
                }
                packed_row[col] = static_cast<uint8_t>((hi_q << 4) | lo_q);
              }
            }
          })
      AT_DISPATCH_CASE(
          at::ScalarType::BFloat16, [&] {
            const auto* input_ptr = flat.data_ptr<c10::BFloat16>();
            for (int64_t row = 0; row < rows; ++row) {
              const auto* row_ptr = input_ptr + row * cols;
              float row_min = static_cast<float>(row_ptr[0]);
              float row_max = row_min;
              for (int64_t col = 1; col < cols; ++col) {
                const float value = static_cast<float>(row_ptr[col]);
                row_min = std::min(row_min, value);
                row_max = std::max(row_max, value);
              }
              const float row_range = row_max - row_min + 1e-6f;
              scale_ptr[row * 2] = at::BFloat16(row_min);
              scale_ptr[row * 2 + 1] = at::BFloat16(row_max);

              auto* packed_row = packed_ptr + row * half;
              for (int64_t col = 0; col < half; ++col) {
                const float hi_val = static_cast<float>(row_ptr[col]);
                const float hi_norm = (hi_val - row_min) / row_range;
                const auto hi_q = static_cast<int>(
                    std::nearbyint(std::clamp(hi_norm * 15.0f, 0.0f, 15.0f)));

                int lo_q = 0;
                const int64_t lo_col = col + half;
                if (lo_col < cols) {
                  const float lo_val = static_cast<float>(row_ptr[lo_col]);
                  const float lo_norm = (lo_val - row_min) / row_range;
                  lo_q = static_cast<int>(std::nearbyint(
                      std::clamp(lo_norm * 15.0f, 0.0f, 15.0f)));
                }
                packed_row[col] = static_cast<uint8_t>((hi_q << 4) | lo_q);
              }
            }
          }));

  packed = packed.reshape(packed_sizes);
  scale = scale.reshape(scale_sizes);
  if (original_sizes.size() == 1) {
    scale = scale.squeeze(0);
  }

  TensorDict result;
  result[prefix + "_packed"] = packed;
  result[prefix + "_scale"] = scale;
  return result;
}

// ---------------------------------------------------------------------------
// compress  (canonical public compressor entrypoint)
// ---------------------------------------------------------------------------
CatKVCompressor::TensorDict CatKVCompressor::compress(
    const std::vector<torch::Tensor>& data) {
  c10::InferenceMode guard;

  auto key = data[0];
  auto value = data[1];

  const auto output_dtype = key.scalar_type();
  const auto original_element_size = key.element_size();
  const auto num_layers = key.size(0);

  const auto seq_len = key.size(1);
  const auto num_heads = key.size(2);
  const auto head_dim = key.size(3);
  const auto hidden_dim = num_heads * head_dim;
  const auto max_elbow_x = std::min(hidden_dim, seq_len);
  const auto total_original_mem_bytes =
      static_cast<double>(num_layers) * static_cast<double>(seq_len) *
      static_cast<double>(hidden_dim) *
      static_cast<double>(original_element_size) * ratio_;

  key = key.reshape({-1, seq_len, hidden_dim});
  value = value.reshape({-1, seq_len, hidden_dim});

  const auto rank = calculate_multi_rank(
      ratio_, num_layers, seq_len, hidden_dim, 1, original_element_size);

  if (key.device().type() != torch::kCPU) {
    key = key.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    value = value.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  } else if (key.scalar_type() != torch::kFloat32) {
    key = key.to(torch::kFloat32);
    value = value.to(torch::kFloat32);
  }
  if (!key.is_contiguous()) key = key.contiguous();
  if (!value.is_contiguous()) value = value.contiguous();

  auto key_sq = key.squeeze(0);
  auto val_sq = value.squeeze(0);
  auto key_result = compute_svd_with_sigma_min(key_sq, rank);
  auto val_result = compute_svd_with_sigma_min(val_sq, rank);

  auto [key_u, key_s, key_v, key_s_min] = std::move(key_result);
  auto [val_u, val_s, val_v, val_s_min] = std::move(val_result);
  auto key_vh = key_v.transpose(-1, -2);
  auto val_vh = val_v.transpose(-1, -2);

  const auto max_residual_rank = std::max(rank - 1, static_cast<int64_t>(0));
  const auto key_residual_dim = std::min(
      calculate_elbow(key_s.unsqueeze(0), max_elbow_x, key_s_min.item().toDouble()),
      max_residual_rank);
  const auto val_residual_dim = std::min(
      calculate_elbow(val_s.unsqueeze(0), max_elbow_x, val_s_min.item().toDouble()),
      max_residual_rank);
  auto [low_rank, high_rank] = allocate_low_rank_multi(
      total_original_mem_bytes,
      hidden_dim,
      seq_len,
      (key_residual_dim + val_residual_dim) / 2);
  const auto final_rank = std::min<int64_t>(
      rank,
      std::max<int64_t>(0, low_rank + high_rank));

  auto u = torch::stack(
               std::vector<torch::Tensor>{
                   key_u.index({S(), S(0, final_rank)}).transpose(0, 1),
                   val_u.index({S(), S(0, final_rank)}).transpose(0, 1),
               },
               0)
               .contiguous()
               .to(output_dtype);
  auto sv = torch::stack(
                std::vector<torch::Tensor>{
                    (key_s.index({S(0, final_rank)}).unsqueeze(1) *
                     key_vh.index({S(0, final_rank), S()})),
                    (val_s.index({S(0, final_rank)}).unsqueeze(1) *
                     val_vh.index({S(0, final_rank), S()})),
                },
                0)
                .contiguous()
                .to(output_dtype);

  auto u_quantized = sq_compress_exact(u, "u");
  auto v_quantized = sq_compress_exact(sv, "v");

  TensorDict result;
  result["u_quantized"] = u_quantized["u_packed"];
  result["u_meta"] = u_quantized["u_scale"];
  result["key_sv_quantized"] =
      v_quantized["v_packed"].index({S(0, 1), S(), S()}).contiguous();
  result["key_sv_meta"] =
      v_quantized["v_scale"].index({S(0, 1), S(), S()}).contiguous();
  result["value_sv_quantized"] =
      v_quantized["v_packed"].index({S(1, 2), S(), S()}).contiguous();
  result["value_sv_meta"] =
      v_quantized["v_scale"].index({S(1, 2), S(), S()}).contiguous();
  result["key_residual_sv"] =
      sv.index({S(0, 1), S(0, key_residual_dim), S()}).contiguous();
  result["value_residual_sv"] =
      sv.index({S(1, 2), S(0, val_residual_dim), S()}).contiguous();
  result["uuid"] = make_uuid_tensor();
  return result;
}

// ---------------------------------------------------------------------------
// compress_multi  (mirrors Python Ours.compress_multi)
// ---------------------------------------------------------------------------
std::vector<CatKVCompressor::TensorDict> CatKVCompressor::compress_multi(
    const std::vector<torch::Tensor>& data,
    const std::vector<std::pair<int64_t, int64_t>>& indices,
    const std::vector<std::string>& uuids) {
  c10::InferenceMode guard;
  if (data.size() != 2) {
    throw std::invalid_argument(
        "compress_multi expects exactly 2 tensors [key, value]");
  }
  if (indices.empty()) {
    throw std::invalid_argument(
        "indices must be provided for compress_multi");
  }
  if (!uuids.empty() && uuids.size() != indices.size()) {
    throw std::invalid_argument(
        "uuids must have the same size as indices when provided");
  }
  if (uuids.empty()) {
    return compress_multi_impl(data, indices);
  }

  using S = torch::indexing::Slice;
  std::vector<TensorDict> outputs(indices.size());
  std::vector<std::size_t> standalone_positions;
  std::vector<std::string> ordered_group_ids;
  std::unordered_map<std::string, std::vector<std::size_t>> grouped_positions;

  for (std::size_t i = 0; i < uuids.size(); ++i) {
    if (uuids[i].empty()) {
      standalone_positions.push_back(i);
      continue;
    }
    auto& positions = grouped_positions[uuids[i]];
    if (positions.empty()) {
      ordered_group_ids.push_back(uuids[i]);
    }
    positions.push_back(i);
  }

  for (std::size_t position : standalone_positions) {
    const auto& [start, end] = indices[position];
    std::vector<torch::Tensor> sliced{
        data[0].index({S(), S(start, end), S(), S()}),
        data[1].index({S(), S(start, end), S(), S()}),
    };
    auto single_outputs = compress_multi_impl(sliced, {{0, end - start}});
    outputs[position] = std::move(single_outputs.front());
  }

  for (const auto& group_id : ordered_group_ids) {
    const auto& positions = grouped_positions[group_id];
    std::vector<torch::Tensor> key_chunks;
    std::vector<torch::Tensor> value_chunks;
    std::vector<std::pair<int64_t, int64_t>> local_indices;
    int64_t offset = 0;
    for (std::size_t position : positions) {
      const auto& [start, end] = indices[position];
      const int64_t chunk_len = end - start;
      local_indices.emplace_back(offset, offset + chunk_len);
      offset += chunk_len;
      key_chunks.push_back(data[0].index({S(), S(start, end), S(), S()}));
      value_chunks.push_back(data[1].index({S(), S(start, end), S(), S()}));
    }
    std::vector<torch::Tensor> merged{
        torch::cat(key_chunks, 1).contiguous(),
        torch::cat(value_chunks, 1).contiguous(),
    };
    auto grouped_outputs = compress_multi_impl(
        merged,
        local_indices,
        catkv_ops::UuidStringToTensor(group_id));
    use_largest_rank_key_sv(grouped_outputs);
    for (std::size_t local_idx = 0; local_idx < positions.size(); ++local_idx) {
      outputs[positions[local_idx]] = std::move(grouped_outputs[local_idx]);
    }
  }

  return outputs;
}

std::vector<CatKVCompressor::TensorDict> CatKVCompressor::compress_multi_impl(
    const std::vector<torch::Tensor>& data,
    const std::vector<std::pair<int64_t, int64_t>>& indices,
    const torch::Tensor& uuid) {
  c10::InferenceMode guard;

  if (indices.empty()) {
    return {};
  }

  auto key = data[0];
  auto value = data[1];
  const auto output_dtype = dtype_;
  const auto original_element_size = key.element_size();

  const auto num_layers = key.size(0);
  const auto seq_len = key.size(1);
  const auto num_heads = key.size(2);
  const auto head_dim = key.size(3);
  const auto hidden_dim = num_heads * head_dim;
  const auto group_count = static_cast<int64_t>(indices.size());
  const auto token_num = seq_len / group_count;
  const auto total_original_mem_bytes =
      static_cast<double>(num_layers) * static_cast<double>(seq_len) *
      static_cast<double>(hidden_dim) *
      static_cast<double>(original_element_size) * ratio_ /
      static_cast<double>(group_count);

  key = key.reshape({-1, seq_len, hidden_dim});
  value = value.reshape({-1, seq_len, hidden_dim});

  const auto rank = calculate_multi_rank(
      ratio_, num_layers, seq_len, hidden_dim, group_count,
      original_element_size);

  if (key.device().type() != torch::kCPU) {
    key = key.to(torch::TensorOptions().dtype(torch::kFloat32)
                     .device(torch::kCPU));
    value = value.to(torch::TensorOptions().dtype(torch::kFloat32)
                         .device(torch::kCPU));
  } else if (key.scalar_type() != torch::kFloat32) {
    key = key.to(torch::kFloat32);
    value = value.to(torch::kFloat32);
  }
  if (!key.is_contiguous()) key = key.contiguous();
  if (!value.is_contiguous()) value = value.contiguous();

  auto [key_u, key_s, key_v, key_s_min] = compute_svd_with_sigma_min(key.squeeze(0), rank);
  key_v = key_v.transpose(-1, -2);
  const auto max_residual_rank = std::max(rank - 1, static_cast<int64_t>(0));
  const auto key_residual_dim = std::min(
      calculate_elbow(key_s.unsqueeze(0), std::min(hidden_dim, seq_len), key_s_min.item().toDouble()),
      max_residual_rank);

  std::vector<TensorDict> group_dicts;
  group_dicts.reserve(indices.size());
  const auto shared_uuid =
      uuid.defined()
          ? uuid.detach().cpu().to(torch::kInt32).contiguous().view({-1})
          : make_uuid_tensor();

  for (const auto& [start, end] : indices) {
    auto value_chunk = value.index({S(), S(start, end), S()}).contiguous().squeeze(0);
    auto [value_u, value_s, value_v, value_s_min] = compute_svd_with_sigma_min(value_chunk, rank);
    value_v = value_v.transpose(-1, -2);
    const auto value_residual_dim = std::min(
        calculate_elbow(
            value_s.unsqueeze(0),
            std::min(hidden_dim, seq_len),
            value_s_min.item().toDouble()),
        max_residual_rank);

    const auto initial_chunk_rank = std::min(rank, end - start);
    auto u = torch::cat(
                 std::vector<torch::Tensor>{
                     key_u.index({S(start, end), S(0, initial_chunk_rank)}).unsqueeze(0),
                     value_u.index({S(), S(0, initial_chunk_rank)}).unsqueeze(0),
                 },
                 0)
                 .to(output_dtype);
    auto s = torch::cat(
        std::vector<torch::Tensor>{
            key_s.index({S(0, initial_chunk_rank)}).unsqueeze(0),
            value_s.index({S(0, initial_chunk_rank)}).unsqueeze(0),
        },
        0);
    auto v = torch::cat(
        std::vector<torch::Tensor>{
            key_v.index({S(0, initial_chunk_rank), S()}).unsqueeze(0),
            value_v.index({S(0, initial_chunk_rank), S()}).unsqueeze(0),
        },
        0);
    auto sv = torch::matmul(torch::diag_embed(s), v).to(output_dtype);

    auto [low_rank, high_rank] = allocate_low_rank_multi(
        total_original_mem_bytes,
        hidden_dim,
        token_num,
        (key_residual_dim + value_residual_dim) / 2);
    const auto final_rank = std::max<int64_t>(0, low_rank + high_rank);

    u = u.index({S(), S(), S(0, final_rank)}).contiguous();
    sv = sv.index({S(), S(0, final_rank), S()}).contiguous();

    auto u_quantized = sq_compress_exact(u.transpose(-1, -2), "u");
    auto v_quantized = sq_compress_exact(sv, "v");

    TensorDict group_dict;
    group_dict["u_quantized"] = u_quantized["u_packed"];
    group_dict["u_meta"] = u_quantized["u_scale"];
    group_dict["key_sv_quantized"] =
        v_quantized["v_packed"].index({S(0, 1), S(), S()}).contiguous();
    group_dict["key_sv_meta"] =
        v_quantized["v_scale"].index({S(0, 1), S(), S()}).contiguous();
    group_dict["value_sv_quantized"] =
        v_quantized["v_packed"].index({S(1, 2), S(), S()}).contiguous();
    group_dict["value_sv_meta"] =
        v_quantized["v_scale"].index({S(1, 2), S(), S()}).contiguous();
    group_dict["key_residual_sv"] =
        sv.index({S(0, 1), S(0, key_residual_dim), S()}).contiguous();
    group_dict["value_residual_sv"] =
        sv.index({S(1, 2), S(0, value_residual_dim), S()}).contiguous();
    group_dict["uuid"] = shared_uuid.clone();
    group_dicts.push_back(std::move(group_dict));
  }

  return group_dicts;
}

}  // namespace disk_manager
