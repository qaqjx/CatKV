/*
 * Fused SQ4 dequantization CUDA kernels for KV cache decompression.
 *
 * Supports both float16 and bfloat16 meta/output tensors.
 * Fuses: nibble unpack + dequantize + (optional) residual overlay + (optional) transpose
 * into a single kernel launch. No shape-dependent recompilation.
 *
 * Kernel 1: fused_dequant_u_transposed
 *   Input:  u_quantized [B, rank, packed_len] uint8, u_meta [B, rank, 2] fp16/bf16
 *   Output: u_out [B, kv_len, rank] fp16/bf16  (transposed!)
 *
 * Kernel 2: fused_dequant_v_residual
 *   Input:  v_quantized [B, rank, packed_hidden] uint8, v_meta [B, rank, 2] fp16/bf16,
 *           key_residual [half_B, key_res_dim, hidden_dim] fp16/bf16,
 *           val_residual [half_B, val_res_dim, hidden_dim] fp16/bf16
 *   Output: v_out [B, rank, hidden_dim] fp16/bf16
 */

#include "fused_dequant.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// Templated kernel 1: Fused u dequantize with transposed output
// ============================================================================
template <typename scalar_t>
__global__ void fused_dequant_u_transposed_kernel(
    const uint8_t* __restrict__ u_quantized,  // [B, rank, packed_len]
    const scalar_t* __restrict__ u_meta,      // [B, rank, 2]
    scalar_t* __restrict__ u_out,             // [B, kv_len, rank] (transposed)
    const int B,
    const int rank,
    const int packed_len,
    const int kv_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * rank * kv_len;
    if (idx >= total) return;

    const int kv_idx = idx % kv_len;
    const int rank_idx = (idx / kv_len) % rank;
    const int batch = idx / (kv_len * rank);

    // Load per-row min/max in native dtype
    const int meta_offset = (batch * rank + rank_idx) * 2;
    const scalar_t s_min = u_meta[meta_offset];
    const scalar_t s_max = u_meta[meta_offset + 1];

    // Unpack nibble from packed uint8
    const int q_base = (batch * rank + rank_idx) * packed_len;
    uint8_t val;
    if (kv_idx < packed_len) {
        val = u_quantized[q_base + kv_idx] >> 4;        // high nibble
    } else {
        val = u_quantized[q_base + kv_idx - packed_len] & 0x0F;  // low nibble
    }

    // Dequantize matching PyTorch: bf16(val) / 15 * (max - min) + min
    // Each intermediate rounds to scalar_t to match PyTorch element-wise semantics
    const scalar_t s_val = static_cast<scalar_t>(static_cast<float>(val));
    const scalar_t normalized = static_cast<scalar_t>(
        static_cast<float>(s_val) / 15.0f);
    const scalar_t s_range = static_cast<scalar_t>(
        static_cast<float>(s_max) - static_cast<float>(s_min));
    const scalar_t scaled = static_cast<scalar_t>(
        static_cast<float>(normalized) * static_cast<float>(s_range));
    const scalar_t dequant = static_cast<scalar_t>(
        static_cast<float>(scaled) + static_cast<float>(s_min));

    // Write in TRANSPOSED layout: u_out[batch, kv_idx, rank_idx]
    const int out_offset = batch * (kv_len * rank) + kv_idx * rank + rank_idx;
    u_out[out_offset] = dequant;
}


// ============================================================================
// Templated kernel 2: Fused v dequantize with residual overlay
// ============================================================================
template <typename scalar_t>
__global__ void fused_dequant_v_residual_kernel(
    const uint8_t* __restrict__ v_quantized,    // [B, rank, packed_hidden]
    const scalar_t* __restrict__ v_meta,        // [B, rank, 2]
    const scalar_t* __restrict__ key_residual,  // [half_B, key_res_dim, hidden_dim]
    const scalar_t* __restrict__ val_residual,  // [half_B, val_res_dim, hidden_dim]
    scalar_t* __restrict__ v_out,               // [B, rank, hidden_dim]
    const int B,
    const int rank,
    const int packed_hidden,
    const int hidden_dim,
    const int half_B,
    const int key_res_dim,
    const int val_res_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * rank * hidden_dim;
    if (idx >= total) return;

    const int col = idx % hidden_dim;
    const int row = (idx / hidden_dim) % rank;
    const int batch = idx / (hidden_dim * rank);

    const bool is_key_batch = (batch < half_B);
    const bool use_key_res = is_key_batch && (row < key_res_dim);
    const bool use_val_res = (!is_key_batch) && (row < val_res_dim);

    scalar_t result;

    if (use_key_res) {
        result = key_residual[(batch * key_res_dim + row) * hidden_dim + col];
    } else if (use_val_res) {
        result = val_residual[((batch - half_B) * val_res_dim + row) * hidden_dim + col];
    } else {
        const int meta_offset = (batch * rank + row) * 2;
        const scalar_t s_min = v_meta[meta_offset];
        const scalar_t s_max = v_meta[meta_offset + 1];

        const int q_base = (batch * rank + row) * packed_hidden;
        uint8_t val;
        if (col < packed_hidden) {
            val = v_quantized[q_base + col] >> 4;
        } else {
            val = v_quantized[q_base + col - packed_hidden] & 0x0F;
        }

        // Match PyTorch: each intermediate rounds to scalar_t
        const scalar_t s_val = static_cast<scalar_t>(static_cast<float>(val));
        const scalar_t normalized = static_cast<scalar_t>(
            static_cast<float>(s_val) / 15.0f);
        const scalar_t s_range = static_cast<scalar_t>(
            static_cast<float>(s_max) - static_cast<float>(s_min));
        const scalar_t scaled = static_cast<scalar_t>(
            static_cast<float>(normalized) * static_cast<float>(s_range));
        result = static_cast<scalar_t>(
            static_cast<float>(scaled) + static_cast<float>(s_min));
    }

    v_out[idx] = result;
}


// ============================================================================
// C++ wrapper functions with dtype dispatch
// ============================================================================

torch::Tensor fused_dequant_u_transposed(
    torch::Tensor u_quantized,   // [B, rank, packed_len] uint8, CUDA
    torch::Tensor u_meta,        // [B, rank, 2] fp16/bf16, CUDA
    int kv_len
) {
    TORCH_CHECK(u_quantized.is_cuda(), "u_quantized must be on CUDA");
    TORCH_CHECK(u_meta.is_cuda(), "u_meta must be on CUDA");
    TORCH_CHECK(u_quantized.dtype() == torch::kUInt8, "u_quantized must be uint8");
    TORCH_CHECK(u_meta.dtype() == torch::kFloat16 || u_meta.dtype() == torch::kBFloat16,
                "u_meta must be float16 or bfloat16");

    const at::cuda::CUDAGuard device_guard(u_quantized.device());

    const int B = u_quantized.size(0);
    const int rank = u_quantized.size(1);
    const int packed_len = u_quantized.size(2);

    auto u_out = torch::empty({B, kv_len, rank}, torch::TensorOptions()
        .dtype(u_meta.dtype()).device(u_quantized.device()));

    const int total = B * rank * kv_len;
    if (total == 0) {
        return u_out;
    }
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream(u_quantized.device().index());

    AT_DISPATCH_SWITCH(u_meta.scalar_type(), "fused_dequant_u",
        AT_DISPATCH_CASE(at::ScalarType::Half,
            [&] {
                fused_dequant_u_transposed_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    u_quantized.data_ptr<uint8_t>(),
                    u_meta.data_ptr<scalar_t>(),
                    u_out.data_ptr<scalar_t>(),
                    B, rank, packed_len, kv_len
                );
            }
        )
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,
            [&] {
                fused_dequant_u_transposed_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    u_quantized.data_ptr<uint8_t>(),
                    u_meta.data_ptr<scalar_t>(),
                    u_out.data_ptr<scalar_t>(),
                    B, rank, packed_len, kv_len
                );
            }
        )
    );

    return u_out;
}


torch::Tensor fused_dequant_v_residual(
    torch::Tensor v_quantized,      // [B, rank, packed_hidden] uint8, CUDA
    torch::Tensor v_meta,           // [B, rank, 2] fp16/bf16, CUDA
    torch::Tensor key_residual,     // [half_B, key_res_dim, hidden_dim] fp16/bf16, CUDA
    torch::Tensor val_residual,     // [half_B, val_res_dim, hidden_dim] fp16/bf16, CUDA
    int hidden_dim
) {
    TORCH_CHECK(v_quantized.is_cuda(), "v_quantized must be on CUDA");
    TORCH_CHECK(v_meta.is_cuda(), "v_meta must be on CUDA");
    TORCH_CHECK(v_meta.dtype() == torch::kFloat16 || v_meta.dtype() == torch::kBFloat16,
                "v_meta must be float16 or bfloat16");

    const at::cuda::CUDAGuard device_guard(v_quantized.device());

    const int B = v_quantized.size(0);
    const int rank = v_quantized.size(1);
    const int packed_hidden = v_quantized.size(2);
    const int half_B = B / 2;
    const int key_res_dim = key_residual.numel() > 0 ? key_residual.size(1) : 0;
    const int val_res_dim = val_residual.numel() > 0 ? val_residual.size(1) : 0;

    auto v_out = torch::empty({B, rank, hidden_dim}, torch::TensorOptions()
        .dtype(v_meta.dtype()).device(v_quantized.device()));

    const int total = B * rank * hidden_dim;
    if (total == 0) {
        return v_out;
    }
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream(v_quantized.device().index());

    AT_DISPATCH_SWITCH(v_meta.scalar_type(), "fused_dequant_v",
        AT_DISPATCH_CASE(at::ScalarType::Half,
            [&] {
                fused_dequant_v_residual_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    v_quantized.data_ptr<uint8_t>(),
                    v_meta.data_ptr<scalar_t>(),
                    key_residual.numel() > 0 ? key_residual.data_ptr<scalar_t>() : nullptr,
                    val_residual.numel() > 0 ? val_residual.data_ptr<scalar_t>() : nullptr,
                    v_out.data_ptr<scalar_t>(),
                    B, rank, packed_hidden, hidden_dim, half_B, key_res_dim, val_res_dim
                );
            }
        )
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,
            [&] {
                fused_dequant_v_residual_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    v_quantized.data_ptr<uint8_t>(),
                    v_meta.data_ptr<scalar_t>(),
                    key_residual.numel() > 0 ? key_residual.data_ptr<scalar_t>() : nullptr,
                    val_residual.numel() > 0 ? val_residual.data_ptr<scalar_t>() : nullptr,
                    v_out.data_ptr<scalar_t>(),
                    B, rank, packed_hidden, hidden_dim, half_B, key_res_dim, val_res_dim
                );
            }
        )
    );

    return v_out;
}
