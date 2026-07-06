#pragma once

#include <torch/extension.h>

torch::Tensor fused_dequant_u_transposed(
    torch::Tensor u_quantized,
    torch::Tensor u_meta,
    int kv_len);

torch::Tensor fused_dequant_v_residual(
    torch::Tensor v_quantized,
    torch::Tensor v_meta,
    torch::Tensor key_residual,
    torch::Tensor val_residual,
    int hidden_dim);
