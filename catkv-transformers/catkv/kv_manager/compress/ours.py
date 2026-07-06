

import logging
import time

import torch
import xxhash
from catkv.kv_manager.compress.abstract_compress import AbstractCompress
from catkv.kv_manager.compress.budget_utils import UniformAllocate, allocate_low_rank
from catkv.kv_manager.compress.residual_utils import calculate_elbow

def timestamp_to_tensor():
    ts = time.time_ns()
    
    h = xxhash.xxh64(str(ts)).intdigest()  # 64-bit
    
    high = (h >> 32) & 0xFFFFFFFF
    low  = h & 0xFFFFFFFF
    
    # Convert to signed int32.
    def to_int32(x):
        return x - (1 << 32) if x >= (1 << 31) else x

    high = to_int32(high)
    low = to_int32(low)

    return torch.tensor([high, low], dtype=torch.int32)

logger = logging.getLogger(__name__)

import catkv_ops as _fused_dequant_ext
_fused_dequant_loaded = False


def _prepare_matrix(A: torch.Tensor) -> torch.Tensor:
    if A.dim() == 3:
        A = A.squeeze(0)
    if A.dtype not in (torch.float32, torch.float64):
        A = A.float()
    return A

def _gram_matrix(A: torch.Tensor) -> torch.Tensor:
    A = _prepare_matrix(A)
    m, n = A.shape
    return A.T @ A if m >= n else A @ A.T

def sigma_min_power_iter(
    A: torch.Tensor,
    eps: float = 1e-12,
    num_iters: int = 4,
) -> torch.Tensor:
    """Fast inverse power iteration for minimum singular value.

    Optimized with:
    - Minimal iterations (6 for speed)
    - LU factorization reuse
    - Minimal allocations
    """
    B = _gram_matrix(A)
    dim = B.shape[0]

    # Initialize random vector
    v = torch.randn(dim, dtype=B.dtype, device=B.device)
    v = v / torch.norm(v)

    # Estimate shift
    trace = torch.trace(B)
    shift = trace / dim * 0.05

    eye = torch.eye(dim, dtype=B.dtype, device=B.device)
    B_shifted = B + shift * eye

    # Pre-factorize LU
    try:
        lu, pivots = torch.linalg.lu_factor(B_shifted)
    except:
        # Fallback
        for _ in range(num_iters):
            v = torch.linalg.solve(B_shifted, v)
            v_norm = torch.norm(v)
            if v_norm < 1e-10:
                break
            v = v / v_norm
        lam_min = torch.dot(v, B @ v)
        lam_min = torch.clamp(lam_min, min=0.0)
        return torch.sqrt(lam_min + eps)

    # LU-accelerated iteration
    for _ in range(num_iters):
        v = torch.linalg.lu_solve(lu, pivots, v.unsqueeze(-1)).squeeze(-1)
        v_norm = torch.norm(v)
        if v_norm < 1e-10:
            break
        v = v / v_norm

    # Rayleigh quotient
    lam_min = torch.dot(v, B @ v)
    lam_min = torch.clamp(lam_min, min=0.0)

    return torch.sqrt(lam_min + eps)

class Ours(AbstractCompress):
    def __init__(self, layer_num: int = 0, device="cuda", config: dict = None):
        super().__init__(device=device)
        config = config or {}
        self.layer_num = layer_num
        self.ratio = config.get("ratio", 0.1)

    def sq_compress(self, data, uuid = None) -> dict:
        u , v = data
        assert u.device != "cpu", "SVD compression only supports CUDA tensors"

        u_min = torch.min(u, dim=-1, keepdim=True).values
        u_max = torch.max(u, dim=-1, keepdim=True).values
        u_range = u_max - u_min + 1e-6
        u_normalized = (u - u_min) / u_range
        u_quantized = torch.clamp((u_normalized * 15).round(), 0, 15).to(torch.uint8)
        
        u_chunk = u_quantized.chunk(dim = -1 , chunks=2)
        # padding chunk 1 to make shapes match
        if u_chunk[0].shape[-1] != u_chunk[1].shape[-1]:
            pad_size = u_chunk[0].shape[-1] - u_chunk[1].shape[-1]
            u_chunk_1_padded = torch.nn.functional.pad(u_chunk[1], (0, pad_size), mode='constant', value=0)        
        else:
            u_chunk_1_padded = u_chunk[1]

        u_quantized = u_chunk[0] * 16 + u_chunk_1_padded
        u_meta = torch.cat([u_min, u_max], dim=-1).cpu()
        
        v_min = torch.min(v, dim=-1, keepdim=True).values
        v_max = torch.max(v, dim=-1, keepdim=True).values
        v_range = v_max - v_min + 1e-6
        v_normalized = (v - v_min) / v_range
        v_quantized = torch.clamp((v_normalized * 15).round(), 0, 15).to(torch.uint8)
        v_chunk = v_quantized.chunk(dim = -1 , chunks=2)
        if v_chunk[0].shape[-1] != v_chunk[1].shape[-1]:
            pad_size = v_chunk[0].shape[-1] - v_chunk[1].shape[-1]
            v_chunk_1_padded = torch.nn.functional.pad(v_chunk[1], (0, pad_size), mode='constant', value=0)
        else:
            v_chunk_1_padded = v_chunk[1]

        v_quantized = v_chunk[0] * 16 + v_chunk_1_padded
        v_meta = torch.cat([v_min, v_max], dim=-1).cpu()

        split_dim = v.size(0) // 2
        residual_key_sv = v[:split_dim,:self.key_residual_dim,:]
        residual_value_sv = v[split_dim:,:self.value_residual_dim,:]
        # print(f"Compressing with ratio {self.ratio}: original u shape {u.shape}, v shape {v.shape}, quantized u shape {u_quantized.shape}, v shape {v_quantized.shape}, residual_key_sv shape {residual_key_sv.shape}, residual_value_sv shape {residual_value_sv.shape}")
        result =  {
            f"u_quantized": u_quantized.cpu().contiguous(),
            f"u_meta": u_meta.cpu(),

            f"key_sv_quantized": v_quantized[:split_dim].cpu().contiguous(),
            f"key_sv_meta": v_meta[:split_dim].cpu(),

            f"value_sv_quantized": v_quantized[split_dim:].cpu().contiguous(),
            f"value_sv_meta": v_meta[split_dim:].cpu(),

            f"key_residual_sv": residual_key_sv.cpu(),
            f"value_residual_sv": residual_value_sv.cpu(),
        }

        if uuid is not None:
            result["uuid"] = uuid.cpu()

        return result

    def sq_decompress(self, compressed_data ,kv_len) -> tuple:
        return self._sq_decompress_combined(compressed_data, kv_len)
    
    def _sq_decompress_combined(self, compressed_data, kv_len) -> tuple:
        u_quantized = compressed_data[f"u_quantized"]
        u_meta = compressed_data[f"u_meta"]
        v_quantized = compressed_data[f"v_quantized"]
        v_meta = compressed_data[f"v_meta"]
        residual_key_v = compressed_data[f"key_residual_sv"]
        residual_value_v = compressed_data[f"value_residual_sv"]

        u_0 = u_quantized // 16
        u_1 = u_quantized % 16
        u_quantized = torch.cat([u_0, u_1], dim=-1).to(torch.uint8)

        split_dim = u_meta.size(-1) // 2
        type = u_meta.dtype
        u_dequantized = u_quantized.to(type) / 15 * (u_meta[: , : , split_dim:] - u_meta[: , : , :split_dim]) + u_meta[: , : , :split_dim]
        u_dequantized = u_dequantized[:, : , :kv_len]
      
        v_0 = v_quantized // 16
        v_1 = v_quantized % 16
        v_quantized = torch.cat([v_0, v_1], dim=-1).to(torch.uint8)
        v_dequantized = v_quantized.to(type) / 15 * (v_meta[: , : , split_dim:] - v_meta[: , : , :split_dim]) + v_meta[: , : , :split_dim]

        split_dim = v_dequantized.size(0) // 2

        residual_key_v_dim = min(v_dequantized.size(1), residual_key_v.size(1))
        residual_value_v_dim = min(v_dequantized.size(1), residual_value_v.size(1))

        v_dequantized[:split_dim, :residual_key_v_dim, :] = residual_key_v[: ,:residual_key_v_dim, :]
        v_dequantized[split_dim:, :residual_value_v_dim, :] = residual_value_v[: ,:residual_value_v_dim, :]
        return u_dequantized, v_dequantized

    def svd(self, x , rank):
        num_type = x.dtype
        # print(f"Performing SVD on tensor of shape {x.shape} with target rank {rank}")
        try:
            u , s , v = torch.svd_lowrank(x.reshape(-1, x.shape[1], x.shape[2] * x.shape[3]).float(), rank)
        except RuntimeError as e:
            u , s , v = torch.svd(x.reshape(-1, x.shape[1], x.shape[2] * x.shape[3]).float())
            u = u[:, : , : rank]
            s = s[:, : rank]
            v = v[:, : rank, :].transpose(-1, -2)
      
        s_min = sigma_min_power_iter(x.reshape(-1, x.shape[1], x.shape[2] * x.shape[3]).float())
        # print(f"fake s_min: {s_min} , actual s_min {s.min()}")
        u = u.to(num_type)
        return u , s , v.transpose(-1, -2) , s_min

    def gpu_compress_bacth(self, kv):
        num_layers , seq_len , num_heads , head_dim = kv[0].shape
        total_original_mem_bytes =  seq_len * num_heads * head_dim * (kv[0].element_size()) * self.ratio
        final_rank = UniformAllocate().allocate(
            total_budget_bytes=total_original_mem_bytes,
                num_layers=1,
                dim=head_dim * num_heads,
                token_num=seq_len
            )
        _ = torch.svd_lowrank(kv[0].reshape(kv[0].shape[0], kv[0].shape[1], -1).float(), final_rank)
        _ = torch.svd_lowrank(kv[1].reshape(kv[1].shape[0], kv[1].shape[1], -1).float(), final_rank)


    def compress(self, all_layer_kv: list, scores: list = None, layer_idx: int = 0 ) -> list:
        """
        Compresses KV caches for all layers with adaptive rank allocation.

        Args:
            all_layer_kv (list): A list of (key, value) tuples.
            all_scale_factors (list): A list of floats, one importance score for each layer.
            global_compression_ratio (float): The target compression ratio for the entire KV cache.

        Returns:
            list: A list of compressed data dictionaries, one for each layer.
        """
        # 1. Calculate original total memory and target budget
        key , value = all_layer_kv

        num_layers , seq_len , num_heads , head_dim = key.shape
        total_original_mem_bytes = num_layers * seq_len * num_heads * head_dim * (key.element_size()) * self.ratio
        # 2. Gather metadata for rank calculation
        final_ranks = UniformAllocate().allocate(
            total_budget_bytes=total_original_mem_bytes,
            num_layers=1,
            dim=head_dim * num_heads,
            token_num=seq_len
        )
        # 3. SVD
        key_u , key_s , key_v , key_s_min = self.svd(key, final_ranks)
        value_u , value_s , value_v , value_s_min = self.svd(value, final_ranks)
        key_residual_dim = int(calculate_elbow(key_s, max_x = min(num_heads*head_dim , seq_len) , min_s = key_s_min))
        value_residual_dim = int(calculate_elbow(value_s, max_x = min(num_heads*head_dim , seq_len) , min_s = value_s_min))

        max_residual_rank = int(max(final_ranks - 1, 0))
        if self.ratio < 0.1:
            max_residual_rank = int(max(final_ranks - 1, 0) * 0.1)
        self.key_residual_dim = min(key_residual_dim, max_residual_rank)
        self.value_residual_dim = min(value_residual_dim, max_residual_rank)
        
        high_rank = (self.key_residual_dim + self.value_residual_dim) // 2
        low_rank, high_rank = allocate_low_rank(total_original_mem_bytes, head_dim * num_heads, seq_len, high_rank)
        final_rank = int(low_rank + high_rank)

        # only one layer data
        u = torch.cat([key_u[:, : , : final_rank], value_u[:, : , : final_rank]], dim=0).transpose(-1, -2)
        s = torch.cat([key_s[0: 1, : final_rank], value_s[0 : 1, : final_rank]], dim=0)
        v = torch.cat([key_v[: , : final_rank, :], value_v[:, : final_rank, :]], dim=0)    
        sv = torch.diag_embed(s) @ v

        # Store for debugging purposes.
        data = self.sq_compress((u, sv.to(u.dtype)),uuid=timestamp_to_tensor())

        return data
    
    def transfer(self, compressed_data):
        for key in compressed_data:
            compressed_data[key] = compressed_data[key].to(self.device, non_blocking=True)
        return compressed_data

    def decompress(self, compressed_data, kv_len): 
        
        rank = min(compressed_data["key_sv_quantized"].shape[-2] , compressed_data["value_sv_quantized"].shape[-2])
        # print(f"Decompressing with rank {rank}: key_sv_quantized shape {compressed_data['key_sv_quantized'].shape}, value_sv_quantized shape {compressed_data['value_sv_quantized'].shape}")
        compressed_data["v_quantized"] = torch.cat([compressed_data["key_sv_quantized"][:,:rank], compressed_data["value_sv_quantized"][:,:rank ]], dim=0)
        compressed_data["v_meta"] = torch.cat([compressed_data["key_sv_meta"][:,:rank], compressed_data["value_sv_meta"][:,:rank]], dim=0)
        compressed_data["u_quantized"] = compressed_data["u_quantized"][:, : rank, :]
        compressed_data["u_meta"] = compressed_data["u_meta"][:, : rank, : ]

        ext = _fused_dequant_ext
        if (
            ext is not None
            and compressed_data["u_quantized"].is_cuda
            and compressed_data["u_meta"].is_cuda
            and compressed_data["v_quantized"].is_cuda
            and compressed_data["v_meta"].is_cuda
        ):
            return self._fused_decompress(ext, compressed_data, kv_len)
        u, sv = self.sq_decompress(compressed_data, kv_len)
        u = u.transpose(-1, -2).contiguous()
        kv = torch.matmul(u, sv)
        return kv[0], kv[1]

    def _fused_decompress(self, ext, compressed_data, kv_len):
        # print(f"keys : {compressed_data.keys()}")
        u_quantized = compressed_data["u_quantized"]
        u_meta = compressed_data["u_meta"]
        v_quantized = compressed_data["v_quantized"]
        v_meta = compressed_data["v_meta"]
        key_residual_sv = compressed_data["key_residual_sv"]
        value_residual_sv = compressed_data["value_residual_sv"]
        hidden_dim = key_residual_sv.shape[-1] if key_residual_sv.numel() > 0 else value_residual_sv.shape[-1]
        u = ext.fused_dequant_u_transposed(u_quantized, u_meta, kv_len)
        sv = ext.fused_dequant_v_residual(v_quantized, v_meta, key_residual_sv, value_residual_sv, hidden_dim)
        # print(f"Fused dequant shapes: u {u.shape}, sv {sv.shape}")
        # print(f"u quantized shape: {u_quantized.shape}, u meta shape: {u_meta.shape}, v quantized shape: {v_quantized.shape}, v meta shape: {v_meta.shape}, key_residual_sv shape: {key_residual_sv.shape}, value_residual_sv shape: {value_residual_sv.shape}")
        kv = torch.matmul(u, sv)
        return kv[0], kv[1]

 
    def compress_multi(self, layer_kv: list, indices = None, uuid = None):
        key , value = layer_kv
        num_layers , seq_len , num_heads , head_dim = key.shape
        group_size = len(indices) 

        total_original_mem_bytes = num_layers * seq_len * num_heads * head_dim * (key.element_size()) * self.ratio / group_size

        final_ranks = UniformAllocate().allocate(
            total_budget_bytes=total_original_mem_bytes,
            num_layers=1,
            dim=head_dim * num_heads,
            token_num=seq_len // group_size
        ) 
        # print(f"compress_multi: total_original_mem_bytes={total_original_mem_bytes}, final_ranks={final_ranks}, group_size={group_size}")
        key_u , key_s , key_v , key_s_min = self.svd(key , final_ranks)
        key_residual_dim = int(calculate_elbow(key_s, max_x = min(num_heads*head_dim , seq_len) , min_s = key_s_min))
        max_residual_rank = int(max(final_ranks - 1, 0))
        if self.ratio < 0.1:
            max_residual_rank = int(max(final_ranks - 1, 0) * 0.1)
        self.key_residual_dim = min(key_residual_dim, max_residual_rank)
        
        data_list = []
        if uuid is None:
            uuid = timestamp_to_tensor()

        for indice in indices:
            value_u , value_s , value_v , value_s_min = self.svd(value[:, indice[0]:indice[1], :, :] , final_ranks)

            value_residual_dim = int(calculate_elbow(value_s, max_x = min(num_heads*head_dim , seq_len) , min_s = value_s_min))
            self.value_residual_dim = min(value_residual_dim, max_residual_rank)

            # only one layer data
            # print(f"key_u shape {key_u.shape} , value_u shape {value_u.shape} , final_ranks {final_ranks}, indice {indice}")
            final_rank = min(final_ranks , indice[1] - indice[0])
            u = torch.cat([key_u[:, indice[0]:indice[1], : final_rank], value_u[:, : , : final_rank]], dim=0).to(torch.bfloat16)
            s = torch.cat([key_s[0: 1, : final_rank], value_s[0 : 1, : final_rank]], dim=0)
            v = torch.cat([key_v[: , : final_rank, :], value_v[:, : final_rank, :]], dim=0)    
            sv = (torch.diag_embed(s) @ v).to(torch.bfloat16)   

            # truncate u and sv for residual
            # compute the rank by compute ratio
            high_rank = (self.key_residual_dim + self.value_residual_dim) // 2
            low_rank, high_rank = allocate_low_rank(total_original_mem_bytes, head_dim * num_heads, seq_len // group_size, high_rank)
            final_rank = int(low_rank + high_rank)
            # print(f"chunk {indice} : key_residual_dim={self.key_residual_dim}, value_residual_dim={self.value_residual_dim}, low_rank={low_rank}, high_rank={high_rank}, final_rank={final_rank}")
            u = u[:, : , : final_rank]
            sv = sv[:, : final_rank, :]
            data_list.append(
                self.sq_compress(
                    ( u.transpose(-1, -2),
                        sv.to(u.dtype)
                    )
                    ,uuid = uuid
                )
            )
        
        return data_list

    
