# GPU Buffer to Paged Memory Timing

## Relevant Code

The key method is `batched_to_gpu` in `lmcache/v1/gpu_connector.py:500-566`.

## Pipeline

The method iterates with:

```python
for layer_id in range(self.num_layers + 2):
```

For a 32-layer model, this loop runs 34 iterations. Each iteration has three
stages.

### Stage 1: Write to Paged Memory

This runs when `layer_id > 1`:

```python
if layer_id > 1:
    lmc_ops.single_layer_kv_transfer(
        self.buffer_mapping[layer_id - 2].tensor,
        self.kvcaches[layer_id - 2],
        slot_mapping_full,
        ...
    )
    del self.buffer_mapping[layer_id - 2]
```

It writes layer `layer_id - 2`.

### Stage 2: RoPE Processing

This runs when `0 < layer_id <= num_layers`:

```python
compute_gpu_buffer_obj, load_gpu_buffer_obj = (
    load_gpu_buffer_obj,
    compute_gpu_buffer_obj,
)
compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(...)
compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0
self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj
```

It processes layer `layer_id - 1`.

### Stage 3: Load CPU Data

This runs when `layer_id < num_layers`:

```python
memory_objs_layer = yield
with torch.cuda.stream(self.load_stream):
    load_gpu_buffer_obj.tensor[0][...].copy_(
        memory_obj.tensor[0],
        non_blocking=True,
    )
```

It loads CPU data for layer `layer_id`.

## Timing Table

For layers 0 through 31:

| layer_id | Paged write | RoPE processing | CPU load |
|----------|-------------|-----------------|----------|
| 0 | none | none | load layer 0 |
| 1 | none | process layer 0 | load layer 1 |
| 2 | write layer 0 | process layer 1 | load layer 2 |
| 3 | write layer 1 | process layer 2 | load layer 3 |
| ... | ... | ... | ... |
| 31 | write layer 29 | process layer 30 | load layer 31 |
| 32 | write layer 30 | process layer 31 | none |
| 33 | write layer 31 | none | none |

## Key Points

The first paged-memory write happens at `layer_id = 2`, where layer 0 is written
after RoPE processing. The pipeline has a two-layer lag: when processing layer
N, the layer written to paged memory is `N - 2`.

The lag is required because CPU data must first arrive in `load_buffer`, then
the buffers are swapped, then RoPE and blending run on `compute_buffer`, and
only then can the result be written to paged memory.

For layer 5, the sequence is:

```text
layer_id=5: load CPU data for layer 5
layer_id=6: process layer 5 and place it in buffer_mapping
layer_id=7: write layer 5 to paged memory
```

## Blending

In blend mode, the buffer is visible through `buffer_mapping`:

```python
old_k, old_v = self.gpu_connector.get_kv(layer_id)
```

The blender updates the buffer for that layer. Two iterations later, the same
buffer is written to paged memory, including the blending changes.

## Summary

Layer N is written to paged memory at `layer_id = N + 2` through
`lmc_ops.single_layer_kv_transfer`. This happens after RoPE recovery and
blending. The design keeps memory usage low by reusing two buffers while
preserving correctness.
