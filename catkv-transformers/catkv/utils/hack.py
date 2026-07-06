# import types

# from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

from catkv.utils.model import attention_forward, decoder_forward, model_forward
from catkv.utils.rope import RotaryEmbeddingESM

def hack_model(
    model,
    attn_kwargs: dict = {},
    **kwargs
):
    attn_kwargs.update(kwargs)
    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    config = model.model.layers[0].self_attn.config
    Attention = model.model.layers[0].self_attn.__class__
    Model = model.model.__class__
    DecoderLayer = model.model.layers[0].__class__
    # RMSNorm = model.model.norm.__class__

    if isinstance(model, LlamaForCausalLM):
        embedding = LlamaRotaryEmbedding(config)
    elif isinstance(model, MistralForCausalLM):
        embedding = MistralRotaryEmbedding(config)
    elif isinstance(model, Qwen2ForCausalLM) or isinstance(model, Qwen3ForCausalLM):
        # NOTE: Add this config to make Qwen2 For Long Context 
        config.rope_scaling = {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn"
        }
        if isinstance(model, Qwen2ForCausalLM):
            embedding = Qwen2RotaryEmbedding(config)
        else:
            embedding = Qwen3RotaryEmbedding(config)
    else:
        raise ValueError("Only supports llama, mistral and qwen2 models.")
    
    if hasattr(config, "head_dim") and config.head_dim is not None:
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads

    # Customized rope
    rope = RotaryEmbeddingESM(
        embedding,
        head_dim,
        model.device
    )
    model.model.position_bias = rope

    def set_catkv_forward(m):
        # if isinstance(m, RMSNorm):
        #     m._old_forward = m.forward
        #     m.forward = types.MethodType(VllmRMSNorm.forward_cuda, m)
        #     m.hidden_size = m.weight.shape[0]
        #     m.variance_size_override = None
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = attention_forward.__get__(m, Attention)
        elif isinstance(m, DecoderLayer):
            m._old_forward = m.forward
            m.forward = decoder_forward.__get__(m, DecoderLayer)
               
    model.apply(set_catkv_forward)

    model.model._old_forward = model.model.forward
    model.model.forward = model_forward.__get__(model.model, Model)

    return model