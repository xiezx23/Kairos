import torch
from transformers import Cache
from edq.quantization import *

# In Qwen2Attention.forward:
# if past_key_value is not None:
#     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

class CacheLayer():
    def __init__(self) -> None:
        self.key_cache, self.val_cache = None, None

    def update(
        self,
        key_states: torch.Tensor,
        val_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache is None:
            self.key_cache = key_states
            self.val_cache = val_states
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=-2)
            self.val_cache = torch.cat([self.val_cache, val_states], dim=-2)
        return self.key_cache, self.val_cache
    
    def get_seq_length(self, cache_position=None) -> int:
        if self.key_cache is None or self.key_cache.numel() == 0:
            return 0
        return self.key_cache.shape[-2]

class CacheManager(Cache):
    def __init__(self) -> None:
        super().__init__()
        self.layers = []
    
    def update(
        self,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs = None
    ):
        while len(self.layers) <= layer_idx:
            self.layers.append(CacheLayer())
        return self.layers[layer_idx].update(key_states, val_states)
    
    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length(cache_position)
    

class QuantCacheLayer(CacheLayer):
    def __init__(self, q_bit = 8) -> None:
        super().__init__()
        self.q_bit = q_bit
        self.key_config = None
        self.val_config = None
        self.return_dtype = None

    def update(
        self,
        key_states: torch.Tensor,
        val_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache is not None:
            key_states = torch.cat([self._dequant_k(), key_states], dim=-2)
            val_states = torch.cat([self._dequant_v(), val_states], dim=-2)
        else:
            self.return_dtype = key_states.dtype
        self._quant_k(key_states)
        self._quant_v(val_states)
        return key_states, val_states
    
    def get_seq_length(self, cache_position=None) -> int:
        if self.key_cache is None or self.key_cache.numel() == 0:
            return 0
        return self.key_cache.shape[-2]

    def _quant_k(self, tensor):
        self.key_cache, self.key_config = quantize_tensor_int8(tensor, 
                                                               return_dtype=self.return_dtype)
    def _quant_v(self, tensor):
        self.val_cache, self.val_config = quantize_tensor_int8(tensor, 
                                                               return_dtype=self.return_dtype)
    def _dequant_k(self):
        return dequantize_tensor_int8(self.key_cache, self.key_config['scale'], return_dtype=self.return_dtype)
    def _dequant_v(self):
        return dequantize_tensor_int8(self.val_cache, self.val_config['scale'], return_dtype=self.return_dtype)


class QuantCacheManager(Cache):
    def __init__(self) -> None:
        super().__init__()
        self.layers = []
    
    def update(
        self,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs = None
    ):
        while len(self.layers) <= layer_idx:
            self.layers.append(QuantCacheLayer())
        return self.layers[layer_idx].update(key_states, val_states)
    
    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length(cache_position)

    def __getitem__(self, layer_idx):
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].key_cache, self.layers[layer_idx].val_cache
        else:
            raise KeyError(f"Cache only has {len(self.layers)} layers, "
                           f"attempted to access layer with index {layer_idx}")

    def __len__(self):
        return len(self.layers)
