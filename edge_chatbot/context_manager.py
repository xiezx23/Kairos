from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import Cache

from utils.color_print import colored_print
from transformers.cache_utils import DynamicCache
from edq.kv_cache import CacheManager, QuantCacheManager


class ContextManager:
    """
    管理多轮对话的上下文，包括对话历史和KV Cache
    支持自动截断以避免超出模型上下文长度限制
    """

    def __init__(
            self,
            tokenizer,
            model_max_length: int = 4096,
            max_ctx_to_keep: Optional[int] = 0.75,
            user_role: str = "user",
            assistant_role: str = "assistant",
            quant_kv: bool = False,
            truncate_kv: bool = False,
            verbose: int = 1,
    ):
        """
        初始化上下文管理器

        Args:
            tokenizer: 模型对应的tokenizer
            model_max_length: 模型支持的最大上下文长度
            max_ctx_to_keep: 要保留的最大上下文长度
            user_role: 用户角色的标识
            assistant_role: 助手角色的标识
            quant_kv: 是否做KV cache量化
            truncate_kv: 是否截断KV cache
            verbose: 打印信息等级
        """
        assert not (quant_kv and truncate_kv), "KV cache quantization does not currently support truncation!"
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        # self.max_ctx_to_keep = max_ctx_to_keep or round(model_max_length * 0.75)
        if isinstance(max_ctx_to_keep, float) and 0 < max_ctx_to_keep < 1:
            self.max_ctx_to_keep = round(model_max_length * max_ctx_to_keep)
        elif isinstance(max_ctx_to_keep, (int, float)) and max_ctx_to_keep > 1:
            # 如果是正整数或大于等于1的浮点数，则表示绝对数值
            self.max_ctx_to_keep = int(max_ctx_to_keep)
        else:
            # 其他情况(包括负数)使用默认值
            self.max_ctx_to_keep = round(model_max_length * 0.75)

        self.user_role = user_role
        self.assistant_role = assistant_role

        self.conversation_history: List[Dict[str, str]] = []
        self.truncate_kv = truncate_kv
        self.quant_kv = quant_kv
        if self.quant_kv:
            self.past_key_values = QuantCacheManager()
        else:
            self.past_key_values = None

    def add_user_message(self, content: str) -> None:
        self.conversation_history.append({"role": self.user_role, "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.conversation_history.append({"role": self.assistant_role, "content": content})

    def get_current_token_count(self) -> int:
        """计算当前对话历史的token长度"""
        # TODO: 减少冗余处理
        prompt = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs.input_ids.shape[1]

    def _truncate_kv_cache(self, truncate_length: int) -> None:
        """
        截断past_key_values，保留从truncate_length开始之后的内容

        Args:
            truncate_length: 要截断的长度（保留从该位置开始的内容）
        """
        if self.past_key_values is None or truncate_length <= 0:
            return
        # if isinstance(self.past_key_values, (DynamicCache, CacheManager, QuantCacheManager)):
        if isinstance(self.past_key_values, (DynamicCache, )):
            # 创建与原缓存相同类型的新缓存实例
            new_cache = type(self.past_key_values)()
            for layer_idx in range(len(self.past_key_values)):
                k, v = self.past_key_values[layer_idx]
                k = k[:, :, truncate_length:, :]
                v = v[:, :, truncate_length:, :]
                new_cache.update(k, v, layer_idx)
            self.past_key_values = new_cache
        elif isinstance(self.past_key_values, (CacheManager, QuantCacheManager)):
            raise NotImplementedError('KV cache quantization does not currently support truncation!')

        else:
            assert isinstance(self.past_key_values, list), "past_key_values must be a list of tuples"
            # 如果是元组列表（legacy格式）
            truncated_kv = []
            for layer_k, layer_v in self.past_key_values:
                # layer_k 和 layer_v 的shape: [batch_size, num_heads, seq_len, head_dim]
                # 沿着序列长度维度进行切片
                truncated_kv.append((
                    layer_k[:, :, truncate_length:, :],
                    layer_v[:, :, truncate_length:, :]
                ))
            self.past_key_values = truncated_kv

    def _truncate_conversation_history(self, tokens_to_remove: int) -> int:
        """
        截断对话历史，从前面开始删除消息，直到减少至少tokens_to_remove个token

        Args:
            tokens_to_remove: 需要减少的token数量

        Returns:
            实际减少的token数量
        """
        if tokens_to_remove <= 0 or len(self.conversation_history) <= 1:
            return 0

        original_length = self.get_current_token_count()
        removed_tokens = 0

        # 从对话历史前面开始删除消息，直到满足长度要求
        while removed_tokens < tokens_to_remove and len(self.conversation_history) > 1:
            # 保存当前状态以便回滚
            old_history = self.conversation_history.copy()

            # 删除最老的消息
            removed_msg = self.conversation_history.pop(0)

            # 计算新的token长度
            # FIXME: 这种做法是否太耗时？
            new_length = self.get_current_token_count()
            removed_this_round = original_length - new_length

            # 如果删除了用户消息，确保也删除对应的助手回复
            if (removed_msg['role'] == self.user_role and
                    len(self.conversation_history) > 0 and
                    self.conversation_history[0]['role'] == self.assistant_role):
                # 删除助手回复
                self.conversation_history.pop(0)
                new_length = self.get_current_token_count()
                removed_this_round = original_length - new_length

            removed_tokens += removed_this_round
            original_length = new_length

            # 如果删除后历史为空，恢复最后一条消息并跳出循环
            if len(self.conversation_history) == 0:
                self.conversation_history = old_history
                break

        return removed_tokens

    def truncate_if_needed(self) -> bool:
        """
        检查并执行必要的截断操作

        Returns:
            bool: 是否执行了截断操作
        """
        if not self.truncate_kv:
            return False

        current_length = self.get_current_token_count()

        if current_length <= self.max_ctx_to_keep:
            return False

        # 计算需要丢弃的token数量
        tokens_to_remove = current_length - self.max_ctx_to_keep

        # 截断对话历史
        actual_removed = self._truncate_conversation_history(tokens_to_remove)

        # 截断KV Cache
        if self.past_key_values is not None and actual_removed > 0:
            current_kv_length = self.past_key_values[0][0].shape[2]
            if actual_removed < current_kv_length:
                self._truncate_kv_cache(actual_removed)
            else:
                # 如果要截断的长度超过现有缓存，重置整个缓存
                self.past_key_values = None

        if self.verbose > 0 and actual_removed > 0:
            remaining_length = self.get_current_token_count()
            colored_print(f"上下文长度溢出: {current_length} / {self.max_ctx_to_keep} tokens", color="yellow")
            colored_print(f"\t需要截断: {tokens_to_remove} tokens", color="yellow")
            colored_print(f"\t实际截断: {actual_removed} tokens", color="yellow")
            colored_print(f"\t剩余长度: {remaining_length} tokens", color="yellow")
        return True

    def get_prompt(self) -> str:
        """生成当前对话历史的prompt"""
        return self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

    def reset(self) -> None:
        """重置对话历史和KV Cache"""
        self.conversation_history.clear()
        self.past_key_values = None

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取当前对话历史"""
        return self.conversation_history.copy()

    def set_past_key_values(self, past_key_values) -> None:
        """设置当前的past_key_values"""
        self.past_key_values = past_key_values

    def get_past_key_values(self):
        """获取当前的past_key_values"""
        return self.past_key_values
