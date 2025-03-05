"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # Skip the input_layernorm
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
        if layer_id == 0:
            del self.input_layernorm
            setattr(self, "input_layernorm", lambda x: x)


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config, i, quant_config=quant_config, prefix=f"model.layers.{i}"
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        hidden_states = self.fc(
            torch.cat((hidden_states, forward_batch.spec_info.hidden_states), dim=-1)
        )

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        return hidden_states + residual


class LlamaForCausalLMEagle(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_config=quant_config)
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if hasattr(config, "hot_vocab_size"):
                self.lm_head = ParallelLMHead(
                    config.hot_vocab_size, config.hidden_size, quant_config=quant_config
                )
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size, config.hidden_size, quant_config=quant_config
                )
        self.logits_processor = LogitsProcessor(config)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
                super().load_weights([(name, loaded_weight)])


EntryClass = [LlamaForCausalLMEagle]
