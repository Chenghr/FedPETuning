"""BaseModel for FedPEFTuning"""

import copy
from abc import ABC
from utils import registry
from models.utils import PromptType

import torch
import torch.nn as nn
from transformers import AutoConfig
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

from opendelta import AutoDeltaConfig
from opendelta.auto_delta import AutoDeltaModel


class BaseModels(nn.Module, ABC):
    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name

        config = registry.get("config")
        self.model_config = config.model_config
        self.training_config = config.training_config
        self.rank = config.federated_config.rank
        self.logger = registry.get("logger")

    def _build_config(self, **kwargs):
        auto_config = AutoConfig.from_pretrained(
            self.model_config.config_name if self.model_config.config_name else self.model_config.model_name_or_path,
            finetuning_task=self.task_name if self.task_name else None,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            **kwargs
        )
        return auto_config

    def _build_model(self):
        backbone = self._add_base_model()
        # if getattr(self.model_config, "permutation_layers", None):
        #     backbone = self._add_permutate_layers(backbone)

        if self.training_config.tuning_library == "opendelta":
            if self.training_config.tuning_type:
                backbone = self._add_delta_model(backbone)
        else:
            backbone = self._add_peft_model(backbone)

        return backbone
    
    def forward(self, inputs):
        raise NotImplementedError

    def _add_base_model(self):
        raise NotImplementedError

    def _add_delta_model(self, backbone):
        # 基于 opendelta 库中的 AutoDeltaConfig 类以及 AutoDeltaModel 实现 PEFT 方法
        if any([True for PType in PromptType if PType in self.training_config.tuning_type]):
            # 如果在配置中指定了前缀调优，可能在 OpenDelta 中
            ...
        else:
            delta_args = registry.get("peft_config")
            delta_config = AutoDeltaConfig.from_dict(delta_args)
            delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backbone)
            delta_model.freeze_module(set_state_dict=True)

        return backbone
    
    def _add_peft_model(self, backbone):
        peft_config = self.training_config.peft_config
        backbone = get_peft_model(backbone, peft_config)
        backbone.print_trainable_parameters()

        return backbone
    
    def _add_permutate_layers(self, backbone):
        # TODO：当前只支持 BERT-NLU (自然语言理解)任务
        bert_modules = self._get_bert_module(backbone)
        old_modules = bert_modules.encoder.layer
        scrambled_modules = torch.nn.ModuleList()

        if self.rank > 0:
            permutation = self.model_config.client_model_layers
        else:
            permutation = self.model_config.server_model_layers
        self.logger.debug(f"model's layer: {permutation}")
        for i in permutation:
            assert i <= 11, permutation
            scrambled_modules.append(old_modules[i])

        backbone_copy = copy.deepcopy(backbone)
        bert_modules_copy = self._get_bert_module(backbone_copy)
        bert_modules_copy.encoder.layer = scrambled_modules
        return backbone_copy

    def _get_bert_module(self, backbone):
        # 获取 BERT 模块的方法
        if self.model_config.model_type == "bert":
            return backbone.bert
        elif self.model_config.model_type == "roberta":
            return backbone.roberta
        elif self.model_config.model_type == "distilbert":
            return backbone.distilbert
        elif self.model_config.model.type == "llama":
            return backbone.llama
        else:
            raise NotImplementedError