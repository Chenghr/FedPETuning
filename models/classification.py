"""SeqClassification Model For FedETuning """

import copy
from abc import ABC
import torch
from utils import registry
from models.base_models import BaseModels
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, LlamaForSequenceClassification


@registry.register_model("seq_classification")
class SeqClassification(BaseModels, ABC):
    def __init__(self, task_name):
        super().__init__(task_name)

        self.num_labels = registry.get("num_labels")
        self.auto_config = self._build_config(num_labels=self.num_labels)
        self.backbone = self._build_model()

    def _add_base_model(self):
        # check whether AutoModelForSequenceClassification support llama.
        if self.model_config.model_type and "llama" in self.model_config.model_type:
            backbone = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name_or_path, 
                config=self.auto_config,
                revision=self.model_config.model_revision,
                use_auth_token=True if self.model_config.use_auth_token else None,
            )
        else:
            backbone = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_config.model_name_or_path),
                config=self.auto_config,
                # cache_dir=self.model_config.cache_dir,
                revision=self.model_config.model_revision,
                use_auth_token=True if self.model_config.use_auth_token else None,
                # ignore_mismatched_sizes=self.model_config.ignore_mismatched_sizes,
            )
        return backbone

    def forward(self, inputs):
        print(inputs.keys())
        if self.model_config.model_type in ["llama"]:
            inputs.pop("token_type_ids", None)
        print(f"input_ids shape: {inputs['input_ids'].shape}")
        print(f"attention_mask shape: {inputs['attention_mask'].shape}")
        print(f"labels shape: {inputs['labels'].shape}")

        # 数组索引越界
        # output = self.backbone(**inputs)
        output = self.backbone(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'],
            )
        return output


@registry.register_model("token_classification")
class TokenClassification(BaseModels, ABC):
    def __init__(self, task_name):
        super().__init__(task_name)

        self.num_labels = registry.get("num_labels")
        self.id2label = registry.get("id2label")
        self.label2id = registry.get("label2id")

        self.auto_config = self._build_config(num_labels=self.num_labels)
        self.backbone = self._build_model()

    def _build_model(self):
        backbone = AutoModelForTokenClassification.from_pretrained(
            self.model_config.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_config.model_name_or_path),
            config=self.auto_config,
            # cache_dir=model_args.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            # ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        backbone.config.label2id = self.label2id
        backbone.config.id2label = self.id2label
        return backbone

    def forward(self, inputs):
        output = self.backbone(**inputs)
        return output


if __name__ == "__main__":
    pass
