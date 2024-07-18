from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments
from peft import PeftType


@dataclass
class TrainArguments(TrainingArguments):
    # tuning type config
    tuning_library: str = field(
        default="opendelta",
        metadata={"help": "The Efficient Fine-tuning libraries, support {opendelta, peft}"}
    )
    tuning_type: str = field(
        default=None,
        metadata={"help": "The Efficient Fine-tuning type, support {adapter, prompt, lora, prefix}"}
    )
    peft_tuning_type: str = field(
        default=None,
        metadata={"help": "The Efficient Fine-tuning type for peft libraries."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "lora specific parameters"}
    )
    lora_alpha: int = field(
        default=8,
        metadata={"help": "lora specific parameters"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora specific parameters"}
    )
    prefix_token_num: int = field(
        default=16,
        metadata={"help": "prefix-tuning specific parameters"}
    )
    bottleneck_dim: int = field(
        default=64,
        metadata={"help": "adapter specific parameters"}
    )

    # training config
    do_reuse: bool = field(
        default=False, metadata={"help": "whether to load last checkpoint"}
    )
    metric_name: str = field(
        default="glue", metadata={"help": "evaluation metrics for tasks"}
    )
    loss_name: str = field(
        default="xent", metadata={"help": "{xent: cross_entropy}"}
    )
    is_decreased_valid_metric: bool = field(
        default=False
    )
    patient_times: int = field(
        default=10,
    )
    do_grid: bool = field(
        default=False, metadata={"help": "whether to do grid search"}
    )

    def __post_init__(self): 
        if self.tuning_library == "peft":
            if "lora" in self.tuning_type:
                self.peft_tuning_type = PeftType.LORA
            elif  "prefix" in self.tuning_type:
                self.peft_tuning_type = PeftType.PREFIX_TUNING
            elif  "prompt" in self.tuning_type:
                self.peft_tuning_type = PeftType.PROMPT_TUNING
            else:
                raise ValueError(f"Not supported peft tuning method {self.tuning_type}")
