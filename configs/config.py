"""config for FedETuning"""

import os
import time
import copy
from abc import ABC
from omegaconf import OmegaConf
from transformers import HfArgumentParser
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

from utils import make_sure_dirs, rm_file
from utils.register import registry
from configs import ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments
from configs.saved_configs.tuning import get_delta_config, get_delta_key

grid_hyper_parameters = [
    "tuning_type", "prefix_token_num", "bottleneck_dim",
    "learning_rate", "dataset_name", "metric_name", "model_output_mode", "seed",
    "alpha", "sample", "num_train_epochs"
]

class Config(ABC):
    def __init__(self, model_args, data_args, training_args, federated_args):
        self.model_config = model_args
        self.data_config = data_args
        self.training_config = training_args
        self.federated_config = federated_args

    def get_peft_configs(self):
        if self.T.tuning_library == "opendelta":
            peft_config = self._get_delta_config()
        elif self.T.tuning_library == "peft":
            peft_config = self._get_peft_config()
        else:
            raise ValueError(f"Not support tuning library {self.T.tuning_library}")
        
        registry.register("peft_config", peft_config)

    def overwrite_configs(self):
        self._overwrite_configs_from_peft_config()
        self._overwrite_configs_from_yaml()

    def check_configs(self):
        # Relevant functions are placed in the post_init function of dataclass.
        self.T.times = registry.get("run_time") 

    def gen_dirs(self):
        self.T.output_dir = os.path.join(self.T.output_dir, self.D.task_name)
        make_sure_dirs(self.T.output_dir)

        if not self.D.cache_dir:
            cache_dir = os.path.join(self.T.output_dir, "cached_data")
            if self.F.rank != -1:
                self.D.cache_dir = os.path.join(
                    cache_dir, f"cached_{self.M.model_type}_{self.F.clients_num}_{self.F.alpha}"
                )
            else:
                self.D.cache_dir = os.path.join(cache_dir, f"cached_{self.M.model_type}_centralized")
        make_sure_dirs(self.D.cache_dir)

        self.T.save_dir = os.path.join(self.T.output_dir, self.F.fl_algorithm.lower())
        make_sure_dirs(self.T.save_dir)
        self.T.checkpoint_dir = os.path.join(self.T.save_dir, "saved_model")
        make_sure_dirs(self.T.checkpoint_dir)
        
    def gen_metrics(self):
        self.T.metric_file = os.path.join(self.T.save_dir, f"{self.M.model_type}_{self.D.task_name}.eval")
        self.T.metric_log_file = os.path.join(self.T.save_dir, f"{self.T.times}_{self.M.model_type}_{self.D.task_name}.eval.log")

        if self.F.do_mimic and self.F.rank == 0:
            server_write_flag_path = os.path.join(self.D.cache_dir, "server_write.flag")
            rm_file(server_write_flag_path)

        # TODO: regenerate grid info here.
        grid_info = ""  
        if self.T.do_grid:
            if self.T.tuning_type == "opendelta":
                key_name, key_abb = get_delta_key(self.T.tuning_type)
                delta_config = registry.get("delta_config")
                if key_name:
                    grid_info = "=".join([key_abb, str(delta_config[key_name])])
        registry.register("grid_info", grid_info)

        self.T.metric_line = f"{self.T.times}_{self.M.model_type}_{self.T.tuning_type}_" \
                                f"seed={self.T.seed}_rounds={self.F.rounds}_" \
                                f"cli={self.F.clients_num}_alp={self.F.alpha}_" \
                                f"sap={self.F.sample}_epo={self.T.num_train_epochs}_" \
                                f"lr={self.T.learning_rate}_{grid_info}_"

    def save_configs(self):
        pass

    def _get_delta_config(self):
        if not self.T.tuning_type or "fine" in self.T.tuning_type:
            delta_config = {"delta_type": "fine-tuning"}
            self.T.tuning_type = ""
        else:
            # delta_args is a dictionary. 
            # Fine-tuning training parameters can be obtained through the task name.
            delta_args = get_delta_config(self.model_config.tuning_type)    
            delta_config = delta_args.get(self.data_config.task_name, delta_args)   
        
        if self.training_config.do_grid:
            # updata delta_config from configs
            for key in delta_config:
                # value = getattr(object, "attribute_name", default_value)
                model_value = getattr(self.model_config, key, None)
                if model_value is not None:
                    delta_config[key] = model_value

                if key in ["learning_rate", "num_train_epochs"]:
                    training_value = getattr(self.training_config, key, None)
                    if training_value is not None:
                        delta_config[key] = training_value

        return delta_config
    
    def _get_peft_config(self):
        # TODO: check here.
        if self.M.peft_tuning_type == PeftType.LORA:
            peft_config = LoraConfig(
                task_type="SEQ_CLS" if self.M.model_output_mode == "seq_classification" else "TOKEN_CLS",
                inference_mode=False, 
                r=self.M.lora_r, 
                lora_alpha=self.M.lora_alpha, 
                lora_dropout=self.M.lora_dropout
            )
        else:
            raise ValueError(f"Not supported peft tuning type {self.M.peft_tuning_type}")

        return peft_config
    
    def _overwrite_configs_from_peft_config(self):
        if self.T.tuning_library == "opendelta":
            delta_config = registry.get("peft_config")

            # Update all configurations with delta_config
            for config in [self.training_config, self.model_config, self.federated_config, self.data_config]:
                for key, value in delta_config.items():
                    if getattr(config, key, None) is not None:
                        setattr(config, key, value)
        else:
            # TODO: check whether need to overwrite for peft configs.
            pass

    def _overwrite_configs_from_yaml(self):
        root_folder = registry.get("root_folder")
        cust_config_path = os.path.join(root_folder, 
                                        f"configs/{self.F.fl_algorithm}/config.yaml")
        if os.path.isfile(cust_config_path):
            cust_config = OmegaConf.load(cust_config_path)
            for key, values in cust_config.items():     # key: data_config、federated_config...
                if values:
                    args = getattr(self, key)
                    for k, v in values.items():
                        if self.T.do_grid and k in grid_hyper_parameters:
                            continue  # grid search not overwrite --arg
                        setattr(args, k, v)

    @property
    def M(self):
        return self.model_config

    @property
    def D(self):
        return self.data_config

    @property
    def T(self):
        return self.training_config

    @property
    def F(self):
        return self.federated_config


def build_config():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments))
    model_args, data_args, training_args, federated_args = parser.parse_args_into_dataclasses()
    
    config = Config(model_args, data_args, training_args, federated_args)
    if config.F.rank > 0:
        time.sleep(2)  # let server firstly start
    
    times = time.strftime("%Y%m%d%H%M%S", time.localtime())
    registry.register("run_time", times)

    phase = "train" if config.T.do_train else "evaluate"
    registry.register("phase", phase)

    config.get_peft_configs()
    config.overwrite_configs()
    config.check_configs()
    config.gen_dirs()
    config.gen_metrics()
    registry.register("config", config)

    logger = registry.get("logger")
    logger.info(f"FL-Algorithm: {config.federated_config.fl_algorithm}")
    logger.info(f"output_dir: {config.training_config.output_dir}")
    logger.info(f"cache_dir: {config.data_config.cache_dir}")
    logger.info(f"save_dir: {config.training_config.save_dir}")
    logger.info(f"checkpoint_dir: {config.training_config.checkpoint_dir}")
    logger.debug(f"TrainBaseInfo: {config.M.model_type}_{config.T.tuning_type}_seed={config.T.seed}_"
                 f"cli={config.F.clients_num}_alp={config.F.alpha}_cr={config.F.rounds}_sap={config.F.sample}_"
                 f"lr={config.T.learning_rate}_epo={config.T.num_train_epochs}")

    return config
