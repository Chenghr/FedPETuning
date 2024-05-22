"""config for FedETuning"""

import os
import time
import copy
from abc import ABC
from omegaconf import OmegaConf
from transformers import HfArgumentParser

from utils import make_sure_dirs, rm_file
from utils.register import registry
from configs import ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments
from configs.tuning import get_delta_config, get_delta_key

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

    def save_configs(self):
        # Implement the save functionality if needed
        pass

    def check_config(self):
        self.config_check_federated()
        self.config_check_model()
        self.config_check_tuning()

    def config_check_federated(self):
        if "cen" in self.federated_config.fl_algorithm:
            if self.federated_config.rank == -1:
                self.federated_config.world_size = 1
            else:
                raise ValueError(f"Must set world_size, but find {self.federated_config.world_size}")
        else:
            if self.federated_config.clients_num % (self.federated_config.world_size - 1) != 0:
                raise ValueError(f"{self.federated_config.clients_num} % {self.federated_config.world_size - 1} != 0")

    def config_check_model(self):
        # Implement the model checking functionality if needed
        pass

    def config_check_tuning(self):
        delta_config = self._get_delta_config()

        # Grid search hard coded parameters
        if self.training_config.do_grid:
            self._update_delta_config_with_grid_search(delta_config)

        registry.register("delta_config", delta_config)

        # Update all configurations with delta_config
        for config in [self.training_config, self.model_config, self.federated_config, self.data_config]:
            for key, value in delta_config.items():
                if getattr(config, key, None) is not None:
                    setattr(config, key, value)

        self.training_config.tuning_type = delta_config["delta_type"]

    def _get_delta_config(self):
        if not self.model_config.tuning_type or "fine" in self.model_config.tuning_type:
            delta_config = {"delta_type": "fine-tuning"}
            self.model_config.tuning_type = ""
        else:
            # ddelta_args is a dictionary. 
            # Fine-tuning training parameters can be obtained through the task name.
            delta_args = get_delta_config(self.model_config.tuning_type)    
            delta_config = delta_args.get(self.data_config.task_name, delta_args)   
        return delta_config

    def _update_delta_config_with_grid_search(self, delta_config):
        for key in delta_config:
            # value = getattr(object, "attribute_name", default_value)
            model_value = getattr(self.model_config, key, None)
            if model_value is not None:
                delta_config[key] = model_value

            if key in ["learning_rate", "num_train_epochs"]:
                training_value = getattr(self.training_config, key, None)
                if training_value is not None:
                    delta_config[key] = training_value

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

def amend_config(model_args, data_args, training_args, federated_args):
    config = Config(model_args, data_args, training_args, federated_args)

    if config.F.rank > 0:
        time.sleep(2)  # let server firstly start

    root_folder = registry.get("root_folder")
    cust_config_path = os.path.join(root_folder, f"run/{config.F.fl_algorithm}/config.yaml")
    if os.path.isfile(cust_config_path):
        cust_config = OmegaConf.load(cust_config_path)
        for key, values in cust_config.items():     # key: data_config、federated_config...
            if values:
                args = getattr(config, key)
                for k, v in values.items():
                    if config.T.do_grid and k in grid_hyper_parameters:
                        continue  # grid search not overwrite --arg
                    setattr(args, k, v)

    config.T.output_dir = os.path.join(config.T.output_dir, config.D.task_name)
    make_sure_dirs(config.T.output_dir)

    if not config.D.cache_dir:
        cache_dir = os.path.join(config.T.output_dir, "cached_data")
        if config.F.rank != -1:
            config.D.cache_dir = os.path.join(
                cache_dir, f"cached_{config.M.model_type}_{config.F.clients_num}_{config.F.alpha}"
            )
        else:
            config.D.cache_dir = os.path.join(cache_dir, f"cached_{config.M.model_type}_centralized")
    make_sure_dirs(config.D.cache_dir)

    config.T.save_dir = os.path.join(config.T.output_dir, config.F.fl_algorithm.lower())
    make_sure_dirs(config.T.save_dir)
    config.T.checkpoint_dir = os.path.join(config.T.save_dir, "saved_model")
    make_sure_dirs(config.T.checkpoint_dir)

    phase = "train" if config.T.do_train else "evaluate"
    registry.register("phase", phase)

    times = time.strftime("%Y%m%d%H%M%S", time.localtime())
    registry.register("run_time", times)
    config.T.times = times
    config.T.metric_file = os.path.join(config.T.save_dir, f"{config.M.model_type}_{config.D.task_name}.eval")
    config.T.metric_log_file = os.path.join(config.T.save_dir, f"{times}_{config.M.model_type}_{config.D.task_name}.eval.log")

    if config.F.do_mimic and config.F.rank == 0:
        server_write_flag_path = os.path.join(config.D.cache_dir, "server_write.flag")
        rm_file(server_write_flag_path)

    if config.F.partition_method is None:
        config.F.partition_method = f"clients={config.F.clients_num}_alpha={config.F.alpha}"

    config.check_config()

    if config.T.do_grid:
        key_name, key_abb = get_delta_key(config.T.tuning_type)
        delta_config = registry.get("delta_config")
        if key_name:
            grid_info = "=".join([key_abb, str(delta_config[key_name])])
        else:
            grid_info = ""
        registry.register("grid_info", grid_info)

        config.T.metric_line = f"{times}_{config.M.model_type}_{config.T.tuning_type}_" \
                               f"seed={config.T.seed}_rounds={config.F.rounds}_" \
                               f"cli={config.F.clients_num}_alp={config.F.alpha}_" \
                               f"sap={config.F.sample}_epo={config.T.num_train_epochs}_" \
                               f"lr={config.T.learning_rate}_{grid_info}_"
    else:
        config.T.metric_line = f"{times}_{config.M.model_type}_{config.T.tuning_type}_" \
                               f"seed={config.T.seed}_rounds={config.F.rounds}_" \
                               f"cli={config.F.clients_num}_alp={config.F.alpha}_" \
                               f"sap={config.F.sample}_rd={config.F.rounds}_epo={config.T.num_train_epochs}_" \
                               f"lr={config.T.learning_rate}_"

    registry.register("config", config)

    return config

def build_config():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments))
    model_args, data_args, training_args, federated_args = parser.parse_args_into_dataclasses()
    config = amend_config(model_args, data_args, training_args, federated_args)
    delta_config = registry.get("delta_config")

    logger = registry.get("logger")
    logger.info(f"FL-Algorithm: {config.federated_config.fl_algorithm}")
    logger.info(f"output_dir: {config.training_config.output_dir}")
    logger.info(f"cache_dir: {config.data_config.cache_dir}")
    logger.info(f"save_dir: {config.training_config.save_dir}")
    logger.info(f"checkpoint_dir: {config.training_config.checkpoint_dir}")
    logger.debug(f"TrainBaseInfo: {config.M.model_type}_{delta_config['delta_type']}_seed={config.T.seed}_"
                 f"cli={config.F.clients_num}_alp={config.F.alpha}_cr={config.F.rounds}_sap={config.F.sample}_"
                 f"lr={config.T.learning_rate}_epo={config.T.num_train_epochs}")

    return config
