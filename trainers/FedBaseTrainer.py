"""
    BaseTrainer 类提供了联邦学习中训练器的基本框架，
    包括初始化配置、定义角色、构建网络、数据、模型、以及实现训练流程等基础功能。
    它设计为抽象基类，需要通过继承并实现特定方法（如_build_server、_build_client等）来创建具体的训练器实现。
"""


# 导入必要的库和模块
from abc import ABC
from utils import registry
from utils import setup_seed
from utils import global_metric_save
from fedlab.core.network import DistNetwork

# BaseTrainer类定义，继承自ABC表示这是一个抽象基类
class BaseTrainer(ABC):
    def __init__(self, *args):
        # 从注册表中获取配置信息，并将其分为模型配置、数据配置、训练配置和联邦学习配置
        config = registry.get("config")
        self.model_config = config.M
        self.data_config = config.D
        self.training_config = config.T
        self.federated_config = config.F

        # 从注册表中获取日志记录器
        self.logger = registry.get("logger")

    # 定义一个属性role，根据rank值确定当前实例的角色
    @property
    def role(self):
        if self.federated_config.rank == 0:
            return "server"
        elif self.federated_config.rank > 0:
            return f"sub-server_{self.federated_config.rank}"
        else:
            return "centralized"

    # 定义建立服务器的方法，具体实现需要在子类中完成
    def _build_server(self):
        raise NotImplementedError

    # 定义建立客户端的方法，具体实现需要在子类中完成
    def _build_client(self):
        raise NotImplementedError

    # 定义建立本地训练器的方法，具体实现需要在子类中完成
    def _build_local_trainer(self, *args):
        raise NotImplementedError

    # 构建网络连接
    def _build_network(self):
        self.network = DistNetwork(
            address=(self.federated_config.ip, self.federated_config.port),
            world_size=self.federated_config.world_size,
            rank=self.federated_config.rank,
            ethernet=self.federated_config.ethernet)

    # 构建数据
    def _build_data(self):
        self.data = registry.get_data_class(self.data_config.dataset_name)()

    # 构建模型
    def _build_model(self):
        self.model = registry.get_model_class(self.model_config.model_output_mode)(
            task_name=self.data_config.task_name
        )

    # 训练前的准备工作
    def _before_training(self):
        # 日志记录和种子设置
        self.logger.info(f"{self.role} set seed {self.training_config.seed}")
        setup_seed(self.training_config.seed)

        # 构建数据和模型
        self.logger.info(f"{self.role} building dataset ...")
        self._build_data()

        self.logger.info(f"{self.role} building model ...")
        self._build_model()

        # 根据rank值构建网络、服务器或客户端
        if self.federated_config.rank != -1:
            self.logger.info(f"{self.role} building network ...")
            self._build_network()

        if self.federated_config.rank == 0:
            self.logger.info("building server ...")
            self._build_server()
        else:
            self._build_client()
            if self.federated_config.rank > 0:
                self.logger.info(f"building client {self.federated_config.rank} ...")
                self.logger.info(f"local rank {self.federated_config.rank}'s client ids "
                                 f"is {list(self.data.train_dataloader_dict.keys())}")
            else:
                self.logger.info("building centralized training")

    # 定义训练方法，具体实现依赖于role
    def train(self):
        if self.federated_config.rank == 0:
            # 服务器端开始训练
            self.logger.debug(f"Server Start ...")
            self.server_manger.run()
            self.on_server_end()

        elif self.federated_config.rank > 0:
            # 子服务器开始训练
            self.logger.debug(f"Sub-Server {self.federated_config.rank} Training Start ...")
            self.client_manager.run()
            self.on_client_end()

        else:
            # 集中式训练开始
            self.logger.debug(f"Centralized Training Start ...")
            self.client_trainer.cen_train()
            self.on_client_end()

    # 服务器训练结束后的处理
    def on_server_end(self):
        self.handler.test_on_server()
        global_metric_save(self.handler, self.training_config, self.logger)

    # 客户端训练结束后的处理，具体实现留给子类
    def on_client_end(self, *args):
        ...
