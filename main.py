"""main for FedETuning"""

from utils import registry
from utils import build_config
from utils import setup_logger, setup_imports


# 主函数定义
def main():
    # 设置导入路径，确保项目内部的模块可以正确导入
    setup_imports()
    # 设置日志，便于记录运行过程中的信息
    setup_logger()
    # 根据配置文件构建配置对象
    config = build_config()
    # 从注册中心获取对应的联邦学习算法类并实例化
    trainer = registry.get_fl_class(config.federated_config.fl_algorithm)()
    # 调用训练器的训练方法开始训练过程
    trainer.train()

if __name__ == "__main__":
    """该部分代码定义了联邦学习调优（FedETuning）的执行流程，
        包括初始化设置、读取配置、获取指定的联邦学习算法，并执行训练过程。
    """
    main()
    
