
class BaseConfigs:
    def __init__(self):
        self.default_configs = {
            "lora": {

            }
        }

class PeftTuningConfigs:
    def __init__(self):
        # 定义每种 PEFT 方法的默认配置
        self.default_configs = {
            "adapter": {
                "learning_rate": 1e-3,
                "bottleneck_dim": 16,
                "unfrozen_modules": ["deltas", "layer_norm", "final_layer_norm", "classifier"]
            },
            "soft_prompt": {
                "learning_rate": 3e-2,
                "soft_token_num": 100,
                "unfrozen_modules": ["deltas", "classifier"]
            },
            "lora": {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            },
            "bitfit": {
                "learning_rate": 3e-4,
                "unfrozen_modules": ["classifier", "deltas"]
            },
            "prefix": {
                "learning_rate": 1e-3,
                "prefix_token_num": 16,
                "unfrozen_modules": ["deltas", "classifier"]
            }
        }

