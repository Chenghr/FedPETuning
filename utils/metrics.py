import numpy as np
from abc import ABC

# 导入评价指标相关的函数和类
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report

# 导入注册工具和 GLUE 数据集评价函数
from utils.register import registry
from tools.glue_scripts.glue_metric import glue_compute_metrics

# 定义一个抽象基类 BaseMetric
class BaseMetric(ABC):
    def __init__(self, task_name, is_decreased_valid_metric=False):
        super().__init__()

        # 初始化任务名称和是否是递减的评价指标
        self.task_name = task_name
        self.is_decreased_valid_metric = is_decreased_valid_metric
        
        # 初始最佳验证指标为正无穷或负无穷
        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        
        # 初始化结果字典
        self.results = {}

    # 计算评价指标，需要子类实现
    def calculate_metric(self, *args):
        raise NotImplementedError

    # 更新评价指标，需要子类实现
    def update_metrics(self, *args):
        raise NotImplementedError

    # 返回最佳评价指标
    @property
    def best_metric(self):
        return self.results

    # 定义抽象属性评价指标名称
    @property
    def metric_name(self):
        raise NotImplementedError

# 注册 GLUE 数据集评价指标类
@registry.register_metric("glue")
class GlueMetric(BaseMetric):
    def __init__(self, task_name, is_decreased_valid_metric=False):
        super().__init__(task_name, is_decreased_valid_metric)

    # 计算 GLUE 数据集的评价指标
    def calculate_metric(self, preds, labels, updated=True):
        results = glue_compute_metrics(self.task_name, preds, labels)
        if updated:
            self.update_metrics(results)
        return results

    # 更新 GLUE 数据集的评价指标
    def update_metrics(self, results):
        cur_valid_metric = results[self.metric_name]
        if self.is_decreased_valid_metric:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric

        if is_best:
            self.results.update(results)
            self.best_valid_metric = cur_valid_metric

    # 返回 GLUE 数据集评价指标名称
    @property
    def metric_name(self):
        glue_metric_name = {
            "cola": "mcc",
            "sst-2": "acc",
            "mrpc": "f1",
            "sts-b": "acc",
            "qqp": "acc",
            "mnli": "acc",
            "mnli-mm": "acc",
            "qnli": "acc",
            "rte": "acc",
            "wnli": "acc"
        }
        return glue_metric_name[self.task_name]

# 注册 CoNLL 数据集评价指标类
@registry.register_metric("conll")
class CoNLLMetric(BaseMetric):
    def __init__(self, task_name, is_decreased_valid_metric=False):
        super().__init__(task_name, is_decreased_valid_metric)

    # 计算 CoNLL 数据集的评价指标
    def calculate_metric(self, preds, labels, label_list, updated=True):
        predictions, labels = preds, labels

        # 移除被忽略的索引（特殊标记）
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # 计算评价指标
        results = {
            "accuary": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions)
        }

        if updated:
            self.update_metrics(results)

        return results

    # 更新 CoNLL 数据集的评价指标
    def update_metrics(self, results):
        cur_valid_metric = results[self.metric_name]
        if self.is_decreased_valid_metric:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric

        if is_best:
            self.results.update(results)
            self.best_valid_metric = cur_valid_metric

    # 返回 CoNLL 数据集评价指标名称
    @property
    def metric_name(self):
        return "f1"
