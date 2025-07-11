from ACBench_base.base_param import TrainingConfig as Param
from ACBench_base.base_metric import TrainingMetrics as Metrics

from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    def __init__(self, lr, momentum, optimizer_name, model, weight_decay):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.optimizer_name = optimizer_name.lower()
        self.weight_decay = weight_decay

    @abstractmethod
    def create_optimizer(self):
        pass

    def get_optimizer(self):
        return self.create_optimizer()