from ACBench_base.base_param import TrainingConfig as Param
from ACBench_base.base_metric import TrainingMetrics as Metrics

from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    def __init__(self, train_data_path, val_data_path, train_batch_size, val_batch_size):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def dataloader(self, thread):
        pass
