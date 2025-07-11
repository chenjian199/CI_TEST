import time,threading
import pynvml
from backends.recorder import recorder
from mindspore import context

class mindspore_recorder(recorder):
    def __init__(self):
        super().__init__()
        self.handle = None
        self.name = None
        self.memory = None
        

    @property
    def dev_name(self):
        return self.name

    @property
    def total_memory(self):
        return self.memory

    def record_init(self):
        pynvml.nvmlInit()

    def set_device(self, index):
        self.index = index
        context.set_context(device_target="GPU", device_id=index)  

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode('utf-8') if isinstance(name, bytes) else name
        self.memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total

    def get_utilization_rates(self):
        return pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu

    def get_power_usage(self):
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # 单位为 W

    def record_free(self):
        pynvml.nvmlShutdown()
