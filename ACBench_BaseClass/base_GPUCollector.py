import time
import threading
from abc import ABC, abstractmethod

class BaseGPUStatsCollector(ABC):
    def __init__(self, gpu_index, empty_load_time, sample_interval, sample_duration):
        self.empty_load_time = empty_load_time
        self.sample_interval = sample_interval
        self.sample_duration = sample_duration
        self.power_draws = []
        self.utilizations = []
        self.running = False
        self.index = gpu_index

        self.gpu_init(gpu_index)
        self.thread = threading.Thread(target=self._collect_stats)

    @property
    @abstractmethod
    def gpu_name(self):
        pass
    
    @property
    @abstractmethod
    def memory_info(self):
        pass

    @abstractmethod
    def get_power_draw(self):
        pass

    @abstractmethod
    def get_utilization(self):
        pass

    @abstractmethod
    def gpu_init():
        pass

    @abstractmethod
    def collect_over():
        pass

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.collect_over()
        self.running = False
        
        self.thread.join()
        

    def get_results(self):
        max_power = f"{max(self.power_draws)} W" if self.power_draws else "0 W"
        avg_power = f"{sum(self.power_draws) / len(self.power_draws)} W" if self.power_draws else "0 W"
        max_utilization = f"{max(self.utilizations)} %" if self.utilizations else "0 %"
        avg_utilization = f"{sum(self.utilizations) / len(self.utilizations)} %" if self.utilizations else "0 %"
        return max_utilization, max_power, avg_utilization, avg_power
    
    def get_gpu_info(self):
        memory_total_in_MB = self.memory_info / 1024 ** 2
        return {
            "gpu_name": self.gpu_name,
            "memory_total": f"{memory_total_in_MB} MB"
        }

    def _collect_stats(self):

        time.sleep(self.empty_load_time)
        start_time = time.time()

        while self.running and time.time() - start_time < self.sample_duration:
            self.power_draws.append(self.get_power_draw())
            self.utilizations.append(self.get_utilization())
            time.sleep(self.sample_interval)

    def get_stats_summary(self):
        max_utilization, max_power, avg_utilization, avg_power = self.get_results()
        gpu_info = self.get_gpu_info()
        stats = {
            f"GPU{self.index}": {
                "gpu_info": gpu_info,
                "gpu_state": {
                    "max_utilization": max_utilization,
                    "max_power": max_power,
                    "avg_utilization": avg_utilization,
                    "avg_power": avg_power
                }
            }
        }
        
        return stats