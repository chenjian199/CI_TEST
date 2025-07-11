from queue import Queue
import threading
import time
from inference.infer_base.base_scenario import BaseScenario
from inference.scenario.offline import Offline

class QueryLoadServer(threading.Thread):
    def __init__(self, scenario:BaseScenario, event:threading.Event, queue:Queue):
        super().__init__()
        self.scenario = scenario
        self.event = event
        self.queue = queue
        self.stop = False
        self.QLS_start_time = time.time()

    def run(self):
        while not self.stop:
            # 等待事件
            self.event.wait()

            if self.stop:
                break
            
            # 重置事件状态
            self.event.clear()

            # 添加数据到队列
            
            num = self.scenario.put_one_query(queue=self.queue)
            print(f"QSL:put one query success:{num} samples")
            