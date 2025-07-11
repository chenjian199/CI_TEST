

from queue import Queue
import time
from ..datasets.dataset import Dataset
from inference.infer_base.base_scenario import BaseScenario


class Offline(BaseScenario):
    def __init__(self, dataset:Dataset, param):
        super().__init__(dataset, param)
        #self.package = self.scenario_schedule()
        self.idx = 0


    def put_one_query(self, queue:Queue):
        num = 0
        print("SCENARIO:Offline")
        idx = self.idx
        for _ in range(self.sample_per_quary):
            num +=1
            if idx >= (len(self.dataset.image_list)):
                if self.param.acc:
                    #print(f"accc:offline:idx-{idx}")
                    print("Accuracy: Already Load All Sample.")
                    data = "Acc",None,None,None
                    queue.put(data)
                    break
                else:
                    idx = 0
            else:
                #print(f"offline:idx-{idx}")
                image, label, id = self.get_one_item(idx=idx)
                data = image, label, id, time.time()
                queue.put(data)
                idx += 1
        data = "Query", None, None, None
        queue.put(data)
        self.idx = idx
        return num

            
        