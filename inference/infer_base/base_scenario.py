import torch
from ..datasets.dataset import Dataset

class BaseScenario:
    def __init__(self,dataset:Dataset, param):
        self.dataset = dataset
        preload = param.preload
        self.param = param
        self.sample_per_quary = param.sample_per_query
        self.cache = self.preload(preload)

    def get_one_item(self, idx):
        if self.cache is not None:
            image, label, id = self.cache[idx]
        
        else:
            image, label, id = self.dataset.get_item(idx)
        
        return image, label, id
    
    def preload(self, preload:bool):
        if preload:
            cache = []
            for idx in range(len(self.dataset.image_list)):
                elem = self.dataset.get_item(idx)
                cache.append(elem)
            print(f"Preload {len(cache)} images success")
            return cache
        else:
            return None

    def __next__(self):
        pass
    
    def put_one_query(self, queue):
        pass
    
    def __len__(self):
        pass 
    

