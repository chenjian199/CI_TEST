import time
from inference.datasets.dataset import Dataset

from inference.infer_base.infer_args import InferenceArguments


class BaseModel():
    def __init__(self, args:InferenceArguments):
        self.args = args
        self.pytorch_model_path = None
        self.onnx_model_path = None
        self.model = None
        self.create_model_path()
        self.input_shape = None
        self.postprocessor = None
        self.datasets:Dataset = self.create_dataset(args.data_dir)
        self.postprocess()
        
    
    def create_model_path(self):
        pass
    
    def create_dataset(self):
        pass
    

    def postprocess(self, output):
        pass

    def acc_compute(self):
        pass