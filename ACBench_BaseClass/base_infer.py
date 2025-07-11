
class BaseInference:
    def __init__(self):
        self.model = None
    
    def inference(self):
        model_path = self.pre_process_model()
        

    def pre_process_model(self):
        pass

    def init_device(self):
        pass

    def load_model(self):
        pass

    def predict(self):
        pass
    
    def data_cache():
        pass
    
    def to_device(self, data):
        pass