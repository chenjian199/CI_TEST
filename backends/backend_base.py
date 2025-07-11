from backends.recorder import recording


class BaseBackend:
    def __init__(self, args, workload):
        self.recording = recording(self.type, args.model)
    
    #可按需实现
    def malloc_pinned_buffer(self, buffer_size): 
        pass

    #可按需实现，若无需pinnde buffer则可以做list to numpy的转换或其它需要在cpu上做处理的流程
    def cp_numpy_to_pinned_buffer(self, images, buffer):
        pass


    #传入的是一个batch的数据，以list形式储存
    def predict(image_list):
        pass

    #如果模型需要一些对结果的后处理，可以实现这个
    def postprocess(self, outputs):
        pass

    def device_sync(self):
        pass