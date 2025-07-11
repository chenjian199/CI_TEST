import torch
import torch.onnx


import torch
import torchvision.models as models
from inference.infer_base.base_model import BaseModel
from inference.infer_base.infer_args import InferenceArguments
from ..backend_base import BaseBackend 
import numpy as np
from backends.recorder import recording
from torch.cuda.amp import autocast

class Pytorch(BaseBackend):
    def __init__(self, args:InferenceArguments, workload:BaseModel):
        self.type = "PYTORCH"
        super().__init__(args, workload)
        model_path = workload.pytorch_model_path
        if args.model == 'resnet50':
                   
            self.model = models.resnet50(pretrained=False)
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.to("cuda").half()
            self.model.eval()
            self.model = torch.compile(self.model,mode="reduce-overhead")
        if args.model == 'inception_v3':
            self.model = models.inception_v3(pretrained=False)
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.to("cuda").half()
            self.model.eval()
            self.model = torch.compile(self.model,mode="reduce-overhead")        
        elif args.model == 'retinanet':
            self.model = torch.jit.load(model_path)
            print("retinanet 加载成功")
            self.model = self.model.to("cuda").half()  # 确保retinanet也转为FP16
            self.model.eval()
        self.model_name = args.model
        
        self.sample = 0
        
        self.recording.set_device(0)

    def malloc_pinned_buffer(self, buffer_size, dtype=np.float32):
        # 保证数据类型与图像数组相匹配
        return buffer_size
    
    def cp_numpy_to_pinned_buffer(self, images: list, buffer):
        images = torch.tensor(np.array(images)).half()
        images = images.pin_memory()
        return images
    
    def predict(self, images):
        images = images.to("cuda").half()  # 仍然可以将输入转为FP16
        self.sample += len(images)

        with torch.no_grad():  # 禁用梯度计算
            with autocast():  # 启用自动混合精度
                output = self.model(images)

        if self.model_name in ["resnet50", "inception_v3"]:
            output = output.to('cpu')
        elif self.model_name == "retinanet":
            output = [
                {key: tensor.cpu() for key, tensor in tensor_dict.items()}
                for tensor_dict in output
            ]

        return output

    
    def postprocess(self, outputs):

        if self.model_name in ["resnet50", "inception_v3"]:
            max_indices_row = torch.argmax(outputs, dim=1)
            return max_indices_row
        else:
            numpy_arrays = [
                {key: tensor.numpy() for key, tensor in tensor_dict.items()}
                for tensor_dict in outputs
            ]
            torch.cuda.empty_cache()
            return numpy_arrays
    
    def device_sync(self):
        return torch.cuda.synchronize()