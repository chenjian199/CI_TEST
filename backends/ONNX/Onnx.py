import torch
import torch.onnx
import onnxruntime as ort

import torch
from torch import Tensor
import torchvision.models as models
from inference.infer_base.base_model import BaseModel
from inference.infer_base.infer_args import InferenceArguments
from ..backend_base import BaseBackend 
import numpy as np
from backends.recorder import recording
from torch.cuda.amp import autocast
from typing import List, Optional, Tuple, Union

from onnx2torch import convert


#！！！ 要求：iou: 0.6 conf: 0.001
class Onnx(BaseBackend):
    def __init__(self, args:InferenceArguments, workload:BaseModel):
        self.type = "ONNX"
        super().__init__(args, workload)
        model_path = workload.pytorch_model_path
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.model_name = args.model

        self.recording.set_device(0)

    def malloc_pinned_buffer(self, buffer_size, dtype=np.float32):
        # 保证数据类型与图像数组相匹配
        return buffer_size
    
    def cp_numpy_to_pinned_buffer(self, images: list, buffer):
        #images = np.expand_dims(images, axis=0)
        return images
    
    def predict(self, images):
        outputs = self.session.run(None, {self.input_name: images})

        return outputs

    
    def postprocess(self, outputs):
        data = yolo_nms_postprocess(outputs)
        bboxes_batch, scores_batch, labels_batch = det_postprocess(data, 1)
        return {
            "bboxes":bboxes_batch,
            "scores":scores_batch,
            "labels":labels_batch,
        }
    
    def device_sync(self):
        return torch.cuda.synchronize()
    
def yolo_nms_postprocess(inputs):
    onnx_output = inputs[0]
    detection_tensor = torch.from_numpy(onnx_output)
    bboxs = detection_tensor[:, :, :4]
    scores = detection_tensor[:, :, 4]
    labels = detection_tensor[:, :, 5]
    num_dets = (scores > 0.0).sum(dim=1, keepdim=True)
    data = (num_dets, bboxs, scores, labels)
    return data

def det_postprocess(data: Tuple[Tensor, Tensor, Tensor, Tensor], batch_size: int):
    """
    处理整个批次的检测输出
    """
    bboxes_batch = []
    scores_batch = []
    labels_batch = []
    
    for i in range(batch_size):
        num_dets = int(data[0][i].item())
        if num_dets == 0:
            bboxes_batch.append(torch.empty((0, 4), device=data[1].device))
            scores_batch.append(torch.empty((0,), device=data[2].device))
            labels_batch.append(torch.empty((0,), dtype=torch.int64, device=data[3].device))
            continue
        
        # 获取当前样本的检测结果
        bboxes = data[1][i][:num_dets]
        scores = data[2][i][:num_dets]
        labels = data[3][i][:num_dets]
        
        # 过滤掉负分数的检测结果
        valid = scores > 0.0
        bboxes = bboxes[valid]
        scores = scores[valid]
        labels = labels[valid]
        
        # 将分数限制在0到1之间
        scores = torch.clamp(scores, min=0.0, max=1.0)
        
        bboxes_batch.append(bboxes)
        scores_batch.append(scores)
        labels_batch.append(labels)
    
    return bboxes_batch, scores_batch, labels_batch