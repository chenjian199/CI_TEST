import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, load_checkpoint, load_param_into_net

from inference.infer_base.base_model import BaseModel
from inference.infer_base.infer_args import InferenceArguments
from ..backend_base import BaseBackend
from backends.recorder import recording

import mindvision.classification.models as models

class MindSpore(BaseBackend):
    def __init__(self, args: InferenceArguments, workload: BaseModel):
        self.type = "MINDSPORE"
        super().__init__(args, workload)

        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
        ms.set_seed(1)

        model_path = workload.pytorch_model_path
        self.model_name = args.model

        if args.model == 'resnet50':
            self.model = models.resnet50(pretrained=False, num_classes=1000)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        param_dict = load_checkpoint(model_path)
        load_param_into_net(self.model, param_dict)
        self.model.set_train(False)

        self.sample = 0
        self.recording.set_device(0)

    def malloc_pinned_buffer(self, buffer_size, dtype=np.float32):
        return None

    def cp_numpy_to_pinned_buffer(self, images: list, buffer):
        arr = np.array(images).astype(np.float16)  
        return Tensor(arr)

    def predict(self, images):
        images = images.astype(ms.float16)
        self.sample += images.shape[0]

        outputs = self.model(images)

        return outputs

    def postprocess(self, outputs):
        if self.model_name in ["resnet50"]:
            pred = outputs.argmax(axis=1)
            return pred

    def device_sync(self):
       pass