import paddle
import numpy as np
import paddle.vision.models as models
from inference.infer_base.base_model import BaseModel
from inference.infer_base.infer_args import InferenceArguments
from ..backend_base import BaseBackend
from backends.recorder import recording


class PaddleBackend(BaseBackend):
    def __init__(self, args: InferenceArguments, workload: BaseModel):
        self.type = "PADDLEPADDLE"
        super().__init__(args, workload)

        model_path = workload.pytorch_model_path  

        if args.model == 'resnet50':
            self.model = models.resnet50(pretrained=False, num_classes=1000)
        elif args.model == 'inception_v3':
            self.model = models.inception_v3(pretrained=False, num_classes=1000)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        self.model.set_state_dict(paddle.load(model_path))
        self.model.eval()
        self.model = self.model.cuda()  
        self.model_name = args.model

        self.sample = 0
        self.recording.set_device(0)

    def malloc_pinned_buffer(self, buffer_size, dtype=np.float32):
        return None

    def cp_numpy_to_pinned_buffer(self, images: list, buffer):
        arr = np.array(images).astype(np.float16)
        return paddle.to_tensor(arr, place=paddle.CUDAPlace(0))

    def predict(self, images):
        images = images.astype('float16')
        self.sample += images.shape[0]
        with paddle.no_grad():
            outputs = self.model(images)
        return outputs

    def postprocess(self, outputs):
        if self.model_name in ["resnet50", "inception_v3"]:
            pred = paddle.argmax(outputs, axis=1)
            return pred.cpu().numpy()

    def device_sync(self):
        pass
