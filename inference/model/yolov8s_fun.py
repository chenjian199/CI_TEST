
from ..infer_base.base_model import BaseModel
from ..datasets.coco2017 import COCO2017Val, COCO2017PostProcess


class Yolov8s(BaseModel):


    def create_model_path(self):
        self.pytorch_model_path = "./checkpoint/Yolov8s/yolov8s_nms.onnx"

    
    def create_dataset(self, val_data_path):
        self.input_shape = (3,640, 640)
        data_path = self.args.data_dir
        self.annotations = data_path+"/annotations/instances_val2017.json"
        if self.args.processed_path is not None:
            preprocessed_dir = self.args.processed_path+"/COCO2017"
        else:
            preprocessed_dir = './npy/Coco2017'
        #pre_process_imagenet = pre_process_imagenet_pytorch
        dataset = COCO2017Val(data_path,
                            name="yolov8s",
                            image_size=[3, 640, 640],
                            use_cache=True,
                            image_format="NCHW",
                            #pre_process=pre_process_imagenet,
                            preprocessed_dir=preprocessed_dir)
        return dataset
    
    

    def postprocess(self):
        self.postprocessor = COCO2017PostProcess(self.annotations)


