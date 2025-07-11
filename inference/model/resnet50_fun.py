
from ..infer_base.base_model import BaseModel
from ..datasets.imagenet import Imagenet
from ..datasets.dataset import pre_process_imagenet_pytorch, PostProcessCommon


class Resnet50(BaseModel):


    def create_model_path(self):
        self.pytorch_model_path = "./checkpoint/ResNet50/resnet50-v1.5.pth"
    
    def create_dataset(self, val_data_path):
        self.input_shape = (3,224,224)
        data_path = val_data_path
        if self.args.processed_path is not None:
            preprocessed_dir = self.args.processed_path
        else:
            preprocessed_dir = './npy/ImageNet'
        pre_process_imagenet = pre_process_imagenet_pytorch
        dataset = Imagenet( data_path, 
                            image_list = None, 
                            name="resnet50",
                            image_size=[224,224,3], 
                            use_cache=True, 
                            image_format="NCHW", 
                            pre_process=pre_process_imagenet,
                            preprocessed_dir=preprocessed_dir
                            )
        return dataset
    

    def postprocess(self):
        self.postprocessor = PostProcessCommon(offset=-1)


