#from backend.NVIDIA.openimage_post import NVIDIA_OpenImage
import torch
import torchvision
import numpy as np
from ..infer_base.base_model import BaseModel
#from ..datasets.openimage_datasets import OpenImagesDataset
from ..datasets.openimage import PostProcessOpenImagesRetinanet, OpenImagesDataset
from ..datasets.dataset import pre_process_openimages_retinanet


class Retinanet(BaseModel):

    def create_model_path(self):
        self.pytorch_model_path = "./checkpoint/RetinaNet/resnext50_32x4d_fpn.pth"

    
    def create_dataset(self, val_data_path):
        self.input_shape = (3,800,800)
        data_path = val_data_path
        if self.args.processed_path is not None:
            preprocessed_dir = self.args.processed_path
        else:
            preprocessed_dir = './npy/OpenImages'
        dataset = OpenImagesDataset(data_path, 
                                        image_list = None, 
                                        name="retinanet", 
                                        use_cache=True,
                                        image_size=[800,800,3], 
                                        image_format="NCHW", 
                                        pre_process=pre_process_openimages_retinanet,
                                        preprocessed_dir=preprocessed_dir
                                        )
        return dataset
    

    def postprocess(self):
        self.postprocessor = PostProcessOpenImagesRetinanet(False,0.05,800,800)
        #self.postprocessor = NVIDIA_OpenImage()
