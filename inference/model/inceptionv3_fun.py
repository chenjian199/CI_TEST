import os
from ..infer_base.base_model import BaseModel
from ..datasets.imagenet_transform_support import Inception_imagenet
from ..datasets.dataset import pre_process_imagenet_pytorch, PostProcessCommon


class Inception_v3(BaseModel):


    def create_model_path(self):
        self.pytorch_model_path = "./checkpoint/Inception_V3/inception_v3.pth"

    
    def create_dataset(self, val_data_path):
        self.input_shape = (3,299,299)
        data_path = val_data_path
        if self.args.processed_path is not None:
            preprocessed_dir = self.args.processed_path
        else:
            preprocessed_dir = './npy/ImageNet'
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize(342),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        # dataset = Imagenet( data_path, 
        #                     image_list = None, 
        #                     name="inception_v3",
        #                     image_size=[299,299,3], 
        #                     use_cache=True, 
        #                     image_format="NCHW", 
        #                     pre_process=pre_process_imagenet,
        #                     preprocessed_dir=preprocessed_dir
        #                     )
        
        mapping_file = os.path.join(data_path, 'val_map.txt')
        dataset = Inception_imagenet(data_path, 
                                     image_list=mapping_file,
                                     pre_process=transform,
                                     preprocessed_dir=preprocessed_dir,
                                     image_format="NCHW",
                                     name="inception_imagenet")
        return dataset
    

    def postprocess(self):
        self.postprocessor = PostProcessCommon(offset=-1)


