import json
import logging
import os
import time
import random
import cv2
import numpy as np
from pycocotools.cocoeval import COCOeval
from  inference.util import pycoco
from .dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class OpenImagesDataset(Dataset):
    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
                 image_format="NHWC", pre_process=None, count=None, cache_dir=None, preprocessed_dir=None, use_label_map=False, threads=os.cpu_count()):
        super().__init__()
        self.image_size = image_size
        self.image_list = []
        self.label_list = []
        self.image_ids = []
        self.image_sizes = []
        self.count = count
        self.use_cache = use_cache
        self.data_path = data_path
        self.pre_process = pre_process
        self.use_label_map=use_label_map

        if not cache_dir:
            cache_dir = os.getcwd()
        if pre_process:
            if preprocessed_dir:
                self.cache_dir = preprocessed_dir
            else:
                self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        else:
            self.cache_dir = cache_dir
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False
        not_found = 0 
        empty_80catageories = 0
        if image_list is None:
            # by default look for val_map.txt
            image_list = os.path.join(data_path, "annotations/openimages-mlperf.json")
            #random.shuffle(image_list)
            
        self.annotation_file = image_list
        if self.use_label_map:
            # for pytorch
            label_map = {}
            with open(self.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1

        os.makedirs(self.cache_dir, exist_ok=True)
        start = time.time()
        images = {}
        with open(image_list, "r") as f:
            openimages = json.load(f)
        for i in openimages["images"]:
            images[i["id"]] = {"file_name": i["file_name"],
                               "height": i["height"],
                               "width": i["width"],
                               "bbox": [],
                               "category": []}
        for a in openimages["annotations"]:
            i = images.get(a["image_id"])
            #print(f"id:{i}")
            if i is None:
                continue
            catagory_ids = label_map[a.get("category_id")] if self.use_label_map else a.get("category_id")
            i["category"].append(catagory_ids)
            i["bbox"].append(a.get("bbox"))

        for image_id, img in images.items():
            
            image_name = img["file_name"]
            if len(img["category"])==0 and self.use_label_map: 
                #if an image doesn't have any of the 81 categories in it    
                empty_80catageories += 1 #should be 48 images - thus the validation sert has 4952 images
                continue 

            if not self.pre_process:
                if not os.path.exists(os.path.join(data_path, image_name) + ".npy"):
                    # if the image does not exists ignore it
                    not_found += 1
                    continue
            else:
                src = os.path.join(data_path, image_name)
                if not os.path.exists(src):
                    # if the image does not exists ignore it
                    not_found += 1
                    continue
                os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_name)), exist_ok=True)
                dst = os.path.join(self.cache_dir, image_name)
                if not os.path.exists(dst + ".npy"):
                    # cache a preprocessed version of the image
                    img_org = cv2.imread(src)
                    processed = self.pre_process(img_org, need_transpose=self.need_transpose, dims=self.image_size)
                    np.save(dst, processed)

            self.image_ids.append(image_id)
            self.image_list.append(image_name)
            self.image_sizes.append((img["height"], img["width"]))
            self.label_list.append((img["category"], img["bbox"]))

            # limit the dataset if requested
        
            if self.count and len(self.image_list) >= self.count:
                break

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)
        if empty_80catageories > 0:
            log.info("reduced image list, %d images without any of the 80 categories", empty_80catageories)

        log.info("loaded {} images, cache={}, already_preprocessed={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, pre_process is None, time_taken))

        self.label_list = np.array(self.label_list, dtype=list)
        #self.shuffle()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """Get image by number in the list."""
        # 获取当前进程的rank
        #rank = dist.get_rank() if dist.is_initialized() else 0
        image_name = self.image_list[idx]
        #print(image_name)
        id = self.image_ids[idx]
        dst = os.path.join(self.cache_dir, image_name)
        img = np.load(dst + ".npy")
        # 打印当前rank和图像名称
        #print(f"Rank {rank}, processing image: {image_name}")
        
        return img, id
    
    def get_item(self, idx):
        """Get image by number in the list."""
        image_name = self.image_list[idx]
        id = self.image_ids[idx]
        dst = os.path.join(self.cache_dir, image_name)
        img = np.load(dst + ".npy")
        label = self.label_list[idx]
        
        return img,  label , id-1

    
    def shuffle(self):
        # 生成一个随机数种子
        random_seed = random.randint(0, 1000)

        # 使用相同的随机数种子对两个list进行shuffle
        random.seed(random_seed)
        random.shuffle(self.label_list)

        random.seed(random_seed)
        random.shuffle(self.image_list)

        random.seed(random_seed)
        random.shuffle(self.image_ids)

        random.seed(random_seed)
        random.shuffle(self.image_sizes)


    def get_item_loc(self, nr):
        src = os.path.join(self.data_path, self.image_list[nr])
        return src


class PostProcessOpenImages:
    """
    Post processing for open images dataset. Annotations should
    be exported into coco format.
    """
    def __init__(self):
        self.results = []
        self.good = 0
        self.total = 0
        self.content_ids = []
        self.use_inv_map = False

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, ids, expected=None, result_dict=None, ):
        # results come as:
        #   tensorflow, ssd-mobilenet: num_detections,detection_boxes,detection_scores,detection_classes
        processed_results = []
        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            # keep the content_id from loadgen to handle content_id's without results
            self.content_ids.append(ids[idx])
            processed_results.append([])
            detection_num = int(results[0][idx])
            detection_boxes = results[1][idx]
            detection_classes = results[3][idx]
            expected_classes = expected[idx][0]
            for detection in range(0, detection_num):
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                processed_results[idx].append([float(ids[idx]),
                                              box[0], box[1], box[2], box[3],
                                              results[2][idx][detection],
                                              float(detection_class)])
                self.total += 1
        return processed_results

    def start(self):
        self.results = []
        self.good = 0
        self.total = 0

    def finalize(self, result_dict, ds=None, output_dir=None):
        result_dict["good"] += self.good
        result_dict["total"] += self.total

        if self.use_inv_map:
            # for pytorch
            label_map = {}
            with open(ds.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1
            inv_map = {v:k for k,v in label_map.items()}

        detections = []
        image_indices = []
        for batch in range(0, len(self.results)):
            image_indices.append(self.content_ids[batch])
            #print(f"aaa{batch}")
            #print(f"image_indices{image_indices}")
            for idx in range(0, len(self.results[batch])):
                detection = self.results[batch][idx]
                # this is the index of the coco image
                image_idx = int(detection[0])

                #if image_idx==24781:
                #    continue
                #print(f"image_idx:{image_idx},content_ids:{self.content_ids[batch]}")
                if image_idx != self.content_ids[batch]:
                    
                    # working with the coco index/id is error prone - extra check to make sure it is consistent
                    log.error("image_idx missmatch, lg={} / result={}".format(image_idx, self.content_ids[batch]))
                # map the index to the coco image id
                detection[0] = ds.image_ids[image_idx]
                height, width = ds.image_sizes[image_idx]
                # box comes from model as: ymin, xmin, ymax, xmax
                ymin = detection[1] * height
                xmin = detection[2] * width
                ymax = detection[3] * height
                xmax = detection[4] * width
                # pycoco wants {imageID,x1,y1,w,h,score,class}
                detection[1] = xmin
                detection[2] = ymin
                detection[3] = xmax - xmin
                detection[4] = ymax - ymin
                if self.use_inv_map:
                    cat_id = inv_map.get(int(detection[6]), -1)
                    if cat_id == -1:
                        # FIXME:
                        log.info("finalize can't map category {}".format(int(detection[6])))
                    detection[6] =  cat_id
                detections.append(np.array(detection))

        # map indices to coco image id's

        image_ids = [ds.image_ids[i]  for i in image_indices]

        self.results = []
        cocoGt = pycoco.COCO(ds.annotation_file)
        cocoDt = cocoGt.loadRes(np.array(detections))
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        result_dict["acc"] = cocoEval.stats[0]


class PostProcessOpenImagesRetinanet(PostProcessOpenImages):
    """
    Post processing required by retinanet / pytorch & onnx
    """
    def __init__(self, use_inv_map=False, score_threshold=0.05, height=800, width=800, dict_format=True):
        """
        Args:
            height (int): Height of the input image
            width (int): Width of the input image
            dict_format (bool): True if the model outputs a dictionary.
                        False otherwise. Defaults to True.
        """
        super().__init__()
        self.use_inv_map = use_inv_map
        self.score_threshold = score_threshold
        self.height = height
        self.width = width
        self.dict_format = dict_format

    def __call__(self, results, ids, expected=None, result_dict=None):

        if self.dict_format:
            # If the output of the model is in dictionary format. This happens
            # for the model retinanet-pytorch
            bboxes_ = [e['boxes'] for e in results]
            labels_ = [e['labels'] for e in results]
            scores_ = [e['scores'] for e in results]
            results = [bboxes_, labels_, scores_]
        else:
            bboxes_ = [results[0]]
            labels_ = [results[1]]
            scores_ = [results[2]]
            results = [bboxes_, labels_, scores_]

        processed_results = []
        content_ids = []
        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            content_ids.append(ids[idx])

            #print(f"ids{ids}")
            processed_results.append([])
            detection_boxes = results[0][idx]
            detection_classes = results[1][idx]
            #print(f"dete_class{detection_classes}")
            expected_classes = expected[idx][0]
            #print(f"exp_classes{expected_classes}")
            scores = results[2][idx]
            for detection in range(0, len(scores)):
                if scores[detection] < self.score_threshold:
                    break
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                # box comes from model as: xmin, ymin, xmax, ymax
                # box comes with dimentions in the range of [0, height] 
                # and [0, width] respectively. It is necesary to scale 
                # them in the range [0, 1]
                processed_results[idx].append(
                    [
                        float(ids[idx]),
                        box[1] / self.height,
                        box[0] / self.width,
                        box[3] / self.height,
                        box[2] / self.width,
                        scores[detection],
                        float(detection_class),
                    ]
                )
                self.total += 1

        self.content_ids.extend(content_ids)
        
        return processed_results
    
    def finalize(self, result_dict, ds=None, output_dir=None):
        result_dict["good"] += self.good
        result_dict["total"] += self.total

        if self.use_inv_map:
            # for pytorch
            label_map = {}
            with open(ds.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1
            inv_map = {v:k for k,v in label_map.items()}

        detections = []
        image_indices = []
        for batch in range(0, len(self.results)):
            image_indices.append(self.content_ids[batch])
            #print(f"aaa{batch}")
            #print(f"image_indices{image_indices}")
            for idx in range(0, len(self.results[batch])):
                detection = self.results[batch][idx]
                # this is the index of the coco image
                image_idx = int(detection[0])

                #if image_idx==24781:
                #    continue
                #print(f"image_idx:{image_idx},content_ids:{self.content_ids[batch]}")
                if image_idx != self.content_ids[batch]:
                    
                    # working with the coco index/id is error prone - extra check to make sure it is consistent
                    log.error("image_idx missmatch, lg={} / result={}".format(image_idx, self.content_ids[batch]))
                # map the index to the coco image id
                detection[0] = ds.image_ids[image_idx]
                height, width = ds.image_sizes[image_idx]
                # box comes from model as: ymin, xmin, ymax, xmax
                ymin = detection[1] * height
                xmin = detection[2] * width
                ymax = detection[3] * height
                xmax = detection[4] * width
                # pycoco wants {imageID,x1,y1,w,h,score,class}
                detection[1] = xmin
                detection[2] = ymin
                detection[3] = xmax - xmin
                detection[4] = ymax - ymin
                if self.use_inv_map:
                    cat_id = inv_map.get(int(detection[6]), -1)
                    if cat_id == -1:
                        # FIXME:
                        log.info("finalize can't map category {}".format(int(detection[6])))
                    detection[6] =  cat_id
                detections.append(np.array(detection))

        # map indices to coco image id's

        image_ids = [ds.image_ids[i]  for i in image_indices]

        self.results = []
        cocoGt = pycoco.COCO(ds.annotation_file)
        cocoDt = cocoGt.loadRes(np.array(detections))
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        result_dict["acc"] = cocoEval.stats[0]