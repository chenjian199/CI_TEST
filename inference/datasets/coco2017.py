import logging
import os
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from .dataset import Dataset
import concurrent.futures
import random
from numpy import ndarray
from typing import List, Tuple, Union
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco2017")

class COCO2017Val(Dataset):
    def __init__(self, coco_path, name, use_cache=0, image_size=None,
                 image_format="NCHW", cache_dir=None, preprocessed_dir=None, threads=os.cpu_count(), device='cpu'):
        super(COCO2017Val, self).__init__()
        self.image_size = [3, 640, 640] if image_size is None else image_size

        coco_gt_path = os.path.join(coco_path, "annotations/instances_val2017.json")
        images_path = os.path.join(coco_path, "val2017")
        self.data_path = images_path
        self.use_cache = use_cache
        self.device = device
        self.W, self.H = self.image_size[1], self.image_size[2]

        self.coco = COCO(coco_gt_path)
        self.image_ids = self.coco.getImgIds()

        if not cache_dir:
            cache_dir = os.getcwd()

        if preprocessed_dir:
            self.cache_dir = preprocessed_dir
        else:
            self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)

        self.image_list = []
        self.label_list = []
        self.count = None

        os.makedirs(self.cache_dir, exist_ok=True)

        image_paths = []
        for image_id in self.image_ids:
            img_info = self.coco.loadImgs(image_id)[0]
            image_path = Path(self.data_path) / img_info['file_name']
            image_paths.append((image_id, image_path))

        self.not_found = 0
        # Scan the directory to find images
        self.image_list = image_paths
        self.label_list = [0] * len(self.image_list)  # Placeholder labels, since COCO labels are in separate files

        N = threads
        CNT = len(self.image_list)
        if N > CNT:
            N = CNT

        start = time.time()

        log.info("Preprocessing {} images using {} threads".format(CNT, N))

        lists = []
        image_lists = []
        for i in range(N):
            lists.append(self.image_list[i::N])
            image_lists.append([])

        executor = concurrent.futures.ThreadPoolExecutor(N)
        futures = [executor.submit(self.process, item, image_lists[lists.index(item)]) for item in lists]
        concurrent.futures.wait(futures)

        self.image_list = []
        for i in range(len(image_lists)):
            self.image_list += image_lists[i]

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if self.not_found > 0:
            log.info("reduced image list, %d images not found", self.not_found)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))

    def process(self, files, image_list):
        for image_id, image_path in files:
            src = str(image_path)
            if not os.path.exists(src):
                # if the image does not exist, ignore it
                self.not_found += 1
                continue

            os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_path.name)), exist_ok=True)
            dst = os.path.join(self.cache_dir, image_path.stem)

            if not os.path.exists(dst + ".npy"):
                # cache a preprocessed version of the image
                tensor, ratio, dwdh = self.preprocess_image(src)
                if tensor is None:
                    self.not_found += 1
                    continue
                np.save(dst + ".npy", {'tensor': tensor.cpu().numpy(), 'ratio': ratio, 'dwdh': dwdh.cpu().numpy(), 'image_id': image_id})

            image_list.append((image_id, image_path.name))
        self.shuffle()

    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return None, None, None

        bgr, ratio, dwdh = letterbox(bgr, (self.W, self.H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        tensor = torch.asarray(tensor, device=self.device).squeeze(0)  # Remove the first dimension
        dwdh = torch.asarray(dwdh, dtype=torch.float32, device=self.device)  # Fix: do not multiply by 2

        return tensor, ratio, dwdh

    def get_item(self, idx):
        """Get image by number in the list."""
        image_id, image_name = self.image_list[idx]
        dst = os.path.join(self.cache_dir, Path(image_name).stem)
        data = np.load(dst + ".npy", allow_pickle=True).item()
        img = data['tensor']
        ratio = data['ratio']
        dwdh = data['dwdh']
        return img, (ratio, dwdh), image_id

    def shuffle(self):
        # Generate a random seed
        random_seed = random.randint(0, 1000)

        # Shuffle both lists using the same seed
        random.seed(random_seed)
        random.shuffle(self.label_list)

        random.seed(random_seed)
        random.shuffle(self.image_list)

    def get_item_loc(self, nr):
        _, image_name = self.image_list[nr]
        src = os.path.join(self.data_path, image_name)
        return src

class COCO2017PostProcess:

    def __init__(self, coco_gt_path):
        self.coco_gt = COCO(coco_gt_path)
        self.category_ids = []
        CLASSES_DET = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
        for class_name in CLASSES_DET:
            cat_ids = self.coco_gt.getCatIds(catNms=[class_name])
            if not cat_ids:
                raise ValueError(f"Class name '{class_name}' not found in COCO categories.")
            self.category_ids.append(cat_ids[0])  # 假设每个类别名对应一个唯一的category_id
        self.results = []

    def __call__(self, results, ids, expected=None):
        bboxes_batch = results["bboxes"]
        scores_batch = results["scores"]
        labels_batch = results["labels"]
        adjusted_bboxes_batch = []

        # 确保expected存在并且长度与bboxes_batch一致
        if expected is None or len(expected) != len(bboxes_batch):
            raise ValueError("Expected must be provided and have the same length as bboxes_batch")

        # 遍历批次中的每个样本
        for bboxes, (ratio, dwdh) in zip(bboxes_batch, expected):
            if bboxes is None or bboxes.numel() == 0:
                adjusted_bboxes_batch.append(bboxes)
                continue

            # 将dwdh和ratio转换为张量（如果尚未转换）
            if not isinstance(dwdh, torch.Tensor):
                dwdh = torch.tensor(dwdh, dtype=torch.float32, device=bboxes.device)
            if not isinstance(ratio, torch.Tensor):
                ratio = torch.tensor(ratio, dtype=torch.float32, device=bboxes.device)

            # 扩展dwdh为形状(4,)
            dwdh = torch.cat([dwdh, dwdh], dim=0)

            # 调整边界框坐标，将其映射回原始图像尺寸
            adjusted_bboxes = (bboxes - dwdh) / ratio

            adjusted_bboxes_batch.append(adjusted_bboxes)

        # 使用调整后的边界框添加预测结果
        self.add_predictions(ids, adjusted_bboxes_batch, scores_batch, labels_batch)


    def add_results(self, results):
        pass

    def add_predictions(self, image_ids, bboxes_batch, scores_batch, labels_batch):
        """Add predictions to the results list."""
        for image_id, bboxes, scores, labels in zip(image_ids, bboxes_batch, scores_batch, labels_batch):
            if bboxes is None or len(bboxes) == 0:
                continue

            for bbox, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = bbox.tolist()
                w = x2 - x1
                h = y2 - y1

                result = {
                    'image_id': image_id,
                    'category_id': self.category_ids[int(label)],
                    'bbox': [x1, y1, w, h],
                    'score': float(score)
                }
                self.results.append(result)

    def evaluate(self):
        if not self.results:
            print("No predictions to evaluate!")
            return {}

        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            'AP@0.5:0.95': coco_eval.stats[0],
            'AP@0.5': coco_eval.stats[1],
            'AP@0.75': coco_eval.stats[2],
            'AP_small': coco_eval.stats[3],
            'AP_medium': coco_eval.stats[4],
            'AP_large': coco_eval.stats[5]
        }
    
    def finalize(self, result_dict, ds):
        results = self.evaluate()
        result_dict["acc"] = results['AP@0.5:0.95']

# Example usage
if __name__ == "__main__":
    # Initialize dataset
    dataset = COCO2017Val(data_path="/path/to/coco/images", name="coco_val")
    
    # Get an image and its metadata
    img, label, img_id, ratio, dwdh = dataset.get_item(0)
    
    # Assuming we have inference results for this batch
    image_ids = [img_id]
    bboxes_batch = [torch.tensor([[10, 20, 100, 200]])]  # Example bounding box
    scores_batch = [torch.tensor([0.9])]  # Example confidence score
    labels_batch = [torch.tensor([1])]  # Example label
    
    # Initialize post-processing and add predictions
    post_process = COCO2017PostProcess(coco_gt_path="/path/to/annotations/instances_val2017.json")
    post_process.add_predictions(image_ids, bboxes_batch, scores_batch, labels_batch)
    
    # Evaluate the predictions
    metrics = post_process.evaluate()
    print(metrics)
