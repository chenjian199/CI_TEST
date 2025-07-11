"""
Implementation of Inception_imagenet dataset for ImageNet validation.
"""

import logging
import os
import re
import time
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import concurrent.futures

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BaseImagenetDataset")


class ImagenetDatasetTransform(Dataset):
    def __init__(self, data_path, image_list=None, name="imagenet", use_cache=0,
                 image_size=None, image_format="NCHW", pre_process=None, count=None,
                 cache_dir=None, preprocessed_dir=None, threads=os.cpu_count()):
        """
        Base class for ImageNet-like datasets.
        :param data_path: 原始数据所在目录
        :param image_list: 图片映射文件路径，默认为 data_path/val_map.txt
        :param name: 数据集名称，用于生成缓存路径
        :param use_cache: 保留原有参数，便于后续扩展
        :param image_size: 图片尺寸，若为空则由派生类设置默认值
        :param image_format: 指定输出图片格式（例如 "NCHW" 或 "NHWC"），主要用于缓存预处理结果时的目录命名
        :param pre_process: 图片预处理函数（例如 torchvision.transforms 组合），若不提供，则认为 data_path 下已有预处理数据
        :param count: 限制加载图片数量
        :param cache_dir: 缓存目录，若为空则使用当前工作目录
        :param preprocessed_dir: 如指定，则直接使用该目录作为缓存目录
        :param threads: 多线程数量，默认为 CPU 核心数
        """
        super(ImagenetDatasetTransform, self).__init__()
        self.image_size = image_size
        if not cache_dir:
            cache_dir = os.getcwd()
        self.image_list = []
        self.label_list = []
        self.count = count
        self.data_path = data_path
        self.pre_process = pre_process  # 若不为 None，则进行预处理并缓存
        self.use_cache = use_cache

        if preprocessed_dir:
            self.cache_dir = preprocessed_dir
        elif pre_process:
            self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        else:
            self.cache_dir = cache_dir

        # 对于 PIL 与 torchvision 的 transform，输出一般为 CHW 顺序，故此处无需额外转置
        self.need_transpose = False

        self.not_found = 0
        # 默认映射文件为 data_path/val_map.txt
        if image_list is None:
            image_list = os.path.join(data_path, "val_map.txt")
        # 统计映射文件中总行数
        with open(image_list, 'r') as fp:
            for count_line, _ in enumerate(fp):
                pass
        total_count = count_line + 1
        CNT = total_count if (count is None or count > total_count) else count

        os.makedirs(self.cache_dir, exist_ok=True)

        start = time.time()
        N = threads if threads <= CNT else CNT

        if self.pre_process:
            log.info("Preprocessing {} images using {} threads".format(CNT, N))
        else:
            log.info("Loading {} preprocessed images using {} threads".format(CNT, N))

        # 将映射文件的每一行分成多个子列表，便于多线程处理
        with open(image_list, 'r') as f:
            lists = []
            for _ in range(N):
                sublist = []
                for _ in range(int(CNT / N)):
                    try:
                        sublist.append(next(f))
                    except StopIteration:
                        break
                lists.append(sublist)
            remainder = CNT % N
            if remainder > 0:
                extra = []
                for _ in range(remainder):
                    try:
                        extra.append(next(f))
                    except StopIteration:
                        break
                lists.append(extra)

        # 为每个子列表准备对应的 image_list 与 label_list
        image_lists = [[] for _ in range(len(lists))]
        label_lists = [[] for _ in range(len(lists))]

        with concurrent.futures.ThreadPoolExecutor(N) as executor:
            futures = []
            for idx, sublist in enumerate(lists):
                futures.append(executor.submit(self.process, data_path, sublist,
                                                 image_lists[idx], label_lists[idx]))
            concurrent.futures.wait(futures)

        for sub in image_lists:
            self.image_list += sub
        for sub in label_lists:
            self.label_list += sub

        time_taken = time.time() - start
        if not self.image_list:
            log.error("No images found in image list")
            raise ValueError("No images found in image list")
        if self.not_found > 0:
            log.info("Reduced image list, %d images not found", self.not_found)

        log.info("Loaded {} images, use_cache={}, pre_process={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, self.pre_process is not None, time_taken))
        self.label_list = np.array(self.label_list)

    def process(self, data_path, lines, image_list, label_list):
        """
        处理每个线程分到的映射文件行
        """
        for line in lines:
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 2:
                continue
            image_name, label = parts[0], parts[1]
            src = os.path.join(data_path, image_name)
            if self.pre_process:
                if not os.path.exists(src):
                    self.not_found += 1
                    continue
                dst = os.path.join(self.cache_dir, image_name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst + ".npy"):
                    try:
                        # 使用 PIL 打开图片并转换为 RGB
                        img_org = Image.open(src).convert("RGB")
                    except Exception as e:
                        log.error("Error opening image %s: %s", src, str(e))
                        self.not_found += 1
                        continue
                    # 调用传入的预处理函数
                    processed = self.pre_process(img_org)
                    # 若返回的是 tensor，则转换为 numpy 数组
                    if hasattr(processed, "numpy"):
                        processed = processed.numpy()
                    np.save(dst, processed)
            else:
                # 若不做预处理，则认为 data_path 中已有 .npy 文件
                if not os.path.exists(os.path.join(data_path, image_name) + ".npy"):
                    self.not_found += 1
                    continue
            image_list.append(image_name)
            try:
                label_int = int(label)
            except ValueError:
                label_int = 0
            label_list.append(label_int)
            # 如果设置了 count 限制，则达到后退出
            if self.count and len(self.image_list) >= self.count:
                break

    def get_item(self, idx):
        """
        根据下标获取图片、标签和 id
        """
        dst = os.path.join(self.cache_dir, self.image_list[idx])
        img = np.load(dst + ".npy")
        label = self.label_list[idx]
        id = idx
        return img, label, id

    def shuffle(self):
        """
        随机打乱 image_list 与 label_list（使用相同随机种子保证顺序一致）
        """
        random_seed = random.randint(0, 1000)
        random.seed(random_seed)
        random.shuffle(self.image_list)
        random.seed(random_seed)
        random.shuffle(self.label_list)

    def get_item_loc(self, nr):
        """
        获取原始图片路径
        """
        return os.path.join(self.data_path, self.image_list[nr])


# Derived class for Inception V3 with default input size 299x299x3
class Inception_imagenet(ImagenetDatasetTransform):
    def __init__(self, data_path, image_list=None, name="inception_imagenet", use_cache=0,
                 image_size=None, image_format="NCHW", pre_process=None, count=None,
                 cache_dir=None, preprocessed_dir=None, threads=os.cpu_count()):
        # 如果未指定 image_size，则默认使用 Inception V3 的输入尺寸
        if image_size is None:
            image_size = [299, 299, 3]
        super(Inception_imagenet, self).__init__(data_path, image_list, name, use_cache,
                                                 image_size, image_format, pre_process,
                                                 count, cache_dir, preprocessed_dir, threads)


