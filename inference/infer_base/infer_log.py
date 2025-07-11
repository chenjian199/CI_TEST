from ast import List
import json
import os
from datetime import datetime
import numpy as np
from .infer_args import InferenceArguments

class InferLogger:
    def __init__(self,
                 args:InferenceArguments,
                 ):
        infer = {}
        
        infer["[WORKLOAD]"] = args.model
        infer["batch Size"] = args.batch_size
        infer["precision"] = args.precision
        infer['framework'] = args.framework
        infer["accuracy"] = args.acc
        self.infer = infer


    def calculate_percentiles(self,time_differences, percentiles=[50, 80, 90, 95, 99, 99.9]):
        """
        Calculate specified percentiles for a list of time differences.

        :param time_differences: List of float, time differences
        :param percentiles: List of float, percentiles to calculate
        :return: Dictionary of percentiles and their values
        """
        percentile_values = np.percentile(time_differences, percentiles)
        return {f"{p}": value for p, value in zip(percentiles, percentile_values)}
    
    def status_judge(self, acc, model):
        BaseLine = {
            'resnet50':0.752499,
            'retinanet':0.37125,
            'inception_v3':0.76725,
            'yolov8s':0.4356
        }
        if acc<BaseLine[model]:
            return 'The accuracy is not up to standard'
        else:
            return 'pass'
        

    def finish(self, metric, gpu_summary):
        infer = self.infer
        infer["accuracy"] = metric.acc_result
        infer["Status"] = self.status_judge(metric.acc_result, infer["[WORKLOAD]"]) if infer["accuracy"] else None
        infer["time"] = metric.processed_time
        infer["samples"] = metric.processed_sample
        infer["throughput"] = metric.throughput
        infer["percentiles"] = self.calculate_percentiles(metric.latency_list)
        
        # 将结果转换成 JSON 格式
        summary = {}
        summary['Status'] = self.status_judge(metric.acc_result, infer["[WORKLOAD]"]) if infer["accuracy"] else None
        summary["[RESULT]"] = infer
        summary["[GPU_SUMMARY]"] = gpu_summary
        
        json_result = {}
        json_result['result'] = summary
        
        # 获取当前时间
        current_time = datetime.now()
        
        # 创建目录名，格式为年-月-日-时-分
        dir_name = current_time.strftime("%Y-%m-%d-%H-%M")
        
        # 创建目录
        os.makedirs(f'./results/{infer["[WORKLOAD]"]}/{dir_name}', exist_ok=True)
        print(f"Throughput: {metric.throughput} images/s")
        # 文件路径
        path = f'./results/{infer["[WORKLOAD]"]}/{dir_name}/{infer["[WORKLOAD]"]}.json'
        
        # 将 json_result 序列化为 JSON 字符串并写入文件
        with open(path, 'w') as file:
            file.write(json.dumps(json_result, indent=4))