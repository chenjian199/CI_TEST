import os
import json
import time
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    def __init__(self, param, mode:str):
        self.records = self.gen_base_structure()
        self.summary_data = {}
        self.log_file = param.log_file
        self.param = param
        self.set_mode(mode)
        self._initialize_log_file()
        self._initialize_json_file()

    def _initialize_log_file(self):
        """Initialize or clear the log file."""
        if os.path.exists(self.log_file):
            # Clear the existing content if the file exists
            open(self.log_file, 'w').close()
        else:
            # Create an empty file if it doesn't exist
            with open(self.log_file, 'w') as f:
                pass

    def _initialize_json_file(self):
        self.json_title = f"{self.mode.upper()}"



    def save_json_file(self, filename):
         
        self.add_json_data("Params",
                           self.param.get_dict()
                            )
        """保存记录到 JSON 文件"""
        data = {self.json_title: self.records}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def get_logs(self, key=None):
        """获取指定键的所有日志，或获取所有日志"""
        if key:
            return self.records.get(key, [])
        return self.records


    def write_log(self, message):
        """以固定的格式写入日志"""
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")

    def set_mode(self, mode: str):
        if mode in ["train", "val", "infer", "base test"]:
            self.mode = mode.title()

        else:
            raise ValueError(f"Invalid mode {mode}. Available modes: train, val, infer, base_test")
  
    def display(self, message):
        """在屏幕上输出消息"""
        print(message)

    
    def add_json_data(self, key, value):
        """在预先定义好的结构中添加或修改元素"""
        keys = key.split('.')
        temp = self.records
        for k in keys[:-1]:
            temp = temp[k]
        temp[keys[-1]] = value


    #预先规定好json格式,生成基本的JSON结构，并确保元素的顺序
    @abstractmethod
    def gen_base_structure(self):
        pass

    