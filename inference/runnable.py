from ast import List
import queue
import threading
import time
import numpy as np
from backends.backend_base import BaseBackend
from inference.infer_base.base_scenario import BaseScenario
from inference.infer_base.base_model import BaseModel
from inference.infer_base.infer_args import InferenceArguments
from inference.infer_base.infer_log import InferLogger
from inference.scenario import QueryLoadServer

class Metrics:
    def __init__(self, infer_time, acc: bool) -> None:
        self.acc = acc
        self.infer_time = infer_time
        self.latency_list: List = []
        self.test_start_time: float = 0.0
        self.processed_sample: int = 0
        self.processed_time: float = 0.0
        self.acc_result = None
        self.batch_first_time: float = 0.0
        self.batch_last_time: float = 0.0

    def batch_processed(self, duration, batch):
        now = time.time()
        batch_time = duration
        self.latency_list.append(batch_time)
        self.processed_sample += batch
        self.processed_time += batch_time
        if self.processed_time >= self.infer_time and not self.acc:
            return "Finished"
        else:
            return "processing"

    def finish(self):
        self.throughput = self.processed_sample / self.processed_time
        #print(f"Throughput: {self.throughput}")

class Runnable:
    def __init__(self, arg: InferenceArguments, scenario: BaseScenario, backend: BaseBackend, workload: BaseModel) -> None:
        self.param = arg
        self.scenario = scenario
        self.event = threading.Event()
        self.queue = queue.Queue()
        self.recorder = backend.recording
        self.malloc_pinned_buffer = backend.malloc_pinned_buffer
        self.cp_image_to_pinned_buffer = backend.cp_numpy_to_pinned_buffer
        self.predict = backend.predict
        self.backendpost = backend.postprocess
        self.sync = backend.device_sync
        self.workload = workload
        self.infer_json = InferLogger(arg)

    def process_batch(self, image_list, label_list, id_list):
        buffer_size = sum(arr.nbytes for arr in image_list)
        pinned_memory = self.malloc_pinned_buffer(buffer_size)
        pinned_images = self.cp_image_to_pinned_buffer(image_list, pinned_memory)
        
        self.recorder.trigger()
        start = time.time()
        outputs = self.predict(pinned_images)
        duration = time.time() - start
        self.recorder.trigger()
       

        if self.param.acc:
            outputs = self.backendpost(outputs)
            batch_results = self.workload.postprocessor(results=outputs, ids=id_list, expected=label_list)
            self.workload.postprocessor.add_results(batch_results)

        return duration
    
    def warmup(self, image_list):
        print("Use 1 batch images to Warmup ... ...")
        buffer_size = sum(arr.nbytes for arr in image_list)
        pinned_memory = self.malloc_pinned_buffer(buffer_size)
        pinned_images = self.cp_image_to_pinned_buffer(image_list, pinned_memory)
        outputs = self.predict(pinned_images)
        print("Warmup Success!")
    
    def warmup_with_fake_data(self, image, batch_size):
        input_shape = image.shape
        fake_data = [np.random.rand(*input_shape).astype(np.float32) for _ in range(batch_size)]
        self.warmup(fake_data)
        
        
    def start_test(self):
        print("QLS init ... ...")
        warmup = True
        QLS = QueryLoadServer(scenario=self.scenario, event=self.event, queue=self.queue)
        QLS.start()
        self.event.set()

        metrics = Metrics(self.param.duration, self.param.acc)
        metrics.test_start_time = QLS.QLS_start_time

        querys = 1
        state = "processing"
        image_list = []
        label_list = []
        id_list = []
        time_list = []
        num = 0

        while state == "processing":
            
            item = self.queue.get()

            if warmup==True:
                self.warmup_with_fake_data(item[0], self.param.batch_size)
                warmup = False

            if isinstance(item[0], np.ndarray):
                image, label, id_, time_ = item
                image_list.append(image)
                label_list.append(label)
                id_list.append(id_)
                time_list.append(time_)

                if len(image_list) >= self.param.batch_size:
                    num += len(image_list)
                    duration = self.process_batch(image_list, label_list, id_list)
                    state = metrics.batch_processed(duration, len(image_list))
                    image_list, label_list, id_list, time_list = [], [], [], []
                

            else:
                #print(f"Stop signal: {item[0]}")
                if item[0] in ["Query", "Acc"]:
                    if (item[0] == "Query" and querys >= self.param.querys) or item[0] == "Acc":
                        if image_list:
                            duration = self.process_batch(image_list, label_list, id_list)
                            metrics.batch_processed(duration, len(image_list))
                        state = "Querys Finished" if item[0] == "Query" else "Acc Finished"
                       # print(state)
                    if item[0] == "Query":
                        querys += 1
                        self.event.set()
        metrics.finish()
        if self.param.acc:
            result_dict = {"good": 0, "total": 0}
            self.workload.postprocessor.finalize(result_dict, ds=self.workload.datasets)
            metrics.acc_result = result_dict["acc"]

        QLS.stop = True
        self.event.set()
        QLS.join()
        record = self.recorder.get_records()
        summary = self.recorder.get_records_summary()
        self.infer_json.finish(metric=metrics, gpu_summary=summary)
        print("Inference SUCCESS!")

        self.recorder.records_dump(mode="inference", records=record, model=self.param.model)
        self.recorder.free()
