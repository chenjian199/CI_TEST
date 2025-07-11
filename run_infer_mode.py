import importlib
from inference.infer_base.base_model import BaseModel as Workload
from inference.scenario import Offline
from inference.infer_base.infer_args import InferenceArguments
from inference.model import Resnet50, Retinanet, Yolov8s, Inception_v3
from inference.runnable import Runnable
from backends.backend_base import BaseBackend

MODEL_MAP={
    "resnet50": Resnet50 , 
    "retinanet": Retinanet,
    "yolov8s": Yolov8s,
    "inception_v3":Inception_v3,
}

SCENARIO_MAP={
    "Offline": Offline
}


def main(args:InferenceArguments):
    
    #workload 初始化
    workload:Workload = MODEL_MAP[args.model](args)
    C,H,W = workload.input_shape
    workload.input_shape = (args.batch_size, C, H, W)

    #Backend 类载入
    backend_file = importlib.import_module(f"backends.{args.backend}")
    backend_module = getattr(backend_file, f"{args.backend.capitalize()}")
    

    #Backend 初始化
    print("Backend init ... ...")
    backend:BaseBackend = backend_module(args, workload)
    print("Backend success ... ...")


    #scenario 初始化
    scenario = SCENARIO_MAP[args.scenario](dataset=workload.datasets, param=args)
    
    #Runnable 初始化
    runnable = Runnable(arg=args, backend=backend, scenario=scenario, workload=workload)
    runnable.start_test()
    print("success!")
    
if __name__ == "__main__":
    args = InferenceArguments()
    args.parse()
    main(args)