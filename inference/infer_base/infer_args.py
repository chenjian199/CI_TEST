import argparse

class InferenceArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Offline inference for ResNet and RetinaNet.")
        self.parser.add_argument("--backend", default="PYTORCH", help="Set Backend" )
        self.parser.add_argument("--model", choices=["resnet50", "retinanet", "yolov8s", "inception_v3"], default="retinanet", help="Choose the model for inference: 'resnet' or 'retinanet'.")
        self.parser.add_argument("--scenario", choices=["Offline"], default="Offline", help="Choose the inference mode: 'Offline'.")
        self.parser.add_argument("--accuracy", type=bool, default=False, help="Run scenario with accuracy")
        self.parser.add_argument("--data_dir", default="path/to/data", help="Directory path.")
        self.parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference.")
        self.parser.add_argument("--duration", type=int, default=60, help="Inference duration in seconds for 'Offline' mode.")
        self.parser.add_argument("--sample_per_query", type=int, default=1000, help="Inference duration in seconds for 'Offline' mode.")
        self.parser.add_argument("--querys", type=int, default=1, help="querys num")
        self.parser.add_argument("--images_processed_path", type=str, default=None, help="path to images processed path e.g.: ./npy/ImageNet")
        self.parser.add_argument("--preload", type=bool, default=False, help="Load all samples to RAM")
        self.parser.add_argument("--precision", type=str, default="fp32", choices=["int8","fp4","fp8", "fp16","fp32"], help="Infernece Precision")
        self.parser.add_argument("--framework", type=str, default="pytorch", help="Framework")

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        self.model = args.model
        self.scenario = args.scenario
        self.acc = args.accuracy
        if args.accuracy:
            self.acc = True
        self.backend = args.backend
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.duration = args.duration
        self.sample_per_query = args.sample_per_query
        self.querys = args.querys
        self.num_workers = 4
        self.preload = args.preload
        self.precision = args.precision
        self.framework = args.framework
        self.processed_path = args.images_processed_path
