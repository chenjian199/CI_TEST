import torch
import torch.distributed as dist
from .base_log import BaseLogger as Logger

class DistributedUtility:
    def __init__(self, param):
        if param.log_file:
            self.logger = Logger(param,"train")
        else:
            self.logger = None

    def distributed_log_write(self, message):
        if self.logger:
            if self._is_main_process():
                self.logger.write_log(message)

    def _is_main_process(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def distributed_print(self, *args, **kwargs):
        """
        Prints the given message only on the main node in a distributed environment.
        If not in a distributed environment, simply prints the message.
        """
        if self._is_main_process():
            print(*args, **kwargs)

    def aggregate_tensor(self, data, average=True):
        """
        Aggregate tensor across all processes.
        """
        try:
            # Convert data to tensor and ensure it's on CUDA
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data).cuda()

            tensor = data.clone().detach().to('cuda')

            if tensor.device.type == 'cuda':
                torch.cuda.synchronize()

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

                if average:
                    tensor /= dist.get_world_size()
        except Exception as e:
            print(f"Error in aggregate_tensor: {e}")
            raise

        return tensor
