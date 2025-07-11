from abc import ABC, abstractmethod
from torch.cuda.amp import autocast

class BaseTrainer(ABC):
    def __init__(self):
        self.loss_function = None
        self.optimizer = None
        self.grad_scaler = None
        self.autocast = None
        self.input = None
        self.input_target = None
        self.loss = 0
        self.total_loss = 0
        self.pre_step = 0
    

    def forward(self):
        return self.model_predict()

    def model_predict(self):
        return self.model(self.input)

    def compute_loss(self,predictions):

        return self.loss_function(predictions, self.input_target)

    def update(self, optimizer, grad_scaler=None):
        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            self.optimizer.step()

    def backward(self):
        if self.grad_scaler is not None:
            scaled_loss = self.grad_scaler.scale(self.loss)
            scaled_loss.backward()
            self.update(self.optimizer, self.grad_scaler)
        else:
            self.loss.backward()
            self.update(self.optimizer)

    def forward_and_process(self):
        self.data_process()
        self.model.train()
        output = self.forward()
        output = self.output_process(output)
        self.loss = self.compute_loss(output)
        self.total_loss += self.loss

    def train_one_step(self):
        self.optimizer.zero_grad()
        if self.grad_scaler is not None:
            with autocast(enabled=True):
                self.forward_and_process()
        else:
            self.forward_and_process()
        
        self.backward()
    
    def log_fn(self, metrics):
        metrics.step += 1
        if metrics.step % metrics.log_steps == 0:
            metrics.end_steps
            this_thread_throughput = ((metrics.global_batchsize * metrics.log_steps) / metrics.steps_duration)
            metrics.throughput = metrics.data_collector(this_thread_throughput,average=False)
            metrics.loss = metrics.data_collector(self.total_loss)
            metrics.train_log_print(pre_step=self.pre_step,
                                        now_step=metrics.step,
                                        duration=metrics.steps_duration,
                                        loss=metrics.loss,
                                        throughput=metrics.throughput)
            metrics.start_steps
            self.pre_step = metrics.step
            self.total_loss = 0



    @abstractmethod
    def data_process(self):
        pass


    @abstractmethod
    def output_process(self,output):
        pass  


    #无返回值 一个epoch结束，希望获得每秒吞吐量、一个epoch的训练时间
    @abstractmethod
    def train_one_epoch(self):
        pass
         

    @abstractmethod
    def device_warm_up(self):
        pass
    
    #无返回值 一次val结束，希望获得 验证时间，准确度,loss
    @abstractmethod
    def validate(self):
        pass
