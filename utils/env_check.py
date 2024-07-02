import os
import torch
from contextlib import contextmanager

@contextmanager
def main_process_first(local_rank):
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

def check_device(device, train_batch_size, val_batch_size = None):
    if val_batch_size is None:
        val_batch_size = train_batch_size
    use_cpu = device == "cpu"
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(num) for num in device)

    use_cuda = not use_cpu and torch.cuda.is_available()

    if use_cuda:
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1 and train_batch_size:
            assert train_batch_size % num_gpu == 0, f"The batch size {train_batch_size} must divided by gpu number {num_gpu}."
        if num_gpu > 1 and val_batch_size:
            assert val_batch_size % num_gpu == 0, f"The batch size {val_batch_size} must divided by gpu number {num_gpu}."

    return torch.device("cuda:0" if use_cuda else "cpu")
