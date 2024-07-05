import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import info_log, main_process_first, prepare_transforms
import torch
import os

def create_dataloader(path, batch_size, shuffle, n_workers, rank, mode, args):
    with main_process_first(rank):
        data_transforms = prepare_transforms(args)
        if rank in [-1, 0]:
            print("{} data_transforms : ".format(mode))
            print(data_transforms)
        dataset = torchvision.datasets.ImageFolder(path, data_transforms[mode])

    batchsize = min(batch_size, len(dataset))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle) if rank != -1 else None
    num_workers = min([os.cpu_count() // args.world_size, batch_size if batch_size > 1 else 0, n_workers])  # number of workers
    loader = DataLoader(dataset,
                        batch_size = batch_size,
                        sampler = sampler if rank != -1 else None,
                        shuffle = shuffle if rank == -1 else None,
                        num_workers = num_workers,
                        pin_memory = True) 

    return loader, dataset 

def load_data(args):
    dataloader = []
    dataset_sizes = []
    trainloader, traindataset = create_dataloader(args.train_dataset_path, 
                                                args.train_batch_size, 
                                                True,
                                                args.train_num_workers,
                                                args.global_rank,
                                                "train", args)

    valloader, valdataset = create_dataloader(args.val_dataset_path, 
                                            args.val_batch_size, 
                                            True,
                                            args.val_num_workers,
                                            args.global_rank, 
                                            "val", args)
    
    # combine
    dataloader = {"train" : trainloader, "val" : valloader}
    dataset_sizes = {"train" : len(trainloader), "val" : len(valloader) if valloader is not None else 0}
    return dataloader, dataset_sizes, None