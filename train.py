import importlib
import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.distributed as dist
import sys
import time
import shutil
import matplotlib.pyplot as plt

from utils import read_args, info_log, cal_cov_component, cal_concept, cal_acc, cal_class_MCP, cal_cov, load_model, check_device, CCD_loss, CKA_loss

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

# =============================================================================
# Get optimizer learning rate
# =============================================================================
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# =============================================================================
# Run one iteration
# =============================================================================
def one_step(model, data, label, loss_funcs, optimizer, args, concept_vectors = None, concept_means = None, class_MCP_dist = None):
    if args.device_id != -1:
        b_data = data.to(args.device_id)
        b_label = label.to(args.device_id)
    else:
        b_data = data
        b_label = label
        
    optimizer.zero_grad() 
    
    # Model forward 
    l1, l2, l3, l4 = model(b_data)

    # calculate loss
    if l1.shape[0] > 2:
        cka_loss = loss_funcs["CKA_loss"]((l1, l2, l3, l4), (1, 2, 3, 4))
    else:
        cka_loss = torch.tensor(0).cuda()

    ccd_loss = loss_funcs["CCD_loss"]((l1, l2, l3, l4), concept_vectors, concept_means, (1, 2, 3, 4), b_label, class_MCP_dist) * args.CCD_weight

    loss = cka_loss + ccd_loss

    loss.backward()

    optimizer.step()
    
    losses = {
                "CKA_loss" : cka_loss.detach(),
                "CCD_loss" : ccd_loss.detach()
             }
    
    return losses

def test(model, data, label, loss_func, args):
    if args.device_id != -1:
        b_data = data.to(args.local_rank)
        b_label = label.cuda(args.global_rank)
    else:
        b_data = data
        b_label = label

    # Model forward
    l1, l2, l3, l4 = model(b_data)
    if args.world_size > 1:
        l1 = torch.cat(GatherLayer.apply(l1.contiguous()), dim = 0)
        l2 = torch.cat(GatherLayer.apply(l2.contiguous()), dim = 0)
        l3 = torch.cat(GatherLayer.apply(l3.contiguous()), dim = 0)
        l4 = torch.cat(GatherLayer.apply(l4.contiguous()), dim = 0)
        b_label = torch.cat(GatherLayer.apply(b_label), dim = 0)

    losses = {
    }
    return losses, l1, l2, l3, l4

# =============================================================================
# Load data, load model (pretrain if needed), define loss function, define optimizer, 
# define learning rate scheduler (if needed), training and validation
# =============================================================================
def runs(args):
    # Load dataset ------------------------------------------------------------
    dataloader = importlib.import_module(args.dataloader)
    dataset, dataset_sizes, all_image_datasets = dataloader.load_data(args)
    # -------------------------------------------------------------------------
    
    # Define tensorboard for recording ----------------------------------------
    if args.global_rank in [-1, 0]:
        with open('{}/logging.txt'.format(args.dst), "a") as f:
            print('Index : {}'.format(args.index), file = f)
            print("dataset : {}".format(args.dataset_name), file = f)
        writer = SummaryWriter('./logs/{}/{}_{}'.format(args.index, args.model.lower(), args.basic_model.lower()))
    # -------------------------------------------------------------------------
    
    start_epoch = 1
    if args.resume:
        resume_data = torch.load(args.weight_path)
        args.concept_cha = resume_data['concept_cha']
        start_epoch = resume_data["Epoch"] + 1

    # Load model (load pretrain if needed) ------------------------------------
    model = load_model(args)
    # -------------------------------------------------------------------------
    
    # Define loss -------------------------------------------------------------
    loss_funcs = {}
    loss_funcs["CCD_loss"] = CCD_loss(args.concept_cha, args.margin)
    loss_funcs["CKA_loss"] = CKA_loss(args.concept_cha)
    if args.global_rank in [0, -1]:
        print(loss_funcs)
    assert len(loss_funcs) != 0, "Miss define loss"
    # -------------------------------------------------------------------------
    
    # Define optimizer --------------------------------------------------------
    train_optimizer = None
    if args.optimizer == "adam":
        train_optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    if args.optimizer == "sgd":
        train_optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = 0.9)
    if args.optimizer == "adamw":
        train_optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    assert train_optimizer is not None, "Miss define optimizer"
    # -------------------------------------------------------------------------
    
    # Define learning rate scheduler ------------------------------------------
    if "lr_scheduler" in args:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(train_optimizer, step_size = args.lr_scheduler, gamma = 0.1)
    # -------------------------------------------------------------------------
    
    # Define Meters -------------------------------------------------------
    max_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
    last_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
    # ---------------------------------------------------------------------
    
    # Train and Validation ---------------------------------------------------------------
    concept_vectors = [[], [], [], []]
    concept_means = [[], [], [],[]]
    first_concept_vectors = [[], [], [], []]
    first_concept_means = [[], [], [], []]
    
    train_transform = dataset["train"].dataset.transform
    val_transform = dataset["val"].dataset.transform
    for epoch in range(start_epoch, args.epoch + 1):
        
        if args.global_rank in [-1, 0]:
            info_log('-' * 15, args.global_rank, args.log_type, args.log)
            info_log('Epoch {}/{}'.format(epoch, args.epoch), args.global_rank, args.log_type, args.log)
            
        cov_xxs = [torch.zeros(args.concept_per_layer[0], args.concept_cha[0], args.concept_cha[0], dtype = torch.float64).cuda(args.global_rank), 
                    torch.zeros(args.concept_per_layer[1], args.concept_cha[1], args.concept_cha[1], dtype = torch.float64).cuda(args.global_rank), 
                    torch.zeros(args.concept_per_layer[2], args.concept_cha[2], args.concept_cha[2], dtype = torch.float64).cuda(args.global_rank), 
                    torch.zeros(args.concept_per_layer[3], args.concept_cha[3], args.concept_cha[3], dtype = torch.float64).cuda(args.global_rank)]
        cov_means = [torch.zeros(args.concept_per_layer[0], args.concept_cha[0], 1, dtype = torch.float64).cuda(args.global_rank), 
                    torch.zeros(args.concept_per_layer[1], args.concept_cha[1], 1, dtype = torch.float64).cuda(args.global_rank), 
                    torch.zeros(args.concept_per_layer[2], args.concept_cha[2], 1, dtype = torch.float64).cuda(args.global_rank), 
                    torch.zeros(args.concept_per_layer[3], args.concept_cha[3], 1, dtype = torch.float64).cuda(args.global_rank)]
        Sum_As = [torch.zeros(args.concept_per_layer[0], dtype = torch.float64).cuda(args.global_rank), 
                    torch.zeros(args.concept_per_layer[1], dtype = torch.float64).cuda(args.global_rank), 
                    torch.zeros(args.concept_per_layer[2], dtype = torch.float64).cuda(args.global_rank), 
                    torch.zeros(args.concept_per_layer[3], dtype = torch.float64).cuda(args.global_rank)]
        Square_Sum_As = [torch.zeros(args.concept_per_layer[0], dtype = torch.float64).cuda(args.global_rank), 
                        torch.zeros(args.concept_per_layer[1], dtype = torch.float64).cuda(args.global_rank), 
                        torch.zeros(args.concept_per_layer[2], dtype = torch.float64).cuda(args.global_rank), 
                        torch.zeros(args.concept_per_layer[3], dtype = torch.float64).cuda(args.global_rank)]
        
        # Inference one time to get the concept ====================================================
        if epoch == 1:
            if args.global_rank in [-1, 0]:
                print("First epoch: Extract the concept vectors and concept means!!")
            dataset["train"].dataset.transform = val_transform
            model.train(False)
            with torch.no_grad():
                nb = len(dataset["train"])
                pbar = enumerate(dataset["train"])
                if args.global_rank in [-1, 0]:  
                    pbar = tqdm.tqdm(pbar, total = nb)  # progress bar

                # Evaluate first time and Extract the concept vector
                for step, (data, label) in pbar:
                    losses, l1, l2, l3, l4 = test(model, data, label, loss_funcs, args)
                    features = [l1, l2, l3, l4]
                    Sum_As, Square_Sum_As, cov_xxs, cov_means = cal_cov_component(features, Sum_As, Square_Sum_As, cov_xxs, cov_means, args)
                    
                if args.world_size > 1:
                    for i in range(len(features)):
                        dist.all_reduce(Sum_As[i], op = dist.ReduceOp.SUM)
                        dist.all_reduce(Square_Sum_As[i], op = dist.ReduceOp.SUM)
                        dist.all_reduce(cov_xxs[i], op = dist.ReduceOp.SUM)
                        dist.all_reduce(cov_means[i], op = dist.ReduceOp.SUM)
                
                covs = []
                for i in range(len(features)):
                    # calculate weighted covariance matrix
                    cov, cov_mean = cal_cov(cov_xxs[i], cov_means[i], Sum_As[i])
                    covs.append(cov)
                    concept_means[i] = cov_mean
                    # eigen decompose
                    concept_vectors[i], concept_means[i] = cal_concept(cov, cov_mean)
                    first_concept_vectors[i] = concept_vectors[i].type(torch.float32).clone()
                    first_concept_means[i] = concept_means[i].type(torch.float32).clone()
                torch.cuda.empty_cache()

                # Calculate the class MCP distributions
                class_MCP = cal_class_MCP(model, concept_vectors, concept_means, dataset["train"], args.category, args)
            print("Finish extract concept and MCP distribution!!")
        torch.cuda.empty_cache()

        # train phase =================================================================================================
        dataset["train"].dataset.transform = train_transform
        model.train(True)
        if args.global_rank != -1:
            dataset["train"].sampler.set_epoch(epoch)
            dataset["val"].sampler.set_epoch(epoch)
        loss_t = AverageMeter()
        loss_detail_t = {}
        nb = len(dataset["train"])
        pbar = enumerate(dataset["train"])
        if args.global_rank in [-1, 0]:
            pbar = tqdm.tqdm(pbar, total=nb)  # progress bar

        for step, (data, label) in pbar:
            losses = one_step(model = model, 
                                data = data,
                                label = label, 
                                loss_funcs = loss_funcs, 
                                optimizer = train_optimizer, 
                                args = args, 
                                concept_vectors = concept_vectors, 
                                concept_means = concept_means,
                                class_MCP_dist = class_MCP)
            # record losses
            loss = 0
            for key in losses.keys():
                loss_i = losses[key]
                dist.all_reduce(loss_i, op = dist.ReduceOp.SUM)
                loss_i = loss_i / args.world_size
                loss += loss_i
                if key not in loss_detail_t.keys():
                    loss_detail_t[key] = AverageMeter()

                if args.global_rank in [-1, 0]:  
                    loss_detail_t[key].update(loss_i, data.size(0) * args.world_size)
                
                losses[key] = losses[key].detach().item()
            if args.global_rank in [-1, 0]:
                loss_t.update(loss, data.size(0) * args.world_size)
                pbar.set_postfix(losses)

        if args.global_rank in [-1, 0]:
            writer.add_scalar('Loss/train', loss_t.avg, epoch)
            for key in loss_detail_t.keys():
                writer.add_scalar('{}/train'.format(key), loss_detail_t[key].avg, epoch)
        
        if epoch == 1:
            for layer_i in range(len(Sum_As)):
                Sum_As[layer_i] = torch.zeros_like(Sum_As[layer_i], dtype = torch.float64)
                Square_Sum_As[layer_i] = torch.zeros_like(Square_Sum_As[layer_i], dtype = torch.float64)
                cov_xxs[layer_i] = torch.zeros_like(cov_xxs[layer_i], dtype = torch.float64)
                cov_means[layer_i] = torch.zeros_like(cov_means[layer_i], dtype = torch.float64)
        torch.cuda.empty_cache()

        # validation =============================================================================================================   
        dataset["train"].dataset.transform = val_transform
        model.train(False)
        for phase in ["train", "val"]:
            correct_t = AverageMeter()
            correct_t5 = AverageMeter()

            loss_t = AverageMeter()
            loss_detail_t = {}
            
            with torch.no_grad():
                total_correct = 0
                total_count = 0
                nb = len(dataset[phase])
                pbar = enumerate(dataset[phase])
                if args.global_rank in [-1, 0]:  
                    pbar = tqdm.tqdm(pbar, total = nb)  # progress bar

                # Evaluate first time and Extract the concept vector
                for step, (data, label) in pbar:
                    if args.global_rank != -1:
                        b_label = label.cuda(args.global_rank)

                    losses, l1, l2, l3, l4 = test(model, data, label, loss_funcs, args)
                    features = [l1, l2, l3, l4]
                    if phase == "train":
                        Sum_As, Square_Sum_As, cov_xxs, cov_means = cal_cov_component(features, Sum_As, Square_Sum_As, cov_xxs, cov_means, args)
                    else:
                        # Calculate val-set acc
                        resp_top1, resp_top5 = cal_acc(features, class_MCP, concept_vectors, concept_means, args)

                    loss = 0
                    for key in losses.keys():
                        loss_i = losses[key]
                        dist.reduce(loss_i, 0, op = dist.ReduceOp.SUM)
                        loss_i = loss_i / args.world_size
                        loss += loss_i
                        if key not in loss_detail_t.keys():
                            loss_detail_t[key] = AverageMeter()

                        if args.global_rank in [-1, 0]:  
                            loss_detail_t[key].update(loss_i, data.size(0) * args.world_size)
                            
                    if args.global_rank in [-1, 0]: 
                        loss_t.update(loss, data.size(0) * args.world_size)

                    if phase == "val":
                        b_label_all = [torch.zeros_like(b_label) for _ in range(args.world_size)]
                        dist.all_gather(b_label_all, b_label)
                        b_label = torch.cat(b_label_all, dim = 0)
                        correct_1 = (resp_top1 == b_label.unsqueeze(1)).sum()
                        total_correct += correct_1
                        total_count += b_label.shape[0]
                        correct_5 = (resp_top5 == b_label.unsqueeze(1)).sum()
                        assert correct_5 >= correct_1, "Error on calulate accuracy"
                        
                        if args.global_rank in [-1, 0]:  
                            correct_t.update(correct_1.item() / b_label.shape[0], b_label.shape[0])
                            correct_t5.update(correct_5.item() / b_label.shape[0], b_label.shape[0])
                    
                if phase == "train":
                    for i in range(4):
                        dist.all_reduce(Sum_As[i], op = dist.ReduceOp.SUM)
                        dist.all_reduce(Square_Sum_As[i], op = dist.ReduceOp.SUM)
                        dist.all_reduce(cov_xxs[i], op = dist.ReduceOp.SUM)
                        dist.all_reduce(cov_means[i], op = dist.ReduceOp.SUM)
                        
                    covs = []
                    sim_vecs = []
                    sim_means = []
                    for i in range(4):
                        # calculate weighted covariance matrix
                        cov, cov_mean = cal_cov(cov_xxs[i], cov_means[i], Sum_As[i])
                        covs.append(cov)
                        concept_means[i] = cov_mean
                        # eigen decompose
                        concept_vectors[i], concept_means[i] = cal_concept(cov, cov_mean)

                    # Calculate the class MCP
                    class_MCP = cal_class_MCP(model, concept_vectors, concept_means, dataset["train"], args.category, args)

            if args.global_rank in [-1, 0]:  
                # Recording loss and accuracy ---------------------------------
                if phase == "val":
                    writer.add_scalar('Loss/{}'.format(phase), loss_t.avg, epoch)
                    for key in losses.keys():
                        writer.add_scalar('{}/{}'.format(key, phase), loss_detail_t[key].avg, epoch)

                writer.add_scalar('Accuracy resp top1/{}'.format(phase), correct_t.avg, epoch)
                writer.add_scalar('Accuracy resp top5/{}'.format(phase), correct_t5.avg, epoch)
                # -------------------------------------------------------------
                
                # Save model --------------------------------------------------
                if max_acc[phase].avg <= correct_t.avg:
                    last_acc[phase] = max_acc[phase]
                    max_acc[phase] = correct_t
                    
                    if phase == 'val':
                        ACCMeters = correct_t
                        LOSSMeters = loss_t
                        info_log('save')

                        optimizers_state_dict= train_optimizer.state_dict()
                        lr_state_dict = lr_scheduler.state_dict()
                            
                        save_data = {"Model" : model.state_dict(),
                                    "Epoch" : epoch,
                                    "Optimizer" : optimizers_state_dict,
                                    "lr_scheduler" : lr_state_dict,
                                    "Best ACC" : max_acc[phase].avg,
                                    "concept_cha" : args.concept_cha}
                        torch.save(save_data, f"{args.dst}/best_model.pkl")
                        MCP_data = {"cent_MCP" : class_MCP,
                                        "concept_covs" : covs,
                                        "concept_means" : concept_means}
                        torch.save(MCP_data, f"{args.dst}/MCP_data.pkl")

                optimizers_state_dict= train_optimizer.state_dict()
                lr_state_dict = lr_scheduler.state_dict()
                save_data = {"Model" : model.state_dict(),
                                "Epoch" : epoch,
                                "Optimizer" : optimizers_state_dict,
                                "Lr_scheduler" : lr_state_dict,
                                "Best ACC" : max_acc[phase].avg,
                                "concept_cha" : args.concept_cha}
                torch.save(save_data, './pkl/{}/{}_{}/last_model.pkl'.format(args.index, args.model.lower(), args.basic_model.lower()))
                # -------------------------------------------------------------
                info_log('Index : {}'.format(args.index), args.global_rank, args.log_type, args.log)
                info_log("dataset : {}".format(args.dataset_name), args.global_rank, args.log_type, args.log)
                info_log("Model name : {}_{}".format(args.model, args.basic_model), args.global_rank, args.log_type, args.log)
                info_log("{} set loss : {:.6f}".format(phase, loss_t.avg), args.global_rank, args.log_type, args.log)
                for key in loss_detail_t.keys():
                    info_log("    {} set {} : {:.6f}".format(phase, key, loss_detail_t[key].avg), args.global_rank, args.log_type, args.log)
                info_log("{} set resp top-1 acc : {:.6f}%".format(phase, correct_t.avg * 100.), args.global_rank, args.log_type, args.log)
                info_log("{} set resp top-5 acc : {:.6f}%".format(phase, correct_t5.avg * 100.), args.global_rank, args.log_type, args.log)
                info_log("{} resp last update : {:.6f}%".format(phase, (max_acc[phase].avg - last_acc[phase].avg) * 100.), args.global_rank, args.log_type, args.log)
                info_log("{} set resp max acc : {:.6f}%".format(phase, max_acc[phase].avg * 100.), args.global_rank, args.log_type, args.log)
                info_log("-" * 10, args.global_rank, args.log_type, args.log)
        lr_scheduler.step()
    # ---------------------------------------------------------------------

    # Show the best result ----------------------------------------------------
    info_log("Best acc : {:.6f} loss : {:.6f}".format(ACCMeters.avg, LOSSMeters.avg), args.global_rank, args.log_type, args.log)

# =============================================================================
# Templet for recording values
# =============================================================================
class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count
        
if __name__ == '__main__':
    args = read_args()
    # Set DDP variables
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # check if it can run on gpu
    device_id = check_device(args.devices, args.train_batch_size, args.val_batch_size)
    args.train_total_batch_size = args.train_batch_size
    args.val_total_batch_size = args.val_batch_size
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device_id = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert args.train_batch_size % args.world_size == 0, 'train_batch_size must be multiple of CUDA device count'
        args.train_batch_size = args.train_total_batch_size // args.world_size

    args.dst = f"{args.saved_dir}/pkl/{args.index}/{args.model.lower()}_{args.basic_model.lower()}"
    args.log = '{}/logging.txt'.format(args.dst)
    if args.global_rank in [-1, 0]:
        first_time = False
        if not os.path.exists(args.dst):
            first_time = True
            os.makedirs(args.dst)
        
        print(f"Args : {args}")
        if not args.resume and not first_time:
            response = input("The experiment already exist ({}/{}_{}). Are you sure you want replace it? (y/n)".format(args.index, args.model.lower(), args.basic_model.lower())).lower()
            while response != 'y' and response != 'n':
                response = input("The experiment already exist ({}/{}_{}). Are you sure you want replace it? (y/n)".format(args.index, args.model.lower(), args.basic_model.lower())).lower()
            if response == 'n':
                sys.exit()

        with open(args.log, "w") as f:
            print(f"Args : {args}", file = f)
        
        print("Save file to ", args.dst)
        shutil.copy(src = os.path.join(os.getcwd(), __file__), dst = args.dst)
        shutil.copy(src = os.path.join(os.getcwd(), "{}.py".format(args.model)), dst = args.dst)

        if args.basic_model == "resnet50":
            shutil.copy(src = os.path.join(os.getcwd(), "ResNet.py"), dst = args.dst)
        elif args.basic_model == "inceptionv3":
            shutil.copy(src = os.path.join(os.getcwd(), "inception_net.py"), dst = args.dst)
        
        shutil.copy(src = os.path.join(os.getcwd(), "arg_reader.py"), dst = args.dst)
        shutil.copy(src = os.path.join(os.getcwd(), "loss.py"), dst = args.dst)

        start = time.time()
    
    args.device_id = device_id
    runs(args)
    
    if args.global_rank in [-1, 0]:
        info_log("Train for {:.1f} hours".format((time.time() - start) / 3600), args.global_rank, args.log_type, args.log)
        
