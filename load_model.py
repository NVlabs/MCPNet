import torch.nn as nn
import torch
import importlib
from torch.nn.parallel import DistributedDataParallel
import sys

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

# =============================================================================
# Return model
# =============================================================================
def load_model(args):
    if args.global_rank in [-1, 0]:
        print(args.model.lower())
    use_cuda = args.device_id != "cpu"
    if args.model.lower() == "resnet":
        print("load ResNet.py")
        model_class = importlib.import_module("ResNet")
        model = model_class.load_model(args.basic_model, num_classes = args.category)
    elif args.model.lower() == "convnext":
        model_class = importlib.import_module("convnext")
        model = model_class.load_model(args.basic_model, num_classes = args.category)
    elif args.model.lower() == "convnext_isotropic":
        model_class = importlib.import_module("convnext_isotropic")
        model = model_class.load_model(args.basic_model, num_classes = args.category)
    elif args.model.lower() == "inception_net":
        model_class = importlib.import_module("inception_net")
        model = model_class.load_model(args.basic_model, num_classes = args.category)
    elif args.model.lower() == "mobilenet":
        model_class = importlib.import_module("mobilenet")
        model = model_class.load_model(args.basic_model, num_classes = args.category)
    elif args.model.lower() == "vit":
        model_class = importlib.import_module("vit")
        model = model_class.load_model(args.basic_model, num_classes = args.category)
    elif args.model.lower() == "vit_wo_cls":
        model_class = importlib.import_module("vit_wo_CLS")
        model = model_class.load_model(args.basic_model, num_classes = args.category)
    elif args.model.lower() == "vgg":
        model_class = importlib.import_module("VGG")
        model = model_class.load_model(args.basic_model, num_classes = args.category)
    elif args.model.lower() == "densenet":
        model_class = importlib.import_module("densenet")
        model = model_class.load_model(args.basic_model, num_classes = args.category)
    else:
        model_class = importlib.import_module(args.model)
        model = model_class.load_model(num_classes = args.category, basic_model = args.basic_model)
    
    # use SyncBatchNorm
    if use_cuda and args.global_rank != -1 and has_batchnorms(model):
        print("Using ", args.device_id)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device_id)

    if "parameter_path" in args and args.parameter_path is not None:
        if args.global_rank in [-1, 0]:
            print("load pretrained : {}".format(args.parameter_path))
        model = model_class.load_pretrained(args.parameter_path, model).to(args.device_id)

    # DDP mode
    if use_cuda and args.global_rank != -1:
        if args.global_rank in [0]:
            print("DDP mode")
        model = DistributedDataParallel(model, device_ids = [args.local_rank])# , output_device = args.local_rank

    if args.resume:
        print("load resume weight : {}".format(args.weight_path))
        model = model_class.load_weight(args.weight_path, model)
    return model


# =============================================================================
# Load pretrain model parameter
# =============================================================================
def load_parameter(model, args):
    import torch
    params = torch.load(args.parameter_path)
    load = []
    not_load = []
    for name, param in params.items():
        if name in model.state_dict():
            try:
                model.state_dict()[name].copy_(param)
                load.append(name)
            except:
                not_load.append(name)
        else:
            not_load.append(name)
    print("Load {} layers".format(len(load)))
    print("Not load {} layers".format(len(not_load)))
    return model
    
# =============================================================================
# Custom model
# =============================================================================
def custom_model():
    pass

# =============================================================================
# Load model weight
# =============================================================================
def load_param(model):
    # load resnet
    params = torch.load("../pretrain/resnet50.pth")
    load = []
    not_load = []
    for name, param in params.items():
        if name in model.state_dict():
            try:
                model.state_dict()[name].copy_(param)
                load.append(name)
            except:
                not_load.append(name)
        else:
            not_load.append(name)
    print("Load {} layers".format(len(load)))
    print("Not load {} layers".format(len(not_load)))
            
    return model