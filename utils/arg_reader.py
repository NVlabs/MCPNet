import argparse
import configparser
from easydict import EasyDict as edict

def read_args():

    cfg = edict()
    parser = argparse.ArgumentParser(description='The options of the MCPNet.')

    parser.add_argument("--index", type = str, default = None, required = True, help = "Name of the experiments")
    parser.add_argument("--saved_dir", default = ".", type = str)
    parser.add_argument("--log_type", default = ["std", "log"], type = str, nargs = "+")

    # training hyper parameters
    parser.add_argument("--local_rank", type = int, default = -1, help = "DDP parameter. (Don't modify !!)")
    parser.add_argument("--devices", type = int, default = None, required = True, nargs = "+")
    parser.add_argument("--epoch", type = int, default = 50)
    parser.add_argument("--optimizer", type = str, default = None, required = True, choices = ["adam", "sgd", "adamw"])
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning rate")
    parser.add_argument("--weight_decay", type = float, default = 1e-4)
    parser.add_argument("--lr_scheduler", type = int, default = 20)
    parser.add_argument("--resume", default = False, action = "store_true")

    # model setting
    parser.add_argument("--parameter_path", type = str, default = None)
    parser.add_argument("--weight_path", type = str, default = None, help = "Resume parameter path.")
    parser.add_argument("--model", type = str, default = None, required = True, help = "File name of the used model.")
    parser.add_argument("--basic_model", type = str, default = None, required = True, help = "Class name of the model.")

    # dataset
    parser.add_argument("--dataloader", type = str, default = "load_data_train_val_classify")
    parser.add_argument("--dataset_name", type = str, default = None, required = True)
    parser.add_argument("--mean", type = float, default = [0.485, 0.456, 0.406])
    parser.add_argument("--std", type = float, default = [0.229, 0.224, 0.225])

    ## training set
    parser.add_argument("--train_batch_size", type = int, default = 64)
    parser.add_argument("--train_num_workers", type = int, default = 8)

    ## validation set
    parser.add_argument("--val_batch_size", type = int, default = 64)
    parser.add_argument("--val_num_workers", type = int, default = 8)

    # MCPNet setting
    parser.add_argument('--concept_per_layer', default = [8, 16, 32, 64], type = int, nargs = "+")
    parser.add_argument('--concept_cha', default = [32, 32, 32, 32], type = int, nargs = "+")
    parser.add_argument("--m_is_concept_num", default = False, action = "store_true")
    
    # Class-aware Concept Distribution (CCD) loss setting
    parser.add_argument("--margin", type = float, default = 0.01, help = "margin")
    parser.add_argument("--CCD_weight", type = float, default = 100.)

    args = vars(parser.parse_args())
    cfg.update(args)
    cfg = edict(cfg)

    if cfg.basic_model == "resnet50":
        cfg.parameter_path = "../pretrained/resnet50.pth"
    elif cfg.basic_model == "resnet18":
        cfg.parameter_path = "../pretrained/resnet18.pth"
    elif cfg.basic_model == "resnet34":
        cfg.parameter_path = "../pretrained/resnet34.pth"
    elif cfg.basic_model == "resnet50_relu":
        cfg.parameter_path = "../pretrained/resnet50.pth"
    elif cfg.basic_model == "resnet152":
        cfg.parameter_path = "../pretrained/resnet152.pth"
    elif cfg.basic_model == "convnext_base":
        cfg.parameter_path = "../pretrained/convnext_base_1k_224_ema.pth"
    elif cfg.basic_model == "convnext_small":
        cfg.parameter_path = "../pretrained/convnext_small_1k_224_ema.pth"
    elif cfg.basic_model == "convnext_tiny":
        cfg.parameter_path = "../pretrained/convnext_tiny_1k_224_ema.pth"
    elif cfg.basic_model == "inceptionv3":
        cfg.parameter_path = "../pretrained/inception_v3.pth"

    if cfg.dataset_name == "CUB_200_2011":
        cfg.category = 200
        cfg.train_dataset_path = "/eva_data_4/bor/datasets/CUB_200_2011/train"
        cfg.val_dataset_path = "/eva_data_4/bor/datasets/CUB_200_2011/val"
    elif cfg.dataset_name == "CUB_200_2011_s":
        cfg.category = 160
        cfg.train_dataset_path = "/eva_data_4/bor/datasets/CUB_200_2011/seen/train"
        cfg.val_dataset_path = "/eva_data_4/bor/datasets/CUB_200_2011/seen/val"
    elif cfg.dataset_name == "Oxford_Flowers_102":
        cfg.category = 102
        cfg.train_dataset_path = "/eva_data_4/bor/datasets/flowers102/train"
        cfg.val_dataset_path = "/eva_data_4/bor/datasets/flowers102/test"
    elif cfg.dataset_name == "AWA2":
        cfg.category = 50
        cfg.train_dataset_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/train"
        cfg.val_dataset_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/val"
    elif cfg.dataset_name == "AWA2_s":
        cfg.category = 40
        cfg.train_dataset_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/seen/train"
        cfg.val_dataset_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/seen/val"
    elif cfg.dataset_name == "Caltech101":
        cfg.category = 101
        cfg.train_dataset_path = "/eva_data_4/bor/datasets/101_ObjectCategories/train"
        cfg.val_dataset_path = "/eva_data_4/bor/datasets/101_ObjectCategories/val"
    elif cfg.dataset_name == "Caltech101_s":
        cfg.category = 81
        cfg.train_dataset_path = "/eva_data_4/bor/datasets/101_ObjectCategories/seen/train"
        cfg.val_dataset_path = "/eva_data_4/bor/datasets/101_ObjectCategories/seen/val"
    elif cfg.dataset_name == "ImageNet-1k-sampled":
        cfg.category = 1000
        cfg.train_dataset_path = "/eva_data_4/bor/datasets/ImageNet2012/train_sampled"
        cfg.val_dataset_path = "/eva_data_4/bor/datasets/ImageNet2012/val"
    elif cfg.dataset_name == "ImageNet-1k":
        cfg.category = 1000
        cfg.train_dataset_path = "/eva_data_4/bor/datasets/ImageNet2012/train"
        cfg.val_dataset_path = "/eva_data_4/bor/datasets/ImageNet2012/val"
    
    if cfg.basic_model != "inceptionv3":
        cfg.train_random_sized_crop = 224
        cfg.val_image_size = 224
    else:
        cfg.train_random_sized_crop = 299
        cfg.val_image_size = 299

    return cfg