import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import torchvision
import numpy as np
import tqdm
import os
import time
import argparse
import glob
from utils.general import get_model_set, get_dataset, load_model, load_concept, load_weight
import importlib

def KL_div(x, y):
    return torch.sum(x * (torch.log2(x) - torch.log2(y)), dim = 1)

def JS_div(x, y):
    return (KL_div(x, (x + y) / 2) + KL_div(y, (x + y) / 2)) / 2

def cal_class_MCP(model, concept_vecs, concept_means, data_transforms, data_path, args, post_name = ""):
    print("Calculate class class MCP")
    dataset = torchvision.datasets.ImageFolder(data_path, transform = data_transforms)
    Class_N = len(dataset.classes)
    N = len(dataset)
    print("Number of classes : ", len(dataset.classes))
    print("Number of images : ", len(dataset))
    class_count = torch.zeros(len(dataset.classes)).cuda()
    for class_i in range(len(dataset.classes)):
        class_count[class_i] = (np.array(dataset.targets) == class_i).sum()
    print("Number of images in each class : \n", class_count)

    dataloader = DataLoader(dataset, batch_size = 64, shuffle = False, num_workers = 16)

    # select the node
    concept_num = [concept_vecs[layer_i].shape[0] for layer_i in range(len(concept_vecs))]
    class_node_resps = [torch.zeros([Class_N] + [concept_num[0]], dtype = torch.float64).cuda(),
                        torch.zeros([Class_N] + [concept_num[1]], dtype = torch.float64).cuda(),
                        torch.zeros([Class_N] + [concept_num[2]], dtype = torch.float64).cuda(),
                        torch.zeros([Class_N] + [concept_num[3]], dtype = torch.float64).cuda()]
    
    with torch.no_grad():
        for iteration, (img, label) in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
            max_responses = []
            img = img.cuda()
            l1, l2, l3, l4 = model(img)
            feats = [l1, l2, l3, l4]
            for layer_i, feat in enumerate(feats):
                feat = feat.flatten(2)
                B, D, N = feat.shape
                feat = feat.reshape(B, args.concept_per_layer[layer_i], args.cha[layer_i], N)
                feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3)
                feat_norm = feat / (torch.norm(feat, dim = 2, keepdim = True) + 1e-16)
            
                # calculate concept vector from covariance matrix
                concept_vector = concept_vecs[layer_i].cuda()
                response = torch.sum(feat_norm * concept_vector.unsqueeze(0).unsqueeze(3), dim = 2)
                response = torch.nn.functional.adaptive_max_pool1d(response, 1)[..., 0]
                max_responses.append(torch.clip((response + 1) / 2, min = 1e-8, max = 1))
                
                class_node_resps[layer_i].index_add_(0, label.cuda(), max_responses[layer_i])
    
    for layer_i in range(len(class_node_resps)):
        class_node_resps[layer_i] = class_node_resps[layer_i] / class_count[:, None]
    class_node_resps = torch.cat(class_node_resps, dim = -1)
    class_node_resps = class_node_resps / torch.sum(class_node_resps, dim = -1, keepdim = True)
    torch.save(class_node_resps, f"./cal_class_MCP_tmp/{case_name}/{args.basic_model}/Class_MCP.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_name', default = "AWA2_Baseline", type = str)
    parser.add_argument('--device', default = "0", type = str)
    parser.add_argument('--concept_mode', type = str, required = True)
    parser.add_argument('--cha', default = [32, 32, 32, 32], type = int, nargs='+')
    parser.add_argument('--concept_per_layer', default = [8, 16, 32, 64], type = int, nargs='+')
    parser.add_argument('--basic_model', default = "resnet50", type = str)
    parser.add_argument('--model', default = "aix_model", type = str)
    parser.add_argument('--w_mode', default = "best", choices=['best', 'last'], type = str)
    parser.add_argument('--all_class', default = False, action = "store_true", help = "if true, generate the whole dataset's classes MCP distribution.")
    parser.add_argument('--few_shot', default = False, action = "store_true")

    args = parser.parse_args()
    print("Calculate the class MCP distribution !!")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    case_name = args.case_name
    args, image_size = get_model_set(args)
    print(args)
    data_path, train_path, val_path, num_class = get_dataset(case_name)

    if not args.all_class:
        if args.few_shot:
            data_path += "few_shot/"
        else:
            data_path += "unseen/"

    print(data_path)

    if args.few_shot:
        num_class = int(num_class * 0.8 + 0.5)
    
    model = load_model(args.model, args.basic_model.lower(), num_class)
    trained_param_path = f"./pkl/{args.case_name}/{args.model.lower()}_{args.basic_model}/{args.w_mode}_model.pkl"
    load_weight(model, trained_param_path)
    model.eval()

    os.makedirs(f"./cal_class_MCP_tmp/{case_name}/{args.basic_model}", exist_ok = True)

    post_name = ""
    if args.few_shot:
        post_name += "_few_shot"
    if args.w_mode == "last":
        post_name += "_last"

    # Calculate basic correlation
    concept_covs = torch.load(f"./PCA_concept_specific_tmp/{args.case_name}/{args.basic_model}/cov_topk2.pkl")
    concept_means = torch.load(f"./PCA_concept_specific_tmp/{args.case_name}/{args.basic_model}/mean_topk2.pkl")
    concept_vecs, concept_means = load_concept(concept_covs, concept_means, concept_mode = args.concept_mode)

    data_transforms = transforms.Compose([transforms.Resize(image_size + 32),
                                     transforms.CenterCrop((image_size, image_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    post_name = ""
    if args.few_shot:
        post_name += "_few_shot"
    if args.w_mode == "last":
        post_name += "_last"

    print("post_name : ", post_name)
    start = time.time()
    cal_class_MCP(model, concept_vecs, concept_means, data_transforms, data_path + train_path, args, post_name = post_name)
    print("Times : {} hrs".format((time.time() - start) / 3600))

