import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import tqdm
import os
from utils.general import get_dataset, get_model_set, load_weight, load_model

import argparse

# Note:
# Calculate the weighted bias based on the weighted sample covariance formulation.
# https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance

def cal_cov_component_weighted(feat, Sum_A, Square_Sum_A, cov_xx, cov_mean, layer_i, args):
    feat = torch.flatten(feat, 2)
    strength = torch.norm(feat, p = 2, dim = 1, keepdim = True)
    ori_feat = feat
    feat = feat * strength
    Sum_A[layer_i] += torch.sum(strength.squeeze(1), dim = 1)
    Square_Sum_A[layer_i] += torch.sum(strength.squeeze(1) ** 2, dim = 1)
    cov_xx[layer_i] += torch.bmm(feat, ori_feat.permute(0, 2, 1))
    cov_mean[layer_i] += torch.sum(feat, dim = -1, keepdim = True)
    return Sum_A, Square_Sum_A, cov_xx, cov_mean

def cal_cov_component(feat, cov_xx, cov_mean, N, layer_i):
    feat = torch.flatten(feat, 2)

    N[layer_i] += feat.shape[-1]
    cov_xx[layer_i] += torch.bmm(feat, feat.permute(0, 2, 1))
    cov_mean[layer_i] += torch.sum(feat, dim = -1, keepdim = True)
    return cov_xx, cov_mean, N

def cal_cov(cov_xx, cov_mean, Sum_A = None, Square_Sum_A = None, N = None):
    cov = []
    for i in range(4):
        if args.weighted:
            cov_xx[i] /= Sum_A[i][:, None, None]
            cov_mean[i] /= Sum_A[i][:, None, None]
            cov.append(cov_xx[i] - torch.bmm(cov_mean[i], cov_mean[i].permute(0, 2, 1))) # * (1 / (1. - (Square_Sum_A[i] / (Sum_A[i] ** 2 + 1e-16))))[:, None, None]
        else:
            cov_xx[i] /= N[i]
            cov_mean[i] /= N[i]
            cov.append(cov_xx[i] - torch.bmm(cov_mean[i], cov_mean[i].permute(0, 2, 1)))
        cov_mean[i] = cov_mean[i][..., 0]
    return cov, cov_mean

def cal_cov_matrix(model, train_loader, args, post_name = "", saved = True):
    os.makedirs(f"./PCA_concept_specific_tmp/{args.case_name}/{args.basic_model}/", exist_ok = True)
    cov_xx = [torch.zeros(args.concept_per_layer[0], args.cha[0], args.cha[0], dtype = torch.float64).cuda(), 
              torch.zeros(args.concept_per_layer[1], args.cha[1], args.cha[1], dtype = torch.float64).cuda(), 
              torch.zeros(args.concept_per_layer[2], args.cha[2], args.cha[2], dtype = torch.float64).cuda(), 
              torch.zeros(args.concept_per_layer[3], args.cha[3], args.cha[3], dtype = torch.float64).cuda()]
    cov_mean = [torch.zeros(args.concept_per_layer[0], args.cha[0], 1, dtype = torch.float64).cuda(), 
                torch.zeros(args.concept_per_layer[1], args.cha[1], 1, dtype = torch.float64).cuda(), 
                torch.zeros(args.concept_per_layer[2], args.cha[2], 1, dtype = torch.float64).cuda(), 
                torch.zeros(args.concept_per_layer[3], args.cha[3], 1, dtype = torch.float64).cuda()]
    
    if args.weighted:
        Sum_A = [torch.zeros(args.concept_per_layer[0], dtype = torch.float64).cuda(), 
                 torch.zeros(args.concept_per_layer[1], dtype = torch.float64).cuda(), 
                 torch.zeros(args.concept_per_layer[2], dtype = torch.float64).cuda(), 
                 torch.zeros(args.concept_per_layer[3], dtype = torch.float64).cuda()]
        Square_Sum_A = [torch.zeros(args.concept_per_layer[0], dtype = torch.float64).cuda(), 
                        torch.zeros(args.concept_per_layer[1], dtype = torch.float64).cuda(),
                        torch.zeros(args.concept_per_layer[2], dtype = torch.float64).cuda(), 
                        torch.zeros(args.concept_per_layer[3], dtype = torch.float64).cuda()]
        N = None
    else:
        Sum_A = None
        Square_Sum_A = None
        N = [0, 0, 0, 0]
    
    with torch.no_grad():
        for iteration, (img, label) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
            img = img.cuda()
            l1, l2, l3, l4 = model(img)
            features = [l1, l2, l3, l4]
            for layer_i, feat in enumerate(features):
                concept_num = args.concept_per_layer[layer_i]
                cha_per_con = args.cha[layer_i]
                B, C, H, W = feat.shape
                feat = feat.reshape(B, concept_num, cha_per_con, H, W).permute(1, 2, 0, 3, 4)
                    
                if args.weighted:
                    Sum_A, Square_Sum_A, cov_xx, cov_mean = cal_cov_component_weighted(feat, Sum_A, Square_Sum_A, cov_xx, cov_mean, layer_i, args)
                else:
                    cov_xx, cov_mean, N = cal_cov_component(feat, cov_xx, cov_mean, N, layer_i)
                
        cov, cov_mean = cal_cov(cov_xx, cov_mean, Sum_A, Square_Sum_A, N)   
        if saved:
            torch.save(cov, f"./PCA_concept_specific_tmp/{args.case_name}/{args.basic_model}/cov_topk2{post_name}.pkl")
            torch.save(cov_mean, f"./PCA_concept_specific_tmp/{args.case_name}/{args.basic_model}/mean_topk2{post_name}.pkl")

    return cov, cov_mean

def load_dataset(data_path: str, image_size: int, args, data_transform = None) -> DataLoader:
    transform = transforms.Compose([transforms.Resize(image_size + 32),
                                     transforms.CenterCrop((image_size, image_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) if data_transform is None else data_transform
    train_dataset = torchvision.datasets.ImageFolder(data_path, transform)
        
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = False, num_workers = 8)
    return train_loader

if __name__ == "__main__":
    print("Extract the concept vector via PCA!! (Only calculate the weighted covariance matrixs)")
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_name', default = "AWA2_Baseline", type = str)
    parser.add_argument('--few_shot', default = False, action = "store_true")
    parser.add_argument('--device', default = "0", type = str)
    parser.add_argument('--cha', default = [32, 32, 32, 32], type = int, nargs='+')
    parser.add_argument('--concept_per_layer', default = [8, 16, 32, 64], type = int, nargs='+')
    parser.add_argument('--weighted', default = True, action = "store_false")
    parser.add_argument('--basic_model', default = "resnet50", type = str)
    parser.add_argument('--model', default = "aix_model", type = str)
    parser.add_argument('--w_mode', default = "best", choices=['best', 'last'], type = str)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    os.makedirs(f"./PCA_concept_specific_tmp/{args.case_name}/{args.basic_model}/", exist_ok = True)
    data_path, train_path, val_path, num_class = get_dataset(args.case_name)
    
    if args.few_shot:
        data_path += "seen/"
    
    if "AWA2" in args.case_name and args.few_shot:
        num_class = 40
    elif "Caltech101" in args.case_name and args.few_shot:
        num_class = 81
    elif "StanfordCar" in args.case_name and args.few_shot:
        num_class = 157
    elif "CUB" in args.case_name and args.few_shot:
        num_class = 160
    data_path = data_path + train_path
    args, image_size = get_model_set(args)
    model = load_model(args.model, args.basic_model, num_class)
    print(args)

    train_loader = load_dataset(data_path, image_size, args)
    trained_param_path = f"./pkl/{args.case_name}/{args.model.lower()}_{args.basic_model}/best_model.pkl"
    load_weight(model, trained_param_path)
        
    post_name = ""
    if args.few_shot:
        post_name += "_few_shot"
    if args.w_mode == "last":
        post_name += "_last"

    print(args.case_name)
    model.eval()

    cal_cov_matrix(model, train_loader, args, post_name)