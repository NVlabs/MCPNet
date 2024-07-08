import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import torchvision
import tqdm
import os
import time
import argparse
sys.path.insert(1, f"{os.path.expanduser('~')}/MCPNet")
from utils.general import get_model_set, get_dataset, load_weight, load_model, load_concept

def KL_div(x, y):
    return torch.mean(x * (torch.log2(x) - torch.log2(y)), dim = 1)

def JS_div(x, y):
    return (KL_div(x, (x + y) / 2) + KL_div(y, (x + y) / 2)) / 2

def cal_JS_sim(img_MCP, class_MCP):
    return JS_div(img_MCP + 1e-8, class_MCP.unsqueeze(0) + 1e-8)

def cal_sim(img_MCP, class_MCP):
    img_MCP = torch.flatten(img_MCP, 1)
    img_MCP = torch.clamp(img_MCP, min = 1e-8, max = 1 - 1e-8)
    class_MCP = torch.clamp(class_MCP, min = 1e-8, max = 1 - 1e-8)
    feat_sim = cal_JS_sim(img_MCP, class_MCP)
    return feat_sim

def cal_acc(model, concept_vecs, concept_means, data_transforms, data_path, args, cent_MCP):
    print("Calculate class MCP distribution")
    deleted_nodes = [args.l1, args.l2, args.l3, args.l4]

    selected_nodes = []
    concept_acc_count = 0
    for layer_i in range(len(concept_vecs)):
        selected_node = torch.arange(concept_vecs[layer_i].shape[0]).cuda()
        mask = torch.ones(concept_vecs[layer_i].shape[0])
        mask[deleted_nodes[layer_i]] = 0
        selected_nodes.append(selected_node[mask.bool()] + concept_acc_count)
        concept_acc_count += concept_vecs[layer_i].shape[0]
    print("Selected nodes : ", selected_nodes)
    selected_nodes = torch.cat(selected_nodes, dim = -1)
    dataset = torchvision.datasets.ImageFolder(data_path, transform = data_transforms)
    print("Number of classes : ", len(dataset.classes))
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size = 64, shuffle = False, num_workers = 16)
    total_count = 0
    total_correct = 0
    total_correct5 = 0
    fc_correct = 0
    with torch.no_grad():
        for iteration, (img, label) in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
            total_count += img.shape[0]
            img = img.cuda()
            l1, l2, l3, l4 = model(img)
            
            top1, top5 = cal_top1_topk([l1, l2, l3, l4], cent_MCP, concept_vecs, concept_means, selected_nodes, args)
            correct_resp = (top1 == label.cuda().unsqueeze(1)).sum()
            correct_resp5 = (top5 == label.cuda().unsqueeze(1)).sum()

            total_correct += correct_resp
            total_correct5 += correct_resp5
    
    fc_correct = fc_correct / total_count
    acc_top1 = total_correct / total_count
    acc_top5 = total_correct5 / total_count
    print(f"Accuracy top1: {acc_top1 * 100:.4f}% ({acc_top1}) top5: {acc_top5 * 100:.4f}% ({acc_top5})")

def cal_top1_topk(feats, cent_MCP, concept_vecs, concept_means, selected_node, args):
    max_responses = []
    B = feats[0].shape[0]
    for layer_i, feat in enumerate(feats):
        concept_num = args.concept_per_layer[layer_i]
        cha_per_con = args.cha[layer_i]
        B, C, H, W = feat.shape
        feat = feat.reshape(B, concept_num, cha_per_con, H, W)
        feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3).unsqueeze(4)
        feat_norm = feat / (torch.norm(feat, dim = 2, keepdim = True) + 1e-16)
        
        # calculate concept vector from covariance matrix
        concept_vector = concept_vecs[layer_i].cuda()
        response = torch.sum(feat_norm * concept_vector.unsqueeze(0).unsqueeze(3).unsqueeze(4), dim = 2)
        
        max_response, max_index = torch.nn.functional.adaptive_max_pool2d(response, output_size = 1, return_indices = True)
        max_responses.append((max_response[..., 0, 0] + 1) / 2)

    Diff_centroid_dist_resp = []
    max_responses = torch.cat(max_responses, dim = -1)
    max_responses = max_responses / torch.sum(max_responses, dim = -1, keepdim = True)
    for class_i in range(len(cent_MCP)):
        img_MCP_dist = max_responses[:, selected_node]
        img_MCP_dist = img_MCP_dist / torch.sum(img_MCP_dist, dim = 1, keepdim = True)

        cent_MCP_dist = cent_MCP[class_i][selected_node]
        cent_MCP_dist = cent_MCP_dist / torch.sum(cent_MCP_dist)

        resp_sims = cal_sim(img_MCP_dist, cent_MCP_dist)
        Diff_centroid_dist_resp.append(resp_sims)
    Diff_centroid_dist_resp = torch.stack(Diff_centroid_dist_resp, dim = 1)
    return torch.topk(-Diff_centroid_dist_resp, dim = 1, k = 1)[1], \
           torch.topk(-Diff_centroid_dist_resp, dim = 1, k = 5)[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_name', default = "AWA2_Baseline", type = str)
    parser.add_argument('--device', default = "0", type = str)
    parser.add_argument('--cha', default = [32, 32, 32, 32], type = int, nargs='+')
    parser.add_argument('--concept_per_layer', default = [8, 16, 32, 64], type = int, nargs='+')
    parser.add_argument('--basic_model', default = "resnet50_relu", type = str)
    parser.add_argument('--model', default = "ResNet", type = str)
    parser.add_argument('--w_mode', default = "best", choices=['best', 'last'], type = str)
    parser.add_argument('--all_class', default = False, action = "store_true")
    parser.add_argument('--few_shot', default = False, action = "store_true")
    parser.add_argument('--l1', default = [], type = int, nargs = "+", help = "Select the node to drop")
    parser.add_argument('--l2', default = [], type = int, nargs = "+", help = "Select the node to drop")
    parser.add_argument('--l3', default = [], type = int, nargs = "+", help = "Select the node to drop")
    parser.add_argument('--l4', default = [], type = int, nargs = "+", help = "Select the node to drop")
    args = parser.parse_args()

    print("Calculate accuracy !!")
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
    load_weight(model, f"./pkl/{case_name}/{args.model.lower()}_{args.basic_model}/best_model.pkl")
    model.eval()

    post_name = ""
    if args.few_shot:
        post_name += "_few_shot"
    if args.w_mode == "last":
        post_name += "_last"

    # Get concepts from weighted Covariance matrix
    concept_covs = torch.load(f"./PCA_concept_specific_tmp/{args.case_name}/{args.basic_model}/cov_topk2.pkl")
    concept_means = torch.load(f"./PCA_concept_specific_tmp/{args.case_name}/{args.basic_model}/mean_topk2.pkl")
    concept_vecs, concept_means = load_concept(concept_covs, concept_means, concept_mode = "pca")

    data_transforms = transforms.Compose([transforms.Resize(image_size + 32),
                                     transforms.CenterCrop((image_size, image_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    print("post_name : ", post_name)
    start = time.time()
    cent_MCP = torch.load(f"./cal_class_MCP_tmp/{args.case_name}/{args.basic_model}/Class_MCP.pkl")
    cal_acc(model, concept_vecs, concept_means, data_transforms, data_path + val_path, args, cent_MCP)
    print("Times : {} hrs".format((time.time() - start) / 3600))

