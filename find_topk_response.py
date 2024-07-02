import torch
import torchvision.transforms as transforms
import sys
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import tqdm
import os
import argparse
import time
from utils.general import load_model, load_concept, get_dataset, load_weight, get_con_num_cha_per_con_num
from torchvision.datasets import ImageFolder
from PIL import Image

class customDataset(ImageFolder):
    def __init__(self, root, transform):
        super(customDataset, self).__init__(root, transform)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        img_name = image_path.split("\\")[-1]
        ori_img = Image.open(image_path).convert('RGB')
        W, H = ori_img.size

        img = self.transform(ori_img)
        for trans in self.transform.transforms[:-1]:
            ori_img = trans(ori_img)
        
        return img, label, ori_img, image_path

if __name__ == "__main__":
    print("Find top-k concept response from the dataset!!!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--case_name', default = ["AWA2_Baseline"], type = str, nargs = "+")
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--basic_model", type = str, required = True)
    parser.add_argument('--device', default = "0", type = str)
    parser.add_argument('--cha', default = [32, 32, 32, 32], type = int, nargs = "+")
    parser.add_argument('--concept_per_layer', default = [32, 32, 32, 32], type = int, nargs = "+")
    parser.add_argument('--eigen_topk', default = 1, type = int)
    parser.add_argument('--use_CLS_token', action = "store_true", default = False)
    args = parser.parse_args()
    
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    case_names = args.case_name

    m_is_concept_num = False
    # dataset_path = "../datasets/stanford car/cars_train"
    
    if args.basic_model == "inceptionv3":
        image_size = 299
    else:
        image_size = 224

    data_transforms = transforms.Compose([transforms.Resize(image_size + 32),
                                     transforms.CenterCrop((image_size, image_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    TOP_RATE = 0.1
    for step, case_name in enumerate(case_names):
        data_path, train_path, val_path, num_class = get_dataset(case_name)
        data_path = data_path + train_path

        # train_dataset = torchvision.datasets.ImageFolder(data_path, data_transforms)
        train_dataset = customDataset(data_path, data_transforms)
        print(len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = False, num_workers = 8)

        concept_covs = torch.load(f"./PCA_concept_specific_tmp/{case_name}/{args.basic_model}/cov_topk2.pkl")
        concept_means = torch.load(f"./PCA_concept_specific_tmp/{case_name}/{args.basic_model}/mean_topk2.pkl")
        concept_vecs, concept_means = load_concept(concept_covs, concept_means, args.eigen_topk)

        model = load_model(args.model, args.basic_model, num_class).cuda()
        trained_param_path = f"./pkl/{case_name}/{args.model.lower()}_{args.basic_model}/best_model.pkl"
        load_weight(model, trained_param_path)
        
        max_resp_path = [np.array([]),
                          np.array([]),
                          np.array([]),
                          np.array([])]
            
        max_resp_value = [torch.tensor([]).cuda(),
                          torch.tensor([]).cuda(),
                          torch.tensor([]).cuda(),
                          torch.tensor([]).cuda()]
        
        max_resp_feat = [torch.tensor([]).cuda(),
                          torch.tensor([]).cuda(),
                          torch.tensor([]).cuda(),
                          torch.tensor([]).cuda()]
        
        max_resp_index = [torch.tensor([], dtype = torch.int64).cuda(),
                          torch.tensor([], dtype = torch.int64).cuda(),
                          torch.tensor([], dtype = torch.int64).cuda(),
                          torch.tensor([], dtype = torch.int64).cuda()]

        os.makedirs(f"./{__file__[:-3]}_tmp/{case_name}/{args.basic_model}", exist_ok = True)
        N = [0, 0, 0, 0]
        with torch.no_grad():
            model.eval()
            tmp = []
            for iteration, (img, label, _, path) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
                path = np.array(path)
                img = img.cuda()
                if args.model.lower() == "aix_model":
                    output, l1, l2, l3, l4 = model(img)
                else:
                    l1, l2, l3, l4 = model(img)

                features = [l1, l2, l3, l4]
                for layer_i, feat in enumerate(features):
                    concept_num = args.concept_per_layer[layer_i]
                    cha_per_con = args.cha[layer_i]
                    B, C, H, W = feat.shape
                    feat = feat.reshape(B, concept_num, cha_per_con, H, W)
                    feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3).unsqueeze(4)
                    feat_idx = torch.arange(B * H * W).reshape(1, -1).cuda() + N[layer_i]

                    feat = torch.flatten(feat.permute(1, 2, 0, 3, 4), 2)
                    feat = feat / (torch.norm(feat, dim = 1, p = 2, keepdim = True) + 1e-16)
                    con_response = torch.sum(feat * concept_vecs[layer_i].unsqueeze(-1), dim = 1)
                    D_per_img = H * W
                    max_resp_value[layer_i] = torch.cat([max_resp_value[layer_i], con_response], dim = 1)
                    max_resp_index[layer_i] = torch.cat([max_resp_index[layer_i], feat_idx.repeat(max_resp_value[layer_i].shape[0], 1)], dim = 1)
                    max_resp_feat[layer_i] = torch.cat([max_resp_feat[layer_i], feat], dim = -1)
                    max_resp_path[layer_i] = np.hstack([max_resp_path[layer_i], path[None, :].repeat([args.concept_per_layer[layer_i]], axis = 0)]) if max_resp_path[layer_i].size else path[None, :].repeat([args.concept_per_layer[layer_i]], axis = 0)
                    topkv, topki = torch.topk(max_resp_value[layer_i], k = min(int(len(train_dataset) * TOP_RATE), max_resp_value[layer_i].shape[1]), dim = 1)
                    max_resp_value[layer_i] = topkv
                    max_resp_index[layer_i] = torch.gather(max_resp_index[layer_i], dim = 1, index = topki)
                    max_resp_feat[layer_i] = torch.gather(max_resp_feat[layer_i], dim = -1, index = topki.unsqueeze(1).repeat(1, args.cha[layer_i], 1))
                    max_resp_path[layer_i] = np.take(max_resp_path[layer_i], np.array(topki.cpu()) // (D_per_img))
                    N[layer_i] += feat.shape[2]
        torch.save(max_resp_value, f"./{__file__[:-3]}_tmp/{case_name}/{args.basic_model}/max_resp_value.pkl")
        torch.save(max_resp_index, f"./{__file__[:-3]}_tmp/{case_name}/{args.basic_model}/max_resp_value_idx.pkl")
        torch.save(max_resp_feat, f"./{__file__[:-3]}_tmp/{case_name}/{args.basic_model}/max_resp_value_feat.pkl")
        # np.save(f"./{__file__[:-3]}_tmp/{case_name}/{args.basic_model}/max_resp_path.npy", max_resp_path)
