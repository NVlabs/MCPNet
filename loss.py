import torch
import torch.nn as nn

class CKA_loss(nn.Module):
    def __init__(self, concept_cha):
        super(CKA_loss, self).__init__()
        self.concept_cha = concept_cha  
    
    def __repr__(self):
        basic = super().__repr__()
        str_show = f"{basic[:-1]}concept_cha={self.concept_cha})"
        return str_show

    def unbiased_HSIC(self, x, y):
        #create the unit **vector** filled with ones
        n = x.shape[1]
        ones = torch.ones(x.shape[0], n, 1).cuda()

        #fill the diagonal entries with zeros 
        # mask = torch.eye(n).repeat(x.shape[0], 1, 1).bool().cuda()
        mask = torch.eye(n).unsqueeze(0).cuda()
        x = x * (1 - mask)
        y = y * (1 - mask)

        #first part in the square brackets
        trace = torch.sum(torch.matmul(x, y.permute(0, 2, 1)) * mask, dim = (-1, -2), keepdim = True)

        #middle part in the square brackets
        nominator1 = torch.sum(x, dim = (-2, -1), keepdim = True)
        nominator2 = torch.sum(y, dim = (-2, -1), keepdim = True)
        denominator = (n - 1) * (n - 2)
        middle = torch.matmul(nominator1, nominator2) / denominator
        
        #third part in the square brackets
        multiplier1 = 2 / (n - 2)
        multiplier2 = torch.matmul(torch.matmul(ones.permute(0, 2, 1), x), torch.matmul(y, ones))
        last = multiplier1 * multiplier2

        #complete equation
        unbiased_hsic = 1 / (n * (n - 3)) * (trace + middle - last)
        return unbiased_hsic
    
    def CKA(self, kernel):
        index = torch.triu_indices(kernel.shape[0], kernel.shape[0], 1)
        nominator = self.unbiased_HSIC(kernel[index[0]], kernel[index[1]])
        denominator1 = self.unbiased_HSIC(kernel[index[0]], kernel[index[0]])
        denominator2 = self.unbiased_HSIC(kernel[index[1]], kernel[index[1]])
        denominator1 = torch.nn.functional.relu(denominator1)
        denominator2 = torch.nn.functional.relu(denominator2)
        denominator = denominator1 * denominator2
        # prevent divide 0
        # mask = (denominator != 0)
        cka = (nominator) / torch.sqrt(torch.clamp(denominator, min = 1e-16))
        return cka
    
    def forward(self, concept_pools, layer):
        # calculate the concept number and channel number of each concept
        
        CKA_loss = 0
        for layer_i, concept_blocks in enumerate(concept_pools):
            concept_num = concept_blocks.shape[1] // self.concept_cha[layer_i]
            cha_per_con = self.concept_cha[layer_i]
            B, C, H, W = concept_blocks.shape
            concept_blocks = torch.flatten(concept_blocks.reshape(B, concept_num, cha_per_con, H, W).permute(1, 0, 2, 3, 4), 2)
            concept_blocks_kernel = torch.matmul(concept_blocks, concept_blocks.permute(0, 2, 1))
            CKA_loss = CKA_loss + torch.mean(torch.abs(self.CKA(concept_blocks_kernel)))
        return CKA_loss

# the loss normalize the response of each concept from [-1, 1] to [0, 1] by (x + 1) / 2
class CCD_loss(nn.Module):
    def __init__(self, concept_cha, margin = 0.01, eps = 1e-8):
        super(CCD_loss, self).__init__()
        self.margin = margin
        self.relu = torch.nn.ReLU()
        self.concept_cha = concept_cha
        self.eps = eps

    def __repr__(self):
        basic = super().__repr__()
        str_show = f"{basic[:-1]}concept_cha={self.concept_cha}, margin={self.margin})"
        return basic

    def KL_div(self, x, y):
        return torch.sum(x * (torch.log2(x) - torch.log2(y)), dim = -1)

    def JS_div(self, img_node, cd_node):
        img_node = img_node.unsqueeze(1)
        cd_node = cd_node.unsqueeze(0)
        M = (img_node + cd_node) / 2
        return (self.KL_div(img_node, M) + self.KL_div(cd_node, M)) / 2

    def forward(self, concept_pool, concept_vector, concept_mean, layer, label, class_MCP_dist):
        max_responses = []
        for layer_i, feat in enumerate(concept_pool):
            concept_num = feat.shape[1] // self.concept_cha[layer_i]
            cha_per_con = self.concept_cha[layer_i]
            B, C, H, W = feat.shape
            feat = feat.reshape(B, concept_num, cha_per_con, H, W)
            feat = feat - concept_mean[layer[layer_i] - 1].unsqueeze(0).unsqueeze(3).unsqueeze(4)
            feat_norm = feat / (torch.norm(feat, dim = 2, keepdim = True) + self.eps)
        
            # calculate concept vector from covariance matrix
            response = torch.sum(feat_norm * concept_vector[layer[layer_i] - 1].unsqueeze(0).unsqueeze(3).unsqueeze(4), dim = 2)
            max_response, max_index = torch.nn.functional.adaptive_max_pool2d(response, output_size = 1, return_indices = True)
            max_responses.append(torch.clip((max_response[..., 0, 0] + 1) / 2, min = self.eps, max = 1))

        img_MCP_dist = torch.cat(max_responses, dim = 1)
        img_MCP_dist = img_MCP_dist / torch.sum(img_MCP_dist, dim = -1, keepdim = True)
        MCP_dist = self.JS_div(img_MCP_dist, class_MCP_dist)
        same_class = torch.mean(torch.gather(MCP_dist, dim = 1, index = label[:, None]))
        mask = torch.ones_like(MCP_dist)
        mask[(torch.arange(label.shape[0]), label)] = 0
        diff_dist = self.relu((self.margin - MCP_dist) * mask)
        denominator = torch.sum(diff_dist != 0, dim = 1)
        # prevent divided by zero
        denominator[denominator == 0] = 1
        diff_class = torch.mean(torch.sum(diff_dist, dim = 1) / denominator)
        total_loss = (same_class + diff_class)
        return total_loss