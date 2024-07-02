# MCPNet: An Interpretable Classifier via Multi-Level Concept Prototypes
[Bor-Shiun Wang](https://eddie221.github.io/),
[Chien-Yi Wang](https://chienyiwang.github.io/)\*,
[Wei-Chen Chiu](https://walonchiu.github.io/)\*

<sup>*Equal Advising</sup>

Official PyTorch implementation of CVPR 2024 paper "[MCPNet: An Interpretable Classifier via Multi-Level Concept Prototypes](https://arxiv.org/abs/2404.08968)".  
<p style="font-size: 0.6em; margin-top: -1em">*Equal Contribution</p>
<p align="center">
<a href="https://arxiv.org/abs/2404.08968"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://eddie221.github.io/MCPNet/"><img src="https://img.shields.io/badge/Project-Website-red"></a>
</p>
<!-- You can visit our project website [here](https://eddie221.github.io/MCPNet/). -->

## Introduction
Recent advancements in post-hoc and inherently interpretable methods have markedly enhanced the explanations of black box classifier models. These methods operate either through post-analysis or by integrating concept learning during model training. Although being effective in bridging the semantic gap between a model's latent space and human interpretation, these explanation methods only partially reveal the model's decision-making process. The outcome is typically limited to high-level semantics derived from the last feature map. We argue that the explanations lacking insights into the decision processes at low and mid-level features are neither fully faithful nor useful. Addressing this gap, we introduce the Multi-Level Concept Prototypes Classifier (MCPNet), an inherently interpretable model. MCPNet autonomously learns meaningful concept prototypes across multiple feature map levels using Centered Kernel Alignment (CKA) loss and an energy-based weighted PCA mechanism, and it does so without reliance on predefined concept labels. Further, we propose a novel classifier paradigm that learns and aligns multi-level concept prototype distributions for classification purposes via Class-aware Concept Distribution (CCD) loss. Our experiments reveal that our proposed MCPNet while being adaptable to various model architectures, offers comprehensive multi-level explanations while maintaining classification accuracy. Additionally, its concept distribution-based classification approach shows improved generalization capabilities in few-shot classification scenarios. 

<div align="center">
  <img src="https://eddie221.github.io/MCPNet/static/images/paper/teaser.png"/>
</div>

## Usage
### Enviroment  

#### Required Python Packages:  
* [Pytorch](https://pytorch.org/get-started/locally/)(includa torchvision, test with Pytorch 1.13.1) 
* Matplotlib  
* OpenCV
* NumPy  
* tqdm  
* argparse  
* easydict  
* importlib  

You can also simply build the evironment via ``.yml`` file.
```
conda env create -f ./environment.yml
```

### Dataset
The code can be applied to any imaging classification data set, structured according to the [Imagefolder format](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder): 

>root/class1/xxx.png  <br /> root/class1/xxy.png  <br /> root/class2/xyy.png <br /> root/class2/yyy.png

Add or update the paths to your dataset in ``arg_reader.py``. 


### Training
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 9560 train.py --index AWA2_test --model ResNet --basic_model resnet50_relu --device 1 --dataset_name AWA2 --margin 0.01 --concept_cha 32 32 32 32 --concept_per_layer 8 16 32 64 --optimizer adam
```

### Visualize concept
To visualize the top-k response images of each concept, we first have to calculate the weighted covariance matrix and weighted means, which are used to calculate the concept prototypes via weighted PCA.
```bash
python extract_prototypes.py --case_name AWA2_test --device 0 --model ResNet --basic_model resnet50_relu --concept_per_layer 8 16 32 64 --cha 32 32 32 32
```
The above code will calculate the weighted convariance matrix and the weighted mean for each corncept pototypes.

Next, the following code will scan the whole training set to find the highest k point for each prototype from the training set. There might be multi-selected points from the same image. Select multiple candidates for each prototype to prevent the visualization from showing the same image. **The code will store the image index list indexed by ``ImageFolder``, so make sure the image set will be the same in this and next step; also, set the ``shuffle=False`` for these two steps.
```bash
python find_topk_response.py --case_name AWA2_test --model ResNet --basic_model resnet50_relu --concept_per_layer 8 16 32 64 --cha 32 32 32 32 --device 0 --eigen_topk 1
```

The final step stores the top-k concept prototype result. Each image will only be selected once to present for each prototype. Precisely, the image of the top responses won't be duplicated for each prototype. 
```bash
python find_topk_area.py --case_name AWA2_test --model ResNet --basic_model resnet50_relu --concept_per_layer 8 16 32 64 --cha 32 32 32 32 --topk 5 --device 0 --eigen_topk 1 --masked --heatmap --individually
```

### Evaluate performance
To evaluate the performance of the trained MCPNet.

(Optional) Firstly, extract the concept prototypes (If the concept prototypes have already been extracted, this step can be passed).
```bash
python extract_prototypes.py --case_name AWA2_test --device 0 --model ResNet --basic_model resnet50_relu --concept_per_layer 8 16 32 64 --cha 32 32 32 32
```

Next, calculated the class Multi-level Concept Prototypes distribution (MCP distribution).
```bash
python cal_class_MCP.py --case_name AWA2_test --device 0 --model ResNet --basic_model resnet50_relu --concept_mode pca --concept_per_layer 8 16 32 64 --cha 32 32 32 32 --all_class
```

Final, classify the image via matching images' MCP distribution to the cloest class MCP distribution.
```bash
python cal_acc_MCP.py --case_name AWA2_test --model ResNet --basic_model resnet50_relu --device 0 --concept_per_layer 8 16 32 64 --cha 32 32 32 32 --all_class
```

#### Trained checkpoint
MCPNet with ResNet50 
* on AWA2 dataset is available [here](https://drive.google.com/drive/folders/1HSaaQkzueYSkCCSkz6RO3abW0nTXxKTY?usp=drive_link)(270MB).
* on Clatech101 dataset is available [here](https://drive.google.com/drive/folders/1HSaaQkzueYSkCCSkz6RO3abW0nTXxKTY?usp=drive_link)(270MB).
* on CUB_200_2011 dataset is available [here](https://drive.google.com/drive/folders/1HSaaQkzueYSkCCSkz6RO3abW0nTXxKTY?usp=drive_link)(270MB).
* To visulize the extracted prototypes, put the checkpoint in ``./pkl`` and follow the step described in ``Visualize concept``.