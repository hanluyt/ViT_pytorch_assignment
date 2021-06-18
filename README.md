Hybrid Structure of ViT and ResNet50x3
===========
In this repository, we implement three models: R50x1-ViT-B_16, ResNet50x3_Params and ResNet50x3_Flops.

Table of Contents
---------
* Introdcution
* Usage
* Visualization

Introduction
-------
ViT model was proposed with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.<br>

This paper shows that Transformers applied directly to image patches and pre-trained on large datasets work really well on image recognition task.

![](https://github.com/hanluyt/ViT_pytorch_assignment/raw/main/Image/figure1.png)

Note that we achieve ViT with the hybrid version so the input sequence will be obtained from feature maps of ResNetV2 instead of raw image patches.

In addition, we implement another two models ResNet50x3_Params and ResNet50x3_Flops to explore pure CNN with the same parameters of ViT and with the same FLOPs of ViT respectively.

Usage
-----
**1. Download Pre-trained models (Google's Official Checkpoint)**
* Imagenet-21K pre-trained models
  *  R50+ViT-B_16, ResNet-50x3
 ```
 wget https://storage.googleapis.com/vit_models/imagenet21k/{model_name}.npz
 ```
**2. Installation** <br>
Make sure you have Python>=3.6 installed on your machine.
