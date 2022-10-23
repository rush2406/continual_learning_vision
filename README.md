# Class Incremental Continual Learning for Computer Vision

> *<div style="text-align: justify"> **Abstract:** Humans have the ability to learn continually throughout their lives. Continual learning,
a subset of machine learning, attempts to replicate humans’ lifelong learning ability in
neural networks. This work investigates the class incremental form of continual learning
for various vision-based applications such as image classification and object detection. Each
learning phase in a class incremental learning (CIL) scenario introduces groups of classes
to a model, where the goal is to learn a unified model that is performant across all classes
seen thus far. Performance of different architectures such as Convolutional Neural Networks
(CNNs) and Vision Transformers (ViT) is studied when used in a continual learning setting.
A hybrid ViT is suitably adapted for continual learning by using an interpretability based
distillation mechanism which maintains the configuration of spatial attention maps as
learning progresses. Eventually this aids in reducing catastrophic forgetting by forcing
the model to focus on the most discriminative regions in an image. When combined with
other methods such as logit adjustments to combat bias, dubbed as D3Former, it shows
considerable improvements across many datasets. Furthermore, the inherent ability of ViTs
to process images as patches is explored in order to reduce exemplar memory by storing
relevant patches instead of images. Besides, continual learning with CNNs is studied from
the perspective of utilizing acquired knowledge in order to provide a better initialization
for learning new knowledge. Apart from image classification, continual learning for object
detection is explored in an open-world setting where unknown classes are present in the
data. The continual learning behaviour is evaluated by improving the unknown identifiable
property using post-processing and proposal sampling strategies. </div>*

#### Directories
D3Former: Adapts a hybrid ViT for Class Incremental Continual Learning

Cluster_Pretrain: Exploiting acquired knowledge for Class Incremental Continual Learning

OWOD: Continual Learning for Object Detection

#### Instructions
Please refer the README in the respective directories
