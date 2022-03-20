整体思路： 

推断过程：1.将gallery数据根据给出标注信息在原图进行裁剪获取ROI，
        并通过feature exacting（分类网络+特征工程）获取特征，建立检索特征库。
        2.将query图像经过目标检测获取目标位置信息， 并将对应的目标在原图上进行裁剪获取ROI，
        ROI进行相同的feature exacting，获取特征。
        3.对query特征与gallery特征进行逐一匹配，修改目标检测的query的类别预测，
        以匹配度最高的gallery类别代替。
（推断过程参考./inference.py）

训练过程：

可以独立的两阶段训练方式：目标检测+图像检索。

图像检索：将训练集的query和gallery根据标注裁剪图像到本地，并用于训练分类网络。
 
目标检测：将训练集的query和gallery训练目标检测网络

（训练过程参考./tools/train.sh）

具体细节：

目标检测（模型和训练参考./model/detection/configs）：

1.model使用cascade_rcnn，使用swin transformer(swinB)作为basebone；

2.使用随机翻转、多尺度、随机裁剪的数据增强方式；

3.使用coco数据集预训练模型。

（https://github.com/llgsdsgll/Swin-Transformer-Object-Detection-0610）

图像检索：

1.模型改进源于reid的一个baseline网络（后附资料）；

（https://github.com/michuanhaohao/reid-strong-baseline），

2.网络basebone尝试添加了多个网络，效果最好仍是swin transformer（swinb-224）；

3.使用ImageNet预训练（https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth）；

4.不使用triplet loss，使用Center loss和CE Loss，Center Loss可以改用AMSoftmaxLoss；

5.对gallery进行数据增强，扩充特征库；

6.学习率阶梯调整，平衡采样（欠采样）训练，labelsmooth，半精度训练加速等；

7.检索借用pyretri工具包，进行特征工程（如特征聚合、正则化、pca等），使用L2作为度量距离。

----------------------------------------------------
reid baseline

论文：http://arxiv.org/abs/1906.08332v2

代码：https://github.com/michuanhaohao/reid-strong-baseline
![img.png](https://pic1.zhimg.com/v2-3f7196351d481a3459fa595132d7da7b_1440w.jpg?source=172ae18b)

-----------------------------------------------------
swin transformer论文及代码地址：

论文：https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.14030

代码：

    分类：https://github.com/microsoft/Swin-Transformer
    
    目标检测：https://github.com/llgsdsgll/Swin-Transformer-Object-Detection-0610
![img.png](https://pic2.zhimg.com/80/v2-a41780d72fc13ef36559acb256c85b91_720w.jpg)

--------------------------------------------------------

pyretri：

论文：https://arxiv.org/abs/2005.02154

代码：https://github.com/PyRetri/PyRetri

![img.png](https://pic1.zhimg.com/80/v2-5db77e48baf545e6020c64276c3bc2f0_720w.png)

