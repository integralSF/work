# PF-Net-Point-Fractal-Network

This repository is still under constructions.

If you have any questions about the code, please email me. Thanks!

This is the Pytorch implement of CVPR2020 PF-Net: Point Fractal Network for 3D Point Cloud Completion. 

https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.pdf

## 0) Environment
Pytorch 1.13.0+cu117
Python 3.8.13

## 1) Dataset
在dataset文件夹下， 目录结构如下
```
|-- own_dataset_processed (初代数据集的处理后存放，后续训练不用，作为备份保存)
|   |-- test
|   |-- train
|   `-- val
|-- own_dataset_pure (初代数据集的原始存放，后续训练不用，作为备份保存)
|   |-- 1-incisor
|   |-- 2-canine
|   |-- 3-premolares
|   `-- 4-molars
|-- temp (尖牙和磨牙分开训练时暂存的地方)
|   |-- train
|   |   |-- gt
|   |   `-- partial
|   `-- val
|       |-- gt
|       `-- partial
|-- test_dataset(原始数据集存放的地方，制作好数据集后就放在这个地方，训练也用的这个)
|   |-- 1-incisor
|   |   |-- gt
|   |   `-- partial
|   |-- 2-canine
|   |   |-- gt
|   |   `-- partial
|   |-- 3-premolares
|   |   |-- gt
|   |   `-- partial
|   `-- 4-molars
|       |-- gt
|       `-- partial
`-- test_dataset_processed（训练之前处理后保存的数据集）
    |-- test
    |   |-- gt
    |   `-- partial
    |-- train
    |   |-- gt
    |   |   |-- 3-premolares
    |   |   `-- 4-molars
    |   `-- partial
    |       |-- 3-premolares
    |       `-- 4-molars
    `-- val
        |-- gt
        |   |-- 3-premolares
        |   `-- 4-molars
        `-- partial
            |-- 3-premolares
            `-- 4-molars
```
* 保证在test_dataset_processed中的partial大小为4000,gt为768。
* 制作数据集后需要分别运行tools下的FPS和UPSAMPLE，会从test_dataset中读取处理后保存在test_dataset_processed中

## 2) Train
在训练之前打开visdom服务
```
python -m visdom.server
```
同时更改Experiment_number，来自定义训练的代号。只要数据集分布如上，则只用该dataroot参数指定就好。
直接运行这个文件就可以执行。
```
python Train_FPNet.py 
```
Change ‘crop_point_num’ to control the number of missing points.
Change ‘point_scales_list ’to control different input resolutions.
Change ‘D_choose’to control without using D-net.


## 3) Inference
测试单个文件的功能在tools/INFERENCE.py中。

可以直接修改infile参数指定需要输入的文件，输入的文件必须要满足点云数为4000，同时infile_real必须为768

如果不满大小可以在tools/FPS和tools/UPSAMPLE中进行手动修改
PARSERS FOR INFERENCE可以快捷定义测试所需要的参数
```
python INFERENCE.py 
```

## 4) Visualization of Examples

If you want to visualize, it is best to use MeshLab.


## 5) TOOLS
各种小工具的使用看 ./tools/README.md