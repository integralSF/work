# PF-Net-Point-Fractal-Network

This repository is still under constructions.

If you have any questions about the code, please email me. Thanks!

This is the Pytorch implement of CVPR2020 PF-Net: Point Fractal Network for 3D Point Cloud Completion. 

https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.pdf

## 0) Environment
Pytorch 1.13.0+cu117
Python 3.8.13

##1) Dataset
```
do it by yourself.
```
## 2) Train
```
python Train_FPNet.py 
```
Change ‘crop_point_num’ to control the number of missing points.
Change ‘point_scales_list ’to control different input resolutions.
Change ‘D_choose’to control without using D-net.


## 3) Visualization of Examples

If you want to visualize, it is best to use MeshLab.
