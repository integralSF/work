#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Convlayer(nn.Module):
    def __init__(self,point_scales):
        super(Convlayer,self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128),2)
        x_256 = torch.squeeze(self.maxpool(x_256),2)
        x_512 = torch.squeeze(self.maxpool(x_512),2)
        x_1024 = torch.squeeze(self.maxpool(x_1024),2)
        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1)
        return x

class Latentfeature(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature,self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3,1,1)       
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self,x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        latentfeature = torch.cat(outs,2)
        latentfeature = latentfeature.transpose(1,2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature,1)
#        latentfeature_64 = F.relu(self.bn1(self.conv1(latentfeature)))  
#        latentfeature = F.relu(self.bn2(self.conv2(latentfeature_64)))
#        latentfeature = F.relu(self.bn3(self.conv3(latentfeature)))
#        latentfeature = latentfeature + latentfeature_64
#        latentfeature_256 = F.relu(self.bn4(self.conv4(latentfeature)))
#        latentfeature = F.relu(self.bn5(self.conv5(latentfeature_256)))
#        latentfeature = F.relu(self.bn6(self.conv6(latentfeature)))
#        latentfeature = latentfeature + latentfeature_256
#        latentfeature = F.relu(self.bn7(self.conv7(latentfeature)))
#        latentfeature = F.relu(self.bn8(self.conv8(latentfeature)))
#        latentfeature = self.maxpool(latentfeature)
#        latentfeature = torch.squeeze(latentfeature,2)
        return latentfeature


class PointcloudCls(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list,k=40):
        super(PointcloudCls,self).__init__()
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class _netG(nn.Module):
    def  __init__(self,num_scales,each_scales_size,point_scales_list,crop_point_num):
        super(_netG,self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        
        self.fc1_1 = nn.Linear(1024,128*512)
        self.fc2_1 = nn.Linear(512,64*128)#nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256,64*3)
        
#        self.bn1 = nn.BatchNorm1d(1024)
#        self.bn2 = nn.BatchNorm1d(512)
#        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
#        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
#        self.bn5 = nn.BatchNorm1d(64*128)
#        
        self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512,256,1)
        self.conv1_3 = torch.nn.Conv1d(256,int((self.crop_point_num*3)/128),1)
        self.conv2_1 = torch.nn.Conv1d(128,6,1)#torch.nn.Conv1d(256,12,1) !
        
#        self.bn1_ = nn.BatchNorm1d(512)
#        self.bn2_ = nn.BatchNorm1d(256)
        
    def forward(self,x):
        x = self.latentfeature(x)
        x_1 = F.relu(self.fc1(x)) #1024
        x_2 = F.relu(self.fc2(x_1)) #512
        x_3 = F.relu(self.fc3(x_2))  #256
        
        
        pc1_feat = self.fc3_1(x_3)
        # pc1_xyz = pc1_feat.reshape(-1,64,3)
        
        # print("PC1_FEAT",pc1_feat.shape)
        
        pc1_xyz = pc1_feat.reshape(-1,64,3) #64x3 center1   # 128 changed
        
        # print("PC1_XYZ",pc1_xyz.shape)
        
        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1,128,64)
        pc2_xyz =self.conv2_1(pc2_feat) #6x64 center2    
        
        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1,512,128) 
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat) #12x128 fine
        
        pc1_xyz_expand = torch.unsqueeze(pc1_xyz,2)
        pc2_xyz = pc2_xyz.transpose(1,2)
        pc2_xyz = pc2_xyz.reshape(-1,64,2,3)
        pc2_xyz = pc1_xyz_expand+pc2_xyz
        # pc2_xyz = pc2_xyz.reshape(-1,128,3)
        pc2_xyz = pc2_xyz.reshape(-1,128,3)  # 256 changed
        
        pc2_xyz_expand = torch.unsqueeze(pc2_xyz,2)
        pc3_xyz = pc3_xyz.transpose(1,2)
        pc3_xyz = pc3_xyz.reshape(-1,128,int(self.crop_point_num/128),3)
        pc3_xyz = pc2_xyz_expand+pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1,self.crop_point_num,3) 
        
        return pc1_xyz,pc2_xyz,pc3_xyz #center1 ,center2 ,fine
    
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.shortcut = nn.Sequential()
        
#         if in_channels != out_channels:
#             # print(in_channels,out_channels)
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, (1, 3)),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         out = self.relu(self.bn(self.conv(x)))
#         # print(out.shape)
#         # print(self.shortcut(x).shape)
#         out = out + self.shortcut(x)
#         return out
    
# class AttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         # 定义三个卷积层，分别用于生成查询、键和值
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
#         self.conv2 = nn.Conv2d(in_channels, in_channels // 8, 1)
#         self.conv3 = nn.Conv2d(in_channels, in_channels, 1)
#         # 定义一个softmax函数，用于计算注意力权重
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         # print("-"*40)
#         # 将输入通过第一个卷积层，得到查询
#         # print("inputX: ", x.shape) # x [1 64 512 1]
#         q = self.conv1(x) # query
#         # 将输入通过第二个卷积层，得到键
#         k = self.conv2(x) # key
#         # 将输入通过第三个卷积层，得到值
#         v = self.conv3(x) # value
#         # 获取输入的形状信息，包括批次大小、通道数、高度和宽度
#         # print("q,k,v", q.shape, k.shape, v.shape)  # [1 8 512 1] [1 8 521 1] [1 64 512 1]
#         b, c, h, w = q.shape
#         # print("B,C,H,W", b,c,h,w)  # 1 8 512 1
#         # 将查询变形为（批次大小、通道数、高度*宽度）
#         q = q.view(b, c, -1) # (b, c, h*w)
#         # print("218Q: ", q.shape)  # [1, 8, 512]
#         # 将键变形为（批次大小、高度*宽度、通道数），并交换第二维和第三维
#         k = k.view(b, c, -1).permute(0, 2, 1) # (b, h*w, c)
#         # print("221Q: ", k.shape)  #  [1, 512, 8]
#         # 将查询和键进行矩阵乘法，得到注意力分数（批次大小、通道数、通道数）
#         a = torch.bmm(k, q) # (b, c, c)
#         # print("224A: ", a.shape)  # [1, 8, 8]
#         # 将注意力分数通过softmax函数，得到注意力权重（注意力图）
#         a = self.softmax(a) # attention map
#         # print("227A: ", a.shape)  # [1, 8, 8]
#         # 将值变形为（批次大小、通道数*高度*宽度）
#         v = v.view(b, -1, h*w) # (b, c*h*w)
#         # print("230V: ", v.shape)  # [1, 64, 512]
#         # 将值和注意力权重进行矩阵乘法，得到注意力输出（批次大小、通道数*高度*宽度、通道数）
#         o = torch.bmm(v, a) # (b, c*h*w, c)
#         # print("233o: ", o.shape)
#         # 将注意力输出变形为（批次大小、通道数*高度*宽度、高度、宽度）
#         o = o.view(b, -1, h, w) # (b, c*h*w, h, w)
#         # print("237o: ", o.shape)
#         # 返回注意力输出
#         return o
    
# class _netlocalD(nn.Module):
#     def __init__(self,crop_point_num):
#         super(_netlocalD,self).__init__()
#         self.crop_point_num = crop_point_num
#         # use residual blocks instead of convolutions
#         self.res1 = ResidualBlock(1, 64, 1)
#         self.res2 = ResidualBlock(64, 64, 1)
#         self.res3 = ResidualBlock(64, 128, 1)
#         self.res4 = ResidualBlock(128, 256, 1)
#         # use attention blocks instead of batch normalization
#         self.att1 = AttentionBlock(64)
#         self.att2 = AttentionBlock(64)
#         self.att3 = AttentionBlock(128)
#         self.att4 = AttentionBlock(256)
#         # use max pooling as before
#         self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
#         # use fully connected layers as before
#         self.fc1 = nn.Linear(448,256)
#         self.fc2 = nn.Linear(256,128)
#         self.fc3 = nn.Linear(128,16)
#         self.fc4 = nn.Linear(16,1)
#         # use batch normalization as before
#         self.bn_1 = nn.BatchNorm1d(256)
#         self.bn_2 = nn.BatchNorm1d(128)
#         self.bn_3 = nn.BatchNorm1d(16)
#         # use dropout to prevent overfitting
#         self.dropout_1 = nn.Dropout(0.5)
#         self.dropout_2 = nn.Dropout(0.5)
#         self.dropout_3 = nn.Dropout(0.5)

#     def forward(self, x):
#         # print("self.res1(x):", x.shape)
#         x = F.relu(self.att1(self.res1(x)))
#         # print("X1: ", x.shape)
#         x_64 = F.relu(self.att2(self.res2(x)))
#         # print("X_64: ",x_64.shape)
#         x_128 = F.relu(self.att3(self.res3(x_64)))
#         # print("X_128: ",x_64.shape)
#         x_256 = F.relu(self.att4(self.res4(x_128)))
#         # print("X_256: ",x_64.shape)

        
#         # print("self.maxpool(x_64)", self.maxpool(x_64).shape)
#         x_64 = torch.squeeze(self.maxpool(x_64))
#         # print("self.maxpool(x_64)", x_64.shape)

#         # print("self.maxpool(x_128)", self.maxpool(x_128).shape)
#         x_128 = torch.squeeze(self.maxpool(x_128))        
#         # print("self.maxpool(x_128)", x_128.shape)

#         # print("self.maxpool(x_256)", self.maxpool(x_256).shape)
#         x_256 = torch.squeeze(self.maxpool(x_256))
#         # print("self.maxpool(x_256)", x_256.shape)

#         Layers = [x_256,x_128,x_64]
#         x = torch.cat(Layers,1)
#         # print("298x: ",x.shape)
#         x = x[:, :, 0].clone()
#         # print("300x: ",x.shape)
#         # print("self.fc1(x))", self.fc1(x).shape)
#         x = F.relu(self.bn_1(self.fc1(x)))
#         # print("303x: ",x.shape)
#         x = self.dropout_1(x)
#         x = F.relu(self.bn_2(self.fc2(x)))
#         # print("306x: ",x.shape)
#         x = self.dropout_2(x)
#         x = F.relu(self.bn_3(self.fc3(x)))
#         # print("309x: ",x.shape)
#         x = self.dropout_3(x)
#         x = self.fc4(x)
#         # print("312x: ",x.shape)
#         return x


class _netlocalD(nn.Module):
    def __init__(self,crop_point_num):
        super(_netlocalD,self).__init__()
        self.crop_point_num = crop_point_num
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x_64 = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x_64)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_64 = torch.squeeze(self.maxpool(x_64))
        x_128 = torch.squeeze(self.maxpool(x_128))
        x_256 = torch.squeeze(self.maxpool(x_256))
        Layers = [x_256,x_128,x_64]
        x = torch.cat(Layers,1)
        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)
        return x

if __name__=='__main__':
    # input1 = torch.randn(64,2048,3)
    input2 = torch.randn(2, 1, 512, 3)
    # input3 = torch.randn(64,256,3)
    # input_ = [input1,input2,input3]
    
    # output = netG(input_)
    # print(output)
    # res1 = ResidualBlock(1, 64, 1)
    # res2 = ResidualBlock(64, 64, 1)
    # res3 = ResidualBlock(64, 128, 1)
    # res4 = ResidualBlock(128, 256, 1)
    # # use attention blocks instead of batch normalization
    # att1 = AttentionBlock(64)
    # att2 = AttentionBlock(64)
    # att3 = AttentionBlock(128)
    # att4 = AttentionBlock(256)
    
    # x = F.relu(att1(res1(input2)))
    
    # x_64 = F.relu(att2(res2(x)))
   
    # x_128 = F.relu(att3(res3(x_64)))
    
    # x_256 = F.relu(att4(res4(x_128)))
    # print("X1: ", x.shape)
    # print("X_64: ",x_64.shape)
    # print("X_128: ",x_64.shape)
    # print("X_256: ",x_64.shape)
    
    netG=_netlocalD(512)
    c = netG(input2).shape
    print(c)