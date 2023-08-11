#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import argparse

import torch
import torch.nn.parallel
import torch.utils.data
import sys
sys.path.append("/home/zjh/workspace/workspace-git/PFNet/")
import utils
from utils import PointLoss

from model_PFNet import _netlocalD,_netG

model_num = '8'
model_iter = '200'
data_num = '62-26' # 60-24 62-26 140-45

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=4000, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=768,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=401, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default=f'./checkpoint/Trained_Model_{model_num}/point_netG{model_iter}.pth', help="path to netG (to continue training)")
parser.add_argument('--infile',type = str, default = f'./test_one/partial-normalize-{data_num}.txt')
parser.add_argument('--infile_real',type = str, default = f'./test_one/gt-normalize-{data_num}.txt')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
# If you want to test your point clouds.  Set the first parameter of '--point_scales_list' equal to (point_number + 512).
parser.add_argument('--point_scales_list',type=list,default=[4000,2000,1000],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
opt = parser.parse_args()
# print(opt)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num) 
point_netG = torch.nn.DataParallel(point_netG)
point_netG.to(device)
point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
point_netG.eval()

input_cropped1 = np.loadtxt(opt.infile,delimiter=' ')
print(input_cropped1.shape)
input_cropped1 = torch.FloatTensor(input_cropped1)
input_cropped1 = torch.unsqueeze(input_cropped1, 0)
# Zeros = torch.zeros(1,768,3)
# input_cropped1 = torch.cat((input_cropped1,Zeros),1)


input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
input_cropped4_idx = utils.farthest_point_sample(input_cropped1,256,RAN = True)
input_cropped4     = utils.index_points(input_cropped1,input_cropped4_idx)
input_cropped2 = input_cropped2.to(device)
input_cropped3 = input_cropped3.to(device)      
input_cropped  = [input_cropped1,input_cropped2,input_cropped3]

# ([2, 2232, 3]) --- torch.Size([2, 1500, 3]) --- torch.Size([2, 768, 3])
print("INPUT_CROPPED:", input_cropped[0].shape, "---", input_cropped[1].shape, "---",input_cropped[2].shape)

fake_center1,fake_center2,fake=point_netG(input_cropped)
print("FAKE",fake.shape)
fake = fake.cuda()
fake_center1 = fake_center1.cuda()
fake_center2 = fake_center2.cuda()


input_cropped2 = input_cropped2.cpu()
input_cropped3 = input_cropped3.cpu()
input_cropped4 = input_cropped4.cpu()

np_crop2 = input_cropped2[0].detach().numpy()
np_crop3 = input_cropped3[0].detach().numpy()
np_crop4 = input_cropped4[0].detach().numpy()

real = np.loadtxt(opt.infile_real,delimiter=' ')
real = torch.FloatTensor(real)
real = torch.unsqueeze(real,0)
real2_idx = utils.farthest_point_sample(real,128, RAN = False)
real2 = utils.index_points(real,real2_idx)
real3_idx = utils.farthest_point_sample(real,256, RAN = True)
real3 = utils.index_points(real,real3_idx)

real = real.cpu()
real2 = real2.cpu()
real3 = real3.cpu()

np_real = real[0].detach().numpy()
np_real2 = real2[0].detach().numpy()
np_real3 = real3[0].detach().numpy()


fake =fake.cpu()
fake_center1 = fake_center1.cpu()
fake_center2 = fake_center2.cpu()
np_fake = fake[0].detach().numpy()
np_fake1 = fake_center1[0].detach().numpy()
np_fake2 = fake_center2[0].detach().numpy()
input_cropped1 = input_cropped1.cpu()
np_crop = input_cropped1[0].numpy() 


# np.savetxt('test_one/np_crop'+'.txt', np_crop, fmt = "%f %f %f")
np.savetxt(f'test_one/{model_num}/{model_num}_{data_num}_test_output_normalize_fake_new200'+'.txt', np_fake, fmt = "%f %f %f")
# np.savetxt('test_one/np_real'+'.txt', np_real, fmt = "%f %f %f")
# np.savetxt('test_one/real_ours_txt1'+'.txt', np_real, fmt = "%f,%f,%f")
# np.savetxt('test_one/real_ours_txt3'+'.txt', np_real, fmt = "%f,%f,%f")