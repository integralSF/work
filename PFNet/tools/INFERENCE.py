# -*- coding: utf-8 -*-

import numpy as np
import argparse
import time
import torch
import torch.nn.parallel
import torch.utils.data
import sys
sys.path.append("/home/zjh/workspace/workspace-git/PFNet/")
import utils
from utils import PointLoss
from model_PFNet import _netG


# ----------PARSER FOR INFERENCE-------------
model_num = '9'
model_iter = '350'
data_num = '101-42' # 60-24 62-26 140-45(MOYA)
                    # 66-13 95-23 101-42(JIAYA)

parser = argparse.ArgumentParser()
# ----------------------------------------INPUT FILE PATH-------------------------------------------------------
parser.add_argument('--infile',type = str, default = f'./test_one(JIAYA)/partial-normalize-{data_num}.txt')
parser.add_argument('--infile_real',type = str, default = f'./test_one(JIAYA)/gt-normalize-{data_num}.txt')
parser.add_argument('--netG', default=f'./checkpoint/Trained_Model_{model_num}/point_netG{model_iter}.pth', help="path to netG (to continue training)")


# ----------------------------------------DEFAULT PARSER--------------------------------------------------------
parser.add_argument('--workers', type=int,default=0, help='number of data loading workers')
parser.add_argument('--crop_point_num',type=int,default=768,help='0 means do not use else use with this weight')
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[4000,2000,1000],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
opt = parser.parse_args()


# --------------------------PERPARE--------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion_PointLoss = PointLoss().to(device)


# -------------------------------LOAD MODEL-----------------------------------
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num) 
point_netG = torch.nn.DataParallel(point_netG)
point_netG.to(device)
point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
point_netG.eval()


# ---------------------READ DATA AND CROP--------------------------------------
input_cropped1 = np.loadtxt(opt.infile,delimiter=' ')
input_cropped1 = torch.FloatTensor(input_cropped1)
input_cropped1 = torch.unsqueeze(input_cropped1, 0)
input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
input_cropped2 = input_cropped2.to(device)
input_cropped3 = input_cropped3.to(device)      
input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
# print("INPUT_CROPPED:", input_cropped[0].shape, "---", input_cropped[1].shape, "---",input_cropped[2].shape)


# -------------------INPUT AND INFERENCE--------------------
fake_center1,fake_center2,fake=point_netG(input_cropped)


# ---------PROCESS FAKE----------
fake =fake.cpu()
np_fake = fake[0].detach().numpy()


# --------------REAL GT DATA-------------------------
real = np.loadtxt(opt.infile_real,delimiter=' ')
real = torch.FloatTensor(real)
real = torch.unsqueeze(real,0)
# if want see other sclar
# real2_idx = utils.farthest_point_sample(real,128, RAN = False)
# real2 = utils.index_points(real,real2_idx)
# real3_idx = utils.farthest_point_sample(real,256, RAN = True)
# real3 = utils.index_points(real,real3_idx)
real = real.cpu()


# ------------------------CALU THE CD LOSS----------------------------------
CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real,1))


# --------------------------SVAE RESULT-------------------------------------
np.savetxt(f'test_one(JIAYA)/{model_num}/{model_num}_{data_num}_test_output_normalize_fake'+'.txt', np_fake, fmt = "%f %f %f")
