# -*- coding: utf-8 -*-

import numpy as np
import argparse
import time
import os
import torch
import torch.nn.parallel
import torch.utils.data
import sys
sys.path.append("/home/zjh/workspace/workspace-git/PFNet/")
import utils
from utils import PointLoss
from model_PFNet import _netG
import NORMALIZE


# ----------PARSER FOR INFERENCE-------------
model_num = '10'
model_iter = '500'
data_num = '99-27' # 60-24 62-26 140-45(MOYA)
                    # 66-13 95-23 101-42(JIAYA)

tooth_type = "MOYA"

parser = argparse.ArgumentParser()
# ----------------------------------------INPUT FILE PATH-------------------------------------------------------
parser.add_argument('--infile_real',type = str, default =
                    f'dataset/test_dataset_processed/val/gt/4-molars/molars-gt-99-27.pts')
parser.add_argument('--infile',type = str, default = 
                    f'dataset/test_dataset_processed/val/partial/4-molars/molars-partial-99-27.pts')
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

input_cropped1 = NORMALIZE.pc_normalize(input_cropped1)

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
time1 = time.time()
fake_center1,fake_center2,fake=point_netG(input_cropped)
print("-------------------------")
print("INFERENCE TIME: %.4f"%(time.time() - time1))

# ---------PROCESS FAKE----------
fake =fake.cpu()
np_fake = fake[0].detach().numpy()


# --------------REAL GT DATA-------------------------
real = np.loadtxt(opt.infile_real,delimiter=' ')

real = NORMALIZE.pc_normalize(real)

real = torch.FloatTensor(real)
real = torch.unsqueeze(real,0)
# if want see other sclar
# real2_idx = utils.farthest_point_sample(real,128, RAN = False)
# real2 = utils.index_points(real,real2_idx)
# real3_idx = utils.farthest_point_sample(real,256, RAN = True)
# real3 = utils.index_points(real,real3_idx)
real = real.cpu()
np_real = real[0].detach().numpy()


# ------------------------CALU THE CD LOSS----------------------------------
CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real,1))
print("CD_LOSS: %.4f" % (CD_LOSS.item()),
      "\n-------------------------")


folder_path = f'test_one({tooth_type})/{model_num}'
os.makedirs(folder_path, exist_ok=True)

# --------------------------SVAE RESULT-------------------------------------
np.savetxt(f'test_one({tooth_type})/{model_num}/{model_num}_{model_iter}_{data_num}_test_output_normalize_fake'+'.txt', np_fake, fmt = "%f %f %f")
np.savetxt(f'test_one({tooth_type})/{model_num}/{model_num}_{model_iter}_{data_num}_test_output_normalize_real'+'.txt', np_real, fmt = "%f %f %f")
