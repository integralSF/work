import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import own_loader
from model_PFNet import _netlocalD,_netG
import numpy as np
from visdom import Visdom
# from tools.chamfer3D import dist_chamfer_3D

# chamLoss = dist_chamfer_3D.chamfer_3DDist()

vis_cd = Visdom()
vis_d = Visdom()
vis_g = Visdom()

vis_cd.line([0.], # Y的第一个点的坐标
          [0.], # X的第一个点的坐标
          win = 'vis_cd', # 窗口的名称
          opts = dict(title = 'vis_cd'),  # 图像的标例,
)

vis_d.line([0.], # Y的第一个点的坐标
          [0.], # X的第一个点的坐标
          win = 'vis_d', # 窗口的名称
          opts = dict(title = 'vis_d'),  # 图像的标例,
)

vis_g.line([0.], # Y的第一个点的坐标
          [0.], # X的第一个点的坐标
          win = 'vis_g', # 窗口的名称
          opts = dict(title = 'vis_g'),  # 图像的标例,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',  default='dataset/test_dataset_processed', help='path to dataset')
parser.add_argument('--workers', type=int,default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=5, help='input batch size')
parser.add_argument('--pnum', type=int, default=4000, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=768,help='0 melearning_ratelrrans do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=801, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='', help="fuck you")
parser.add_argument('--netD', default='', help="fuck")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[4000,2000,1000],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.95,help='0 means do not use else use with this weight')
opt = parser.parse_args()
# print(opt)


# ------------------------------ PREPARE ----------------------------------
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
point_netD = _netlocalD(opt.crop_point_num)
cudnn.benchmark = True
resume_epoch=0

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) 

if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    point_netG = torch.nn.DataParallel(point_netG)
    point_netD = torch.nn.DataParallel(point_netD)
    point_netG.to(device) 
    point_netG.apply(weights_init_normal)
    point_netD.to(device)
    point_netD.apply(weights_init_normal)
if opt.netG != '' :
    point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
if opt.netD != '' :
    point_netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']


# -------------- SET SEED ----------------
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


# ---------------------- SET DATALOADER ----------------------------------
dset = own_loader.my_LoadData(opt.dataroot, "train")
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
                                         shuffle=True,num_workers = int(opt.workers))

print("DATASET LEN :", len(dset))
# test_dset = shapenet_part_loader.PartDataset( root='./dataset/dataset/',classification=True, class_choice=None, npoints=opt.pnum, split='test')
# test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
#                                          shuffle=True,num_workers = int(opt.workers))

# print(point_netG)
# print(point_netD)


criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)

# setup optimizer
optimizerD = torch.optim.Adam(point_netD.parameters(), lr=opt.learning_rate,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
optimizerG = torch.optim.Adam(point_netG.parameters(), lr=opt.learning_rate,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

real_label = 1
fake_label = 0

crop_point_num = int(opt.crop_point_num)
input_cropped1 = torch.FloatTensor(opt.batchSize, opt.pnum, 3)
label = torch.FloatTensor(opt.batchSize)


num_batch = len(dset) / opt.batchSize
###########################
#  G-NET and T-NET
##########################  
if opt.D_choose == 1:
    for epoch in range(resume_epoch,opt.niter):

        if epoch<30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2
        _epoch_cd_loss = 0
        _epoch_d_loss = 0
        _epoch_g_loss = 0
        
        for i, data in enumerate(dataloader, 0):
            
            partial, gt, labels = data
            
            batch_size = partial.size()[0]
            
            input_cropped1 = partial
            real_center = gt
                        
            label.resize_([batch_size,1]).fill_(real_label)
            # real_point = real_point.to(device)
            real_center = real_center.to(device)
            # print("REAL_POINT:", real_point.shape)
            # print("输入大小:", real_center.shape)
            # print("gt大小:", input_cropped1.shape)
            
            # print(real_center in real_point)
            input_cropped1 = input_cropped1.to(device)
            label = label.to(device)
            ############################
            # (1) data prepare
            ###########################   
            
            # ------------------------原来的gt----------------------------------------   ([2, 512, 3])
            real_center = Variable(real_center,requires_grad=True)
            real_center = torch.squeeze(real_center,1)
            # print("原来的gt:", real_center.shape)
            # ----------------------------------------------------------------------------
            
            # ------------------------下采样2次的gt----------------------------------------([2, 64, 3])
            real_center_key1_idx = utils.farthest_point_sample(real_center,128,RAN = False)
            real_center_key1 = utils.index_points(real_center,real_center_key1_idx)
            real_center_key1 =Variable(real_center_key1,requires_grad=True)
            # print("下采样2次的gt:", real_center_key1.shape)
            # ----------------------------------------------------------------------------
            
            # ------------------------下采样1次的gt----------------------------------------([2, 128, 3])
            real_center_key2_idx = utils.farthest_point_sample(real_center,256,RAN = True)
            real_center_key2 = utils.index_points(real_center,real_center_key2_idx)
            real_center_key2 =Variable(real_center_key2,requires_grad=True)
            # print("下采样1次的gt:", real_center_key2.shape)
            # ----------------------------------------------------------------------------

            input_cropped1 = torch.squeeze(input_cropped1,1) # ([2, 2048, 3])
            # print("输入:", input_cropped1.shape)
            
            input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True) 
            input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)  # ([2, 1024, 3])
            # print("经过1次采样的输入:", input_cropped2.shape)
            
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
            input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx) # ([2, 512, 3])
            # print("经过2次采样的输入:", input_cropped3.shape)
            
            input_cropped1 = Variable(input_cropped1,requires_grad=True)
            input_cropped2 = Variable(input_cropped2,requires_grad=True)
            input_cropped3 = Variable(input_cropped3,requires_grad=True)
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)      
            
            input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
            # print("INPUT_CRO.PPED:", input_cropped[0].shape, "---", input_cropped[1].shape, "---",input_cropped[2].shape)
                  
            point_netG = point_netG.train()
            point_netD = point_netD.train()
            ############################
            # (2) Update D network
            ###########################        
            point_netD.zero_grad()
            real_center = torch.unsqueeze(real_center,1)  
            # print("REAL_CENTER: ", real_center.shape)
            output = point_netD(real_center)
            errD_real = criterion(output,label)
            errD_real.backward()
            fake_center1,fake_center2,fake = point_netG(input_cropped)
            fake = torch.unsqueeze(fake,1)
            label.data.fill_(fake_label)
            output = point_netD(fake.detach())
            errD_fake = criterion(output, label)
            # torch.autograd.set_detect_anomaly(True)
            errD_fake.backward()
            # print("ERRD_REAL:", errD_real, " ERRED_FAKE:", errD_fake)
            errD = errD_real + errD_fake
            optimizerD.step()
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            point_netG.zero_grad()
            label.data.fill_(real_label)
            output = point_netD(fake)
            errG_D = criterion(output, label)
            errG_l2 = 0
            CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
        
            errG_l2 = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))\
            +alpha1*criterion_PointLoss(fake_center1,real_center_key1)\
            +alpha2*criterion_PointLoss(fake_center2,real_center_key2)
            
            # ------------------------ CHANGE THE WAY TO CALU CD -----------------------------
            # dist1, dist2, idx1, idx2 = chamLoss(torch.squeeze(fake,1), torch.squeeze(real_center,1))
            # batch_cd_loss = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2  # cd_p
            # CD_LOSS = torch.mean(batch_cd_loss)
            # CD_LOSS = CD_LOSS * 100
            
            # dist1_l1, dist2_l1, idx1_l1, idx2_l1 = chamLoss(fake_center1,real_center_key1)
            # batch_cd_loss_l1 = (torch.sqrt(dist1_l1).mean(1) + torch.sqrt(dist2_l1).mean(1)) / 2
            # errG_l1 = torch.mean(batch_cd_loss_l1)
            # errG_l1 = errG_l1 * alpha1 * 100
            
            # dist1_l2, dist2_l2, idx1_l2, idx2_l2 = chamLoss(fake_center2,real_center_key2)
            # batch_cd_loss_l2 = (torch.sqrt(dist1_l2).mean(1) + torch.sqrt(dist2_l2).mean(1)) / 2
            # errG_l2 = torch.mean(batch_cd_loss_l2)
            # errG_l2 = errG_l2 * alpha2 * 100

            
            # errG_total_cd = CD_LOSS + errG_l1 + errG_l2
            
            errG = (1 - opt.wtl2) * errG_D + opt.wtl2 * errG_l2
            errG.backward()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f'
                  % (epoch, opt.niter, i, len(dataloader), 
                     errD.data, errG_D.data,errG_l2,errG,CD_LOSS))
            f=open('checkpoint/Trained_Model_9/loss_PFNet_9.txt','a')
            f.write('\n'+'[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f'
                  % (epoch, opt.niter, i, len(dataloader), 
                     errD.data, errG_D.data,errG_l2,errG,CD_LOSS))
            
            _epoch_cd_loss += CD_LOSS
            _epoch_d_loss += errD.data
            _epoch_g_loss += errG
            
        vis_cd.line([((_epoch_cd_loss.cpu().detach().numpy())/len(dataloader))],[epoch],win = 'vis_cd',update = 'append')
        vis_d.line([((_epoch_d_loss.cpu().detach().numpy())/len(dataloader))],[epoch],win = 'vis_d',update = 'append')
        vis_g.line([((_epoch_g_loss.cpu().detach().numpy())/len(dataloader))],[epoch],win = 'vis_g',update = 'append')
        
        print(f"============================FINSH {epoch} EPOCH===============================")
            # if i % 10 ==0:
            #     print('After, ',i,'-th batch')
            #     f.write('\n'+'After, '+str(i)+'-th batch')
            #     for i, data in enumerate(test_dataloader, 0):
            #         real_point, target = data
                    
            
            #         batch_size = real_point.size()[0]
            #         real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
            #         input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
            #         input_cropped1 = input_cropped1.data.copy_(real_point)
            #         real_point = torch.unsqueeze(real_point, 1)
            #         input_cropped1 = torch.unsqueeze(input_cropped1,1)
                    
            #         p_origin = [0,0,0]
                    
            #         if opt.cropmethod == 'random_center':
            #             choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                        
            #             for m in range(batch_size):
            #                 index = random.sample(choice,1)
            #                 distance_list = []
            #                 p_center = index[0]
            #                 for n in range(opt.pnum):
            #                     distance_list.append(distance_squre(real_point[m,0,n],p_center))
            #                 distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])                         
            #                 for sp in range(opt.crop_point_num):
            #                     input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
            #                     real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]  
            #         real_center = real_center.to(device)
            #         real_center = torch.squeeze(real_center,1)
            #         input_cropped1 = input_cropped1.to(device) 
            #         input_cropped1 = torch.squeeze(input_cropped1,1)
            #         input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
            #         input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
            #         input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
            #         input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
            #         input_cropped1 = Variable(input_cropped1,requires_grad=False)
            #         input_cropped2 = Variable(input_cropped2,requires_grad=False)
            #         input_cropped3 = Variable(input_cropped3,requires_grad=False)
            #         input_cropped2 = input_cropped2.to(device)
            #         input_cropped3 = input_cropped3.to(device)      
            #         input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
            #         point_netG.eval()
            #         fake_center1,fake_center2,fake  =point_netG(input_cropped)
            #         CD_loss = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
            #         print('test result:',CD_loss)
            #         f.write('\n'+'test result:  %.4f'%(CD_loss))
            #         break
            # f.close()
        
        schedulerD.step()
        schedulerG.step()
        
        if epoch% 50 == 0:   
            print("============SAVE MODEL==============")
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netG.state_dict()},
                        'checkpoint/Trained_Model_9/point_netG'+str(epoch)+'.pth' )
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netD.state_dict()},
                        'checkpoint/Trained_Model_9/point_netD'+str(epoch)+'.pth' )  
            print("====================================")
    