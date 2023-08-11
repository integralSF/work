import os
import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import transforms3d
import random


class my_LoadData(data.Dataset):
    def __init__(self, path, prefix="train"):
        if prefix=="train":
            self.file_path = os.path.join(path,'train') 
        elif prefix=="val":
            self.file_path = os.path.join(path,'val')  
        elif prefix=="test":
            self.file_path = os.path.join(path,'test') 
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
      
        self.input_data_dict = self.get_dict(self.file_path)
        
        self.input_data_list = list(self.input_data_dict.keys())
        random.shuffle(self.input_data_list)
        
        self.len = len(self.input_data_dict)

    def __len__(self):
        return self.len

    def read_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        return points
    
    def read_pts(self, path):
        point = np.loadtxt(path).astype(np.float32)
        return point
    

    def get_dict(self, path):
        temp_path = os.path.join(path, "partial")
        cls = os.listdir(temp_path)
        data = {}
        for c in cls:
            objs = os.listdir(os.path.join(temp_path, c))
            for obj in objs:
                file_path_partial = os.path.join(path,"partial", c, obj)
                obj_gt = obj.replace("partial", "gt")
                file_path_gt = os.path.join(path,"gt", c, obj_gt)
                data.update({file_path_partial:file_path_gt})
        return data


    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, index):

        partial_path = self.input_data_list[index]
        gt_path = self.input_data_dict[partial_path]
        
        label = os.path.basename(partial_path)[:3]
        partial = self.read_pts(partial_path)
        gt = self.read_pts(gt_path)
        
        # print(partial_path, "-->", gt_path)
        partial = self.pc_normalize(partial)
        gt = self.pc_normalize(gt)
        return partial, gt, label
        

if __name__ == "__main__":
    dataloader = my_LoadData("/home/zjh/workspace/workspace-git/PFNet/dataset/test_dataset_processed", "train")
    
    print(dataloader[30])