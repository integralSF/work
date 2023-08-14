import numpy as np
import numpy as np
from plyfile import PlyData
import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

def farthest_point_sample(points, num_samples):
    """
    Args:
        points: (N, 3) tensor
        num_samples
    Returns:
        new_points: (num_samples, 3) tensor
    """
    N = points.shape[0]
    new_points_idx = torch.zeros(num_samples, dtype=torch.long)
    distances = torch.ones(N, dtype=torch.float32) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long).item()

    for i in range(num_samples):
        new_points_idx[i] = farthest
        new_point = points[farthest].view(1, -1)
        dists = torch.sum((points - new_point) ** 2, dim=1)
        mask = dists < distances
        distances[mask] = dists[mask]
        farthest = torch.max(distances, dim=0)[1].item()

    new_points = points[new_points_idx, :]

    return new_points

def read_data(path):
    """read ply data"""
    plydata = PlyData.read(path)
    vertex = np.array([list(x)[:3] for x in plydata['vertex'].data])
    v_torch = torch.from_numpy(vertex)
    return v_torch

if __name__ == "__main__":
    count = 0
    name_list = ["1-incisor","2-canine","3-premolares", "4-molars"]
    for name in name_list:
        print("gt current :", name)
        
        # ---------------------------- SET PATH --------------------------------------------
        path = f"/home/zjh/workspace/workspace-git/PFNet/dataset/test_dataset/{name}/gt"
        save_path = f"/home/zjh/workspace/workspace-git/PFNet/dataset/test_dataset_processed/train/gt/{name}/"
        file_names = os.listdir(path)
        pbar1 = tqdm(total=len(file_names), desc="Running GT FPS")
        
        
        for i in file_names:
            pbar1.update(1)
            count += 1
            
            # -------------- OUTPUT PATH -------------------------
            file_name = os.path.join(path, i)
            file_class_name = i[:-4] + '.pts'
            file_save = os.path.join(save_path, file_class_name)
            
            
            # ----------------READ DATA ------------------------
            data = read_data(path=file_name)
            
            
            # ---------------- PROCESS ------------------------
            fps_data_torch = farthest_point_sample(data, 768)
            fps_data_np = fps_data_torch.numpy()
            
            
            np.savetxt(file_save, fps_data_np, delimiter=" ")
        pbar1.close()
        print("------------------------")
    print(f" PROCESS NUM: {count} ")
    
   