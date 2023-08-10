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
    使用farthest point sampling从点云中采样指定数量的点。
    Args:
        points: (N, 3) tensor，输入点云的坐标。
        num_samples: 采样点的数量。
    Returns:
        new_points: (num_samples, 3) tensor，采样得到的点云坐标。
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
    plydata = PlyData.read(path)
    vertex = np.array([list(x)[:3] for x in plydata['vertex'].data])
    v_torch = torch.from_numpy(vertex)
    return v_torch

if __name__ == "__main__":
    # path = '/mnt/d/orthtest/2023-07-31_11_27_39-Teeth-22.ply'
    # data = read_data(path=path)

    # fps_data_torch = farthest_point_sample(data, 822)

    # fps_data_np = fps_data_torch.numpy()

    # # 保存numpy数组为.pts文件
    # # np.savetxt("/mnt/d/temp/p9/22.ply", fps_data_np, delimiter=" ")
    # pcd = o3d.geometry.PointCloud()
    # # 将numpy数组赋值给PointCloud对象的points属性
    # pcd.points = o3d.utility.Vector3dVector(fps_data_np)
    # # 将PointCloud对象保存为ply文件

    # o3d.io.write_point_cloud("/mnt/d/temp/p9/22.ply", pcd)
    count = 0
    name_list = ["1-incisor","2-canine","3-premolares", "4-molars"]
    for name in name_list:
        print("gt current :", name)
        path = f"/home/zjh/workspace/workspace-git/PFNet/dataset/test_dataset/{name}/gt"
        save_path = f"/home/zjh/workspace/workspace-git/PFNet/dataset/test_dataset_processed/train/gt/{name}/"
        file_names = os.listdir(path)
        pbar1 = tqdm(total=len(file_names), desc="Running GT FPS")
        for i in file_names:
            pbar1.update(1)
            count += 1
            file_name = os.path.join(path, i)
            file_class_name = i[:-4] + '.pts'
            file_save = os.path.join(save_path, file_class_name)
            
            data = read_data(path=file_name)
            
            fps_data_torch = farthest_point_sample(data, 768)
            
            fps_data_np = fps_data_torch.numpy()
            np.savetxt(file_save, fps_data_np, delimiter=" ")
        pbar1.close()
        print("------------------------")
    print(f"======={count}=========")
        
        
    # for name in name_list:
    #     print("partial current :", name)
    #     path = f"/home/zjh/workspace/workspace-git/PFNet/dataset/own_dataset_pure/{name}/partial"
    #     save_path = f"/home/zjh/workspace/workspace-git/PFNet/dataset/own_dataset_processed/train/partial/{name}/"
    #     file_names = os.listdir(path)
    #     pbar2 = tqdm(total=len(file_names), desc="Running Partial FPS")
    #     for i in file_names:
    #         pbar2.update(1)
            
    #         file_name = os.path.join(path, i)
    #         file_class_name = i[:-4] + '.pts'
    #         file_save = os.path.join(save_path, file_class_name)
    
    #         data = read_data(path=file_name)
            
    #         fps_data_torch = farthest_point_sample(data, 2048)

    #         fps_data_np = fps_data_torch.numpy()
    #         np.savetxt(file_save, fps_data_np, delimiter=" ")
    #     pbar2.close()
    #     print("------------------------")