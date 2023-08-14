import numpy as np
from pyntcloud import PyntCloud
import os
from tqdm import tqdm
import numpy as np
import plyfile

def increase_point_cloud(input_file, output_file, target_num_points):
    """
    unsample, make (n, 3) --> (targer_num_points, 3)
    Args:
        input_file : path
        output_file : path
        target_num_points (int): result points num
    """
    plydata = plyfile.PlyData.read(input_file)
    point_cloud = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T

    num_points_current = point_cloud.shape[0]


    num_points_target = target_num_points  # set result points

    # 如果当前点数少于目标点数，则生成新的点
    if num_points_current < num_points_target:
        num_points_to_generate = num_points_target - num_points_current

        # 随机选择要复制的点
        points_to_duplicate = point_cloud[np.random.choice(num_points_current, num_points_to_generate)]

        # 对每个点添加小的随机偏移
        random_offsets = np.random.normal(scale=0.01, size=points_to_duplicate.shape)  # 偏移的规模可以根据需要调整
        new_points = points_to_duplicate + random_offsets

        # 将新的点添加到点云中
        point_cloud = np.vstack([point_cloud, new_points])

    # 将点云保存为 PTS 文件
    np.savetxt(output_file, point_cloud, fmt='%f')
    
# 示例
# increase_point_cloud("dataset/test/partial/partial-test-1-36.ply", "dataset/test_dataset_processed/test/partial/partial-test-1-36.pts", 4000)

count = 0
name_list = ["1-incisor","2-canine","3-premolares", "4-molars"]
for name in name_list:
    print("partial current :", name)
    
    
    # --------------------- SET PATH -------------------------------
    path = f"dataset/test_dataset/{name}/partial"
    save_path = f"dataset/test_dataset_processed/train/partial/{name}/"
    file_names = os.listdir(path)
    
    
    pbar1 = tqdm(total=len(file_names), desc="Running PARTIAL UPSAMPLE")
    for i in file_names:
        count+=1
        pbar1.update(1)
        
        
        # ------- SAVE PATH -------------
        file_name = os.path.join(path, i)
        file_class_name = i[:-4] + '.pts'
        file_save = os.path.join(save_path, file_class_name)
        
        
        # ------------- PROCESS ----------------------
        increase_point_cloud(file_name, file_save, 4000)
        
    pbar1.close()

    print("------------------------")
    
print(f" PROCESS NUM: {count} ")

