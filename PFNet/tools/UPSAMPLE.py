import numpy as np
from pyntcloud import PyntCloud
import os
from tqdm import tqdm
import numpy as np
import plyfile

def increase_point_cloud(input_file, output_file, target_num_points):
    # 读取 ply 文件
    
    # 读取 PLY 文件
    plydata = plyfile.PlyData.read(input_file)
    # 获取点云数据
    point_cloud = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T

    # 计算当前点云的数量
    num_points_current = point_cloud.shape[0]

    # 定义期望的点数
    num_points_target = target_num_points  # 可以修改为你需要的点数

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
# 示例tt
# increase_point_cloud("dataset/own_dataset_pure/4-molars/partial/molars-partial-62-26.ply", "output.pts", 4000)

count = 0
name_list = ["1-incisor","2-canine","3-premolares", "4-molars"]
for name in name_list:
    print("partial current :", name)
    path = f"/home/zjh/workspace/workspace-git/PFNet/dataset/test_dataset/{name}/partial"
    save_path = f"/home/zjh/workspace/workspace-git/PFNet/dataset/test_dataset_processed/train/partial/{name}/"
    file_names = os.listdir(path)
    pbar1 = tqdm(total=len(file_names), desc="Running PARTIAL UPSAMPLE")

    for i in file_names:
        count+=1
        pbar1.update(1)

        file_name = os.path.join(path, i)
        file_class_name = i[:-4] + '.pts'
        file_save = os.path.join(save_path, file_class_name)
        # print(file_name, ' ',file_save)
        increase_point_cloud(file_name, file_save, 4000)
    pbar1.close()
    print("------------------------")
print(f"=========={count}==============")

# import numpy as np
# from plyfile import PlyData

# def increase_point_cloud(input_file, output_file, target_num_points):
#     # 读取 ply 文件
#     import numpy as np
#     import plyfile

#     # 读取 PLY 文件
#     plydata = plyfile.PlyData.read(input_file)
#     # 获取点云数据
#     point_cloud = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T

#     # 计算当前点云的数量
#     num_points_current = point_cloud.shape[0]

#     # 定义期望的点数
#     num_points_target = target_num_points  # 可以修改为你需要的点数

#     # 如果当前点数少于目标点数，则生成新的点
#     if num_points_current < num_points_target:
#         num_points_to_generate = num_points_target - num_points_current

#         # 随机选择要复制的点
#         points_to_duplicate = point_cloud[np.random.choice(num_points_current, num_points_to_generate)]

#         # 对每个点添加小的随机偏移
#         random_offsets = np.random.normal(scale=0.1, size=points_to_duplicate.shape)  # 偏移的规模可以根据需要调整
#         new_points = points_to_duplicate + random_offsets

#         # 将新的点添加到点云中
#         point_cloud = np.vstack([point_cloud, new_points])

#     # 将点云保存为 PTS 文件
#     np.savetxt('output_file', point_cloud, fmt='%f')



# # 示例用法
# increase_point_cloud('dataset/test_dataset/3-premolares/partial/premolares-partial-89-14.ply', '/mnt/d/temp/temp1/test.pts', 4000)
