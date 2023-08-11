from DIFF import *
from FPS import *
from tqdm import tqdm
import time

idx = 140
p = "p16"
#       记得换idx
num = "46" # 记得换idx 44 45 46 
#       记得换idx
pbar1 = tqdm(total=10, desc=f"{num} Running PROCESS PIPELINE")

pbar1.update(1)
time.sleep(0.05)

root_path = f'/mnt/d/orthtest/'
path = ''
all_teeths = os.listdir(root_path)
for teeth in all_teeths:
   if teeth[-6:-4] == num:
       path = os.path.join(root_path, teeth)
       break
if not path:
    raise Exception('no path exist')
data = read_data(path=path)

pbar1.update(1)
time.sleep(0.05)

fps_data_torch = farthest_point_sample(data, 888)

pbar1.update(1)
time.sleep(0.05)


fps_data_np = fps_data_torch.numpy()

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(fps_data_np)

o3d.io.write_point_cloud(f"/mnt/d/temp/{p}/{num}.ply", pcd)

pbar1.update(1)
time.sleep(0.05)

# 读取PLY文件

pbar1.update(1)
time.sleep(0.05)

cloud1 = o3d.io.read_point_cloud(f"/mnt/d/temp/{p}/completion.ply")
cloud2 = o3d.io.read_point_cloud(f"/mnt/d/temp/{p}/{num}.ply")

pbar1.update(1)
time.sleep(0.05)

# 获取点云中的点
points1 = np.asarray(cloud1.points)
points2 = np.asarray(cloud2.points)

# 计算距离并剔除距离近的点
threshold = 0.33

pbar1.update(1)
time.sleep(0.05)

new_points = remove_close_points(points1, points2, threshold)

pbar1.update(1)
time.sleep(0.05)

# 保存新的PLY文件
new_cloud = o3d.geometry.PointCloud()
new_cloud.points = o3d.utility.Vector3dVector(new_points)

pbar1.update(1)
time.sleep(0.05)

o3d.io.write_point_cloud(f"/mnt/d/temp/{p}/not-cropping-Teeth-{num}.ply", new_cloud)

pbar1.update(1)
time.sleep(0.05)


pbar1.close()