import open3d as o3d
import numpy as np
from scipy.spatial import distance

def remove_close_points(cloud1, cloud2, threshold):
    dist_matrix = distance.cdist(cloud1, cloud2)
    min_dists = np.min(dist_matrix, axis=1)
    mask = min_dists > threshold
    return cloud1[mask]

if __name__ == "__main__":
    p = "p9"
    num = "21"


    # 读取PLY文件
    cloud1 = o3d.io.read_point_cloud(f"/mnt/d/temp/{p}/completion.ply")
    cloud2 = o3d.io.read_point_cloud(f"/mnt/d/temp/{p}/{num}.ply")

    # 获取点云中的点
    points1 = np.asarray(cloud1.points)
    points2 = np.asarray(cloud2.points)

    # 计算距离并剔除距离近的点
    threshold = 0.5
    new_points = remove_close_points(points1, points2, threshold)

    # 保存新的PLY文件
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(new_points)
    o3d.io.write_point_cloud(f"/mnt/d/temp/{p}/not-cropping-Teeth-{num}.ply", new_cloud)


# import open3d as o3d
# import numpy as np

# # 读取两个PLY文件
# cloud1 = o3d.io.read_point_cloud('/mnt/d/temp/p1/completion.ply')
# cloud2 = o3d.io.read_point_cloud('/mnt/d/temp/p1/2023-07-28_09_30_26-Teeth-16.ply')

# # 将点云转换为numpy数组
# points1 = np.asarray(cloud1.points)
# points2 = np.asarray(cloud2.points)

# # 找出存在于cloud1中，但不在cloud2中的点
# diff_points = [point for point in points1 if point not in points2]
# print(len(points1), len(points2), len(diff_points))
# # 创建一个新的点云，包含差异点
# diff_cloud = o3d.geometry.PointCloud()
# diff_cloud.points = o3d.utility.Vector3dVector(diff_points)

# # 保存差异点云为PLY文件
# o3d.io.write_point_cloud('/mnt/d/temp/p1/not-cropping-Teeth-16.ply', diff_cloud)


