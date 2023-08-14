import open3d as o3d
import numpy as np
from scipy.spatial import distance

def remove_close_points(cloud1, cloud2, threshold):
    """
    Args:
        cloud1 
        cloud2
        threshold
    Returns:
        _type_: a new cloud
    """
    dist_matrix = distance.cdist(cloud1, cloud2)
    min_dists = np.min(dist_matrix, axis=1)
    mask = min_dists > threshold
    return cloud1[mask]

if __name__ == "__main__":
    p = "p9"
    num = "21"

    # ------------------------- READ DATA ----------------------------------
    cloud1 = o3d.io.read_point_cloud(f"/mnt/d/temp/{p}/completion.ply")
    cloud2 = o3d.io.read_point_cloud(f"/mnt/d/temp/{p}/{num}.ply")
    points1 = np.asarray(cloud1.points)
    points2 = np.asarray(cloud2.points)

    # ------------------------ SET THREHOLD TO REMOVE --------------------
    threshold = 0.5
    new_points = remove_close_points(points1, points2, threshold)

    # ----------------------- SAVE THE RESULT ---------------------------
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(new_points)
    o3d.io.write_point_cloud(f"/mnt/d/temp/{p}/not-cropping-Teeth-{num}.ply", new_cloud)

