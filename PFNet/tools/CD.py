import numpy as np
import torch

# -------------- READ DATA --------------------------------
point_cloud_1 = np.loadtxt("test_one/gt_noramlize.txt")
point_cloud_2 = np.loadtxt("test_one/output_noramlize.txt")

point_cloud_1 = torch.from_numpy(point_cloud_1)
point_cloud_2 = torch.from_numpy(point_cloud_2)
point_cloud_1 = point_cloud_1.unsqueeze(0)
point_cloud_2 = point_cloud_2.unsqueeze(0)


from chamfer3D import dist_chamfer_3D


# --------- BUILD CALU --------------------
chamLoss = dist_chamfer_3D.chamfer_3DDist()


# ------------------- TURN DEVICE --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
point_cloud_1 = point_cloud_1.to(device)
point_cloud_2 = point_cloud_2.to(device)
point_cloud_1=point_cloud_1.to(torch.float32)
point_cloud_2=point_cloud_2.to(torch.float32)


# -------------------- CALU THE DIFFERENCE CD ------------------------
dist1, dist2, idx1, idx2 = chamLoss(point_cloud_1, point_cloud_2)
cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
cd_t = (dist1.mean(1) + dist2.mean(1))

print(cd_p)
