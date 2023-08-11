# import numpy as np
# from sklearn.preprocessing import MinMaxScaler



# # read the .pts file and convert it to a numpy array
# data = np.loadtxt("dataset/test_dataset_processed/train/gt/3-premolares/premolares-gt-140-45.pts")

# # create a scaler object that can normalize each feature to [0,1] range
# scaler = MinMaxScaler()

# # normalize the data and save it as a new numpy array
# normalized_data = scaler.fit_transform(data)

# # save the normalized data as a new .pts file
# np.savetxt("test_one/gt-normalize-140-45.txt", normalized_data)


import numpy as np

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def read_pts(filename):
    """ Read a .pts file and return a Nx3 numpy array """
    with open(filename, 'r') as f:
        lines = f.readlines()
    points = [list(map(float, line.strip().split())) for line in lines]
    return np.array(points)

def write_pts(filename, points):
    """ Write a Nx3 numpy array to a .pts file """
    with open(filename, 'w') as f:
        for point in points:
            f.write(' '.join(map(str, point)) + '\n')

def main(input_filename, output_filename):
    points = read_pts(input_filename)
    points_normalized = pc_normalize(points)
    write_pts(output_filename, points_normalized)

if __name__ == "__main__":# 60-24 62-26 140-45
    main("dataset/test_dataset_processed/train/partial/4-molars/molars-partial-62-26.pts",
         "test_one/partial-normalize-62-26.txt")
    # import sys
    # if len(sys.argv) != 3:
    #     print(f"Usage: {sys.argv[0]} <input.pts> <output.pts>")
    #     sys.exit(1)
    # main(sys.argv[1], sys.argv[2])