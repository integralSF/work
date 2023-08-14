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
    """main process"""
    points = read_pts(input_filename)
    points_normalized = pc_normalize(points)
    write_pts(output_filename, points_normalized)

if __name__ == "__main__":# 60-24 62-26 140-45
    main("dataset/test_dataset_processed/train/gt/1-incisor/incisor-gt-101-42.pts",
         "test_one(JIAYA)/gt-normalize-101-42.txt")
   