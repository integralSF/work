import numpy as np
from sklearn.preprocessing import MinMaxScaler



# read the .pts file and convert it to a numpy array
data = np.loadtxt("test_one/gt.txt")

# create a scaler object that can normalize each feature to [0,1] range
scaler = MinMaxScaler()

# normalize the data and save it as a new numpy array
normalized_data = scaler.fit_transform(data)

# save the normalized data as a new .pts file
np.savetxt("test_one/gt_noramlize.txt", normalized_data)
