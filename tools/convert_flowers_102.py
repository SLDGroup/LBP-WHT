import os
import sys
import numpy as np
from scipy.io import loadmat

label_path = os.path.join(sys.argv[1], "imagelabels.mat")
split_path = os.path.join(sys.argv[1], "setid.mat")

train_path = os.path.join(sys.argv[1], "train.txt")
test_path = os.path.join(sys.argv[1], "test.txt")

image_labels = loadmat(label_path)['labels'].flatten()
split = loadmat(split_path)
trainval_id = np.concatenate(
    [split['trnid'].flatten(), split['valid'].flatten()])
test_id = split['tstid'].flatten()

with open(train_path, 'w') as f:
    for idx in trainval_id:
        f.write(f"image_{idx:05d}.jpg {image_labels[idx-1] - 1}\n")

with open(test_path, 'w') as f:
    for idx in test_id:
        f.write(f"image_{idx:05d}.jpg {image_labels[idx-1] - 1}\n")
