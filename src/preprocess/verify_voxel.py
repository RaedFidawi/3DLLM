import numpy as np
import os

VOXELIZED_DIR = '../../voxelized'

for file in os.listdir(VOXELIZED_DIR):
    if file.endswith('.npy'):
        voxels = np.load(os.path.join(VOXELIZED_DIR, file))
        print(f"{file}: shape = {voxels.shape}")
