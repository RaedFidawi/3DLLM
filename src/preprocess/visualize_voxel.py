import numpy as np
import matplotlib.pyplot as plt

import sys

filename = sys.argv[1]

voxel_grid = np.load("../../voxelized/" + filename)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(voxel_grid, edgecolor='k')

plt.savefig(f"../../visualize/voxel_visualization_{filename}.png")
print("Saved visualization to voxel_visualization.png")
