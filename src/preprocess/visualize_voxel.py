import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
from dotenv import load_dotenv
import re

load_dotenv()

def is_clean_filename(filename):
    # Check if filename (without extension) does NOT contain _rot or _flip
    name = os.path.splitext(filename)[0]
    return not re.search(r'_(rot|flip)\d*', name)

def filter_clean_filenames(file_list):
    return [f for f in file_list if is_clean_filename(f)]
    
def visualize(filename, DATA_PATH, VOXEL_RESOLUTION):
    voxel_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(voxel_path):
        print(f"[WARN] File not found: {voxel_path}")
        return

    voxel_grid = np.load(voxel_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, edgecolor='k')

    ax.set_xlim(0, VOXEL_RESOLUTION)
    ax.set_ylim(0, VOXEL_RESOLUTION)
    ax.set_zlim(0, VOXEL_RESOLUTION)

    try:
        ax.set_box_aspect([1, 1, 1])
    except:
        def set_axes_equal(ax):
            limits = np.array([
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ])
            spans = limits[:, 1] - limits[:, 0]
            centers = np.mean(limits, axis=1)
            max_span = max(spans)
            half_span = max_span / 2
            new_limits = np.array([
                centers - half_span,
                centers + half_span
            ]).T
            ax.set_xlim3d(new_limits[0])
            ax.set_ylim3d(new_limits[1])
            ax.set_zlim3d(new_limits[2])
        set_axes_equal(ax)

    out_dir = f"../../visualize/RES_{VOXEL_RESOLUTION}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filename}_res{VOXEL_RESOLUTION}.png")
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")
    plt.close()

# --- Main logic ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python script.py <filename(s)> <VOXEL_RESOLUTION>")
        print("  python script.py --random N <RES1> [RES2 ...]")
        sys.exit(1)

    if sys.argv[1] == "--random":
        try:
            count = int(sys.argv[2])
            resolutions = [int(r) for r in sys.argv[3:]]

            if not resolutions:
                raise ValueError("At least one resolution must be provided")

            # Pick files from the first resolution's path
            first_res = resolutions[0]
            base_path = os.getenv(f"VOXELIZED_DATA_DIR_{first_res}")
            all_files = [f for f in os.listdir(base_path) if f.endswith('.npy')]
            clean_files = filter_clean_filenames(all_files)

            selected_files = random.sample(clean_files, min(count, len(all_files)))

            for res in resolutions:
                DATA_PATH = os.getenv(f"VOXELIZED_DATA_DIR_{res}")
                for fname in selected_files:
                    visualize(fname, DATA_PATH, res)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    else:
        # Specific filenames + 1 resolution at the end
        selected_files = sys.argv[1:-1]
        VOXEL_RESOLUTION = int(sys.argv[-1])
        DATA_PATH = os.getenv(f"VOXELIZED_DATA_DIR_{VOXEL_RESOLUTION}")
        for fname in selected_files:
            visualize(fname, DATA_PATH, VOXEL_RESOLUTION)
