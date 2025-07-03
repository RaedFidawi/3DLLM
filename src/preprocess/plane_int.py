import numpy as np
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def clean_filename(filename: str) -> str:
    """
    Remove substrings like *rot*, *flip* from the filename (without extension)
    and return the cleaned name.
    """
    name = os.path.splitext(filename)[0]
    cleaned_name = re.sub(r'_(rot|flip)\d*', '', name)
    return cleaned_name

def generate_random_plane(resolution):
    normal = np.random.randn(3)
    normal /= np.linalg.norm(normal)
    point = np.random.uniform(0, resolution, size=3)
    return normal, point

def voxel_intersects_plane(voxel_min, voxel_max, normal, point, tol=1e-8):
    """
    Returns True if the plane (normal, point) intersects the voxel defined by voxel_min and voxel_max.
    The voxel is axis-aligned, so we check the signed distances at all 8 corners.
    """
    corners = np.array([
        [voxel_min[0], voxel_min[1], voxel_min[2]],
        [voxel_min[0], voxel_min[1], voxel_max[2]],
        [voxel_min[0], voxel_max[1], voxel_min[2]],
        [voxel_min[0], voxel_max[1], voxel_max[2]],
        [voxel_max[0], voxel_min[1], voxel_min[2]],
        [voxel_max[0], voxel_min[1], voxel_max[2]],
        [voxel_max[0], voxel_max[1], voxel_min[2]],
        [voxel_max[0], voxel_max[1], voxel_max[2]],
    ])
    dists = np.dot(corners - point, normal)
    # If the plane passes through the voxel, there must be both positive and negative distances
    return (np.any(dists < -tol) and np.any(dists > tol)) or np.any(np.abs(dists) < tol)

def find_intersecting_plane(full, resolution, max_plane_retries=100):
    filled_indices = np.argwhere(full > 0)
    if len(filled_indices) == 0:
        raise RuntimeError("No filled voxels to intersect.")
    # 8 corners relative to voxel min
    corner_offsets = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
    ], dtype=float)
    for _ in range(max_plane_retries):
        normal = np.random.randn(3)
        normal /= np.linalg.norm(normal)
        idx = filled_indices[np.random.randint(len(filled_indices))]
        point = idx.astype(float) + 0.5
        # Vectorized: all voxel mins (N, 1, 3) + (1, 8, 3) => (N, 8, 3)
        voxel_corners = filled_indices[:, None, :] + corner_offsets[None, :, :]
        dists = np.dot(voxel_corners - point, normal)  # (N, 8)
        # For each voxel: does it have both positive and negative distances?
        has_pos = (dists > 1e-8).any(axis=1)
        has_neg = (dists < -1e-8).any(axis=1)
        crosses = has_pos & has_neg
        intersecting = filled_indices[crosses]
        if len(intersecting) > 0:
            return normal, point, intersecting
    raise RuntimeError("Failed to find an intersecting plane after max retries.")

def generate_partial_from_full(full, resolution, num_planes=3, max_retries=100):
    for n_planes in range(num_planes, 0, -1):
        for _ in range(max_retries):
            partial = full.copy()
            try:
                for _ in range(n_planes):
                    normal, point, intersecting_voxels = find_intersecting_plane(partial, resolution, max_plane_retries=max_retries)
                    for idx in intersecting_voxels:
                        partial[tuple(idx)] = 0
                    if np.count_nonzero(partial) == 0:
                        break  # This attempt failed, try again
                if np.count_nonzero(partial) > 0:
                    return partial
            except RuntimeError:
                continue  # Try again with a new set of planes
    raise RuntimeError("Failed to generate a valid partial voxel cut after reducing planes.")


def process_voxel_files(input_dir: Path, output_dir: Path, resolution: int, num_planes: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of files to process
    files = list(input_dir.glob("*.npy"))
    
    # Process files with progress bar
    with tqdm(files, desc="Processing") as pbar:
        for file in pbar:
            # Update progress bar description with current filename
            clean_name = clean_filename(file.name)
            pbar.set_description(f"Processing: {clean_name}")
            
            full = np.load(file)
            partial = generate_partial_from_full(full, resolution, num_planes)
            out_file = output_dir / (file.stem + ".npz")
            np.savez_compressed(out_file, complete=full, partial=partial)

def main():
    VOXEL_RESOLUTION = 32
    NUM_PLANES = 5
    COMPLETE_VOXELIZED_DIR = Path(os.getenv(f"VOXELIZED_DATA_DIR_{VOXEL_RESOLUTION}"))
    PARTIAL_VOXELIZED_DIR = Path(os.getenv(f"PARTIAL_DATA_DIR_{VOXEL_RESOLUTION}_PLANE_{NUM_PLANES}"))
    
    print(f"Loading complete voxels from: {COMPLETE_VOXELIZED_DIR}")
    print(f"Saving partial voxels to: {PARTIAL_VOXELIZED_DIR}")
    
    process_voxel_files(COMPLETE_VOXELIZED_DIR, PARTIAL_VOXELIZED_DIR, VOXEL_RESOLUTION, NUM_PLANES)

if __name__ == "__main__":
    main()