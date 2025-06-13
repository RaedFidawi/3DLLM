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

def create_mask(shape, planes):
    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    coords = np.stack([x, y, z], axis=-1)
    mask = np.zeros(shape, dtype=bool)
    
    for normal, point in planes:
        vec = coords - point
        dot = np.sum(vec * normal, axis=-1)
        cut_mask = dot > 0
        mask |= cut_mask
    
    return ~mask  # voxels on negative side (keep these)

def generate_partial_from_full(full, resolution, num_planes=3, max_retries=100):
    full_voxel_count = np.count_nonzero(full)
    
    for _ in range(max_retries):
        planes = [generate_random_plane(resolution) for _ in range(num_planes)]
        mask = create_mask(full.shape, planes)
        partial_voxel_count = np.count_nonzero(full[mask])
        
        if 0 < partial_voxel_count < full_voxel_count:
            partial = full.copy()
            partial[~mask] = 0
            return partial
    
    raise RuntimeError("Failed to generate a valid partial voxel cut after max retries.")

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
    VOXEL_RESOLUTION = 16
    NUM_PLANES = 3
    COMPLETE_VOXELIZED_DIR = Path(os.getenv(f"VOXELIZED_DATA_DIR_{VOXEL_RESOLUTION}"))
    PARTIAL_VOXELIZED_DIR = Path(os.getenv(f"PARTIAL_DATA_DIR_{VOXEL_RESOLUTION}"))
    
    print(f"Loading complete voxels from: {COMPLETE_VOXELIZED_DIR}")
    print(f"Saving partial voxels to: {PARTIAL_VOXELIZED_DIR}")
    
    process_voxel_files(COMPLETE_VOXELIZED_DIR, PARTIAL_VOXELIZED_DIR, VOXEL_RESOLUTION, NUM_PLANES)

if __name__ == "__main__":
    main()