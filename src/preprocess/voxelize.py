import os
import numpy as np
import trimesh
from itertools import combinations
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform

RAW_DATA_DIR = os.path.abspath('../../raw_data')
OUTPUT_DIR = os.path.abspath('../../voxelized')
VOXEL_RESOLUTION = 16

def normalize_and_center_mesh(mesh, resolution):
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / np.max(mesh.extents)
    mesh.apply_scale(scale)
    mesh.apply_translation([0.5, 0.5, 0.5])
    voxelized = mesh.voxelized(pitch=1.0 / resolution)
    voxels = np.zeros((resolution, resolution, resolution), dtype=bool)
    filled = np.round(voxelized.points * resolution).astype(int)
    for x, y, z in filled:
        if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
            voxels[x, y, z] = True
    return voxels

def rotate_voxel_arbitrary(voxel, axis, degrees):
    """
    Rotate voxel by arbitrary degrees around specified axis
    axis: 0 (x), 1 (y), 2 (z)
    degrees: rotation angle in degrees
    """
    # Create rotation matrix
    if axis == 0:  # X-axis
        rotation_vector = [np.radians(degrees), 0, 0]
    elif axis == 1:  # Y-axis
        rotation_vector = [0, np.radians(degrees), 0]
    else:  # Z-axis
        rotation_vector = [0, 0, np.radians(degrees)]
    
    rot = R.from_rotvec(rotation_vector)
    rotation_matrix = rot.as_matrix()
    
    # Get voxel center
    center = np.array(voxel.shape) / 2
    
    # Create coordinate grids
    coords = np.mgrid[0:voxel.shape[0], 0:voxel.shape[1], 0:voxel.shape[2]]
    coords = coords.reshape(3, -1).T
    
    # Center coordinates
    coords_centered = coords - center
    
    # Apply rotation
    coords_rotated = coords_centered @ rotation_matrix.T + center
    
    # Use affine transformation for interpolation
    # We need the inverse transformation matrix
    inv_rotation_matrix = rotation_matrix.T
    
    # Create affine transformation matrix (4x4)
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = inv_rotation_matrix
    
    # Apply translation to account for rotation around center
    offset = center - center @ inv_rotation_matrix.T
    affine_matrix[:3, 3] = offset
    
    # Apply transformation
    rotated_voxel = affine_transform(
        voxel.astype(float), 
        affine_matrix[:3, :3], 
        offset=affine_matrix[:3, 3],
        output_shape=voxel.shape,
        order=1,  # Linear interpolation
        cval=0.0
    )
    
    # Threshold back to boolean
    return rotated_voxel > 0.5

def rotate_voxel_90_degree(voxel, axis, k=1):
    """Keep the original 90-degree rotation function for efficiency"""
    # axis: 0 (x), 1 (y), 2 (z)
    return np.rot90(voxel, k=k, axes=[(1,2), (0,2), (0,1)][axis])

def generate_rotations(voxel):
    """Generate rotations for specified angles"""
    rotations = []
    
    # Define the rotation angles you want
    rotation_angles = [30, 60, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    
    for axis in [0, 1, 2]:  # X, Y, Z axes
        for angle in rotation_angles:
            if angle % 90 == 0 and angle != 360:
                # Use efficient 90-degree rotation for 90, 180, 270
                k = (angle // 90) % 4
                if k != 0:  # Skip 0 degrees (k=0 means no rotation)
                    rotated = rotate_voxel_90_degree(voxel, axis, k)
                    rotations.append((f"rot{axis}_{angle}", rotated))
            elif angle != 360:  # Skip 360 degrees (same as 0)
                # Use arbitrary angle rotation for other angles
                rotated = rotate_voxel_arbitrary(voxel, axis, angle)
                rotations.append((f"rot{axis}_{angle}", rotated))
    
    return rotations

def generate_flips(voxel):
    flips = []
    axes = [0, 1, 2]  # X, Y, Z
    for r in range(1, 4):
        for combo in combinations(axes, r):
            flipped = np.copy(voxel)
            for axis in combo:
                flipped = np.flip(flipped, axis=axis)
            name = "flip" + "".join(str(a) for a in combo)
            flips.append((name, flipped))
    return flips

def save_voxel(name, voxel):
    output_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(output_path, voxel)

def process_all_stl_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.lower().endswith('.stl'):
            file_path = os.path.join(RAW_DATA_DIR, filename)
            try:
                mesh = trimesh.load(file_path)
                base_voxel = normalize_and_center_mesh(mesh, VOXEL_RESOLUTION)
                base_name = os.path.splitext(filename)[0]
                
                # Save original
                save_voxel(base_name, base_voxel)

                # Also flip original
                for flip_name, flipped in generate_flips(base_voxel):
                    save_voxel(f"{base_name}_{flip_name}", flipped)
                
                # Generate and save rotations
                for rot_name, rotated in generate_rotations(base_voxel):
                    save_voxel(f"{base_name}_{rot_name}", rotated)
                    
                    # For each rotation, also apply all flips
                    for flip_name, flipped_rot in generate_flips(rotated):
                        save_voxel(f"{base_name}_{rot_name}_{flip_name}", flipped_rot)

                print(f"Augmented and saved: {base_name}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    process_all_stl_files()