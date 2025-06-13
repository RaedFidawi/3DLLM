import trimesh
from itertools import combinations
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

VOXEL_RESOLUTION = 32
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
OUTPUT_DIR = os.getenv(f"VOXELIZED_DATA_DIR_{VOXEL_RESOLUTION}")

def normalize_and_center_mesh(mesh, resolution, padding_voxels=1):
    """
    Normalize and center mesh with voxel-based padding to prevent edge clipping
    
    Args:
        mesh: trimesh object
        resolution: voxel grid resolution
        padding_voxels: number of voxels to use as padding on each side (default: 1)
    """
    # Center the mesh at origin
    mesh.apply_translation(-mesh.centroid)
    
    # Calculate scale factor with voxel-based padding
    # Reserve padding_voxels on each side, so usable voxels = resolution - 2*padding_voxels
    usable_voxels = resolution - (2 * padding_voxels)
    usable_space = usable_voxels / resolution  # Convert back to unit cube fraction
    
    scale = usable_space / np.max(mesh.extents)
    mesh.apply_scale(scale)
    
    # Translate to center of unit cube (this centers it in the padded space)
    mesh.apply_translation([0.5, 0.5, 0.5])
    
    # Voxelize with the same resolution
    voxelized = mesh.voxelized(pitch=1.0 / resolution)
    
    # Create voxel grid - same size as before
    voxels = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # Fill voxels
    filled = np.round(voxelized.points * resolution).astype(int)
    for x, y, z in filled:
        if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
            voxels[x, y, z] = True
    
    return voxels

def voxels_equal(voxel1, voxel2, tolerance=1e-6):
    """Check if two voxel arrays are effectively identical"""
    if voxel1.dtype != voxel2.dtype:
        # Convert to same type for comparison
        voxel1 = voxel1.astype(float)
        voxel2 = voxel2.astype(float)
    
    if voxel1.dtype == bool:
        return np.array_equal(voxel1, voxel2)
    else:
        return np.allclose(voxel1, voxel2, atol=tolerance)

def rotate_mesh_arbitrary(mesh, axis, degrees):
    """
    Rotate mesh by arbitrary degrees around specified axis
    axis: 0 (x), 1 (y), 2 (z)
    degrees: rotation angle in degrees
    Returns: rotated mesh copy
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
    
    # Create a copy of the mesh and apply rotation
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(np.vstack([
        np.hstack([rotation_matrix, [[0], [0], [0]]]),
        [0, 0, 0, 1]
    ]))
    
    return rotated_mesh

def rotate_voxel_90_degree(voxel, axis, k=1):
    """Keep the original 90-degree rotation function for efficiency"""
    # axis: 0 (x), 1 (y), 2 (z)
    return np.rot90(voxel, k=k, axes=[(1,2), (0,2), (0,1)][axis])

def analyze_symmetries(voxel):
    """Analyze the symmetries of a voxel to provide insights"""
    symmetries = {
        'rotational': [],
        'reflective': []
    }
    
    # Test rotational symmetries (90, 180, 270 degrees on each axis)
    test_angles = [90, 180, 270]
    for axis in range(3):
        for angle in test_angles:
            k = (angle // 90) % 4
            rotated = rotate_voxel_90_degree(voxel, axis, k)
            if voxels_equal(rotated, voxel):
                symmetries['rotational'].append(f"{angle}° around axis {axis}")
    
    # Test reflective symmetries
    for axis in range(3):
        flipped = np.flip(voxel, axis=axis)
        if voxels_equal(flipped, voxel):
            symmetries['reflective'].append(f"axis {axis}")
    
    return symmetries

def save_voxel(name, voxel):
    output_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(output_path, voxel.astype(np.uint8))

def generate_all_augmentations(mesh, resolution):
    """Generate all augmentations from the original mesh while tracking global duplicates"""
    all_variants = []
    global_seen_voxels = []  # Track ALL generated voxels globally
    original_count = 0
    
    # Generate base voxel from original mesh
    base_voxel = normalize_and_center_mesh(mesh.copy(), resolution)
    all_variants.append(("original", base_voxel))
    global_seen_voxels.append(("original", base_voxel))
    
    # Generate flips of original voxel (keep these as voxel operations since they're exact)
    rotation_angles = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    original_count += 1 + 7  # original + 7 possible flips
    
    axes = [0, 1, 2]  # X, Y, Z
    for r in range(1, 4):
        for combo in combinations(axes, r):
            flipped = np.copy(base_voxel)
            for axis in combo:
                flipped = np.flip(flipped, axis=axis)
            
            # Check against ALL previously seen voxels
            is_duplicate = False
            for _, seen_voxel in global_seen_voxels:
                if voxels_equal(flipped, seen_voxel):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                flip_name = "flip" + "".join(str(a) for a in combo)
                all_variants.append((flip_name, flipped))
                global_seen_voxels.append((flip_name, flipped))
    
    # Generate rotations from MESH (not voxel) for perfect detail preservation
    original_count += 33  # 11 angles × 3 axes
    for axis in range(3):  # X=0, Y=1, Z=2 axes
        for angle in rotation_angles:
            if angle % 90 == 0:
                # For 90-degree rotations, still use efficient voxel rotation
                k = (angle // 90) % 4
                rotated = rotate_voxel_90_degree(base_voxel, axis, k)
            else:
                # For arbitrary angles, rotate the original mesh then voxelize
                rotated_mesh = rotate_mesh_arbitrary(mesh, axis, angle)
                rotated = normalize_and_center_mesh(rotated_mesh, resolution)
            
            # Check against ALL previously seen voxels
            is_duplicate = False
            for _, seen_voxel in global_seen_voxels:
                if voxels_equal(rotated, seen_voxel):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                rotation_name = f"rot{axis}_{angle}"
                all_variants.append((rotation_name, rotated))
                global_seen_voxels.append((rotation_name, rotated))
                
                # For this unique rotation, generate flips (on the voxel, since flips are exact)
                original_count += 7  # 7 possible flips per rotation
                for r in range(1, 4):
                    for combo in combinations(axes, r):
                        flipped_rot = np.copy(rotated)
                        for flip_axis in combo:
                            flipped_rot = np.flip(flipped_rot, axis=flip_axis)
                        
                        # Check against ALL previously seen voxels
                        is_duplicate = False
                        for _, seen_voxel in global_seen_voxels:
                            if voxels_equal(flipped_rot, seen_voxel):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            flip_name = "flip" + "".join(str(a) for a in combo)
                            combined_name = f"{rotation_name}_{flip_name}"
                            all_variants.append((combined_name, flipped_rot))
                            global_seen_voxels.append((combined_name, flipped_rot))
    
    return all_variants, original_count

def process_all_stl_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_saved = 0
    total_skipped = 0
    
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.lower().endswith('.stl'):
            file_path = os.path.join(RAW_DATA_DIR, filename)
            try:
                mesh = trimesh.load(file_path)
                base_voxel = normalize_and_center_mesh(mesh.copy(), VOXEL_RESOLUTION)
                base_name = os.path.splitext(filename)[0]
                
                # Analyze symmetries for reporting
                symmetries = analyze_symmetries(base_voxel)
                
                # Generate all unique augmentations (passing mesh instead of voxel)
                all_variants, original_count = generate_all_augmentations(mesh, VOXEL_RESOLUTION)
                
                # Save all unique variants
                for variant_name, variant_voxel in all_variants:
                    if variant_name == "original":
                        save_voxel(base_name, variant_voxel)
                    else:
                        save_voxel(f"{base_name}_{variant_name}", variant_voxel)
                
                saved_count = len(all_variants)
                skipped_count = original_count - saved_count
                total_saved += saved_count
                total_skipped += skipped_count
                
                print(f"Processed {base_name}:")
                print(f"  - Saved: {saved_count} variants")
                print(f"  - Skipped: {skipped_count} duplicates")
                if symmetries['rotational']:
                    print(f"  - Rotational symmetries: {', '.join(symmetries['rotational'])}")
                if symmetries['reflective']:
                    print(f"  - Reflective symmetries: {', '.join(symmetries['reflective'])}")
                print()
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    
    print(f"Summary:")
    print(f"Total variants saved: {total_saved}")
    print(f"Total duplicates skipped: {total_skipped}")
    print(f"Storage efficiency: {total_skipped/(total_saved + total_skipped)*100:.1f}% reduction")

if __name__ == "__main__":
    process_all_stl_files()