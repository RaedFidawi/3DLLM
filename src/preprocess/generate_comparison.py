#!/usr/bin/env python3
"""
Script to compare images from RES_16 and RES_32 folders side by side.
Reads matching images from both folders and creates comparison images.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

def create_comparison_image(img16_path, img32_path, output_path, filename):
    """Create a side-by-side comparison of two images."""
    try:
        # Open both images
        img16 = Image.open(img16_path)
        img32 = Image.open(img32_path)
        
        # Get dimensions
        w16, h16 = img16.size
        w32, h32 = img32.size
        
        # Calculate the size for the comparison image
        max_height = max(h16, h32)
        total_width = w16 + w32 + 60  # 60 pixels for spacing and labels
        
        # Create new image with white background
        comparison = Image.new('RGB', (total_width, max_height + 60), 'white')
        
        # Paste images side by side
        comparison.paste(img16, (10, 30))
        comparison.paste(img32, (w16 + 50, 30))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
            
        # Add labels
        draw.text((10 + w16//2 - 20, 5), "RES_16", fill='black', font=font)
        draw.text((w16 + 50 + w32//2 - 20, 5), "RES_32", fill='black', font=font)
        draw.text((total_width//2 - 50, max_height + 35), re.sub(r'_(rot|flip)\d*.*$', '', filename), fill='black', font=font)
        
        # Save comparison
        comparison.save(output_path)
        print(f"Created comparison: {filename}")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

def main():
    # Define paths
    res16_path = Path("../../visualize/RES_16")
    res32_path = Path("../../visualize/RES_32")
    output_path = Path("../../compare_16_32")
    
    # Check if input directories exist
    if not res16_path.exists():
        print(f"Error: Directory {res16_path} does not exist")
        return
    
    if not res32_path.exists():
        print(f"Error: Directory {res32_path} does not exist")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified: {output_path}")
    
    # Get sorted list of image files from both directories
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    res16_files = sorted([f for f in res16_path.iterdir() 
                         if f.suffix.lower() in image_extensions])
    res32_files = sorted([f for f in res32_path.iterdir() 
                         if f.suffix.lower() in image_extensions])

    if len(res16_files) != len(res32_files):
        print(f"Warning: Different number of files - RES_16: {len(res16_files)}, RES_32: {len(res32_files)}")
    
    # Process by index
    num_files = min(len(res16_files), len(res32_files))
    print(f"Processing {num_files} image pairs")
    
    for i in range(num_files):
        img16_path = res16_files[i]
        img32_path = res32_files[i]
        
        # Create output filename
        name_part = img16_path.stem
        output_filename = f"compare_{i:03d}_{name_part}.png"
        output_file_path = output_path / output_filename
        
        # Create comparison
        create_comparison_image(img16_path, img32_path, output_file_path, img16_path.name)
    
    print(f"\nComparison complete! Check {output_path} for results.")

if __name__ == "__main__":
    main()