from typing import Tuple, List
from dataclasses import dataclass
from tqdm import tqdm
from datetime import timedelta
import zipfile
import shutil
import tempfile
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os
import glob
from torch.utils.data import Subset, DataLoader
from vis_helper import *
PRINT_LOGS = True

class VoxelDataLoader:
    """Loads and processes NPZ voxel data directly from a zip file (no extraction)"""
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.zip_file = zipfile.ZipFile(zip_path, 'r')
        # List all .npz files in the zip
        self.npz_files = [f for f in self.zip_file.namelist() if f.endswith('.npz')]
        print(f"Found {len(self.npz_files)} total NPZ files in zip: {zip_path}")
        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found in zip file {zip_path}")
        # remove shuffling for consistency in data transfer (HPC and LOCAL machine)
        # random.shuffle(self.npz_files)
        self.npz_files.sort()

    def __del__(self):
        try:
            self.zip_file.close()
        except Exception:
            pass
    def load_single_file(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # file_path is the internal path in the zip
        with self.zip_file.open(file_path) as f:
            data = np.load(f)
            if 'complete' not in data or 'partial' not in data:
                raise ValueError(f"NPZ file {file_path} must contain both 'complete' and 'partial' arrays")
            complete = torch.from_numpy(data['complete']).float()
            partial = torch.from_numpy(data['partial']).float()
            if complete.shape != partial.shape:
                raise ValueError(f"Shape mismatch in {file_path}: complete {complete.shape} vs partial {partial.shape}")
            return complete, partial
    def get_all_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        all_data = []
        for file_path in self.npz_files:
            complete, partial = self.load_single_file(file_path)
            all_data.append((complete, partial))
        return all_data
    def get_voxel_grids(self, index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if index >= len(self.npz_files):
            raise IndexError(f"Index {index} out of range. Only {len(self.npz_files)} files available.")
        return self.load_single_file(self.npz_files[index])

class VoxelDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for voxel completion"""
    def __init__(self, zip_path: str, transform=None):
        self.data_loader = VoxelDataLoader(zip_path)
        self.transform = transform
    def __len__(self):
        return len(self.data_loader.npz_files)
    def __getitem__(self, idx):
        complete, partial = self.data_loader.get_voxel_grids(idx)
        complete = (complete > 0).float()
        partial = (partial > 0).float()
        if self.transform:
            complete, partial = self.transform(complete, partial)
        return complete, partial

def create_data_loader(zip_path: str, batch_size: int = 1, shuffle: bool = True, num_workers: int = 0):
    dataset = VoxelDataset(zip_path)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.2, seed=42):
    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_trainval = int(n * 0.8)
    n_test = n - n_trainval
    trainval_indices = indices[:n_trainval]
    test_indices = indices[n_trainval:]
    n_train = int(len(trainval_indices) * 0.8)
    train_indices = trainval_indices[:n_train]
    val_indices = trainval_indices[n_train:]
    return train_indices, val_indices, test_indices

def create_data_loaders(zip_path, batch_size=1, shuffle=True, num_workers=0, seed=42):
    dataset = VoxelDataset(zip_path)
    print(f"Dataset size: {len(dataset)}")
    train_idx, val_idx, test_idx = split_dataset(dataset, seed=seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

class SpatialAttention3D(nn.Module):
    """
    True 3D spatial attention with proper windowing that maintains 3D structure throughout.
    Unlike the original implementation, this version never flattens the 3D windows,
    preserving spatial relationships within each attention window.
    """
    def __init__(self, d_model: int, num_heads: int = 4, window_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0
        
        self.qkv = nn.Conv3d(d_model, d_model * 3, kernel_size=1)
        self.proj = nn.Conv3d(d_model, d_model, kernel_size=1)
        self.scale = self.head_dim ** -0.5
        self.attn_weights = None
        
        # 3D positional embeddings for window positions (optional enhancement)
        self.use_pos_embed = True
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1, self.head_dim, 1, 1, 1, window_size, window_size, window_size)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, window_size=None):
        B, C, D, H, W = x.shape
        
        tensor_to_txt(x, output_path="4. initial_input_attention.txt", max_channels=C, print_logs=PRINT_LOGS)

        # Generate Q, K, V
        qkv = self.qkv(x)  # [B, 3*C, D, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # Each: [B, C, D, H, W]

        tensor_to_txt(qkv, output_path="5. qkv_attention.txt", max_channels=3*C, print_logs=PRINT_LOGS)
        tensor_to_txt(q, output_path="5. q_attention.txt", max_channels=C, print_logs=PRINT_LOGS)
        tensor_to_txt(k, output_path="5. k_attention.txt", max_channels=C, print_logs=PRINT_LOGS)
        tensor_to_txt(v, output_path="5. v_attention.txt", max_channels=C, print_logs=PRINT_LOGS)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, D, H, W)
        k = k.view(B, self.num_heads, self.head_dim, D, H, W)
        v = v.view(B, self.num_heads, self.head_dim, D, H, W)
        
        multi_head_tensor_to_txt(q, '6. q_reshaped_attention.txt', max_heads=self.num_heads, max_channels=self.head_dim, print_logs=PRINT_LOGS)
        multi_head_tensor_to_txt(k, '6. k_reshaped_attention.txt', max_heads=self.num_heads, max_channels=self.head_dim, print_logs=PRINT_LOGS)
        multi_head_tensor_to_txt(q, '6. v_reshaped_attention.txt', max_heads=self.num_heads, max_channels=self.head_dim, print_logs=PRINT_LOGS)

        # Extract windows efficiently using unfold
        ws = window_size if window_size is not None else self.window_size
        pad = ws // 2
        
        # Pad the tensors
        # q_pad = F.pad(q, [pad]*6, mode='constant', value=0)
        k_pad = F.pad(k, [pad]*6, mode='constant', value=0)
        v_pad = F.pad(v, [pad]*6, mode='constant', value=0)
        # multi_head_tensor_to_txt(q_pad, '7. q_padded.txt', max_heads=self.num_heads, max_channels=self.head_dim, print_logs=PRINT_LOGS)
        multi_head_tensor_to_txt(k_pad, '7. k_padded.txt', max_heads=self.num_heads, max_channels=self.head_dim, print_logs=PRINT_LOGS)
        multi_head_tensor_to_txt(v_pad, '7. v_padded.txt', max_heads=self.num_heads, max_channels=self.head_dim, print_logs=PRINT_LOGS)

        # Extract windows - maintains 3D structure
        def extract_windows(tensor):
            # tensor: [B, heads, head_dim, D_pad, H_pad, W_pad]
            windows = tensor.unfold(3, ws, 1).unfold(4, ws, 1).unfold(5, ws, 1)
            # Result: [B, heads, head_dim, D, H, W, ws, ws, ws]
            return windows.contiguous()
        
        # q_win = extract_windows(q_pad)  # [B, heads, head_dim, D, H, W, ws, ws, ws]
        k_win = extract_windows(k_pad)
        v_win = extract_windows(v_pad)

        # multi_head_window_tensor_to_txt(q_win, output_path="8. q_windowed.txt", batch_idx=0, max_heads=self.num_heads, max_head_dim=self.head_dim, print_logs=PRINT_LOGS)
        multi_head_window_tensor_to_txt(k_win, output_path="8. k_windowed.txt", batch_idx=0, max_heads=self.num_heads, max_head_dim=self.head_dim, print_logs=PRINT_LOGS)
        multi_head_window_tensor_to_txt(v_win, output_path="8. v_windowed.txt", batch_idx=0, max_heads=self.num_heads, max_head_dim=self.head_dim, print_logs=PRINT_LOGS)

        # center = ws // 2
        # q_center = q_win[:, :, :, :, :, :, center, center, center]  # [B, heads, head_dim, D, H, W]
        # multi_head_tensor_to_txt(q_center, '9. q_center_attention.txt', print_logs=PRINT_LOGS)

        if self.use_pos_embed:
            k_win = k_win + self.pos_embed[:, :, :, :, :, :, :ws, :ws, :ws]
        

        q_expanded = q.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, heads, head_dim, D, H, W, 1, 1, 1]
        q_expanded = q_expanded.expand(-1, -1, -1, -1, -1, -1, ws, ws, ws)  # [B, heads, head_dim, D, H, W, ws, ws, ws]
        
        multi_head_window_tensor_to_txt(q_expanded, output_path="9. q_expanded_attention.txt", batch_idx=0, print_logs=PRINT_LOGS)

        # Compute attention scores maintaining 3D structure
        # Element-wise multiplication and sum over head_dim dimension
        attn_scores = (q_expanded * k_win).sum(dim=2) * self.scale  # [B, heads, D, H, W, ws, ws, ws]
        multi_head_single_dim_window_tensor_to_txt(attn_scores, output_path="10. attn_scores.txt", batch_idx=0, max_heads=self.num_heads, max_depth=D, max_height=H, max_width=W, max_ws=ws, print_logs=PRINT_LOGS)

        # Apply softmax over the 3D window
        # We need to flatten only for softmax computation, then reshape back
        original_shape = attn_scores.shape
        attn_scores_flat = attn_scores.view(B, self.num_heads, D, H, W, -1)  # [B, heads, D, H, W, ws³]
        attn_weights = F.softmax(attn_scores_flat, dim=-1)
        attn_weights_3d = attn_weights.view(original_shape)  # [B, heads, D, H, W, ws, ws, ws]
        multi_head_single_dim_window_tensor_to_txt(attn_scores, output_path="11. attn_scores.txt", batch_idx=0, max_heads=self.num_heads, max_depth=D, max_height=H, max_width=W, max_ws=ws, print_logs=PRINT_LOGS)

        # Apply attention to values while maintaining 3D structure
        # attn_weights_3d: [B, heads, D, H, W, ws, ws, ws]
        # v_win: [B, heads, head_dim, D, H, W, ws, ws, ws]
        # We need to broadcast attn_weights across the head_dim dimension
        attn_weights_expanded = attn_weights_3d.unsqueeze(2)  # [B, heads, 1, D, H, W, ws, ws, ws]
        multi_head_window_tensor_to_txt(attn_weights_expanded, output_path="12. attn_weights_expanded.txt", batch_idx=0, max_heads=self.num_heads, max_depth=D, max_height=H, max_width=W, max_ws=ws, print_logs=PRINT_LOGS)

        # Weighted sum over the 3D window
        attn_out = (attn_weights_expanded * v_win).sum(dim=(-3, -2, -1))  # [B, heads, head_dim, D, H, W]
        
        multi_head_tensor_to_txt(attn_out, output_path="13. attn_output.txt", batch_idx=0, max_heads=self.num_heads, max_channels=self.head_dim, print_logs=PRINT_LOGS)
        # Reshape back to original format
        attn_out = attn_out.view(B, C, D, H, W)
        tensor_to_txt(attn_out, output_path="14. attn_output_reshaped.txt", batch_idx=0, max_channels=C, print_logs=PRINT_LOGS)

        # Final projection
        out = self.proj(attn_out)
        tensor_to_txt(out, output_path="15. output_projected.txt", batch_idx=0, max_channels=C, print_logs=PRINT_LOGS)

        # Store attention weights for analysis (reshape for compatibility)
        self.attn_weights = attn_weights  # [B, heads, D, H, W, ws³]


        return out


class VoxelTransformerLayer3D(nn.Module):
    """
    Complete transformer layer with proper normalization and residuals.
    Now supports dynamic window size for attention.
    """
    def __init__(self, d_model: int, num_heads: int = 8, window_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # Layer normalization (adapted for 3D)
        self.norm1 = nn.GroupNorm(1, d_model)  # GroupNorm works better for 3D than LayerNorm
        self.norm2 = nn.GroupNorm(1, d_model)
        
        # Attention
        self.attention = SpatialAttention3D(d_model, num_heads, window_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Conv3d(d_model, d_model * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(d_model * 4, d_model, kernel_size=1),
            nn.Dropout3d(dropout)
        )
        
        self.dropout = nn.Dropout3d(dropout)
        
    def forward(self, x, window_size=None):
        global PRINT_LOGS

        # Attention block with residual connection
        norm_x = self.norm1(x)

        ################# Visualize the input ###################
        tensor_to_txt(norm_x, output_path="3. norm_x_attention.txt", max_channels=self.d_model, print_logs=PRINT_LOGS)
        #############################################################

        attn_out = self.attention(norm_x, window_size=window_size)
        x = x + self.dropout(attn_out)
        tensor_to_txt(x, output_path="16. dropout_x.txt", max_channels=self.d_model, print_logs=PRINT_LOGS)
        # FFN block with residual connection
        norm_x = self.norm2(x)
        tensor_to_txt(norm_x, output_path="17. norm_x.txt", max_channels=self.d_model, print_logs=PRINT_LOGS)
        ffn_out = self.ffn(norm_x)
        tensor_to_txt(ffn_out, output_path="18. ffn_output.txt", max_channels=self.d_model, print_logs=PRINT_LOGS)
        x = x + ffn_out
        tensor_to_txt(x, output_path="19. x_output.txt", max_channels=self.d_model, print_logs=PRINT_LOGS)
        
        PRINT_LOGS = False

        return x


class PositionalEncoding3D(nn.Module):
    """
    Learned 3D positional encoding for voxel grids.
    """
    def __init__(self, d_model: int, max_grid_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        # Learnable positional embedding for each voxel position
        self.pos_embed = nn.Parameter(
            torch.zeros(1, d_model, max_grid_size, max_grid_size, max_grid_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, d_model, D, H, W]
        _, _, D, H, W = x.shape
        return self.pos_embed[:, :, :D, :H, :W]


class VoxelCompletionTransformer(nn.Module):
    """
    Improved 3D transformer for voxel completion.
    Predicts in a single level at the given window size.
    """
    def __init__(self, d_model: int = 64, num_heads: int = 8, num_layers: int = 4,
                 max_grid_size: int = 16, window_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        self.num_layers = num_layers
        self.window_size = window_size
        # Input projection
        self.input_proj = nn.Conv3d(1, d_model, kernel_size=1)
        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(d_model, max_grid_size)
        # Transformer layers
        self.layers = nn.ModuleList([
            VoxelTransformerLayer3D(d_model, num_heads, window_size, dropout)
            for _ in range(num_layers)
        ])
        # Output projection
        self.output_norm = nn.GroupNorm(1, d_model)
        self.output_proj = nn.Conv3d(d_model, 1, kernel_size=1)
        # Initialize weights
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x, window_size=None):
        # x: [B, 1, D, H, W]
        global PRINT_LOGS
        tensor_to_txt(x, output_path="1. input_tensor.txt", max_channels=1, print_logs=PRINT_LOGS)
        x = self.input_proj(x)  # [B, d_model, D, H, W]
        tensor_to_txt(x, output_path="2. input_tensor_projected.txt", max_channels=self.d_model, print_logs=PRINT_LOGS)
        x = x + self.pos_encoding(x)
        tensor_to_txt(x, output_path="3. encoded_tensor.txt", max_channels=self.d_model, print_logs=PRINT_LOGS)

        ws = window_size if window_size is not None else self.window_size
        for layer in self.layers:
            x = layer(x, window_size=ws)
        x = self.output_norm(x)
        x = self.output_proj(x)  # [B, 1, D, H, W]
        
        PRINT_LOGS = True
        tensor_to_txt(x, output_path="20. FINAL_X.txt", max_channels=1, print_logs=PRINT_LOGS)

        return x

MODEL_SAVE_PATH = "../../models/model_rm_40_new_att/trained_model_rm_40_dmodel_96_ws_3_new_att.pth"

import numpy as np
import torch
import glob
import json 

zip_path = "../../chunk_data_16_flood_fill_rm_40.zip"
dataset = VoxelDataset(zip_path)

test_dir = "../../test_data/test_data_rm_40_new_att/"
test_indices_file = os.path.join(test_dir, "test_indices.json")

with open(test_indices_file, "r") as f:
    test_idx = json.load(f)

test_samples = []
for i in range(100):
    complete, partial = dataset[i]
    test_samples.append((complete, partial))

print(f"Loaded {len(test_samples)} test samples from {test_dir}")

# --- Updated test_model to use test_set ---
from torchviz import make_dot
import torch
from torchinfo import summary

def test_model(model_path, test_set, sample_idx=0, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoxelCompletionTransformer(
        d_model=96,        
        num_heads=6,       
        num_layers=6,      
        window_size=3,    
        dropout=0.1
    ).to(device)

    # If using multi-gpu
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Remove 'module.' prefix if present (from DataParallel)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    
    # if using one gpu
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    complete, partial = test_set[sample_idx]

    filled_complete = (complete > 0).sum()
    # Of those, how many are also filled in partial?
    filled_partial = ((complete > 0) & (partial > 0)).sum()
    
    missing_percent = (filled_complete - filled_partial) / filled_complete

    # if missing_percent > 0.3:
    print("Missing Percentage: ", missing_percent)

    ptl = partial
    partial = partial.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D, H, W]
    with torch.no_grad():

        output = model(partial)
        output = torch.sigmoid(output)
        
        # with open("model_summary.txt", "w") as f:
            # f.write(str(summary(model, input_size=(1, 1, 16, 16, 16))))

        # output[0, 0][ptl == 1] = 1.0
        output = output.squeeze().cpu()

    print("Inference complete.")
    print("Partial shape:", partial.shape)
    print("Output shape:", output.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for k in range(output.shape[2]):
                if output[i, j, k] > threshold:
                    output[i, j, k] = 1.0
                else:
                    output[i, j, k] = 0.0
    out_path = "output_voxel.npy"
    complete_path = "complete_voxel.npy"
    partial_path = "partial_voxel.npy"
    np.save(out_path, output.numpy())
    np.save(complete_path, complete)
    np.save(partial_path, ptl)
    print("Sample Index: ", sample_idx)
    print(f"Output saved to {out_path}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_model(
    model_path=MODEL_SAVE_PATH,
    test_set=test_samples,
    sample_idx=
    # 0,
    random.randint(0, 100 - 1),
    threshold=0.5,
    device=device
)
