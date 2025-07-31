# Voxel Completion Training Script (converted from pos_weight.ipynb)
from typing import Tuple, List
from dataclasses import dataclass
from tqdm import tqdm
from datetime import timedelta
import zipfile
import shutil
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
import json

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
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # [B, 3*C, D, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # Each: [B, C, D, H, W]
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, D, H, W)
        k = k.view(B, self.num_heads, self.head_dim, D, H, W)
        v = v.view(B, self.num_heads, self.head_dim, D, H, W)
        
        # Extract windows efficiently using unfold
        ws = window_size if window_size is not None else self.window_size
        pad = ws // 2
        
        # Pad the tensors
        q_pad = F.pad(q, [pad]*6, mode='constant', value=0)
        k_pad = F.pad(k, [pad]*6, mode='constant', value=0)
        v_pad = F.pad(v, [pad]*6, mode='constant', value=0)
        
        # Extract windows - maintains 3D structure
        def extract_windows(tensor):
            # tensor: [B, heads, head_dim, D_pad, H_pad, W_pad]
            windows = tensor.unfold(3, ws, 1).unfold(4, ws, 1).unfold(5, ws, 1)
            # Result: [B, heads, head_dim, D, H, W, ws, ws, ws]
            return windows.contiguous()
        
        q_win = extract_windows(q_pad)  # [B, heads, head_dim, D, H, W, ws, ws, ws]
        k_win = extract_windows(k_pad)
        v_win = extract_windows(v_pad)
        
        # Get center query for each position
        center = ws // 2
        q_center = q_win[:, :, :, :, :, :, center, center, center]  # [B, heads, head_dim, D, H, W]
        
        # Add positional embeddings to keys (optional enhancement)
        if self.use_pos_embed:
            k_win = k_win + self.pos_embed[:, :, :, :, :, :, :ws, :ws, :ws]
        
        # TRUE 3D ATTENTION COMPUTATION - NO FLATTENING
        # Expand q_center to match window dimensions for element-wise operations
        q_expanded = q_center.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, heads, head_dim, D, H, W, 1, 1, 1]
        q_expanded = q_expanded.expand(-1, -1, -1, -1, -1, -1, ws, ws, ws)  # [B, heads, head_dim, D, H, W, ws, ws, ws]
        
        # Compute attention scores maintaining 3D structure
        # Element-wise multiplication and sum over head_dim dimension
        attn_scores = (q_expanded * k_win).sum(dim=2) * self.scale  # [B, heads, D, H, W, ws, ws, ws]
        
        # Apply softmax over the 3D window
        # We need to flatten only for softmax computation, then reshape back
        original_shape = attn_scores.shape
        attn_scores_flat = attn_scores.view(B, self.num_heads, D, H, W, -1)  # [B, heads, D, H, W, ws³]
        attn_weights = F.softmax(attn_scores_flat, dim=-1)
        attn_weights_3d = attn_weights.view(original_shape)  # [B, heads, D, H, W, ws, ws, ws]
        
        # Apply attention to values while maintaining 3D structure
        # attn_weights_3d: [B, heads, D, H, W, ws, ws, ws]
        # v_win: [B, heads, head_dim, D, H, W, ws, ws, ws]
        # We need to broadcast attn_weights across the head_dim dimension
        attn_weights_expanded = attn_weights_3d.unsqueeze(2)  # [B, heads, 1, D, H, W, ws, ws, ws]
        
        # Weighted sum over the 3D window
        attn_out = (attn_weights_expanded * v_win).sum(dim=(-3, -2, -1))  # [B, heads, head_dim, D, H, W]
        
        # Reshape back to original format
        attn_out = attn_out.view(B, C, D, H, W)
        
        # Final projection
        out = self.proj(attn_out)
        
        # Store attention weights for analysis (reshape for compatibility)
        self.attn_weights = attn_weights  # [B, heads, D, H, W, ws³]
        
        return out

class VoxelTransformerLayer3D(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, window_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.norm1 = nn.GroupNorm(1, d_model)
        self.norm2 = nn.GroupNorm(1, d_model)
        self.attention = SpatialAttention3D(d_model, num_heads, window_size)
        self.ffn = nn.Sequential(
            nn.Conv3d(d_model, d_model * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(d_model * 4, d_model, kernel_size=1),
            nn.Dropout3d(dropout)
        )
        self.dropout = nn.Dropout3d(dropout)
    def forward(self, x, window_size=None):
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, window_size=window_size)
        x = x + self.dropout(attn_out)
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        return x

class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model: int, max_grid_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, d_model, max_grid_size, max_grid_size, max_grid_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    def forward(self, x):
        _, _, D, H, W = x.shape
        return self.pos_embed[:, :, :D, :H, :W]

class VoxelCompletionTransformer(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 8, num_layers: int = 4,
                 max_grid_size: int = 16, window_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.input_proj = nn.Conv3d(1, d_model, kernel_size=1)
        self.pos_encoding = PositionalEncoding3D(d_model, max_grid_size)
        self.layers = nn.ModuleList([
            VoxelTransformerLayer3D(d_model, num_heads, window_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.GroupNorm(1, d_model)
        self.output_proj = nn.Conv3d(d_model, 1, kernel_size=1)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x, window_size=None):
        x = self.input_proj(x)
        x = x + self.pos_encoding(x)
        ws = window_size if window_size is not None else self.window_size
        for layer in self.layers:
            x = layer(x, window_size=ws)
        x = self.output_norm(x)
        x = self.output_proj(x)
        return x

def masked_bce_loss(preds, targets, partial_grid, criterion):
    unknown_mask = (partial_grid == 0)
    masked_loss = criterion(preds * unknown_mask, targets * unknown_mask)
    denom = unknown_mask.float().sum() + 1e-6
    return (masked_loss * unknown_mask.float()).sum() / denom

def consistency_loss(preds, partial_grid):
    known_mask = (partial_grid == 1)
    return F.mse_loss(preds * known_mask, partial_grid * known_mask)

def compute_pos_weight(dataset, sample_size=100):
    total_occupied = 0
    total_empty = 0
    n = min(sample_size, len(dataset))
    for i in range(n):
        complete, _ = dataset[i]
        total_occupied += (complete > 0.5).sum().item()
        total_empty += (complete <= 0.5).sum().item()
    if total_occupied == 0:
        return torch.tensor([1.0])
    return torch.tensor([total_empty / total_occupied])

def train_model(
    model: nn.Module,
    train_set,
    val_set,
    num_epochs: int = 50,
    batch_size: int = 1,
    window_size: int = 3,
    lambda_consistency: float = 1.0,
    seed: int = 42
):
    print(f"Batch Size: {batch_size}, Window Size: {window_size}")
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}")
    pos_weight = compute_pos_weight(train_set)
    print(f"Using pos_weight for BCEWithLogitsLoss: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    total_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_samples_processed = 0
        model.train()
        epoch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True, unit='sample')
        for batch_idx, (complete_grid, partial_grid) in enumerate(epoch_pbar):
            complete_grid = complete_grid.to(device, non_blocking=True)
            partial_grid = partial_grid.to(device, non_blocking=True)
            optimizer.zero_grad()
            if partial_grid.dim() == 4:
                partial_grid = partial_grid.unsqueeze(1)
            if complete_grid.dim() == 4:
                complete_grid = complete_grid.unsqueeze(1)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                preds = model(partial_grid, window_size=window_size)
                masked_loss = masked_bce_loss(preds, complete_grid, partial_grid, criterion)
                cons_loss = consistency_loss(preds, partial_grid)
                total_batch_loss = masked_loss + lambda_consistency * cons_loss
            if scaler is not None:
                scaler.scale(total_batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_batch_loss.backward()
                optimizer.step()
            total_loss += total_batch_loss.item()
            num_samples_processed += 1
            del complete_grid, partial_grid, preds, masked_loss, cons_loss, total_batch_loss
            torch.cuda.empty_cache()
            epoch_pbar.set_postfix({
                'train_loss': f'{total_loss/num_samples_processed:.4f}',
                'samples': num_samples_processed,
                'lr': optimizer.param_groups[0]['lr']
            })
        avg_train_loss = total_loss / max(num_samples_processed, 1)
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for complete_grid, partial_grid in val_loader:
                complete_grid = complete_grid.to(device, non_blocking=True)
                partial_grid = partial_grid.to(device, non_blocking=True)
                if partial_grid.dim() == 4:
                    partial_grid = partial_grid.unsqueeze(1)
                if complete_grid.dim() == 4:
                    complete_grid = complete_grid.unsqueeze(1)
                preds = model(partial_grid, window_size=window_size)
                masked_loss = masked_bce_loss(preds, complete_grid, partial_grid, criterion)
                cons_loss = consistency_loss(preds, partial_grid)
                total_batch_loss = masked_loss + lambda_consistency * cons_loss
                val_loss += total_batch_loss.item()
                val_samples += 1
                del complete_grid, partial_grid, preds, masked_loss, cons_loss, total_batch_loss
                torch.cuda.empty_cache()
        avg_val_loss = val_loss / max(val_samples, 1)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {timedelta(seconds=int(epoch_time))}, Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}, Samples: {num_samples_processed}")
    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {timedelta(seconds=int(total_time))}")
    print(f"Average time per epoch: {timedelta(seconds=int(total_time/num_epochs))}")

# --- Main script ---
if __name__ == "__main__":
    zip_path = "/home/rnf14/thesis/3DLLM/data/chunk_data_16_flood_fill_rm_40.zip"
    test_dir = "/home/rnf14/thesis/3DLLM/data/test_data/test_data_rm_40/"
    test_indices_file = os.path.join(test_dir, "test_indices.json")

    dataset = VoxelDataset(zip_path)
    print(f"Total dataset size: {len(dataset)}")
    train_idx, val_idx, test_idx = split_dataset(dataset, seed=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    # Save test data to test_dir
    os.makedirs(test_dir, exist_ok=True)
    print(f"Saving {len(test_idx)} test samples to {test_dir}")
    with open(test_indices_file, "w") as f:
        json.dump(test_idx, f)

    print(f"Test data saved to {test_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = VoxelCompletionTransformer(
        d_model=96,
        num_heads=6,
        num_layers=6,
        window_size=3,
        dropout=0.1
    )
    # .to(device)
    print("Starting training...")
    torch.cuda.empty_cache()
    # Attempt 1: d_model 48, 
    # nh 6, nl 6, 
    # ws 3,
    # 5 epochs, bs 32 (0.2 final loss, not that good results)

    # Attempt 2: d_model 96,
    # nh 6, nl 6,
    # ws 3,
    # 5 epochs, bs 64 (0.26 final loss, not that good results)

    # Attempt 3: d_model 96
    # nh 6, nl 6,
    # ws 4
    # 5 epochs, bs 32

    # Use DataParallel for multi-GPU training (2 GPUs)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel...")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    train_model(model, train_set, val_set, window_size=3, num_epochs=5, batch_size=8, lambda_consistency=1)
    MODEL_SAVE_PATH = "/home/rnf14/thesis/3DLLM/model/trained_model_rm_40_dmodel_96_ws_3_new_att.pth"
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
