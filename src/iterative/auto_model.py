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

# ------------------------------
# Data loader / dataset (unchanged)
# ------------------------------
class VoxelDataLoader:
    """Loads and processes NPZ voxel data directly from a zip file (no extraction)"""
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.zip_file = zipfile.ZipFile(zip_path, 'r')
        self.npz_files = [f for f in self.zip_file.namelist() if f.endswith('.npz')]
        print(f"Found {len(self.npz_files)} total NPZ files in zip: {zip_path}")
        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found in zip file {zip_path}")
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
        # Binarize to 0/1
        complete = (complete > 0).float()
        partial = (partial > 0).float()
        if self.transform:
            complete, partial = self.transform(complete, partial)
        return complete, partial

def create_data_loaders(zip_path, batch_size=1, shuffle=True, num_workers=0, seed=42):
    dataset = VoxelDataset(zip_path)
    print(f"Dataset size: {len(dataset)}")
    # simple split as before
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
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

# ------------------------------
# Positional encoding (fixed for [B, D, H, W, d_model])
# ------------------------------
class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model: int, max_grid_size: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        # stored as (D, H, W, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_grid_size, max_grid_size, max_grid_size, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, D, H, W, d_model]
        B, D, H, W, _ = x.shape
        pos = self.pos_embed[:D, :H, :W, :].unsqueeze(0)  # [1, D, H, W, d_model]
        return x + pos

    def get_encoding(self, D, H, W):
        return self.pos_embed[:D, :H, :W, :]

# ------------------------------
# Local attention (your class, unchanged)
# ------------------------------
class LocalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, window_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        assert d_model % num_heads == 0
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = self.head_dim ** -0.5

    def forward(self, target_embedding, neighbor_embeddings, mask):
        """
        Args:
            target_embedding: [B, d_model] - Embedding of the voxel to predict.
            neighbor_embeddings: [B, ws, ws, ws, d_model] - Embeddings of the neighborhood.
            mask: [B, ws, ws, ws] - Boolean mask (True for known, False for unknown).
        """
        B = target_embedding.shape[0]
        ws = self.window_size

        neighbor_flat = neighbor_embeddings.view(B, ws * ws * ws, self.d_model)
        mask_flat = mask.view(B, ws * ws * ws)  # [B, ws^3]

        # Query from target embedding: [B, 1, d_model]
        q = self.q_proj(target_embedding.unsqueeze(1))  # [B, 1, d_model]
        # Keys and values from neighbors: [B, ws^3, d_model]
        k = self.k_proj(neighbor_flat)
        v = self.v_proj(neighbor_flat)

        # Reshape for multi-head attention
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
        k = k.view(B, ws * ws * ws, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, ws^3, head_dim]
        v = v.view(B, ws * ws * ws, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, ws^3, head_dim]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, 1, ws^3]

        # Apply mask (True means allowed)
        mask_expanded = mask_flat.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, 1, -1)
        scores = scores.masked_fill(~mask_expanded, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, num_heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model).squeeze(1)
        return self.out_proj(out)

# ------------------------------
# Voxel transformer layer that uses LocalAttention per voxel
# ------------------------------
class VoxelTransformerLayer3D(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, window_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = LocalAttention(d_model, num_heads, window_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, neighborhood_fn, mask_fn):
        """
        x: [B, D, H, W, d_model]
        neighborhood_fn: callable(grid, d, h, w, window_size) -> [B, ws, ws, ws, d_model]
        mask_fn: callable(D, H, W, d, h, w, window_size) -> [ws,ws,ws] or [1,ws,ws,ws] or [B,ws,ws,ws]
        """
        B, D, H, W, C = x.shape
        out = torch.zeros_like(x)

        for dd in range(D):
            for hh in range(H):
                for ww in range(W):
                    target = x[:, dd, hh, ww, :]  # [B, d_model]
                    neighbors = neighborhood_fn(x, dd, hh, ww, self.window_size)  # [B, ws, ws, ws, d_model]

                    # === robust mask handling ===
                    mask = mask_fn(D, H, W, dd, hh, ww, self.window_size)
                    # mask can be one of: [ws,ws,ws], [1,ws,ws,ws], [B,ws,ws,ws]
                    if mask.dim() == 3:
                        # [ws,ws,ws] -> [1,ws,ws,ws]
                        mask = mask.unsqueeze(0)
                    if mask.shape[0] == 1 and B > 1:
                        # [1,ws,ws,ws] -> [B,ws,ws,ws]
                        mask = mask.expand(B, -1, -1, -1).contiguous()
                    # now mask is guaranteed to be [B, ws, ws, ws]

                    # attention (per-voxel)
                    tgt_norm = self.norm1(target)
                    attn_out = self.attention(tgt_norm, neighbors, mask)
                    target = target + self.dropout(attn_out)

                    # ffn
                    tgt_norm = self.norm2(target)
                    ffn_out = self.ffn(tgt_norm)
                    target = target + ffn_out

                    out[:, dd, hh, ww, :] = target
        return out

# ------------------------------
# Stack layers
# ------------------------------
class VoxelTransformer3D(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, window_size: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            VoxelTransformerLayer3D(d_model, num_heads, window_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, neighborhood_fn, mask_fn):
        """
        x: [B, D, H, W, d_model]
        """
        for layer in self.layers:
            x = layer(x, neighborhood_fn, mask_fn)
        return x

# ------------------------------
# Helper functions for voxel candidate selection
# ------------------------------
def get_voxel_candidates(complete_grid, partial_grid, max_voxels: int = 256):
    """
    Returns a balanced *sample* of candidate voxels to predict.
    
    Args:
        complete_grid: [B, 1, D, H, W]
        partial_grid: [B, 1, D, H, W]
        max_voxels: maximum number of candidate voxels per batch element.
    Returns:
        list of voxel coords: (b, d, h, w, label)
    """
    B, _, D, H, W = complete_grid.shape
    candidates = []

    for b in range(B):
        filled = ((complete_grid[b,0]==1) & (partial_grid[b,0]==0)).nonzero(as_tuple=False)
        empty  = ((complete_grid[b,0]==0) & (partial_grid[b,0]==0)).nonzero(as_tuple=False)

        # balance between filled and empty
        k = max_voxels // 2
        filled_k = min(len(filled), k)
        empty_k  = min(len(empty),  k)

        if filled_k > 0:
            filled_idx = torch.randperm(len(filled))[:filled_k]
            filled = filled[filled_idx]
        else:
            filled = []

        if empty_k > 0:
            empty_idx = torch.randperm(len(empty))[:empty_k]
            empty = empty[empty_idx]
        else:
            empty = []

        for f in filled:
            candidates.append((b, f[0].item(), f[1].item(), f[2].item(), 1))
        for e in empty:
            candidates.append((b, e[0].item(), e[1].item(), e[2].item(), 0))

    return candidates


def compute_density(grid, d, h, w, window_size):
    """Counts filled neighbors around voxel. grid: [D,H,W]"""
    D, H, W = grid.shape
    r = window_size // 2
    d0, d1 = max(0, d-r), min(D, d+r+1)
    h0, h1 = max(0, h-r), min(H, h+r+1)
    w0, w1 = max(0, w-r), min(W, w+r+1)
    patch = grid[d0:d1, h0:h1, w0:w1]
    return float(patch.sum().item())

def sort_voxels(candidates, complete_grid, window_size):
    """Sorts voxel list by density then by distance to origin."""
    sorted_list = []
    for (b,d,h,w,label) in candidates:
        density = compute_density(complete_grid[b,0], d,h,w, window_size)
        dist = d + h + w  # Manhattan distance to origin
        sorted_list.append(((b,d,h,w,label), density, dist))
    sorted_list.sort(key=lambda x: (-x[1], x[2]))  # high density first, then close to origin
    return [item[0] for item in sorted_list]

# ------------------------------
# neighborhood_raw: returns raw patch (channel-first) and known_mask
# ------------------------------
def neighborhood_raw(grid, b, d, h, w, window_size):
    """
    grid: [B, 1, D, H, W] (binary partial grid with 1=known occupied and 0=unknown/empty)
    returns:
        patch: [1, 1, ws, ws, ws]  (channel-first, ready for Conv3d)
        known_mask: [1, ws, ws, ws] (boolean True where neighbor is known)
    """
    # grid: [B, C, D, H, W] where C==1
    _, C, D, H, W = grid.shape
    assert C == 1
    r = window_size // 2
    d0, d1 = max(0, d - r), min(D, d + r + 1)
    h0, h1 = max(0, h - r), min(H, h + r + 1)
    w0, w1 = max(0, w - r), min(W, w + r + 1)

    patch = grid[b:b+1, :, d0:d1, h0:h1, w0:w1]  # [1,1,d_patch,h_patch,w_patch]

    # pads: (W_left,W_right,H_top,H_bottom,D_front,D_back)
    pad_d0 = max(0, r - d)
    pad_h0 = max(0, r - h)
    pad_w0 = max(0, r - w)
    pad_d1 = max(0, (d + r + 1) - D)
    pad_h1 = max(0, (h + r + 1) - H)
    pad_w1 = max(0, (w + r + 1) - W)
    pad = (pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1)
    if any(p > 0 for p in pad):
        patch = F.pad(patch, pad)  # pad with zeros

    # patch now [1,1,ws,ws,ws]
    # known mask: True if this position was observed as known in partial grid
    known = (patch[:, 0] == 1).to(torch.bool)  # [1, ws, ws, ws]

    return patch.contiguous(), known.contiguous()

# ------------------------------
# IterativeVoxelModel: projection + positional encoding + small transformer on the patch
# ------------------------------
class IterativeVoxelModel(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, num_layers: int = 3,
                 window_size: int = 3, max_grid_size: int = 32, dropout: float = 0.1):
        """
        This model expects per-call patch input:
            neighbors_patch: [B, 1, ws, ws, ws]  (binary observed values or zeros)
            known_mask: [B, ws, ws, ws] (boolean)
        The model:
            - projects neighbors -> d_model via Conv3d(1 -> d_model)
            - permutes to [B, Dp, Hp, Wp, d_model], adds positional encoding
            - runs VoxelTransformer3D on the patch
            - extracts center voxel embedding and returns a scalar logit (per batch)
        """
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.input_proj = nn.Conv3d(1, d_model, kernel_size=1)  # works on [B,1,ws,ws,ws]
        self.pos_encoding = PositionalEncoding3D(d_model, max_grid_size=max_grid_size)
        # use the same transformer as defined above, but it expects patch in [B, D, H, W, d_model]
        self.transformer = VoxelTransformer3D(num_layers=num_layers, d_model=d_model,
                                              num_heads=num_heads, window_size=window_size,
                                              dropout=dropout)
        # output head: from d_model to one logit
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, neighbors_patch, known_mask):
        """
        neighbors_patch: [B, 1, ws, ws, ws]
        known_mask: [B, ws, ws, ws] boolean - True where neighbor is observed/known
        Returns:
            logits: [B, 1] (logit for center voxel occupancy)
        """
        B = neighbors_patch.shape[0]
        ws = self.window_size
        assert neighbors_patch.shape[2:] == (ws, ws, ws), f"neighbors size mismatch {neighbors_patch.shape}"
        # project
        emb = self.input_proj(neighbors_patch)  # [B, d_model, ws, ws, ws]
        # permute to [B, D, H, W, d_model]
        emb = emb.permute(0, 2, 3, 4, 1).contiguous()
        # add positional encoding
        emb = self.pos_encoding(emb)  # [B, ws, ws, ws, d_model]

        # define patch-local neighborhood_fn and mask_fn used by VoxelTransformerLayer3D
        def neighborhood_fn_patch(grid, dd, hh, ww, window_size):
            # grid: [B, Dp, Hp, Wp, d_model] where Dp=ws
            B2, Dp, Hp, Wp, C = grid.shape
            r = window_size // 2
            d0, d1 = max(0, dd - r), min(Dp, dd + r + 1)
            h0, h1 = max(0, hh - r), min(Hp, hh + r + 1)
            w0, w1 = max(0, ww - r), min(Wp, ww + r + 1)
            patch_local = grid[:, d0:d1, h0:h1, w0:w1, :]  # may be smaller than ws on boundaries
            # pad if necessary to shape [B, ws, ws, ws, d_model]
            pd0 = max(0, r - dd); pd1 = max(0, (dd + r + 1) - Dp)
            ph0 = max(0, r - hh); ph1 = max(0, (hh + r + 1) - Hp)
            pw0 = max(0, r - ww); pw1 = max(0, (ww + r + 1) - Wp)
            if any([pd0,pd1,ph0,ph1,pw0,pw1]):
                # permute to channel-first temporarily to use F.pad on last 3 dims
                # patch_local: [B, d_patch, h_patch, w_patch, C] -> permute to [B, C, d,h,w]
                tmp = patch_local.permute(0, 4, 1, 2, 3).contiguous()
                pad = (pw0, pw1, ph0, ph1, pd0, pd1)
                tmp = F.pad(tmp, pad)
                patch_local = tmp.permute(0, 2, 3, 4, 1).contiguous()
            # ensure final shape is [B, ws, ws, ws, d_model]
            return patch_local

        def mask_fn_patch(Dp, Hp, Wp, dd, hh, ww, window_size):
            # We'll use the known_mask passed to the outer forward to compute mask per patch position
            # known_mask: [B, ws, ws, ws] - but voxel transformer expects mask independent of batch or same across batch
            # We return per-batch masks in VoxelTransformerLayer3D; it will expand as needed.
            # compute boundaries within patch
            r = window_size // 2
            d0, d1 = max(0, dd - r), min(Dp, dd + r + 1)
            h0, h1 = max(0, hh - r), min(Hp, hh + r + 1)
            w0, w1 = max(0, ww - r), min(Wp, ww + r + 1)
            # Because we already have known_mask for the entire patch, we'll extract the appropriate patch of known_mask
            # BUT VoxelTransformerLayer3D will call mask_fn_patch(Dp,Hp,Wp,dd,hh,ww,ws) which doesn't give batch index.
            # To keep simple, return a mask that will be interpreted per-batch inside VoxelTransformerLayer3D.
            # We'll return the mask for a single (arbitrary) batch element shape [ws,ws,ws], but VoxelTransformerLayer3D
            # will expand to [B, ws, ws, ws].
            # Build a mask with all True (attend all) — the LocalAttention will still receive per-batch mask when it is used.
            # Here, deliver a mask that only marks valid (in-range) positions (True) — padding areas will be False.
            mask = torch.ones((window_size, window_size, window_size), dtype=torch.bool, device=emb.device)
            # pad areas relative to patch edges:
            pd0 = max(0, r - dd); pd1 = max(0, (dd + r + 1) - Dp)
            ph0 = max(0, r - hh); ph1 = max(0, (hh + r + 1) - Hp)
            pw0 = max(0, r - ww); pw1 = max(0, (ww + r + 1) - Wp)
            if pd0 > 0:
                mask[:pd0, :, :] = False
            if pd1 > 0:
                mask[-pd1:, :, :] = False
            if ph0 > 0:
                mask[:, :ph0, :] = False
            if ph1 > 0:
                mask[:, -ph1:, :] = False
            if pw0 > 0:
                mask[:, :, :pw0] = False
            if pw1 > 0:
                mask[:, :, -pw1:] = False
            return mask.unsqueeze(0)  # [1, ws, ws, ws]

        # Now run the transformer on emb patch
        # emb: [B, ws, ws, ws, d_model]
        # we must give neighborhood_fn_patch and mask_fn_patch to transformer
        out_patch = self.transformer(emb, neighborhood_fn_patch, mask_fn_patch)  # [B, ws, ws, ws, d_model]

        # extract center voxel index
        center = ws // 2
        center_emb = out_patch[:, center, center, center, :]  # [B, d_model]
        logits = self.output_head(center_emb)  # [B, 1]
        return logits


def neighborhood_fn(grid, b, d, h, w, window_size):
    """
    Extract a cubic neighborhood around voxel (d,h,w) from batch element b.
    grid: [B, C, D, H, W]
    Returns: [1, C, ws, ws, ws]
    """
    assert grid.dim() == 5, f"Expected 5D grid, got {grid.shape}"
    assert window_size % 2 == 1, "window_size must be odd."
    radius = window_size // 2

    # Pad grid symmetrically
    padded = F.pad(grid[b:b+1], (radius, radius, radius, radius, radius, radius), mode="constant", value=0)

    # shift indices because of padding
    d, h, w = d + radius, h + radius, w + radius

    # slice out the patch
    patch = padded[:, :, d-radius:d+radius+1, h-radius:h+radius+1, w-radius:w+radius+1]
    return patch  # [1, C, ws, ws, ws]


def mask_fn(grid_shape, d, h, w, window_size):
    """
    Create a binary mask for the cubic neighborhood around (d,h,w).
    grid_shape: (D,H,W)
    Returns: [1, ws, ws, ws]
    """
    D, H, W = grid_shape
    assert window_size % 2 == 1, "window_size must be odd."
    radius = window_size // 2

    mask = torch.ones((window_size, window_size, window_size), dtype=torch.float32)

    # Prevent model from "seeing" the target voxel itself
    mask[radius, radius, radius] = 0.0

    # Handle boundaries (mask out-of-range neighbors)
    for dd in range(window_size):
        for hh in range(window_size):
            for ww in range(window_size):
                gd, gh, gw = d + dd - radius, h + hh - radius, w + ww - radius
                if not (0 <= gd < D and 0 <= gh < H and 0 <= gw < W):
                    mask[dd, hh, ww] = 0.0

    return mask.unsqueeze(0)  # [1, ws, ws, ws]


def train_model_voxelwise(
    model: nn.Module,
    train_set,
    val_set,
    num_epochs: int = 50,
    batch_size: int = 1,
    window_size: int = 3,
    lambda_consistency: float = 1.0,
    seed: int = 42
):
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for complete_grid, partial_grid in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            complete_grid = complete_grid.to(device)
            partial_grid = partial_grid.to(device)

            # Ensure 5D [B,1,D,H,W]
            if complete_grid.dim() == 4:
                complete_grid = complete_grid.unsqueeze(1)
            if partial_grid.dim() == 4:
                partial_grid = partial_grid.unsqueeze(1)

            # ---- candidate voxels (implement separately) ----
            candidates = get_voxel_candidates(complete_grid, partial_grid, max_voxels=256)
            candidates = sort_voxels(candidates, complete_grid, window_size)

            if len(candidates) == 0:
                continue

            patches, known_masks, labels = [], [], []

            for (b, d, h, w, label) in candidates:
                patch, known_mask = neighborhood_raw(partial_grid, b, d, h, w, window_size)
                patches.append(patch)
                known_masks.append(known_mask)
                labels.append(label)

            # stack into tensors
            patches = torch.cat(patches, dim=0).to(device)        # [N,1,ws,ws,ws]
            known_masks = torch.cat(known_masks, dim=0).to(device)  # [N,ws,ws,ws]
            labels = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)  # [N,1]

            # ---- forward + loss ----
            logits = model(patches, known_masks)  # [N,1]
            loss = criterion(logits, labels)

            # ---- backward ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

# ------------------------------
# Example main (instantiate and run)
# ------------------------------
if __name__ == "__main__":
    zip_path = "/home/rnf14/thesis/3DLLM/data/chunk_data_32_flood_fill_rm_20.zip"
    dataset = VoxelDataset(zip_path)
    print(f"Total dataset size: {len(dataset)}")
    train_idx, val_idx, test_idx = [], [], []
    # quick split (use existing split logic in your code if preferred)
    n = len(dataset)
    indices = list(range(n))
    random.Random(42).shuffle(indices)
    n_trainval = int(n * 0.8)
    trainval_indices = indices[:n_trainval]
    test_idx = indices[n_trainval:]
    n_train = int(len(trainval_indices) * 0.8)
    train_idx = trainval_indices[:n_train]
    val_idx = trainval_indices[n_train:]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    model = IterativeVoxelModel(
        d_model=96,
        num_heads=6,
        num_layers=3,
        window_size=3,
        max_grid_size=32,
        dropout=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    train_model_voxelwise(model, train_set, val_set, num_epochs=2, batch_size=8, window_size=3)
    MODEL_SAVE_PATH = "./iterative_model.pth"
    torch.save({'model_state_dict': model.state_dict()}, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
