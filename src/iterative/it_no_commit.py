from typing import Tuple, List
from tqdm import tqdm
import zipfile
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import os

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
        assert d_model % num_heads == 0
        self.window_size = window_size
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = self.head_dim ** -0.5

    def forward(self, target_embedding, neighbor_embeddings, mask):
        """
        target_embedding:    [B, d_model]
        neighbor_embeddings: [B, ws, ws, ws, d_model]
        mask:                [B, ws, ws, ws] float {0,1}
        """
        B = target_embedding.shape[0]
        ws = self.window_size

        # numeric -> boolean
        mask_bool = (mask > 0.5)

        neighbor_flat = neighbor_embeddings.view(B, ws*ws*ws, self.d_model)
        mask_flat = mask_bool.view(B, ws*ws*ws)

        q = self.q_proj(target_embedding.unsqueeze(1))                         # [B,1,C]
        k = self.k_proj(neighbor_flat)                                         # [B,ws^3,C]
        v = self.v_proj(neighbor_flat)

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)        # [B,h,1,d]
        k = k.view(B, ws*ws*ws, self.num_heads, self.head_dim).transpose(1, 2) # [B,h,ws^3,d]
        v = v.view(B, ws*ws*ws, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale             # [B,h,1,ws^3]

        mask_expanded = mask_flat.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, 1, -1)
        scores = scores.masked_fill(~mask_expanded, float('-inf'))

        # all-false safety: avoid NaNs when no known neighbors
        all_false = ~mask_expanded.any(dim=-1, keepdim=True)                   # [B,h,1,1]
        scores = torch.where(all_false, torch.zeros_like(scores), scores)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)                                    # [B,h,1,d]
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model).squeeze(1)
        return self.out_proj(out)

# ------------------------------
# Voxel transformer layer that uses LocalAttention per voxel
# ------------------------------
class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, window_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.attn = LocalAttention(d_model, num_heads, window_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        self.d_model = d_model

    def forward(self, x, neighborhood_fn, mask_fn):
        """
        x: [B, ws, ws, ws, d_model]  (patch embeddings)
        neighborhood_fn: (grid, dd, hh, ww, ws) -> [B, ws, ws, ws, d_model]
        mask_fn:         (Dp,Hp,Wp, dd,hh,ww, ws) -> [B, ws, ws, ws]  (0/1 numeric; known-mask)
        Returns:
            x_center: [B, d_model]  (updated center embedding only)
        """
        B, Dp, Hp, Wp, C = x.shape
        r = self.window_size // 2
        # target (center) token
        target = x[:, r, r, r, :]  # [B, d_model]

        # neighborhood embeddings and known mask for the center position
        neighbors = neighborhood_fn(x, r, r, r, self.window_size)  # [B, ws, ws, ws, d_model]
        mask      = mask_fn(Dp, Hp, Wp, r, r, r, self.window_size) # [B, ws, ws, ws] (0/1)

        # attention (center queries neighbors)
        tgt = self.norm1(target)
        attn_out = self.attn(tgt, neighbors, mask)                 # [B, d_model]
        target = target + self.dropout(attn_out)

        # ffn on center
        tgt2 = self.norm2(target)
        ffn_out = self.ffn(tgt2)                                   # [B, d_model]
        target = target + ffn_out

        return target  # [B, d_model]


# ------------------------------
# Stack layers
# ------------------------------
class Transformer(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, window_size: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, window_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, neighborhood_fn, mask_fn):
        """
        x: [B, D, H, W, d_model]
        """
        center = None
        for layer in self.layers:
            center = layer(x, neighborhood_fn, mask_fn)
            x = x.clone()
            r = x.shape[1] // 2
            x[:, r, r, r, :] = center
        return center

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
def neighborhood_raw(occ_grid, known_grid, b, d, h, w, window_size):
    """
    occ_grid:   [B,1,D,H,W] float {0,1}
    known_grid: [B,1,D,H,W] float {0,1}

    Returns:
        patch_occ:   [1,1,ws,ws,ws] float {0,1}
        patch_known: [1,  ws,ws,ws] float {0,1}
    """
    _, C, D, H, W = occ_grid.shape
    assert C == 1
    r = window_size // 2

    d0, d1 = max(0, d-r), min(D, d+r+1)
    h0, h1 = max(0, h-r), min(H, h+r+1)
    w0, w1 = max(0, w-r), min(W, w+r+1)

    patch_occ   = occ_grid[b:b+1, :, d0:d1, h0:h1, w0:w1]          # [1,1,dp,hp,wp]
    patch_known = known_grid[b:b+1, :, d0:d1, h0:h1, w0:w1]        # [1,1,dp,hp,wp]

    # pads: (W_left, W_right, H_top, H_bottom, D_front, D_back)
    pad = (
        max(0, r - w),               max(0, (w + r + 1) - W),
        max(0, r - h),               max(0, (h + r + 1) - H),
        max(0, r - d),               max(0, (d + r + 1) - D),
    )

    if any(p > 0 for p in pad):
        patch_occ   = F.pad(patch_occ,   pad, value=0.0)  # unknown outside -> 0 (empty value; but also unknown)
        patch_known = F.pad(patch_known, pad, value=0.0)  # outside is unknown => 0

    # squeeze channel for known to [1,ws,ws,ws]
    patch_known = patch_known[:, 0]
    return patch_occ.contiguous(), patch_known.contiguous()


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
        self.transformer = Transformer(num_layers=num_layers, d_model=d_model,
                                              num_heads=num_heads, window_size=window_size,
                                              dropout=dropout)
        # output head: from d_model to one logit
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, neighbors_patch, known_mask):
        """
        neighbors_patch: [B,1,ws,ws,ws] float {0,1}
        known_mask:      [B,  ws,ws,ws] float {0,1}
        """
        B = neighbors_patch.shape[0]
        ws = self.window_size

        emb = self.input_proj(neighbors_patch)              # [B,C,ws,ws,ws]
        emb = emb.permute(0, 2, 3, 4, 1).contiguous()       # [B,ws,ws,ws,C]
        emb = self.pos_encoding(emb)

        def neighborhood_fn_patch(grid, dd, hh, ww, window_size):
            B2, Dp, Hp, Wp, C = grid.shape
            r = window_size // 2
            d0, d1 = max(0, dd-r), min(Dp, dd+r+1)
            h0, h1 = max(0, hh-r), min(Hp, hh+r+1)
            w0, w1 = max(0, ww-r), min(Wp, ww+r+1)
            patch_local = grid[:, d0:d1, h0:h1, w0:w1, :]   # [B,dp,hp,wp,C]

            pd0 = max(0, r - dd); pd1 = max(0, (dd + r + 1) - Dp)
            ph0 = max(0, r - hh); ph1 = max(0, (hh + r + 1) - Hp)
            pw0 = max(0, r - ww); pw1 = max(0, (ww + r + 1) - Wp)
            if any([pd0,pd1,ph0,ph1,pw0,pw1]):
                tmp = patch_local.permute(0, 4, 1, 2, 3)     # [B,C,d,h,w]
                tmp = F.pad(tmp, (pw0,pw1,ph0,ph1,pd0,pd1))
                patch_local = tmp.permute(0, 2, 3, 4, 1).contiguous()
            return patch_local                               # [B,ws,ws,ws,C]

        def mask_fn_patch(Dp, Hp, Wp, dd, hh, ww, window_size):
            # Use the provided known_mask; it is already the patch’s 0/1 known map (padded=0).
            # Expand to [B, ws, ws, ws] for the attention call.
            return known_mask  # numeric 0/1

        center_emb = self.transformer(emb, neighborhood_fn_patch, mask_fn_patch)  # [B, d_model]

        logits = self.output_head(center_emb)
        return logits

def build_static_patch_batch(
    complete_batch: torch.Tensor,   # [B,1,D,H,W], float {0,1}
    partial_batch: torch.Tensor,    # [B,1,D,H,W], float {0,1}
    window_size: int,
    max_voxels_per_sample: int = 256,
):
    """
    Returns:
      patches: [N,1,ws,ws,ws]
      knowns : [N,  ws,ws,ws]
      labels : [N,1]
    Static context: patches come from 'partial_batch'; known = (partial>0.5).
    """
    device = complete_batch.device
    B = complete_batch.shape[0]

    patches, knowns, labels = [], [], []

    # static training context
    occ_grids   = partial_batch          # [B,1,D,H,W]
    known_grids = (partial_batch > 0.5).float()  # [B,1,D,H,W]

    for b in range(B):
        cands_b = get_voxel_candidates(
            complete_batch[b:b+1], partial_batch[b:b+1], max_voxels=max_voxels_per_sample
        )
        cands_b = sort_voxels(cands_b, complete_batch[b:b+1], window_size)

        # extract patches from the static grids (NO commits here)
        for (_, d, h, w, label) in cands_b:
            patch_occ, patch_known = neighborhood_raw(occ_grids, known_grids, b, d, h, w, window_size)
            patches.append(patch_occ)                          # [1,1,ws,ws,ws]
            knowns.append(patch_known)                         # [1,ws,ws,ws]
            labels.append([float(label)])                      # [[0]] or [[1]]

    if len(patches) == 0:
        return (None, None, None)

    patches = torch.cat(patches, dim=0).to(device)             # [N,1,ws,ws,ws]
    knowns  = torch.cat(knowns,  dim=0).to(device)             # [N,ws,ws,ws]
    labels  = torch.tensor(labels, dtype=torch.float32, device=device)  # [N,1]
    return patches, knowns, labels

def train(
    model: nn.Module,
    train_set,
    val_set=None,
    *,
    num_epochs: int = 20,
    batch_size: int = 8,
    window_size: int = 3,
    max_voxels_per_sample: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    pos_weight: float = 1.0,
    amp: bool = True,
    seed: int = 42,
    progress: bool = True,
):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = None
    if val_set is not None:
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, batches = 0.0, 0

        # progress bar with loss display
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=not progress)

        for complete_batch, partial_batch in pbar:
            # Ensure shapes [B,1,D,H,W]
            if complete_batch.dim() == 4: complete_batch = complete_batch.unsqueeze(1)
            if partial_batch.dim()  == 4: partial_batch  = partial_batch.unsqueeze(1)
            complete_batch = complete_batch.float().to(device, non_blocking=True)
            partial_batch  = partial_batch.float().to(device,  non_blocking=True)

            patches, knowns, labels = build_static_patch_batch(
                complete_batch, partial_batch, window_size, max_voxels_per_sample
            )
            if patches is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logits = model(patches, knowns)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batches += 1

            # Update progress bar text
            avg_loss = running_loss / batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        if batches > 0:
            print(f"Epoch {epoch}: train_loss = {running_loss / batches:.4f}")

        # ---------- optional validation ----------
        if val_loader is not None:
            model.eval()
            val_loss, val_batches = 0.0, 0
            correct, total = 0, 0
            with torch.no_grad():
                for complete_batch, partial_batch in val_loader:
                    if complete_batch.dim() == 4: complete_batch = complete_batch.unsqueeze(1)
                    if partial_batch.dim()  == 4: partial_batch  = partial_batch.unsqueeze(1)
                    complete_batch = complete_batch.float().to(device, non_blocking=True)
                    partial_batch  = partial_batch.float().to(device,  non_blocking=True)

                    patches, knowns, labels = build_static_patch_batch(
                        complete_batch, partial_batch, window_size, max_voxels_per_sample
                    )
                    if patches is None:
                        continue

                    logits = model(patches, knowns)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    val_batches += 1

                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    correct += (preds == labels).sum().item()
                    total   += labels.numel()

            if val_batches > 0:
                print(f"           val_loss   = {val_loss / val_batches:.4f} | val_acc = {correct/total:.4f}")

    return model


# ------------------------------
# Example main (instantiate and run)
# ------------------------------
if __name__ == "__main__":
    zip_path = "/home/rnf14/thesis/3DLLM/data/chunk_data_32_flood_fill_rm_20.zip"
    train_list = "/home/rnf14/thesis/3DLLM/train_list.txt"
    dataset = VoxelDataset(zip_path)
    print(f"Total dataset size: {len(dataset)}")
    train_idx, val_idx, test_idx = [], [], []
    # quick split (use existing split logic in your code if preferred)
    n = len(dataset)
    indices = list(range(n))
    random.Random(42).shuffle(indices)

    DATA_PER = 0.3

    n_trainval = int(n * DATA_PER)
    trainval_indices = indices[:n_trainval]
    test_idx = indices[n_trainval:]
    n_train = int(len(trainval_indices) * DATA_PER)
    train_idx = trainval_indices[:n_train]
    val_idx = trainval_indices[n_train:]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    # === Save list of training file names ===
    npz_files = dataset.data_loader.npz_files
    train_files = [npz_files[i] for i in train_idx]

    os.makedirs(os.path.dirname(train_list), exist_ok=True)
    with open(train_list, "w") as f:
        for name in train_files:
            f.write(name + "\n")
    print(f"Saved training file list to: {train_list}")
    print(f"Training samples: {len(train_idx)} | Validation samples: {len(val_idx)}")
    # ========================================

    WINDOW_SIZE = 5
    D_MODEL = 96
    NUM_HEADS = 6
    NUM_LAYERS = 6
    DROPOUT = 0.1

    model = IterativeVoxelModel(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        window_size=WINDOW_SIZE,
        dropout=DROPOUT
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    EPOCHS = 10
    BATCH_SIZE = 32

    train(model, train_set, val_set, num_epochs=EPOCHS, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)
    MODEL_SAVE_PATH = "./it.pth"
    torch.save({'model_state_dict': model.state_dict()}, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
