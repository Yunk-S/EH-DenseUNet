import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import cv2

class LiverTumorDataset2D(Dataset):
    def __init__(self, image_dir, label_dir, config, augment=True):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.clip_min, self.clip_max = config['preprocessing']['intensity_clip']
        self.resample_size = config['preprocessing']['resample_size']
        self.augment = augment

        self.index = []
        for i, img_path in enumerate(self.image_paths):
            img = nib.load(img_path).get_fdata()
            num_slices = img.shape[2]
            for z in range(1, num_slices - 1):
                self.index.append((i, z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.index[idx]
        img = nib.load(self.image_paths[vol_idx]).get_fdata()
        lbl = nib.load(self.label_paths[vol_idx]).get_fdata()

        img = np.clip(img, self.clip_min, self.clip_max)
        img = (img - self.clip_min) / (self.clip_max - self.clip_min)

        H, W = self.resample_size
        slices = [img[:, :, slice_idx - 1], img[:, :, slice_idx], img[:, :, slice_idx + 1]]
        img_stack = np.stack([cv2.resize(s, (W, H)) for s in slices], axis=0)

        label = lbl[:, :, slice_idx]
        label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)

        return torch.from_numpy(img_stack).float(), torch.from_numpy(label).long()


class LiverTumorDataset3D(Dataset):
    def __init__(self, image_dir, label_dir, config, training=True):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.clip_min, self.clip_max = config['preprocessing']['intensity_clip']
        self.resample_size = config['preprocessing']['resample_size']
        self.training = training
        # 从配置文件读取深度尺寸，而不是硬编码
        self.depth_size = config['preprocessing'].get('depth_size', 16)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata()
        lbl = nib.load(self.label_paths[idx]).get_fdata()

        img = np.clip(img, self.clip_min, self.clip_max)
        img = (img - self.clip_min) / (self.clip_max - self.clip_min)

        H, W = self.resample_size
        D = img.shape[2]
        img_resized = np.stack([cv2.resize(img[:, :, i], (W, H)) for i in range(D)], axis=0)
        lbl_resized = np.stack([cv2.resize(lbl[:, :, i], (W, H), interpolation=cv2.INTER_NEAREST) for i in range(D)],
                               axis=0)

        # 使用配置文件中的深度尺寸
        D_fixed = self.depth_size
        if D > D_fixed:
            if self.training:
                # 训练时使用中心裁切以提高稳定性，而不是随机裁切
                start = (D - D_fixed) // 2
            else:
                # 推理时总是使用中心裁切
                start = (D - D_fixed) // 2
            img_resized = img_resized[start:start + D_fixed]
            lbl_resized = lbl_resized[start:start + D_fixed]
        elif D < D_fixed:
            # 优化填充策略：使用对称填充而非零填充
            pad_before = (D_fixed - D) // 2
            pad_after = D_fixed - D - pad_before
            pad_width = ((pad_before, pad_after), (0, 0), (0, 0))
            img_resized = np.pad(img_resized, pad_width, mode='edge')  # 使用边缘值填充
            lbl_resized = np.pad(lbl_resized, pad_width, mode='constant', constant_values=0)

        volume = img_resized[np.newaxis, ...]  # [1, D, H, W]
        mask = lbl_resized[np.newaxis, ...]  # [1, D, H, W]

        return torch.from_numpy(volume).float(), torch.from_numpy(mask).long()