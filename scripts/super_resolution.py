"""
scripts/super_resolution.py: Super-Resolution Module (Real-ESRGAN x4)
=====================================================================
Encapsulates deep super-resolution via Standalone RRDBNet (Real-ESRGAN x4).
Derives 4x native and downsampled 2x from a single forward pass, with
automatic RGB/BGR conversion and Lanczos interpolation fallbacks.
"""

import os
from typing import Tuple, Optional
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return (self.rdb3(self.rdb2(self.rdb1(x))) * 0.2) + x


class StandaloneRRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RealESRGANEngine:
    """
    Super-Resolution Module:
    Encapsulates deep super-resolution via Standalone RRDBNet (Real-ESRGAN x4).
    Produces native 4x or downsampled 2x, with automatic RGB/BGR conversion
    and graceful Lanczos interpolation fallback.
    """
    def __init__(self, device: torch.device, model: Optional[nn.Module] = None):
        self.device = device
        self.available = False
        if model is not None:
            self.model = model.to(device)
            self.model.eval()
            self.available = True
            print("[✓] [SuperResolutionModule] Using preloaded Real-ESRGAN model.")
        else:
            print("[*] [SuperResolutionModule] Initializing Standalone Real-ESRGAN x4...")
            try:
                local_cand = os.path.join("models", "RealESRGAN_x4.pth")
                if os.path.isfile(local_cand):
                    ckpt_path = local_cand
                    print(f"[✓] [SuperResolutionModule] Loaded local weights from: {ckpt_path}")
                else:
                    ckpt_path = hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir="models")
                self.model = StandaloneRRDBNet(in_nc=3, out_nc=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                checkpoint = torch.load(ckpt_path, map_location=device)
                state_dict = checkpoint["params_ema"] if "params_ema" in checkpoint else (checkpoint["params"] if "params" in checkpoint else checkpoint)
                self.model.load_state_dict(state_dict, strict=True)
                self.model.to(device)
                self.model.eval()
                self.available = True
                print("[✓] [SuperResolutionModule] Standalone Real-ESRGAN x4 loaded.")
            except Exception as e:
                print(f"[!] [SuperResolutionModule] Real-ESRGAN initialization failed ({e}).")

    def upscale(self, crop_bgr: np.ndarray, target_scale: float = 4.0) -> Tuple[np.ndarray, str]:
        h, w = crop_bgr.shape[:2]
        if not self.available or max(h, w) > 512:
            tw, th = int(w * target_scale), int(h * target_scale)
            return cv2.resize(crop_bgr, (tw, th), interpolation=cv2.INTER_LANCZOS4), f"LANCZOS_FALLBACK_{target_scale:.1f}X"

        try:
            img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            img_f = img_rgb.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Note: RRDBNet has 23 unnormalized residual dense blocks which overflow in FP16.
            # Run strictly in FP32 for numerical stability.
            with torch.inference_mode():
                sr_tensor = self.model(tensor.float())

            sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
            sr_rgb = (sr_np * 255.0).round().astype(np.uint8)
            sr_bgr_native = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)

            if abs(target_scale - 4.0) < 1e-3:
                return sr_bgr_native, "REAL_ESRGAN_NATIVE_4X"
            else:
                tw, th = int(w * target_scale), int(h * target_scale)
                sr_bgr_down = cv2.resize(sr_bgr_native, (tw, th), interpolation=cv2.INTER_LANCZOS4)
                return sr_bgr_down, f"REAL_ESRGAN_4X_DOWNSAMPLED_TO_{target_scale:.1f}X"
        except Exception as e:
            tw, th = int(w * target_scale), int(h * target_scale)
            return cv2.resize(crop_bgr, (tw, th), interpolation=cv2.INTER_LANCZOS4), f"LANCZOS_FALLBACK_ON_ERROR_{target_scale:.1f}X"

