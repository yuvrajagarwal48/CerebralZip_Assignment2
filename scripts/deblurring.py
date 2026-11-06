"""
scripts/deblurring.py: Learned Motion Deblurring Module (NAFNet)
==============================================================
Encapsulates learned motion deblurring via Megvii NAFNet (Nonlinear Activation Free Network).
Allows easy substitution with other restorative models (Restormer, MPRNet, etc.).
"""

import os
from typing import Optional
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return y * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors
        N, C, H, W = grad_output.size()
        w = weight.view(1, C, 1, 1)
        g = grad_output * w
        mean_g = g.mean(1, keepdim=True)
        mean_gy = (g * y).mean(1, keepdim=True)
        gx = 1.0 / (var + eps).sqrt() * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=[0, 2, 3]), grad_output.sum(dim=[0, 2, 3]), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, bias=True)
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.conv1(self.norm1(inp))
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        mod_pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        x_pad = F.pad(inp, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        x = self.intro(x_pad)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + x_pad
        return x[:, :, :H, :W]


class DeblurringModule:
    """
    Deblurring Module:
    Encapsulates learned motion deblurring via NAFNet.
    Allows easy substitution with other restorers (e.g., Restormer, MPRNet, or identity bypass).
    """
    def __init__(
        self,
        device: torch.device,
        model: Optional[nn.Module] = None,
        repo_id: str = "mikestealth/nafnet-models",
        filename: str = "NAFNet-GoPro-width32.pth"
    ):
        self.device = device
        if model is not None:
            self.model = model.to(device)
            self.model.eval()
            print("[✓] [DeblurringModule] Using preloaded NAFNet model.")
        else:
            print(f"[*] [DeblurringModule] Initializing NAFNet weights ({filename})...")
            local_cand = os.path.join("models", filename)
            if os.path.isfile(local_cand):
                ckpt_path = local_cand
                print(f"[✓] [DeblurringModule] Loaded local weights from: {ckpt_path}")
            else:
                ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models")
            self.model = NAFNet(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(device)
            self.model.eval()
            print("[✓] [DeblurringModule] NAFNet model loaded.")

    def deblur(self, full_bgr: np.ndarray, max_dim: int = 1024) -> np.ndarray:
        """
        Executes learned motion deblurring on a BGR image with memory protection:
        - Downscales high-resolution scenes (max(H, W) > max_dim) to avoid catastrophic out-of-memory.
        - Scales restored output back to original dimensions preserving aspect ratio and scale.
        - Catches out-of-memory / allocation errors and falls back safely to original image.
        """
        if full_bgr is None or full_bgr.size == 0:
            return full_bgr

        orig_h, orig_w = full_bgr.shape[:2]

        needs_downscale = max(orig_h, orig_w) > max_dim
        if needs_downscale:
            scale = max_dim / float(max(orig_h, orig_w))
            target_w = max(16, int(orig_w * scale))
            target_h = max(16, int(orig_h * scale))
            pad_size = getattr(self.model, "padder_size", 16)
            target_w = ((target_w + pad_size - 1) // pad_size) * pad_size
            target_h = ((target_h + pad_size - 1) // pad_size) * pad_size
            input_scaled = cv2.resize(full_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            input_scaled = full_bgr

        try:
            img_rgb = cv2.cvtColor(input_scaled, cv2.COLOR_BGR2RGB)
            img_f = img_rgb.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(self.device)

            use_amp = (self.device.type == "cuda")
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(tensor)
                else:
                    out = self.model(tensor)

            out_np = out.squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
            restored_rgb = (out_np * 255.0).round().astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

            if needs_downscale:
                return cv2.resize(restored_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            return restored_bgr

        except (RuntimeError, MemoryError) as e:
            print(f"[!] [DeblurringModule] Memory limit exceeded ({e}). Falling back safely to input frame.")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return full_bgr


def load_official_nafnet(device: torch.device) -> NAFNet:
    module = DeblurringModule(device)
    return module.model


def deblur_full_image_nafnet(model: nn.Module, device: torch.device, full_bgr: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    if full_bgr is None or full_bgr.size == 0:
        return full_bgr

    orig_h, orig_w = full_bgr.shape[:2]
    needs_downscale = max(orig_h, orig_w) > max_dim

    if needs_downscale:
        scale = max_dim / float(max(orig_h, orig_w))
        target_w = max(16, int(orig_w * scale))
        target_h = max(16, int(orig_h * scale))
        pad_size = getattr(model, "padder_size", 16)
        target_w = ((target_w + pad_size - 1) // pad_size) * pad_size
        target_h = ((target_h + pad_size - 1) // pad_size) * pad_size
        input_scaled = cv2.resize(full_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        input_scaled = full_bgr

    try:
        img_rgb = cv2.cvtColor(input_scaled, cv2.COLOR_BGR2RGB)
        img_f = img_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
        out_np = out.squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
        restored_rgb = (out_np * 255.0).round().astype(np.uint8)
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        if needs_downscale:
            return cv2.resize(restored_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        return restored_bgr
    except (RuntimeError, MemoryError) as e:
        print(f"[!] [DeblurringModule] Memory limit exceeded ({e}). Returning original image.")
        return full_bgr

