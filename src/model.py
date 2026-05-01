"""
model.py - Neural Network Architectures for MRI-to-CT Synthesis

Implements M1-M4 ablation variants: Base, CBAM, SA, and CBAM+SA UNets.
Features dropout injected into convolution blocks for Monte Carlo Dropout uncertainty.

Medical ML Rationale:
Standard UNet provides strong localized feature extraction, but lacks global
context. CBAM adds channel/spatial attention while Self-Attention (SA) captures
non-local structural correlations bridging distant anatomical landmarks.
"""

import math
import logging
from typing import Dict, Type, Any

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)

# =============================================================================
# Core Building Blocks
# =============================================================================

class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => LeakyReLU => Dropout2d) * 2"""
    def __init__(self, in_channels: int, out_channels: int, p_drop: float = 0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=p_drop),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=p_drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(pool))

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class SelfAttention2D(nn.Module):
    """
    Self-Attention for spatial features with memory safety guards.
    HARD CONSTRAINT: Guard H*W > 1024 with chunking/skipping.
    """
    def __init__(self, in_channels: int, safe_mode: bool = True):
        super().__init__()
        self.safe_mode = safe_mode
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        N = H * W
        
        # Hard memory constraint wrapper
        if self.safe_mode and N > 1024:
            # Skip full O(N^2) attention if dimensions are too massive (to prevent OOM on MPS)
            # Alternatively could chunk, but skipping avoids kernel panic on MPS for massive feature maps
            logger.debug(f"SelfAttention2D skipped due to constraint H*W={N} > 1024")
            return x

        q = self.query(x).view(B, -1, N).permute(0, 2, 1)  # [B, N, C/8]
        k = self.key(x).view(B, -1, N)                     # [B, C/8, N]
        v = self.value(x).view(B, -1, N)                   # [B, C, N]

        attn = torch.bmm(q, k)                             # [B, N, N]
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))          # [B, C, N]
        out = out.view(B, C, H, W)
        return self.gamma * out + x


# =============================================================================
# Architectures
# =============================================================================

class UNetBase(nn.Module):
    """M1: Base UNet with MC Dropout"""
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_f: int = 64, p_drop: float = 0.1):
        super().__init__()
        self.p_drop = p_drop
        
        self.down1 = DoubleConv(in_ch, base_f, p_drop)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(base_f, base_f*2, p_drop)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(base_f*2, base_f*4, p_drop)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(base_f*4, base_f*8, p_drop)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bot   = DoubleConv(base_f*8, base_f*16, p_drop)
        
        self.up4   = nn.ConvTranspose2d(base_f*16, base_f*8, 2, stride=2)
        self.conv4 = DoubleConv(base_f*16, base_f*8, p_drop)
        
        self.up3   = nn.ConvTranspose2d(base_f*8, base_f*4, 2, stride=2)
        self.conv3 = DoubleConv(base_f*8, base_f*4, p_drop)
        
        self.up2   = nn.ConvTranspose2d(base_f*4, base_f*2, 2, stride=2)
        self.conv2 = DoubleConv(base_f*4, base_f*2, p_drop)
        
        self.up1   = nn.ConvTranspose2d(base_f*2, base_f, 2, stride=2)
        self.conv1 = DoubleConv(base_f*2, base_f, p_drop)
        
        self.outc  = nn.Conv2d(base_f, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        
        b  = self.bot(self.pool4(d4))
        
        u4 = self.up4(b)
        c4 = self.conv4(torch.cat([u4, d4], dim=1))
        
        u3 = self.up3(c4)
        c3 = self.conv3(torch.cat([u3, d3], dim=1))
        
        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))
        
        return torch.tanh(self.outc(c1))  # [-1, 1] bounds for CT Output
        
    def enable_dropout(self):
        """Forces dropout layers active during eval mode for MCD."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


class UNetCBAM(UNetBase):
    """M2: UNet + CBAM at decoder skips"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_f = 64 # hardcoded for variant builder simplicity unless pulled from kwargs
        if 'base_f' in kwargs: base_f = kwargs['base_f']
        
        self.cbam4 = CBAM(base_f*8)
        self.cbam3 = CBAM(base_f*4)
        self.cbam2 = CBAM(base_f*2)
        self.cbam1 = CBAM(base_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        
        b  = self.bot(self.pool4(d4))
        
        u4 = self.up4(b)
        c4 = self.conv4(torch.cat([u4, self.cbam4(d4)], dim=1))
        
        u3 = self.up3(c4)
        c3 = self.conv3(torch.cat([u3, self.cbam3(d3)], dim=1))
        
        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, self.cbam2(d2)], dim=1))
        
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, self.cbam1(d1)], dim=1))
        
        return torch.tanh(self.outc(c1))


class UNetSA(UNetBase):
    """M3: UNet + Self-Attention at bottleneck"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_f = kwargs.get('base_f', 64)
        # Apply SA at bottleneck where H*W is smallest
        self.sa_bot = SelfAttention2D(base_f*16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        
        b  = self.bot(self.pool4(d4))
        b  = self.sa_bot(b)
        
        u4 = self.up4(b)
        c4 = self.conv4(torch.cat([u4, d4], dim=1))
        
        u3 = self.up3(c4)
        c3 = self.conv3(torch.cat([u3, d3], dim=1))
        
        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))
        
        return torch.tanh(self.outc(c1))


class UNetCBAM_SA(UNetCBAM):
    """M4: Full Model (CBAM + SA)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_f = kwargs.get('base_f', 64)
        self.sa_bot = SelfAttention2D(base_f*16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        
        b  = self.bot(self.pool4(d4))
        b  = self.sa_bot(b)
        
        u4 = self.up4(b)
        c4 = self.conv4(torch.cat([u4, self.cbam4(d4)], dim=1))
        
        u3 = self.up3(c4)
        c3 = self.conv3(torch.cat([u3, self.cbam3(d3)], dim=1))
        
        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, self.cbam2(d2)], dim=1))
        
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, self.cbam1(d1)], dim=1))
        
        return torch.tanh(self.outc(c1))


# =============================================================================
# Registry & Builder
# =============================================================================

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "unet":         UNetBase,
    "unet_cbam":    UNetCBAM,
    "unet_sa":      UNetSA,
    "unet_cbam_sa": UNetCBAM_SA,
}

def build_model(arch: str, in_ch: int = 1, out_ch: int = 1, base_f: int = 64, p_drop: float = 0.1) -> nn.Module:
    """Factory builder for model architectures."""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(MODEL_REGISTRY.keys())}")
        
    return MODEL_REGISTRY[arch](
        in_ch=in_ch,
        out_ch=out_ch,
        base_f=base_f,
        p_drop=p_drop
    )
