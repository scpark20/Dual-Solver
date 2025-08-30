# dit.py
# -*- coding: utf-8 -*-
# DiT backbone + CTScheduler (from DiTPipeline only)

import numpy as np
import torch
import torch.nn as nn
from diffusers import DiTPipeline
from ..solvers.scheduler import CTScheduler

class DiT(nn.Module):
    """
    Minimal DiT backbone for the runner:
      - __init__(dtype, model_id)
      - get_scheduler(labels, steps, cfg, null_id)  -> CTScheduler
      - get_noise(seeds)                            -> [B,4,32,32]
      - decode_vae(latents, raw_output, pil_output) -> dict
      - attributes: device, C, S, pipe
    """
    def __init__(self, dtype: torch.dtype = torch.bfloat16, model_id: str | None = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mid = model_id or "facebook/DiT-XL-2-256"
        self.pipe = DiTPipeline.from_pretrained(mid, torch_dtype=dtype).to(self.device)
        self.pipe.vae.enable_slicing()
        self.dtype = dtype
        self.C = int(self.pipe.transformer.config.in_channels)   # 4
        self.S = int(self.pipe.transformer.config.sample_size)   # 32 (for 256x256)

    def get_scheduler(self, labels, steps: int = 10, cfg: float = 4.0, null_id: int = 1000) -> CTScheduler:
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, device=self.device, dtype=torch.long)
        return CTScheduler(self.pipe, labels.to(self.device), steps=steps, cfg=cfg, null_id=null_id, dtype=self.dtype)

    @torch.no_grad()
    def get_noise(self, seeds):
        B = len(seeds)
        x = torch.empty(B, self.C, self.S, self.S, device=self.device, dtype=self.dtype)
        for i, sd in enumerate(seeds):
            g = torch.Generator(device=self.device).manual_seed(int(sd))
            x[i] = torch.randn((self.C, self.S, self.S), generator=g, device=self.device, dtype=self.dtype)
        return x

    @torch.no_grad()
    def decode_vae(self, latents: torch.Tensor, raw_output: bool = True, pil_output: bool = True):
        imgs = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample  # [-1,1], [B,3,H,W]
        out = {}
        if raw_output:
            out["raw_output"] = imgs
        if pil_output:
            arr = ((imgs.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            from PIL import Image
            out["pil_output"] = [Image.fromarray(a, mode="RGB") for a in arr]
        return out
