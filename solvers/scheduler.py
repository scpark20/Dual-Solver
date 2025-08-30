# solvers/ct_scheduler.py
# -*- coding: utf-8 -*-
import numpy as np
import torch

class CTScheduler:
    """
    Build continuous-time coeffs & CFG predictions directly from Diffusers DiTPipeline:
      - τ grid: 1 → 0 (nodes = steps+1)
      - α_i, σ_i via geometric interpolation of ᾱ(k)=∏(1-β_k)
      - t_int[i] = floor(τ_i * N) for model calls (integer timestep)
      - eps_i(x,i): CFG ε̂ at node i  (learned_sigma=True handled)
      - x0_i(x,i) : x̂0 at node i
    Exposed fields: alphas[S], sigmas[S], t_int[S], S
    """
    def __init__(self, pipe, labels: torch.LongTensor, steps: int = 10,
                 cfg: float = 4.0, null_id: int = 1000, dtype: torch.dtype = torch.float16):
        self.pipe   = pipe
        self.labels = labels
        self.null   = torch.full_like(labels, null_id)
        self.cfg    = float(cfg)
        self.dtype  = dtype
        self.C      = int(pipe.transformer.config.in_channels)  # epsilon channels (learned_sigma-aware)

        # --- betas → log ᾱ(k), k=0..N (loga[0]=0) ---
        s = pipe.scheduler
        betas = s.betas if isinstance(s.betas, torch.Tensor) else torch.as_tensor(s.config.trained_betas)
        b = betas.detach().cpu().numpy().reshape(-1)
        self.N = int(b.size)
        loga = np.zeros(self.N + 1, dtype=np.float64)
        loga[1:] = np.cumsum(np.log(1.0 - b + 1e-12))
        self.loga = torch.as_tensor(loga, device=labels.device, dtype=torch.float64)

        # --- τ grid & coeffs ---
        self.S  = int(steps) + 1
        tau = torch.linspace(1.0, 0.0, self.S, device=labels.device, dtype=torch.float64)  # [S]
        u   = tau * self.N
        k   = torch.floor(u).long().clamp(0, self.N - 1)  # [S]
        f   = (u - k.to(torch.float64))                   # [S]
        logabar = (1. - f) * self.loga[k] + f * self.loga[k + 1]
        a = torch.sqrt(torch.exp(logabar)).to(self.dtype)
        s = torch.sqrt(torch.clamp(1. - a*a, min=0.0)).to(self.dtype)

        self.alphas = a   # [S]
        self.sigmas = s   # [S]
        self.t_int  = k   # [S] for model calls

    @torch.no_grad()
    def eps_i(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Classifier-free guided ε̂ at node i."""
        B = x.size(0)
        tvec = self.t_int[i].expand(B)  # [B] long
        tr = self.pipe.transformer
        out_c = tr(hidden_states=x, timestep=tvec, class_labels=self.labels.to(x.device), return_dict=True).sample
        out_u = tr(hidden_states=x, timestep=tvec, class_labels=self.null.to(x.device),   return_dict=True).sample
        # learned_sigma=True → ε는 in_channels까지만
        eps_c = out_c[:, :self.C].to(x.dtype)
        eps_u = out_u[:, :self.C].to(x.dtype)
        return eps_u + self.cfg * (eps_c - eps_u)

    @torch.no_grad()
    def x0_i(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """x̂0 at node i, computed from ε̂ and (α_i, σ_i)."""
        eps = self.eps_i(x, i)
        ai  = self.alphas[i].view(1, *([1]*(x.ndim-1)))
        si  = self.sigmas[i].view(1, *([1]*(x.ndim-1)))
        return (x - si * eps) / (ai + 1e-12)
