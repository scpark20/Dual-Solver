import numpy as np
import torch
import torch.nn as nn

class CTScheduler:
    """Continuous-time coeffs & CFG predictions directly from DiTPipeline.scheduler.betas."""
    def __init__(self, pipe, labels: torch.LongTensor, steps=10, cfg=4.0, null_id=1000, dtype=torch.float16):
        self.pipe   = pipe
        self.labels = labels
        self.null   = torch.full_like(labels, null_id)
        self.cfg    = float(cfg)
        self.dtype  = dtype
        self.C      = int(pipe.transformer.config.in_channels)  # epsilon channels (learned_sigma-aware)

        # betas → log ᾱ(k), k=0..N (loga[0]=0)
        s = pipe.scheduler
        betas = s.betas if isinstance(s.betas, torch.Tensor) else torch.as_tensor(s.config.trained_betas)
        b = betas.detach().cpu().numpy().reshape(-1)
        self.N = int(b.size)
        loga = np.zeros(self.N + 1, dtype=np.float64)
        loga[1:] = np.cumsum(np.log(1.0 - b + 1e-12))
        self.loga = torch.as_tensor(loga, device=labels.device, dtype=torch.float64)

        # τ grid (steps+1 nodes) → α_i, σ_i, t_int[i]
        self.S  = int(steps) + 1
        tau = torch.linspace(1.0, 0.0, self.S, device=labels.device, dtype=torch.float64)  # [S]
        u   = tau * self.N
        k   = torch.floor(u).long().clamp(0, self.N - 1)  # [S]
        f   = (u - k.to(torch.float64))
        logabar = (1. - f) * self.loga[k] + f * self.loga[k + 1]
        a = torch.sqrt(torch.exp(logabar)).to(self.dtype)
        s = torch.sqrt(torch.clamp(1. - a*a, min=0.0)).to(self.dtype)

        self.alphas = a   # [S]
        self.sigmas = s   # [S]
        self.t_int  = k   # [S]

    @torch.no_grad()
    def eps_i(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """CFG ε̂ at node i."""
        B = x.size(0)
        tvec = self.t_int[i].expand(B)  # [B] long
        tr = self.pipe.transformer
        out_c = tr(hidden_states=x, timestep=tvec, class_labels=self.labels.to(x.device), return_dict=True).sample
        out_u = tr(hidden_states=x, timestep=tvec, class_labels=self.null.to(x.device),   return_dict=True).sample
        return (out_u[:, :self.C] + self.cfg * (out_c[:, :self.C] - out_u[:, :self.C])).to(x.dtype)

    @torch.no_grad()
    def x0_i(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """x̂0 at node i, from ε̂ and (α_i, σ_i)."""
        eps = self.eps_i(x, i)
        ai  = self.alphas[i].view(1, *([1]*(x.ndim-1)))
        si  = self.sigmas[i].view(1, *([1]*(x.ndim-1)))
        return (x - si * eps) / (ai + 1e-12)

