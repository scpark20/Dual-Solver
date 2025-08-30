# dpm.py
# -*- coding: utf-8 -*-
# DPM-Solver++ 2M (data prediction only) — uses CTScheduler from dit.py

import torch

class DPMSolver:
    """
    DPM-Solver++ 2M:
      - step 0: Euler
      - steps 1..: AB2-style 2M with x̂0 differences
      - optional final_lower_order: last step uses Euler (order-1) fallback
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    @torch.no_grad()
    def sample(self, scheduler, latents: torch.Tensor, final_lower_order: bool = True) -> torch.Tensor:
        x = latents
        a, s, S = scheduler.alphas, scheduler.sigmas, scheduler.S
        lam = (a.clamp_min(1e-12).log() - s.clamp_min(1e-12).log()).to(torch.float32)  # [S]

        # step 0: Euler
        h = lam[1] - lam[0]
        phi1 = torch.expm1(-h).to(x.dtype)
        x0_s = scheduler.x0_i(x, 0)
        x    = (s[1] / s[0]) * x - a[1] * phi1 * x0_s

        x0_prev1 = x0_s
        x0_prev0 = scheduler.x0_i(x, 1)

        for k in range(1, S - 1):
            h0 = lam[k]   - lam[k-1]
            h  = lam[k+1] - lam[k]
            phi1 = torch.expm1(-h).to(x.dtype)
            is_final = (k == S - 2)
            if final_lower_order and is_final:
                x = (s[k+1] / s[k]) * x - a[k+1] * phi1 * x0_prev0
            else:
                r0 = h0 / (h + self.eps)
                D1 = (x0_prev0 - x0_prev1) / (r0 + self.eps)
                x  = (s[k+1] / s[k]) * x - a[k+1] * phi1 * (x0_prev0 + 0.5 * D1)
            if not is_final:
                x0_prev1, x0_prev0 = x0_prev0, scheduler.x0_i(x, k+1)
        return x
