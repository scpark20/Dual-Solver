import torch
from .scheduler import CTScheduler

# ==================== DPMSolver (DPM++ 2M, data-pred only) ====================
# Drop-in replacement for the previous DPMSolver (DPM++ 2M, data-pred only)
# Adds: final_lower_order — use a 1st-order (Euler) fallback on the LAST step for stability.

class DPMSolver:
    """
    DPM-Solver++ 2M (data prediction only)
      - step 0: Euler
      - steps 1..: 2M (AB2) with x̂0 differences
      - FINAL STEP (k = S-2): if final_lower_order=True, use Euler (order-1) fallback
    Requires: CTScheduler (provides alphas/sigmas/x0_i).
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    @torch.no_grad()
    def sample(self, scheduler, latents: torch.Tensor, final_lower_order: bool = True) -> torch.Tensor:
        x = latents
        a, s, S = scheduler.alphas, scheduler.sigmas, scheduler.S
        # λ_i from α,σ (for φ1(-h))
        lam = (a.clamp_min(1e-12).log() - s.clamp_min(1e-12).log()).to(torch.float32)  # [S]
        
        # --- step 0: Euler ---
        h = lam[1] - lam[0]
        phi1 = torch.expm1(-h).to(x.dtype)
        x0_s = scheduler.x0_i(x, 0)
        x    = (s[1] / s[0]) * x - a[1] * phi1 * x0_s

        # cache last two x̂0
        x0_prev1 = x0_s
        x0_prev0 = scheduler.x0_i(x, 1)

        if S <= 2:
            return x  # nothing else to do

        # --- steps 1..S-2 ---
        for k in range(1, S - 1):
            h0 = lam[k]   - lam[k-1]
            h  = lam[k+1] - lam[k]
            phi1 = torch.expm1(-h).to(x.dtype)

            is_final = (k == S - 2)
            if final_lower_order and is_final:
                # order-1 fallback (Euler) on the last step:
                # x_{k+1} = (σ_{k+1}/σ_k) x_k − α_{k+1} φ1(-h) · x̂0_k
                x = (s[k+1] / s[k]) * x - a[k+1] * phi1 * x0_prev0
            else:
                # 2M (AB2) update:
                r0 = h0 / (h + self.eps)
                D1 = (x0_prev0 - x0_prev1) / (r0 + self.eps)
                x  = (s[k+1] / s[k]) * x - a[k+1] * phi1 * (x0_prev0 + 0.5 * D1)

            if not is_final:
                # advance x̂0 window for next step
                x0_prev1, x0_prev0 = x0_prev0, scheduler.x0_i(x, k+1)

        return x