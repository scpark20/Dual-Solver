import torch
from .scheduler import CTScheduler

# ==================== EulerSolver (DDIM η=0) ====================
class EulerSolver:
    """x0=(x−σ ε)/α ;  x' = α' x0 + σ' ε."""
    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    @torch.no_grad()
    def sample(self, scheduler: CTScheduler, latents: torch.Tensor, mode: str = "noise") -> torch.Tensor:
        x = latents
        a, s = scheduler.alphas, scheduler.sigmas
        for i in range(scheduler.S - 1):
            ai = a[i].view(1, *([1]*(x.ndim-1))); si = s[i].view(1, *([1]*(x.ndim-1)))
            if mode == "noise":
                eps = scheduler.eps_i(x, i)
                x0  = (x - si * eps) / (ai + self.eps)
            else:  # "data"
                x0  = scheduler.x0_i(x, i)
                eps = (x - ai * x0) / (si + self.eps)
            aj = a[i+1].view(1, *([1]*(x.ndim-1))); sj = s[i+1].view(1, *([1]*(x.ndim-1)))
            x  = aj * x0 + sj * eps
        return x
