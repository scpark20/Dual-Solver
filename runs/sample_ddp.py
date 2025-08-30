# sample_ddp.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DDP sampling with DiT + (Euler | DPM++ 2M), saves Inception features per sample.
# Files required alongside this:
#   - dit.py   (DiT backbone + CTScheduler)
#   - euler.py (EulerSolver)
#   - dpm.py   (DPMSolver: DPM++ 2M, data-pred)

import os, re, json, math, argparse
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from ..fid.fid import FIDInception

from ..backbones.dit import DiT  # our backbone wrapper
from ..solvers.euler import EulerSolver
from ..solvers.dpm import DPMSolver


# ---------------------- CLI ----------------------
def build_parser():
    p = argparse.ArgumentParser(description="DiT sampling (DDP) → save inception_feature")
    p.add_argument('--out_dir',      type=str, required=True, help="Output folder")
    p.add_argument('--model_id',     type=str, default='facebook/DiT-XL-2-256')
    p.add_argument('--steps',        type=int, default=10, help="NFE (nodes = steps+1 for CTScheduler)")
    p.add_argument('--solver',       type=str, default='euler', choices=['euler','dpmpp2m'])
    p.add_argument('--final_lower',  action='store_true', help="DPM++ 2M: last step Euler fallback")
    p.add_argument('--n_samples',    type=int, default=100)
    p.add_argument('--batch_size',   type=int, default=10)
    p.add_argument('--cfg',          type=float, default=4.0)
    p.add_argument('--seed_offset',  type=int, default=0)
    p.add_argument('--dtype',        type=str, default='bf16', choices=['bf16','fp16','fp32'])
    return p

def resolve_dtype(name: str) -> torch.dtype:
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[name.lower()]

# ---------------------- DDP utils ----------------------
def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
        rank = dist.get_rank()
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0

def barrier():
    if dist.is_available() and dist.is_initialized(): dist.barrier()

def bcast_obj(obj, src=0):
    if not (dist.is_available() and dist.is_initialized()): return obj
    box = [obj]; dist.broadcast_object_list(box, src=src); return box[0]

# ---------------------- helpers ----------------------
def get_save_dir(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    i = max([int(m.group(1)) for d in os.listdir(root) if (m := re.match(r'^run_(\d+)$', d))] or [-1])
    path = os.path.join(root, f"run_{i+1}"); os.makedirs(path, exist_ok=True); return path

def imagenet_labels(n, seed=0):
    per = n // 1000
    arr = np.repeat(np.arange(1000, dtype=np.int64), per)
    rem = n - arr.size
    if rem > 0: arr = np.concatenate([arr, np.random.default_rng(seed).choice(1000, size=rem, replace=True)])
    rng = np.random.default_rng(seed); rng.shuffle(arr); return arr.tolist()

# ---------------------- main ----------------------
def main():
    args = build_parser().parse_args()
    rank, world, local = init_dist()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    # save dir
    out_root = args.out_dir if os.path.isabs(args.out_dir) else os.path.abspath(args.out_dir)
    save_dir = get_save_dir(out_root) if rank == 0 else None
    save_dir = bcast_obj(save_dir, src=0)
    if rank == 0:
        with open(os.path.join(save_dir, "config.json"), "w") as f: json.dump(vars(args), f, indent=2)
    barrier()

    # model
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dtype = resolve_dtype(args.dtype)
    dit = DiT(dtype=dtype, model_id=args.model_id)
    C, S, pipe = dit.C, dit.S, dit.pipe

    # inception
    inception = FIDInception().to(device)

    # shard indices
    all_idx = list(range(args.n_samples))
    my_idx  = all_idx[rank::world]
    n_iters = math.ceil(len(my_idx) / args.batch_size)
    pbar = tqdm(total=n_iters, desc=f"rank{rank}", disable=(rank != 0))

    # global labels once, broadcast to all ranks
    labels_all = imagenet_labels(args.n_samples, seed=0)
    labels_all = bcast_obj(labels_all, src=0)

    ptr = 0
    while ptr < len(my_idx):
        batch_indices = my_idx[ptr: ptr + args.batch_size]; ptr += args.batch_size
        if not batch_indices: break

        labels = torch.as_tensor([labels_all[i] for i in batch_indices], device=dit.device, dtype=torch.long)
        sched = dit.get_scheduler(labels=labels, steps=args.steps, cfg=args.cfg, null_id=1000)

        seeds   = [args.seed_offset + gidx for gidx in batch_indices]
        latents = dit.get_noise(seeds)

        if args.solver == 'euler':
            latents = EulerSolver().sample(sched, latents, mode="noise")
        else:
            latents = DPMSolver().sample(sched, latents, final_lower_order=args.final_lower)

        # decode → inception features
        with torch.no_grad():
            imgs = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample  # [-1,1], [B,3,H,W]
        arr = ((imgs.clamp(-1,1)+1)*127.5).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
        from PIL import Image
        pils = [Image.fromarray(a, mode="RGB") for a in arr]
        feats = inception(pils).detach().cpu()  # [B,2048]

        for j, gidx in enumerate(batch_indices):
            torch.save({'inception_feature': feats[j]}, os.path.join(save_dir, f"{gidx}.pt"))

        pbar.update(1)

    pbar.close()
    barrier()
    if rank == 0:
        print(f"Done. Saved to: {save_dir}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
