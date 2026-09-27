#!/usr/bin/env python3
"""
scripts/hold_gpu.py

GPU Reservation and Execution Utility.
Prevents other users/processes on shared clusters from taking GPUs by allocating
dummy VRAM buffers. Supports both static holding and wrapping command execution.

Usage:
  1. Hold specific GPU(s) completely:
     python scripts/hold_gpu.py --gpus 2

  2. Hold all GPUs that currently have at least 15 GB free:
     python scripts/hold_gpu.py --auto --min-free-gb 15

  3. Hold GPU 2 but leave 8 GB free for lightweight jobs:
     python scripts/hold_gpu.py --gpus 2 --leave-free-gb 8

  4. Wrap-and-Run mode (holds GPU until ready, runs command, re-holds upon exit):
     python scripts/hold_gpu.py --gpus 2 --run "python experiments/hybrid_er/orchestration/run_5k.py"
"""

import sys
import time
import signal
import argparse
import subprocess
import torch

def get_free_memory(gpu_id: int) -> float:
    """Returns free VRAM in GiB for a given GPU index."""
    try:
        free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
        return free_bytes / (1024 ** 3)
    except Exception:
        return 0.0

def allocate_holding_tensors(gpu_ids: list, leave_free_gb: float = 1.0) -> list:
    """Allocates tensors on the specified GPUs to hold available VRAM."""
    tensors = []
    for gid in gpu_ids:
        free_gb = get_free_memory(gid)
        target_hold_gb = max(0.0, free_gb - leave_free_gb)
        if target_hold_gb <= 0.5:
            print(f"[GPU {gid}] Only {free_gb:.2f} GB free. Skipping allocation.")
            continue

        # Calculate number of float32 elements (4 bytes per element)
        num_elements = int((target_hold_gb * (1024 ** 3)) / 4)
        print(f"[GPU {gid}] Allocating {target_hold_gb:.2f} GB buffer (leaving ~{leave_free_gb:.1f} GB free)...")
        try:
            device = torch.device(f"cuda:{gid}")
            tensor = torch.empty((num_elements,), dtype=torch.float32, device=device)
            tensors.append(tensor)
            print(f"[GPU {gid}] Successfully locked {target_hold_gb:.2f} GB.")
        except torch.OutOfMemoryError:
            # Fall back to 80% of target if fragmentation prevented full allocation
            fallback_gb = target_hold_gb * 0.8
            num_elements = int((fallback_gb * (1024 ** 3)) / 4)
            print(f"[GPU {gid}] Full allocation failed due to fragmentation; falling back to {fallback_gb:.2f} GB...")
            try:
                device = torch.device(f"cuda:{gid}")
                tensor = torch.empty((num_elements,), dtype=torch.float32, device=device)
                tensors.append(tensor)
                print(f"[GPU {gid}] Locked {fallback_gb:.2f} GB.")
            except Exception as e:
                print(f"[GPU {gid}] Failed to allocate: {e}")
    return tensors

def main():
    parser = argparse.ArgumentParser(description="Hold GPU memory on shared cluster")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU indices to hold (e.g. '2,3')")
    parser.add_argument("--auto", action="store_true", help="Automatically hold all GPUs with sufficient free memory")
    parser.add_argument("--min-free-gb", type=float, default=15.0, help="Min free GB required for --auto selection")
    parser.add_argument("--leave-free-gb", type=float, default=1.0, help="VRAM (GB) to leave unallocated (default: 1.0 GB)")
    parser.add_argument("--run", type=str, default="", help="Command to run while holding GPU before and after execution")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        sys.exit(1)

    num_devices = torch.cuda.device_count()

    target_gpus = []
    if args.gpus:
        target_gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    elif args.auto:
        print(f"Scanning {num_devices} GPUs for at least {args.min_free_gb} GB free memory...")
        for i in range(num_devices):
            free_gb = get_free_memory(i)
            print(f"  GPU {i}: {free_gb:.2f} GB free")
            if free_gb >= args.min_free_gb:
                target_gpus.append(i)
    else:
        print("Please specify either --gpus <list> or --auto. Example: python scripts/hold_gpu.py --gpus 2")
        sys.exit(1)

    if not target_gpus:
        print("No eligible GPUs found to hold.")
        sys.exit(1)

    print(f"Holding GPU(s): {target_gpus}")

    # If --run is supplied, we allocate hold buffer, release it, run the command, and re-allocate hold buffer
    if args.run:
        print(f"Holding GPUs {target_gpus} until command launch...")
        tensors = allocate_holding_tensors(target_gpus, leave_free_gb=args.leave_free_gb)
        print("\nReleasing holding buffers to execute command...")
        del tensors
        torch.cuda.empty_cache()
        time.sleep(1)

        print(f"Executing: {args.run}\n" + "=" * 60)
        ret = subprocess.run(args.run, shell=True)
        print("=" * 60 + f"\nCommand completed with exit code {ret.returncode}. Re-allocating GPU hold...")
        tensors = allocate_holding_tensors(target_gpus, leave_free_gb=args.leave_free_gb)

    else:
        tensors = allocate_holding_tensors(target_gpus, leave_free_gb=args.leave_free_gb)
        if not tensors:
            print("Failed to lock any GPU memory.")
            sys.exit(1)

    print(f"\n[ACTIVE] Holding memory on GPUs {target_gpus}. Press Ctrl+C to release and exit.")

    def handle_exit(sig, frame):
        print("\nReleasing all GPU memory allocations...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    while True:
        time.sleep(5)

if __name__ == "__main__":
    main()
