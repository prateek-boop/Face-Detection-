"""
Performance profiler — identifies bottlenecks in the detection pipeline.

Uses torch.profiler to measure GPU time per operator, data loader vs forward
pass ratio, and latency across batch sizes.  Outputs Chrome trace JSON.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Model profiling
# ---------------------------------------------------------------------------

def profile_model(cfg: dict, checkpoint_path: str, output_dir: str = "profiler_output"):
    """Profile model forward pass and report top operators."""
    os.makedirs(output_dir, exist_ok=True)

    from training.train import FaceDetectionModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaceDetectionModel(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    img_size = cfg["model"]["image_size"]
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)

    # Profile
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("model_forward"):
            for _ in range(20):
                with torch.no_grad():
                    model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()

    # Print top operators
    print("\n" + "=" * 70)
    print("Top 10 operators by total time:")
    print("=" * 70)
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))

    # Memory summary
    if device.type == "cuda":
        print("\n" + "=" * 70)
        print("Top 10 operators by GPU memory:")
        print("=" * 70)
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    # Save Chrome trace
    trace_path = os.path.join(output_dir, "trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\n📁 Chrome trace saved to {trace_path}")
    print(f"   Open chrome://tracing and load this file to visualise\n")


# ---------------------------------------------------------------------------
# Batch size latency sweep
# ---------------------------------------------------------------------------

def batch_size_sweep(cfg: dict, checkpoint_path: str,
                     batch_sizes: list = None):
    """Measure latency at different batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]

    from training.train import FaceDetectionModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaceDetectionModel(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    img_size = cfg["model"]["image_size"]

    print("\n" + "=" * 50)
    print("Batch Size Latency Sweep")
    print("=" * 50)
    print(f"{'Batch':>6} | {'Latency (ms)':>14} | {'Throughput':>12}")
    print("-" * 50)

    for bs in batch_sizes:
        try:
            dummy = torch.randn(bs, 3, img_size, img_size, device=device)

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Timed
            times = []
            for _ in range(30):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            mean_ms = np.mean(times) * 1000
            throughput = bs / (np.mean(times))
            print(f"{bs:>6} | {mean_ms:>11.2f} ms | {throughput:>8.1f} img/s")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{bs:>6} | {'OOM':>14} | {'—':>12}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

    print()


# ---------------------------------------------------------------------------
# Data loader profiling
# ---------------------------------------------------------------------------

def profile_dataloader(cfg: dict, num_batches: int = 50):
    """Measure data loading time vs model forward time ratio."""
    from data.dataset import WIDERFaceDataset, collate_fn
    from data.augment import apply_augmentations
    from torch.utils.data import DataLoader

    paths = cfg["paths"]
    train_cfg = cfg["train"]
    aug_cfg = cfg.get("augment", {})

    transform = lambda img, boxes: apply_augmentations(img, boxes, aug_cfg)

    dataset = WIDERFaceDataset(
        images_root=paths["train_images"],
        ann_path=paths["train_ann"],
        img_size=cfg["model"]["image_size"],
        stride=32,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    data_times = []
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        data_times.append(time.perf_counter() - t0)
        if i >= num_batches:
            break
        t0 = time.perf_counter()

    avg_data_ms = np.mean(data_times) * 1000

    print("\n" + "=" * 50)
    print("Data Loader Profiling")
    print("=" * 50)
    print(f"  Average batch loading time: {avg_data_ms:.2f} ms")
    print(f"  Batches measured: {len(data_times)}")

    if avg_data_ms > 100:
        print("  ⚠️  Data loading is likely the bottleneck!")
        print("      Consider: more workers, pre-cached data, faster storage")
    else:
        print("  ✅ Data loading is not the bottleneck")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile face detection model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="profiler_output")
    parser.add_argument("--batch-sweep", action="store_true",
                        help="Run batch size latency sweep")
    parser.add_argument("--dataloader", action="store_true",
                        help="Profile data loader performance")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Model profiling
    profile_model(cfg, args.checkpoint, args.output_dir)

    # Batch sweep
    if args.batch_sweep:
        batch_size_sweep(cfg, args.checkpoint)

    # Data loader
    if args.dataloader:
        profile_dataloader(cfg)


if __name__ == "__main__":
    main()
