"""
Sweep VPQ performance over post-processing parameters.

Varies nms_kernel_size and conf_threshold independently, keeping the other fixed
at its default value. Saves results to logs/sweep_vpq_<timestamp>.json.

Usage (inside Docker container):
    uv run scripts/sweep_vpq.py
    uv run scripts/sweep_vpq.py --sweep conf_threshold
    uv run scripts/sweep_vpq.py --sweep nms_kernel_size
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Parameter ranges (edit these to change the sweep) ──────────────────────────
CONF_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
NMS_KERNEL_SIZES = [3, 5, 7, 9, 11, 13, 15, 17, 19]


# Default values used when a parameter is held fixed
DEFAULT_CONF = 0.1
DEFAULT_KERNEL = "null"  # Hydra null → Python None → auto round(350/spatial_extent)
# ───────────────────────────────────────────────────────────────────────────────


def run_validation(conf_threshold: float, nms_kernel_size) -> dict | None:
    """Run a single validation and return the metric dict, or None on failure."""
    cmd = [
        "uv", "run", "tgrip/val.py",
        f"model.postproc_kwargs.conf_threshold={conf_threshold}",
        f"model.postproc_kwargs.nms_kernel_size={nms_kernel_size}",
        "hydra.run.dir=logs/sweep_vpq/${now:%Y-%m-%d_%H-%M-%S}",
    ]
    print(f"\n[sweep] conf_threshold={conf_threshold}  nms_kernel_size={nms_kernel_size}")
    print(f"[sweep] cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

    # Echo output so the user can see Lightning progress
    print(result.stdout, end="")

    # Extract the JSON line printed by val.py
    match = re.search(r"SWEEP_METRICS:(\{.*\})", result.stdout)
    if not match:
        print("[sweep] WARNING: could not find SWEEP_METRICS in output", file=sys.stderr)
        return None

    metrics = json.loads(match.group(1))
    return {
        "pq": metrics.get("val_vpq_metric"),
        "sq": metrics.get("val_sq_metric"),
        "rq": metrics.get("val_rq_metric"),
    }


def print_table(rows: list[dict], param_name: str) -> None:
    header = f"{'  ' + param_name:<20}  {'PQ':>8}  {'SQ':>8}  {'RQ':>8}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for row in rows:
        val = row["param_value"]
        pq = row["pq"]
        sq = row["sq"]
        rq = row["rq"]
        pq_s = f"{pq:.4f}" if pq is not None else "  FAIL"
        sq_s = f"{sq:.4f}" if sq is not None else "  FAIL"
        rq_s = f"{rq:.4f}" if rq is not None else "  FAIL"
        print(f"  {str(val):<18}  {pq_s:>8}  {sq_s:>8}  {rq_s:>8}")
    print("─" * len(header))


def sweep_param(param: str) -> list[dict]:
    results = []
    if param == "conf_threshold":
        values = CONF_THRESHOLDS
        fixed_name, fixed_val = "nms_kernel_size", DEFAULT_KERNEL
    else:
        values = NMS_KERNEL_SIZES
        fixed_name, fixed_val = "conf_threshold", DEFAULT_CONF

    print(f"\n{'='*60}")
    print(f"Sweeping {param}  (fixed {fixed_name}={fixed_val})")
    print(f"{'='*60}")

    for v in values:
        if param == "conf_threshold":
            metrics = run_validation(conf_threshold=v, nms_kernel_size=fixed_val)
        else:
            metrics = run_validation(conf_threshold=fixed_val, nms_kernel_size=v)

        row = {"param_name": param, "param_value": v, **(metrics or {"pq": None, "sq": None, "rq": None})}
        results.append(row)

    print_table(results, param)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        choices=["conf_threshold", "nms_kernel_size", "both"],
        default="both",
        help="Which parameter to sweep (default: both)",
    )
    args = parser.parse_args()

    all_results = {}

    if args.sweep in ("conf_threshold", "both"):
        all_results["conf_threshold"] = sweep_param("conf_threshold")

    if args.sweep in ("nms_kernel_size", "both"):
        all_results["nms_kernel_size"] = sweep_param("nms_kernel_size")

    # Save results
    out_path = Path("logs") / f"sweep_vpq_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[sweep] Results saved to {out_path}")


if __name__ == "__main__":
    main()
