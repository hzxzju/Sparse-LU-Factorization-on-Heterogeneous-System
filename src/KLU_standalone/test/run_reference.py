#!/usr/bin/env python3
"""Batch CPU KLU reference solves of Ax=ones(n), with timings and solutions."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time


KLU_DIR = Path(__file__).resolve().parents[1]
FIELDS = ["matrix", "status", "exit_code", "n", "nnz", "analyze_s", "factor_s",
          "solve_s", "total_s", "process_wall_s", "residual_l2", "rhs_l2",
          "klu_status", "solution_path", "log_path", "note"]
TIMING = re.compile(r"analyze_s=(\S+) factor_s=(\S+) solve_s=(\S+) total_s=(\S+)")
RESIDUAL = re.compile(r"n=(\d+) nnz=(\d+) \|\|Ax-b\|\|2=(\S+) "
                      r"\|\|b\|\|2=(\S+) status=(-?\d+)")


def remove_file(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def read_metrics(log_path):
    metrics = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIMING.fullmatch(line)
        if match:
            metrics.update(zip(("analyze_s", "factor_s", "solve_s", "total_s"),
                               map(float, match.groups())))
        match = RESIDUAL.fullmatch(line)
        if match:
            n, nnz, residual, rhs, status = match.groups()
            metrics.update(n=int(n), nnz=int(nnz), residual_l2=float(residual),
                           rhs_l2=float(rhs), klu_status=int(status))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=KLU_DIR.parent / "matrix")
    parser.add_argument("--output-dir", type=Path, default=KLU_DIR / "test_result")
    parser.add_argument("--matrix", action="append", metavar="NAME",
                        help="select a filename or stem; repeat to select several matrices")
    parser.add_argument("--timeout", type=float, default=0,
                        help="maximum seconds per matrix; 0 disables timeout")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be finite and nonnegative")
    matrix_dir = args.matrix_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not matrix_dir.is_dir():
        parser.error("matrix directory does not exist: {}".format(matrix_dir))
    matrices = sorted(p for p in matrix_dir.glob("*.mtx") if p.is_file())
    if args.matrix:
        requested = set(args.matrix)
        missing = requested - {name for p in matrices for name in (p.name, p.stem)}
        if missing:
            parser.error("unknown matrices: " + ", ".join(sorted(missing)))
        matrices = [p for p in matrices if p.name in requested or p.stem in requested]
    if not matrices:
        parser.error("no *.mtx matrices selected")

    # Rebuild changed sources before measuring; compilation is never timed.
    subprocess.run(["make", "-C", str(KLU_DIR), "klu_demo"], check=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    failed = 0
    total_s = 0.0
    process_wall_s = 0.0
    batch_started = time.perf_counter()
    print("Matrices: {} | Results: {}".format(len(matrices), output_dir), flush=True)
    with summary_path.open("w", newline="", encoding="utf-8") as summary:
        writer = csv.DictWriter(summary, fieldnames=FIELDS)
        writer.writeheader()
        summary.flush()
        for index, matrix in enumerate(matrices, 1):
            solution = output_dir / (matrix.stem + "_x.txt")
            pending = output_dir / (matrix.stem + "_x.txt.partial")
            log_path = output_dir / (matrix.stem + ".log")
            timing_path = output_dir / (matrix.stem + "_timing.json")
            # A failed rerun must never leave an older solution looking current.
            remove_file(solution)
            remove_file(pending)
            row = dict(matrix=matrix.name, status="ok", exit_code=None,
                       solution_path="", log_path=log_path.name, note="")
            print("[{}/{}] {}".format(index, len(matrices), matrix.name), flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                started = time.perf_counter()
                try:
                    result = subprocess.run(
                        [str(KLU_DIR / "klu_demo"), str(matrix), str(pending)],
                        stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout or None)
                    row["exit_code"] = result.returncode
                    if result.returncode:
                        row.update(status="failed", note="KLU process returned a nonzero exit code")
                except subprocess.TimeoutExpired:
                    row.update(status="timeout", note="exceeded {} seconds".format(args.timeout))
                except OSError as exc:
                    row.update(status="failed", note=str(exc))
                    log.write("\nBatch runner: {}\n".format(exc))
                finally:
                    row["process_wall_s"] = time.perf_counter() - started
            metrics = read_metrics(log_path)
            row.update(metrics)
            if row["status"] == "ok":
                required = ("n", "nnz", "analyze_s", "factor_s", "solve_s", "total_s",
                            "residual_l2", "rhs_l2", "klu_status")
                if any(k not in metrics for k in required):
                    row.update(status="failed", note="incomplete KLU metrics")
                elif (any(not math.isfinite(v) or v < 0 for v in metrics.values())
                      or metrics["klu_status"] != 0):
                    row.update(status="failed", note="nonfinite metrics or unsuccessful KLU status")
                elif not pending.is_file() or pending.stat().st_size == 0:
                    row.update(status="failed", note="solution file missing or empty")
            if row["status"] == "ok":
                pending.replace(solution)
                row["solution_path"] = solution.name
                total_s += metrics["total_s"]
            else:
                failed += 1
            remove_file(pending)
            process_wall_s += row["process_wall_s"]
            # Keep JSON standards-compliant even when KLU reports NaN/Inf.
            record = {k: None if isinstance(v, float) and not math.isfinite(v) else v
                      for k, v in row.items()}
            timing_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n",
                                   encoding="utf-8")
            writer.writerow(row)
            summary.flush()
            print("  {} | total_s={} | process_wall_s={:.6f}".format(
                row["status"], row.get("total_s", ""), row["process_wall_s"]), flush=True)

    totals = dict(matrices=len(matrices), succeeded=len(matrices) - failed, failed=failed,
                  total_s=total_s, process_wall_s=process_wall_s,
                  batch_wall_s=time.perf_counter() - batch_started,
                  timing_definition="total_s = analyze_s + factor_s + solve_s; successful matrices only",
                  rhs="ones(n)", expand_symmetric=False)
    (output_dir / "total_time.json").write_text(json.dumps(totals, indent=2) + "\n",
                                              encoding="utf-8")
    print("Finished: {} ok, {} failed | KLU total: {:.6f} s | {}".format(
        totals["succeeded"], failed, total_s, summary_path), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; completed rows remain in summary.csv.", file=sys.stderr)
        sys.exit(130)
    except (OSError, subprocess.CalledProcessError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        sys.exit(1)
