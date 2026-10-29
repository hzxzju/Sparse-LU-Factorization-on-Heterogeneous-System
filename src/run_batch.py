#!/usr/bin/env python3
"""Run lu_cmd sequentially and collect stage timings and residuals."""

import argparse
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import subprocess
import sys


SOURCE_DIR = Path(__file__).resolve().parent
METRICS = {
    "Matrix Row:": ("n", int),
    "Original nonzero:": ("nnz", int),
    "Input/preprocessing time:": ("preprocessing_time", float),
    "Symbolic time:": ("symbolic_time", float),
    "CSR time:": ("csr_time", float),
    "PredictLU time:": ("predictlu_time", float),
    "Leveling time:": ("leveling_time", float),
    "Symbolic preparation time (total):": ("symbolic_preparation_time", float),
    "Numeric LU time (including GPU setup/transfers):": ("numeric_lu_time", float),
    "Triangular solve time (CPU):": ("triangular_solve_time", float),
    "Task 1 total solve time (T_gpu):": ("total_solve_ms", float),
    "Ax-b (1-norm):": ("residual_l1", float),
    "Ax-b (2-norm):": ("residual_l2", float),
    "Ax-b (infinite-norm):": ("residual_linf", float),
}
FIELDS = [
    "matrix", "update_strategy", "scheduling", "perturb", "symbolic_backend", "numeric_sm_padding", "status", "exit_code",
    "n", "nnz", "preprocessing_time", "symbolic_time", "csr_time",
    "predictlu_time", "leveling_time", "symbolic_preparation_time",
    "numeric_lu_time", "triangular_solve_time", "total_solve_ms",
    "residual_l1", "residual_l2", "residual_linf", "log_path", "solution_path", "note",
]


def parse_metrics(log_path):
    metrics = {}
    with log_path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            line = line.strip()
            for label, (field, convert) in METRICS.items():
                if line.startswith(label):
                    try:
                        metrics[field] = convert(line[len(label):].split()[0])
                    except (ValueError, IndexError):
                        pass
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=SOURCE_DIR / "matrix",
                        help="matrix directory, searched recursively for *.mtx")
    parser.add_argument("--solver", type=Path, default=SOURCE_DIR / "lu_cmd")
    parser.add_argument("--output-dir", type=Path,
                        help="new output directory (must not already exist)")
    parser.add_argument("--update-strategy", choices=["right-looking", "left-looking"],
                        default="right-looking")
    parser.add_argument("--scheduling", choices=["level", "synchronization-free"],
                        default="level")
    parser.add_argument("--perturb", action="store_true", help="pass -p to lu_cmd")
    parser.add_argument("--sf-workers", type=int, default=0,
                        help="SF workers; 0 selects automatically, 1 helps diagnose inter-worker waits")
    parser.add_argument("--numeric-sm-padding", action="store_true",
                        help="experimental one-block-per-SM numeric grid with idle padding (LL-SF only; default off)")
    parser.add_argument("--progress", action="store_true",
                        help="flush solver stage diagnostics to solver.log while running")
    parser.add_argument("--residual-tol", type=float, default=1e-6,
                        help="absolute ||Ax-b||_2 target (default: 1e-6)")
    parser.add_argument("--max-refine-iters", default="n",
                        help="maximum correction solves; n = matrix order (default), 0 disables corrections")
    parser.add_argument("--timeout", type=float, default=0,
                        help="maximum seconds per matrix; 0 disables timeout")
    parser.add_argument("--symbolic-backend", choices=["auto", "cpu", "gpu", "gpu-sf"], default="auto",
                        help="symbolic factorization backend (default: auto selects CPU/GPU-SF by input structure)")
    parser.add_argument("--symbolic-sf-workers", type=int, default=0,
                        help="symbolic SF workers, independent of numeric --sf-workers")
    parser.add_argument("--symbolic-batch-cols", type=int, default=128,
                        help="positive graph GPU workspace slot limit (default: 128)")
    parser.add_argument("--symbolic-pool-entries", type=int, default=0)
    parser.add_argument("--symbolic-max-pool-entries", type=int, default=0)
    parser.add_argument("--symbolic-block-size", type=int, choices=[64, 128, 256], default=256)
    parser.add_argument("--symbolic-verify", action="store_true")
    parser.add_argument("--symbolic-no-fallback", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.symbolic_batch_cols <= 4294967295:
        parser.error("--symbolic-batch-cols must be between 1 and 4294967295")
    for name in ("symbolic_sf_workers", "symbolic_pool_entries", "symbolic_max_pool_entries"):
        if not 0 <= getattr(args, name) <= 4294967295:
            parser.error("--{} must be between 0 and 4294967295".format(name.replace("_", "-")))
    if args.symbolic_backend not in ("auto", "gpu-sf") and any((args.symbolic_sf_workers,
            args.symbolic_pool_entries, args.symbolic_max_pool_entries)):
        parser.error("symbolic workers/pool options require --symbolic-backend auto or gpu-sf")
    if args.symbolic_max_pool_entries and args.symbolic_pool_entries > args.symbolic_max_pool_entries:
        parser.error("initial symbolic pool exceeds maximum pool entries")
    if args.symbolic_backend == "cpu" and (args.symbolic_verify or args.symbolic_no_fallback):
        parser.error("symbolic verification/no-fallback requires auto or a GPU backend")
    if not math.isfinite(args.residual_tol) or args.residual_tol <= 0:
        parser.error("--residual-tol must be finite and positive")
    if args.max_refine_iters != "n":
        value = args.max_refine_iters
        if not value or any(ch not in "0123456789" for ch in value) or len(value.lstrip("0")) > 10:
            parser.error("--max-refine-iters must be an unsigned integer or n")
        args.max_refine_iters = int(value.lstrip("0") or "0")
        if args.max_refine_iters > 4294967295:
            parser.error("--max-refine-iters must not exceed 4294967295")
    if not 0 <= args.sf_workers <= 4294967295:
        parser.error("--sf-workers must be between 0 and 4294967295")
    if args.sf_workers and args.scheduling != "synchronization-free":
        parser.error("--sf-workers requires --scheduling synchronization-free")
    if args.numeric_sm_padding and (args.update_strategy != "left-looking" or
                                    args.scheduling != "synchronization-free"):
        parser.error("--numeric-sm-padding requires --update-strategy left-looking --scheduling synchronization-free")
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be finite and nonnegative")
    if args.scheduling == "synchronization-free" and args.update_strategy != "left-looking":
        parser.error("synchronization-free requires --update-strategy left-looking")

    solver = args.solver.resolve()
    matrix_dir = args.matrix_dir.resolve()
    if not solver.is_file() or not os.access(str(solver), os.X_OK):
        parser.error("solver is not executable: {}; build with make -C src MAIN".format(solver))
    if not matrix_dir.is_dir():
        parser.error("matrix directory does not exist: {}".format(matrix_dir))
    matrices = sorted(path for path in matrix_dir.rglob("*.mtx") if path.is_file())
    if not matrices:
        parser.error("no *.mtx files found in {}".format(matrix_dir))
    output_dir = (args.output_dir or SOURCE_DIR / "batch_results" /
                  datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error("cannot create fresh output directory: {}".format(exc))

    env = os.environ.copy()
    hpcc_path = env.get("HPCC_PATH", "/opt/hpcc")
    env["HPCC_PATH"] = hpcc_path
    env["LD_LIBRARY_PATH"] = str(Path(hpcc_path) / "lib") + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    summary = output_dir / "summary.csv"
    failed = 0
    print("Matrices: {} | Results: {}".format(len(matrices), output_dir), flush=True)
    with summary.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        csv_file.flush()
        for index, matrix in enumerate(matrices, 1):
            # Each process has its own cwd so lu_cmd's x.dat never gets overwritten.
            run_dir = output_dir / "{:04d}_{}".format(index, matrix.stem)
            run_dir.mkdir()
            log_path = run_dir / "solver.log"
            solution_path = run_dir / "x.dat"
            command = [str(solver), "-i", str(matrix), "-u", args.update_strategy,
                       "-s", args.scheduling, "--residual-tol", str(args.residual_tol),
                       "--max-refine-iters", str(args.max_refine_iters), "--symbolic-backend", args.symbolic_backend]
            for name in ("symbolic_sf_workers", "symbolic_pool_entries", "symbolic_max_pool_entries"):
                if getattr(args, name):
                    command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
            if args.symbolic_block_size != 256:
                command.extend(["--symbolic-block-size", str(args.symbolic_block_size)])
            if args.symbolic_batch_cols != 128:
                command.extend(["--symbolic-batch-cols", str(args.symbolic_batch_cols)])
            if args.symbolic_verify:
                command.append("--symbolic-verify")
            if args.symbolic_no_fallback:
                command.append("--symbolic-no-fallback")
            if args.perturb:
                command.append("-p")
            if args.sf_workers:
                command.extend(["--sf-workers", str(args.sf_workers)])
            if args.numeric_sm_padding:
                command.append("--numeric-sm-padding")
            if args.progress:
                command.append("--progress")
            row = dict(matrix=str(matrix.relative_to(matrix_dir)),
                       update_strategy=args.update_strategy, scheduling=args.scheduling,
                       symbolic_backend=args.symbolic_backend,
                       numeric_sm_padding=int(args.numeric_sm_padding),
                       perturb=int(args.perturb), status="ok", exit_code="",
                       log_path=str(log_path.relative_to(output_dir)), solution_path="",
                       note="")
            print("[{}/{}] {}".format(index, len(matrices), row["matrix"]), flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(command, cwd=str(run_dir), env=env,
                                            stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.timeout or None)
                    row["exit_code"] = result.returncode
                    if result.returncode != 0:
                        if result.returncode == 2:
                            row.update(status="residual_not_converged",
                                       note="solver did not reach the residual L2 target")
                        else:
                            row.update(status="failed", note="solver returned a nonzero exit code")
                except subprocess.TimeoutExpired:
                    row.update(status="timeout", note="exceeded {} seconds".format(args.timeout))
                except OSError as exc:
                    row.update(status="failed", note=str(exc))
                    log.write("\nBatch runner: {}\n".format(exc))

            metrics = parse_metrics(log_path)
            row.update(metrics)
            missing = [field for field, _ in METRICS.values() if field not in metrics]
            invalid = [field for field, value in metrics.items()
                       if not math.isfinite(value) or value < 0 or (field == "n" and value == 0)]
            if row["status"] == "ok":
                if missing:
                    row.update(status="missing_metrics", note="missing: " + ", ".join(missing))
                elif invalid:
                    row.update(status="invalid_metrics", note="invalid: " + ", ".join(invalid))
                elif metrics["residual_l2"] > args.residual_tol:
                    row.update(status="residual_not_converged",
                               note="residual L2 exceeds {}".format(args.residual_tol))
            if solution_path.is_file():
                row["solution_path"] = str(solution_path.relative_to(output_dir))
            writer.writerow(row)
            csv_file.flush()
            failed += row["status"] != "ok"
            print("  {} | total_solve_ms={}".format(
                row["status"], row.get("total_solve_ms", "")),
                flush=True)

    print("Finished: {} ok, {} failed. CSV: {}".format(
        len(matrices) - failed, failed, summary), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; completed rows remain in summary.csv.", file=sys.stderr)
        sys.exit(130)
