#!/usr/bin/env python3
"""Compare CPU, graph GPU and SF symbolic on identical solver inputs; retain logs."""
import argparse
import csv
import math
import os
from pathlib import Path
import statistics
import subprocess

SF_METRICS = ("symbolic_gpu_total_ms", "symbolic_gpu_compute_ms", "symbolic_gpu_pack_ms",
              "symbolic_pred_merges", "symbolic_workers", "symbolic_peak_workspace_bytes",
              "symbolic_predecessor_rows_read", "symbolic_task_parks")
SHAPE_METRICS = ("Matrix Row", "Original nonzero", "Symbolic nonzero")


def metrics(text):
    result = {}
    for line in text.splitlines():
        label, sep, value = line.partition(":")
        if sep:
            try:
                result[label.strip()] = float(value.split()[0])
            except (ValueError, IndexError):
                pass
    return result


def p90(values):
    return sorted(values)[max(0, math.ceil(len(values) * .9) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrices", nargs="+", type=Path)
    parser.add_argument("--solver", type=Path, default=Path(__file__).resolve().parent / "lu_cmd")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--perturb", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or not 0 <= args.workers <= 4294967295:
        parser.error("invalid warmup/repeat/worker count")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    solver = args.solver.resolve()
    if not solver.is_file() or not os.access(str(solver), os.X_OK):
        parser.error("solver is not executable")
    if any(not matrix.is_file() for matrix in args.matrices):
        parser.error("a matrix file is missing")
    output = args.output_dir.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(str(exc))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(Path(env.get("HPCC_PATH", "/opt/hpcc")) / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    columns = ["matrix", "backend", "status", "samples", "n", "nnzA", "nnzSymbolic",
               "e2e_median_ms", "e2e_p90_ms", "worker_median_ms", "pack_median_ms",
               "speedup_vs_cpu", "workers", "peak_workspace_bytes", "pred_rows_read", "parks", "note"]
    failed = False
    with (output / "symbolic.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, matrix in enumerate(args.matrices):
            cpu_median = None
            reference_shape = None
            for backend in ("cpu", "gpu", "gpu-sf"):
                samples, compute, pack = [], [], []
                last, problems = {}, []
                for repeat in range(args.warmups + args.repeats):
                    run = output / "{:04d}_{}_{}".format(index, backend, repeat)
                    run.mkdir()
                    command = [str(solver), "-i", str(matrix.resolve()), "-u", "left-looking",
                               "-s", "synchronization-free", "--symbolic-backend", backend]
                    if backend != "cpu":
                        command.append("--symbolic-no-fallback")
                    if backend == "gpu-sf" and args.workers:
                        command += ["--symbolic-sf-workers", str(args.workers)]
                    if args.perturb:
                        command.append("-p")
                    with (run / "solver.log").open("w") as log:
                        try:
                            process = subprocess.run(command, cwd=str(run), env=env, stdout=log,
                                                     stderr=subprocess.STDOUT, timeout=args.timeout)
                            # Exit 2 still produced the factors but did not meet
                            # the solve residual target; keep that visible below.
                            if process.returncode not in (0, 2):
                                problems.append("solver exit {}".format(process.returncode))
                                break
                            if process.returncode == 2:
                                problems.append("numeric residual target missed")
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            problems.append(str(exc))
                            break
                    data = metrics((run / "solver.log").read_text(errors="replace"))
                    required = ("Symbolic time",) + SHAPE_METRICS
                    if backend == "gpu-sf":
                        required += SF_METRICS
                    missing = [key for key in required if key not in data]
                    if missing:
                        problems.append("missing metrics: " + ", ".join(missing))
                        break
                    invalid = [key for key in required if not math.isfinite(data[key]) or data[key] < 0]
                    if invalid:
                        problems.append("invalid metrics: " + ", ".join(invalid))
                        break
                    shape = tuple(data[key] for key in SHAPE_METRICS)
                    if reference_shape is None:
                        reference_shape = shape
                    if shape != reference_shape:
                        problems.append("matrix/nnz differs from CPU reference run")
                        break
                    if repeat >= args.warmups:
                        # Full CLI fill_in call includes validation and commit
                        # on all backends. Verification is off for measurement.
                        samples.append(data["Symbolic time"])
                        if backend == "gpu-sf":
                            compute.append(data["symbolic_gpu_compute_ms"])
                            pack.append(data["symbolic_gpu_pack_ms"])
                        last = data
                median = statistics.median(samples) if samples else None
                complete = not problems and len(samples) == args.repeats
                if backend == "cpu" and complete:
                    cpu_median = median
                row = dict(matrix=str(matrix), backend=backend,
                           status="ok" if complete else "incomplete",
                           samples=len(samples), note="; ".join(sorted(set(problems))))
                if median is not None:
                    row.update(e2e_median_ms=median, e2e_p90_ms=p90(samples),
                               n=last.get("Matrix Row", ""), nnzA=last.get("Original nonzero", ""),
                               nnzSymbolic=last.get("Symbolic nonzero", ""))
                    if complete and cpu_median and median > 0:
                        row["speedup_vs_cpu"] = cpu_median / median
                if compute:
                    row.update(worker_median_ms=statistics.median(compute), pack_median_ms=statistics.median(pack),
                        workers=last["symbolic_workers"], peak_workspace_bytes=last["symbolic_peak_workspace_bytes"],
                        pred_rows_read=last["symbolic_predecessor_rows_read"], parks=last["symbolic_task_parks"])
                writer.writerow(row)
                stream.flush()
                failed |= row["status"] != "ok"
                print("{} {}: {} median={} ms".format(matrix.name, backend, row["status"], median), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
