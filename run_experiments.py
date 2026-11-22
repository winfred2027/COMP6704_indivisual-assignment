"""Run serial, paired experiments and persist all inputs and observations."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
from pathlib import Path
import json
import platform
import subprocess
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
import matplotlib
from datetime import datetime, timezone
from src.transport import toy, generate, model, solve, certificate


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT/"config.json"))
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for folder in ("data", "results/history", "results/solutions", "results/figures"):
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown") if Path("/proc/cpuinfo").exists() else platform.processor()
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       cpu=cpu, logical_cpus=os.cpu_count(), numpy=np.__version__,
                       scipy=scipy.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                       threads=1, parallel=False, tolerances=dict(primal=1e-8, dual=1e-8, ipm=1e-9),
                       presolve=True)
    environment["run_started_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        from scipy.optimize._highspy._core import _Highs
        environment["highs"] = _Highs().version()
    except ImportError:
        environment["highs"] = "not exposed by this SciPy build"
    (ROOT/"results/environment.json").write_text(json.dumps(environment, indent=2))
    versions = [f"{package}=={version}" for package, version in
                (("numpy", np.__version__), ("scipy", scipy.__version__),
                 ("pandas", pd.__version__), ("matplotlib", matplotlib.__version__))]
    (ROOT/"requirements.txt").write_text("\n".join(versions)+"\n")
    # Warm both compiled solver paths before collecting any times.
    for algorithm in ("NW", "LC", "MODI-NW", "MODI-LC", "DS", "IPM"):
        solve(*toy(), algorithm)
    rows = []

    def run_case(instance_id, C, s, d, algorithms, experiment, seed=-1,
                 cost_type="toy", coords=None, tau=np.nan):
        metadata = dict(C=C, s=s, d=d)
        if coords:
            metadata.update(warehouses=coords[0], customers=coords[1])
        np.savez_compressed(ROOT/f"data/{instance_id}.npz", **metadata)
        begin = perf_counter()
        matrices = model(s, d)
        build_time = perf_counter()-begin
        # Separate reference validation from timed calls; reference time is not reported.
        reference = solve(C, s, d, "DS", matrices=matrices)
        if not reference["success"]:
            raise RuntimeError(f"Reference failed: {instance_id}")
        zref = reference["objective"]
        ordering = np.random.default_rng(12345+max(seed, 0))
        for repeat in range(cfg["timing_repeats"]):
            for algorithm in ordering.permutation(algorithms):
                begin = perf_counter()
                try:
                    result = solve(C, s, d, algorithm, matrices, track=False,
                                   time_limit=cfg["time_limit_s"])
                    elapsed = perf_counter()-begin
                    metrics = certificate(result, C, s, d)
                    gap = (result["objective"]-zref)/max(1, abs(zref))
                    valid = (metrics["primal_residual"] <= 1e-7
                             and metrics["min_x"] >= -1e-7)
                    if algorithm not in ("NW", "LC"):
                        valid = valid and abs(gap) <= 1e-7 and metrics["dual_residual"] <= 1e-7 and metrics["duality_gap"] <= 1e-7
                    row = dict(objective=result["objective"], reference=zref,
                               gap_percent=100*gap, solve_time_s=elapsed,
                               nit=result["nit"], crossover_nit=result["crossover_nit"],
                               status=result["status"], message=result.get("message", ""),
                               success=result["success"], verified=bool(valid and result["success"]),
                               degenerates=result.get("degenerates", np.nan), **metrics)
                    if repeat == 0 and result["X"] is not None:
                        np.savez_compressed(ROOT/f"results/solutions/{instance_id}_{algorithm}.npz",
                                            X=result["X"], **({k: result[k] for k in ("u", "v")} if "u" in result else {}))
                except Exception as error:
                    row = dict(objective=np.nan, reference=zref, gap_percent=np.nan,
                               solve_time_s=perf_counter()-begin, nit=np.nan,
                               crossover_nit=np.nan, success=False, verified=False,
                               status="exception", message=repr(error))
                rows.append(dict(experiment=experiment, instance_id=instance_id,
                                 seed=seed, m=len(s), n=len(d), cost_type=cost_type,
                                 algorithm=algorithm, repeat=repeat, tau=tau,
                                 build_time_s=build_time, **row))
            pd.DataFrame(rows).to_csv(ROOT/"results/raw.csv", index=False)
        print(instance_id, "completed", flush=True)

    all_algorithms = ["NW", "LC", "MODI-NW", "MODI-LC", "DS", "IPM"]
    run_case("toy", *toy(), all_algorithms, "toy")
    for m, n in cfg["sizes"]:
        algorithms = all_algorithms if m <= 30 else ["NW", "LC", "DS", "IPM"]
        for seed in cfg["seeds"]:
            C, s, d, w, q = generate(m, n, seed)
            run_case(f"geo_{m}_{n}_{seed}", C, s, d, algorithms, "scale", seed,
                     "geographic", (w, q))
    for kind in ("uniform", "dynamic"):
        for seed in cfg["seeds"]:
            C, s, d, w, q = generate(30, 60, seed, kind)
            run_case(f"{kind}_30_60_{seed}", C, s, d, all_algorithms,
                     "distribution", seed, kind, (w, q))
    m, n = cfg["large_size"]
    for seed in cfg["large_seeds"]:
        C, s, d, w, q = generate(m, n, seed)
        run_case(f"geo_{m}_{n}_{seed}", C, s, d, ["DS", "IPM"], "scale_large",
                 seed, "geographic", (w, q))
    C0, s, d = toy()
    for tau in cfg["sensitivity_tau"]:
        C = C0.copy()
        C[0, 1] += tau
        run_case(f"route_tau_{tau:g}", C, s, d, ["DS", "IPM"],
                 "sensitivity_route", tau=tau)
    for tau in (0, 1, 2, 3):
        C = C0.copy()
        C[0] += tau
        run_case(f"row_tau_{tau}", C, s, d, ["DS", "IPM"],
                 "sensitivity_row", tau=tau)
        assert abs(solve(C, s, d, "DS")["objective"]-(1060+35*tau)) < 1e-6
    # Separate traced runs avoid charging history instrumentation to MODI alone.
    for seed in cfg["seeds"]:
        C, s, d = generate(10, 20, seed)[:3]
        for initialization in ("NW", "LC"):
            result = solve(C, s, d, f"MODI-{initialization}", track=True)
            pd.DataFrame(result["history"]).to_csv(
                ROOT/f"results/history/geo_10_20_{seed}_MODI-{initialization}.csv", index=False)
    raw = pd.DataFrame(rows)
    per_instance = raw.groupby(["experiment", "instance_id", "m", "n", "cost_type", "algorithm"], sort=False).agg(
        time_s=("solve_time_s", "median"), objective=("objective", "first"),
        gap_percent=("gap_percent", "first"), primal_residual=("primal_residual", "max"),
        dual_residual=("dual_residual", "max"), duality_gap=("duality_gap", "max"),
        nit=("nit", "first"), crossover_nit=("crossover_nit", "first"),
        verified=("verified", "all"), degenerates=("degenerates", "first")).reset_index()
    per_instance.to_csv(ROOT/"results/per_instance.csv", index=False)
    summary = per_instance.groupby(["m", "n", "cost_type", "algorithm"]).agg(
        instances=("verified", "size"), success_rate=("verified", "mean"),
        median_time_s=("time_s", "median"), q25_time_s=("time_s", lambda x: x.quantile(.25)),
        q75_time_s=("time_s", lambda x: x.quantile(.75)), median_gap_percent=("gap_percent", "median"),
        median_nit=("nit", "median"), max_primal_residual=("primal_residual", "max"),
        max_dual_residual=("dual_residual", "max"), max_duality_gap=("duality_gap", "max")).reset_index()
    summary.to_csv(ROOT/"results/summary.csv", index=False)
    print("DONE:", len(rows), "timed runs; verified", int(raw.verified.sum()), flush=True)


if __name__ == "__main__":
    main()
