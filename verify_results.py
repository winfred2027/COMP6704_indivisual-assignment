"""Audit saved solutions, row counts, certificates, and sensitivity identities.

This audits supplied observations without solving benchmark instances again.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.transport import certificate, toy


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root/"results/raw.csv")
    assert raw.verified.all() and raw.success.all()
    assert not raw.duplicated(["instance_id", "algorithm", "repeat"]).any()
    assert (raw.groupby(["instance_id", "algorithm"]).size() == 3).all()
    checked = 0
    for _, row in raw[raw.repeat == 0].iterrows():
        data = np.load(root/f"data/{row.instance_id}.npz")
        solution = np.load(root/f"results/solutions/{row.instance_id}_{row.algorithm}.npz")
        C, s, d = data["C"], data["s"], data["d"]
        X = solution["X"]
        objective = float((C*X).sum())
        assert np.isclose(objective, row.objective, rtol=1e-12)
        result = dict(X=X, objective=objective)
        if "u" in solution:
            result.update(u=solution["u"], v=solution["v"])
        metrics = certificate(result, C, s, d)
        assert metrics["primal_residual"] <= 1e-7
        assert metrics["min_x"] >= -1e-7
        if "u" in result:
            assert metrics["dual_residual"] <= 1e-7
            assert metrics["duality_gap"] <= 1e-7
        checked += 1
    C0, s, d = toy()
    for tau in (0, .25, .5, 1, 2, 4, 8, 12):
        C = C0.copy()
        C[0, 1] += tau
        t = min(tau, 2)
        u, v = np.array([6, 9, 9-t]), np.array([0, t, 4, -4+t])
        assert np.max(u[:, None]+v-C) <= 1e-10
        predicted = min(1060+15*tau, 1090)
        assert abs(s@u+d@v-predicted) < 1e-8
        observed = raw[(raw.experiment == "sensitivity_route")&(raw.tau == tau)].objective
        assert np.max(np.abs(observed-predicted)) < 1e-8
    out = dict(timed_runs=len(raw), saved_pairs_verified=checked,
               exact_runs=int((~raw.algorithm.isin(["NW", "LC"])).sum()),
               cases=raw.instance_id.nunique(), status="passed",
               checks=["all saved primal solutions", "saved primal-dual certificates",
                       "all timing repeats accounted for", "analytical route sensitivity"])
    (root/"results/audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
