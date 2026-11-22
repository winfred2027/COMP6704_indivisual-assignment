"""Meaningful validation: fixed optimum, degeneracy, ties, and random LPs."""
import numpy as np
from src.transport import toy, generate, solve, certificate


def run():
    cases = [toy(), (np.ones((4, 4)), np.ones(4), np.ones(4)),
             (np.array([[4., 1], [1, 4]]), np.ones(2), np.ones(2)),
             (np.array([[12.,9,8,4],[5,1,2,1],[3,12,10,13],[8,9,14,11]]),
              np.ones(4), np.ones(4))]
    for seed in range(12):
        for cost_type in ("geographic", "uniform", "dynamic"):
            cases.append(generate(5, 9, seed, cost_type)[:3])
    for index, (C, s, d) in enumerate(cases):
        reference = solve(C, s, d, "DS")
        assert reference["success"]
        for algorithm in ("NW", "LC", "MODI-NW", "MODI-LC", "IPM"):
            result = solve(C, s, d, algorithm, track=algorithm.startswith("MODI"))
            metrics = certificate(result, C, s, d)
            assert result["success"], (index, algorithm)
            assert metrics["primal_residual"] < 1e-8, (index, algorithm, metrics)
            assert metrics["min_x"] >= -1e-8
            if algorithm not in ("NW", "LC"):
                assert abs(result["objective"]-reference["objective"])/max(1, abs(reference["objective"])) < 1e-8
                assert metrics["dual_residual"] < 1e-8
                assert metrics["duality_gap"] < 1e-8
            if result["history"]:
                values = [r["objective"] for r in result["history"]]
                assert np.max(np.diff(values), initial=0) < 1e-7*max(1, values[0])
    expected = {"NW": 1210, "LC": 1110, "MODI-NW": 1060, "MODI-LC": 1060, "DS": 1060, "IPM": 1060}
    for algorithm, value in expected.items():
        assert abs(solve(*toy(), algorithm)["objective"]-value) < 1e-6
    # This constructed integer example forces two zero-step pivots from NW.
    assert solve(*cases[3], "MODI-NW")["degenerates"] == 2
    print(f"PASS: {len(cases)} cases, 6 algorithms; primal/dual certificates and monotonicity.")


if __name__ == "__main__":
    run()
