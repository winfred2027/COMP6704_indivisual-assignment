# Transportation linear programming: reproducible Python project



## Installation

Tested with Python 3.12.14. Create an independent virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Exact tested versions: NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, Matplotlib 3.10.8.
The installed SciPy extension reports HiGHS 1.8.0; we do not assume that a
different wheel or a separate `highspy` installation contains the same solver.
The private `_Highs` import is used only for reporting that version; solving
uses the public `scipy.optimize.linprog` interface.

## Reproduction

Run from the project root:

```bash
python validate.py
python verify_results.py
python run_experiments.py --config config.json
python analyze.py
```

`validate.py` checks 40 cases, including a forced two-degenerate-pivot example,
with primal feasibility, dual feasibility, objective agreement, and monotonicity.

`verify_results.py` independently audits 320 saved input/solution pairs and the
analytical sensitivity formula without solving the benchmarks again.

`run_experiments.py` runs all cases serially, overwriting generated data and
result files. Copy `results/` first if preserving the supplied measurement run.
It uses 60-second optimization limits and a deterministic shuffled method order.
The supplied run produced 960 timing records over 68 cases, all verified.
There are 55 randomly generated benchmark networks, one fixed example, and
twelve sensitivity cases. Timing repeats are not independent network samples.
Runtime depends on the host; bitwise identical times are not expected.
