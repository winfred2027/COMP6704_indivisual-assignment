"""Balanced transportation LP and independently implemented initializers/MODI.

Variable order is row-major (k = i*n + j). Basis membership is maintained
separately from positive flow, which is essential for degenerate solutions.
"""
from collections import deque
from time import perf_counter
import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, OptimizeWarning


def toy():
    return (np.array([[8, 6, 10, 9], [9, 12, 13, 7], [14, 9, 16, 5.]], float),
            np.array([35., 50, 40]), np.array([30., 25, 40, 30]))


def generate(m, n, seed, cost_type="geographic"):
    rng = np.random.default_rng(seed)
    warehouses, customers = rng.uniform(0, 100, (m, 2)), rng.uniform(0, 100, (n, 2))
    a, b = rng.uniform(.5, 1.5, m), rng.uniform(.5, 1.5, n)
    s, d = 1000*a/a.sum(), 1000*b/b.sum()
    s[-1], d[-1] = 1000-s[:-1].sum(), 1000-d[:-1].sum()
    if cost_type == "geographic":
        C = 2 + .1*np.linalg.norm(warehouses[:, None]-customers[None, :], axis=2)
    elif cost_type == "uniform":
        C = rng.uniform(1, 20, (m, n))
    elif cost_type == "dynamic":
        C = 10**rng.uniform(0, 4, (m, n))
    else:
        raise ValueError(cost_type)
    return C, s, d, warehouses, customers


def model(s, d):
    m, n = len(s), len(d)
    A = sparse.vstack([
        sparse.kron(sparse.eye(m), np.ones((1, n))),
        sparse.kron(np.ones((1, m)), sparse.eye(n))], format="csr")
    b = np.r_[s, d]
    return A[:-1], b[:-1]


def residual(X, s, d):
    return max(np.max(np.abs(X.sum(axis=1)-s)),
               np.max(np.abs(X.sum(axis=0)-d)))/max(1, np.max(np.r_[s, d]))


def initialize(C, s, d, method="NW"):
    """Return a feasible flow using northwest corner or sorted least cost."""
    m, n = C.shape
    a, b, X = s.copy(), d.copy(), np.zeros_like(C)
    # A stable sort keeps row-major order for cost ties.
    order = range(m*n) if method == "NW" else np.argsort(C.ravel(), kind="stable")
    for k in order:
        i, j = divmod(int(k), n)
        if a[i] <= 0 or b[j] <= 0:
            continue
        q = min(a[i], b[j])
        X[i, j] += q
        a[i] -= q
        b[j] -= q
    if residual(X, s, d) > 1e-10:
        raise RuntimeError("Initializer failed supply/demand validation")
    return X


def complete_tree(X):
    """Complete the support forest by ZERO basic edges without cycles."""
    m, n = X.shape
    parent = list(range(m+n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def add(i, j):
        a, b = find(i), find(m+j)
        if a == b:
            return False
        parent[a] = b
        return True

    B = set()
    for k in np.flatnonzero(X.ravel() > 0):
        i, j = divmod(int(k), n)
        if not add(i, j):
            raise RuntimeError("Initial support contains a cycle")
        B.add(int(k))
    for k in range(m*n):
        if len(B) == m+n-1:
            break
        i, j = divmod(k, n)
        if k not in B and add(i, j):
            B.add(k)
    assert len(B) == m+n-1
    return B


def tree_potentials(C, B):
    m, n = C.shape
    adj = [[] for _ in range(m+n)]
    for k in sorted(B):
        i, j = divmod(k, n)
        adj[i].append((m+j, k))
        adj[m+j].append((i, k))
    potential = np.full(m+n, np.nan)
    potential[0] = 0
    queue = deque([0])
    while queue:
        a = queue.popleft()
        for b, k in adj[a]:
            if np.isnan(potential[b]):
                potential[b] = C.ravel()[k] - potential[a]
                queue.append(b)
    if np.isnan(potential).any():
        raise RuntimeError("Disconnected basis")
    return potential[:m], potential[m:], adj


def path_edges(adj, start, end):
    parent = {start: (None, None)}
    queue = deque([start])
    while queue:
        a = queue.popleft()
        if a == end:
            break
        for b, k in adj[a]:
            if b not in parent:
                parent[b] = a, k
                queue.append(b)
    edges, node = [], end
    while node != start:
        previous, edge = parent[node]
        edges.append(edge)
        node = previous
    return edges[::-1]


def modi(C, s, d, initialization="NW", time_limit=60, maxiter=100000,
         track=False):
    """Transportation simplex with Bland entering/leaving tie rules.

    The iteration history contains measured states, not interpolated values.
    Only exact minimum ratio ties are resolved by index; near-ties are not
    permitted to remove an edge carrying positive flow.
    """
    start = perf_counter()
    X = initialize(C, s, d, initialization)
    m, n = C.shape
    B = complete_tree(X)
    history, degenerates = [], 0
    tol = 1e-9*max(1, np.max(np.abs(C)))
    theta_previous = np.nan
    for iteration in range(maxiter+1):
        u, v, adj = tree_potentials(C, B)
        reduced = C-u[:, None]-v[None, :]
        reduced.ravel()[list(B)] = np.inf
        min_reduced = float(np.min(reduced)) if B else np.nan
        if track:
            history.append(dict(iteration=iteration, objective=float(np.sum(C*X)),
                                min_reduced_cost=min_reduced, theta=theta_previous,
                                primal_residual=residual(X, s, d),
                                elapsed_s=perf_counter()-start))
        candidates = np.flatnonzero(reduced.ravel() < -tol)
        if not len(candidates):
            return dict(X=X, objective=float(np.sum(C*X)), nit=iteration,
                        crossover_nit=0, success=True, status="optimal",
                        u=u, v=v, history=history, degenerates=degenerates)
        if perf_counter()-start > time_limit or iteration == maxiter:
            return dict(X=X, objective=float(np.sum(C*X)), nit=iteration,
                        crossover_nit=0, success=False, status="limit",
                        history=history, degenerates=degenerates)
        entering = int(candidates[0])
        i, j = divmod(entering, n)
        path = path_edges(adj, i, m+j)
        minus, plus = path[::2], path[1::2]
        theta = min(X.ravel()[k] for k in minus)
        leaving = min(k for k in minus if X.ravel()[k] == theta)
        X.ravel()[entering] += theta
        for k in plus:
            X.ravel()[k] += theta
        for k in minus:
            X.ravel()[k] -= theta
        B.remove(leaving)
        B.add(entering)
        theta_previous = float(theta)
        degenerates += int(theta == 0)
    raise RuntimeError("Unexpected MODI exit")


def highs(C, s, d, method, matrices=None, time_limit=60):
    if matrices is None:
        matrices = model(s, d)
    A, b = matrices
    with warnings.catch_warnings():
        # SciPy forwards these HiGHS-native options verbatim; document them.
        warnings.filterwarnings("ignore", category=OptimizeWarning,
                                message="Unrecognized options detected.*")
        result = linprog(C.ravel(), A_eq=A, b_eq=b, bounds=(0, None), method=method,
                         options=dict(presolve=True, time_limit=time_limit,
                                      primal_feasibility_tolerance=1e-8,
                                      dual_feasibility_tolerance=1e-8,
                                      ipm_optimality_tolerance=1e-9,
                                      threads=1, parallel=False))
    out = dict(X=None if result.x is None else result.x.reshape(C.shape),
               objective=np.nan if result.fun is None else float(result.fun),
               nit=result.nit, crossover_nit=getattr(result, "crossover_nit", 0),
               success=bool(result.success), status=str(result.status),
               message=result.message, history=[])
    if result.success:
        y = np.r_[result.eqlin.marginals, 0.]
        out.update(u=y[:len(s)], v=y[len(s):])
    return out


def solve(C, s, d, algorithm, matrices=None, track=False, time_limit=60):
    if algorithm in ("NW", "LC"):
        X = initialize(C, s, d, algorithm)
        return dict(X=X, objective=float(np.sum(C*X)), success=True,
                    status="feasible", nit=np.nan, crossover_nit=np.nan, history=[])
    if algorithm.startswith("MODI"):
        return modi(C, s, d, algorithm.split("-")[1], track=track, time_limit=time_limit)
    return highs(C, s, d, {"DS": "highs-ds", "IPM": "highs-ipm"}[algorithm],
                 matrices, time_limit)


def certificate(result, C, s, d):
    X = result.get("X")
    if X is None:
        return dict(primal_residual=np.nan, min_x=np.nan,
                    dual_residual=np.nan, duality_gap=np.nan)
    out = dict(primal_residual=residual(X, s, d), min_x=float(X.min()),
               dual_residual=np.nan, duality_gap=np.nan)
    if "u" in result:
        u, v = result["u"], result["v"]
        out["dual_residual"] = max(0, float(np.max(u[:, None]+v[None, :]-C)))/max(1, abs(C).max())
        out["duality_gap"] = abs(result["objective"]-s@u-d@v)/max(1, abs(result["objective"]))
    return out
