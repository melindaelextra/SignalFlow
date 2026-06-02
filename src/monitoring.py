from __future__ import annotations

import time
import pandas as pd


def benchmark_model(model, X: pd.DataFrame, n_runs: int = 5) -> dict:
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict_proba(X)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed / max(1, len(X)) * 1000)

    series = pd.Series(latencies)
    return {
        "avg_latency_ms": float(series.mean()),
        "p95_latency_ms": float(series.quantile(0.95)),
        "rows": int(len(X)),
    }
