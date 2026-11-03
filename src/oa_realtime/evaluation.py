"""Agreement measures and interleaved single-observation timing."""
import time
import os
import platform
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits


def hardware_info():
    cpuinfo = Path('/proc/cpuinfo')
    models = []
    if cpuinfo.exists():
        models = sorted({line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                         if line.startswith('model name')})
    product = Path('/sys/class/dmi/id/product_name')
    try:
        server = product.read_text().strip()
    except OSError:
        server = None
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    return {'cpu_models': models, 'server_model': server, 'cpu_affinity': affinity,
            'logical_cpus_available': len(affinity) if affinity is not None else os.cpu_count(),
            'python': platform.python_version()}


def agreement(estimate, reference):
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    if estimate.shape != reference.shape or not np.isfinite(estimate).all() or not np.isfinite(reference).all():
        raise ValueError('Estimates and references must be aligned and finite')
    mae = float(np.mean(np.abs(estimate - reference)))
    mean = float(np.mean(reference))
    return {'mae': mae, 'normalized_mae_percent': 100 * mae / mean if mean else float('nan')}


def benchmark(calls, n_observations, repetitions=128, seed=20261102):
    """Time loaded callables with randomized method order and one BLAS thread."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_observations, size=repetitions)
    rows = []
    with threadpool_limits(limits=1):
        for call in calls.values():
            call(0)
        for i in indices:
            for name in rng.permutation(list(calls)):
                started = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
                calls[name](int(i))
                rows.append({'index': int(i), 'method': str(name),
                             'seconds': time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - started})
    return rows
