"""Rolling-window inputs and the EPA PMF ME-2 interface."""
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def write_me2_matrix(concentration: np.ndarray, uncertainty: np.ndarray, path: Path) -> None:
    if concentration.shape != uncertainty.shape:
        raise ValueError("Concentration and uncertainty matrices must have the same shape")
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for matrix in (concentration, uncertainty):
            for row in matrix:
                handle.write(" ".join(f"{float(value):.10g}" for value in row) + "\n")


def read_q(target: Path) -> pd.DataFrame:
    rows = []
    for line in (target / "PMFreport.txt").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        parts = line.split()
        if len(parts) == 10 and parts[0].isdigit():
            rows.append(
                {
                    "task": int(parts[0]),
                    "iterations": int(parts[1]),
                    "q_robust": float(parts[2]),
                    "q_main": float(parts[3]),
                    "q_aux": float(parts[4]),
                    "q_true": float(parts[5]),
                    "gradnorm": float(parts[6]),
                    "endcode": int(parts[9]),
                }
            )
    if not rows:
        raise ValueError(f"No Q rows found in {target / 'PMFreport.txt'}")
    return pd.DataFrame(rows)


def parse_matrix_block(
    lines: list[str], start: int, row_count: int, column_count: int
) -> tuple[np.ndarray, int]:
    values: list[list[float]] = []
    index = start
    while index < len(lines) and len(values) < row_count:
        parts = lines[index].split()
        if len(parts) >= column_count:
            try:
                values.append([float(value) for value in parts[-column_count:]])
            except ValueError:
                if values:
                    raise ValueError(f"Matrix block ended at line {index + 1}")
        index += 1
    if len(values) != row_count:
        raise ValueError(f"Expected {row_count} matrix rows, found {len(values)}")
    return np.asarray(values, dtype=float), index


def parse_factors(
    target: Path, prefix: str, task: int, sample_count: int, species_count: int, factor_count: int
) -> tuple[np.ndarray, np.ndarray]:
    lines = (target / f"{prefix}.txt").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    active = False
    contributions = profiles = None
    index = 0
    while index < len(lines):
        match = re.search(r"Results written by postprocessing for task #\s+(\d+)", lines[index])
        if match:
            active = int(match.group(1)) == task
        if active and "Factor matrix AA" in lines[index]:
            contributions, index = parse_matrix_block(
                lines, index + 1, sample_count, factor_count
            )
            continue
        if active and "Factor matrix BB" in lines[index]:
            profiles, index = parse_matrix_block(
                lines, index + 1, species_count, factor_count
            )
            break
        index += 1
    if contributions is None or profiles is None:
        raise ValueError(f"Could not parse task {task} from {prefix}.txt")
    return contributions, profiles


def window_indices(times, cutoff, days=14, include_current=True):
    """Select the 14-day window ending at an observation or before an update."""
    times = np.asarray(times, dtype='datetime64[s]')
    cutoff = np.datetime64(cutoff, 's')
    end = times <= cutoff if include_current else times < cutoff
    return np.flatnonzero((times >= cutoff - np.timedelta64(days, 'D')) & end)


def prepare_run(x, u, factor_count, directory, runtime, run_id='pmf', starts=20, seed=26801):
    """Prepare the study's unconstrained ME-2 run using an installed EPA runtime."""
    x, u = np.asarray(x, dtype=float), np.asarray(u, dtype=float)
    if x.ndim != 2 or x.shape != u.shape or not np.isfinite(x).all() or not np.isfinite(u).all() or (u <= 0).any():
        raise ValueError('Expected finite spectra and positive matching uncertainties')
    if not re.fullmatch(r'[A-Za-z0-9]+', run_id):
        raise ValueError('run_id must contain only letters and digits')
    directory, runtime = Path(directory), Path(runtime)
    directory.mkdir(parents=True, exist_ok=False)
    write_me2_matrix(x, u, directory / 'pmfdata.txt')
    for name in ('me2gfP4_1345c4.exe', 'me2key.key'):
        shutil.copy2(runtime / name, directory / name)
    ini = (runtime / 'PMF_bs_6f8xx_sealed_GUI.ini').read_text(encoding='utf-8', errors='replace')
    (directory / 'PMF_bs_6f8xx_sealed_GUI.ini').write_text(ini.replace("'gui'=1,", "'gui'=0,"), encoding='utf-8', newline='\r\n')
    seeds = ' '.join(str(seed + j) for j in range(5))
    parameters = f"""161110
0
1 4 4 20 {starts} 1
1 11 0
0 0 1 1.5 0 1
0.00 0
{seeds}
{len(x)} {x.shape[1]} {factor_count} 0.0 0.0 -12
0 0 20 1.2 0.35 1.1 0.50
'pmfdata.txt' 'PMF_prev_res.dat' '{run_id}'
0 1 0 -1 0
161110
"""
    (directory / 'iniparams.txt').write_text(parameters, encoding='ascii')
    return directory


def run_me2(directory, command=('wine',)):
    directory = Path(directory)
    with (directory / 'me2_stdout.txt').open('w', encoding='utf-8') as log:
        subprocess.run([*command, 'me2gfP4_1345c4.exe', 'PMF_bs_6f8xx_sealed_GUI.ini'],
                       cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)


def select_solution(directory, run_id, n_samples, n_channels, n_factors, starts=20):
    """Select the converged, nonnegative start with the lowest robust Q."""
    q = read_q(Path(directory)).sort_values('task')
    if set(q.task.astype(int)) != set(range(1, starts + 1)):
        raise ValueError('Incomplete ME-2 starts')
    usable = []
    for row in q.itertuples(index=False):
        g, f = parse_factors(Path(directory), run_id, int(row.task), n_samples, n_channels, n_factors)
        valid = (np.isfinite(g).all() and np.isfinite(f).all() and
                 (g >= 0).all() and (f >= 0).all() and (f.sum(axis=0) > 0).all())
        if row.endcode == 4 and np.isfinite(row.q_robust) and valid:
            usable.append((float(row.q_robust), int(row.task), g, f))
    if not usable:
        raise ValueError('No converged usable ME-2 solution')
    q_best, task, g, f = min(usable, key=lambda z: (z[0], z[1]))
    scale = f.sum(axis=0)
    return {'task': task, 'q_robust': q_best, 'contributions': g * scale[None, :],
            'profiles': (f / scale).T}


def match_barcelona(profiles, reference):
    """Match four factors to HOA, COA, and the combined remaining factors."""
    profiles, reference = np.asarray(profiles), np.asarray(reference)
    if profiles.shape[0] != 4 or reference.shape[0] < 2:
        raise ValueError('Expected four factors and HOA/COA reference profiles')
    similarity = reference[:2] @ profiles.T / np.outer(np.linalg.norm(reference[:2], axis=1),
                                                       np.linalg.norm(profiles, axis=1))
    if not np.isfinite(similarity).all():
        raise ValueError('Invalid spectral similarity')
    rows, cols = linear_sum_assignment(-similarity)
    if not np.array_equal(rows, np.arange(2)):
        raise ValueError('Joint assignment error')
    coa = int(np.argmax(similarity[1]))
    hoa = max((i for i in range(4) if i != coa), key=lambda i: (similarity[0, i], -i))
    pairs = {'joint_prior': [int(cols[0]), int(cols[1])], 'coa_first': [hoa, coa]}
    return {name: [[p[0]], [p[1]], sorted(set(range(4)) - set(p))] for name, p in pairs.items()}
