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
    """Read all three ME-2 factors and absorb CC into the contributions.

    Each task in the ME-2 .dat file contains AA, BB and one CC row.  The
    reconstructed observations are ``(AA * CC) @ BB.T``.  CC can differ from
    one in constrained continuations, so omitting it changes the result.
    """
    if task < 1 or min(sample_count, species_count, factor_count) < 1:
        raise ValueError('Task and matrix dimensions must be positive')
    values = np.loadtxt(Path(target) / f'{prefix}.dat', ndmin=2)
    block_rows = sample_count + species_count + 1
    if (values.shape[1] != factor_count or values.shape[0] % block_rows or
            task > values.shape[0] // block_rows or not np.isfinite(values).all()):
        raise ValueError(f'Unexpected ME-2 factor blocks in {prefix}.dat')
    offset = (task - 1) * block_rows
    aa = values[offset:offset + sample_count]
    bb = values[offset + sample_count:offset + sample_count + species_count]
    cc = values[offset + sample_count + species_count]
    return aa * cc, bb


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


def match_profiles(reference_profiles, current_profiles):
    """Match each reference factor to one current factor by spectral cosine.

    Return current-factor indices in reference order and their cosine values.
    Matching defines correspondence; it does not establish source identity.
    """
    reference = np.asarray(reference_profiles, dtype=float)
    current = np.asarray(current_profiles, dtype=float)
    if (reference.ndim != 2 or reference.shape != current.shape or
            not reference.size or not np.isfinite(reference).all() or
            not np.isfinite(current).all()):
        raise ValueError('Expected matching finite factor-by-channel matrices')
    norms = np.outer(np.linalg.norm(reference, axis=1), np.linalg.norm(current, axis=1))
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('Factor profiles must have finite nonzero norms')
    similarity = reference @ current.T / norms
    rows, cols = linear_sum_assignment(-similarity)
    if not np.array_equal(rows, np.arange(len(reference))):
        raise ValueError('Could not match all factor profiles')
    return cols, similarity[rows, cols]


def prepare_history_constrained_run(base_dir, historical_profiles, current_profiles,
                                    task, output_dir, run_id='bounded', a_value=0.5):
    """Continue a converged ME-2 task with bounds around matched historical profiles.

    Both profile matrices have one unit-sum factor per row. The true profiles,
    if known from a simulation, must not be passed to this function.
    """
    base_dir, output_dir = Path(base_dir), Path(output_dir)
    historical = np.asarray(historical_profiles, dtype=float)
    current = np.asarray(current_profiles, dtype=float)
    if (historical.ndim != 2 or historical.shape != current.shape or
            not np.isfinite(historical).all() or not np.isfinite(current).all() or
            (historical < 0).any() or (current < 0).any() or
            not np.allclose(historical.sum(axis=1), 1) or
            not np.allclose(current.sum(axis=1), 1) or not 0 <= a_value <= 1):
        raise ValueError('Expected matched unit-sum profiles and an a-value in [0, 1]')
    if not re.fullmatch(r'[A-Za-z0-9]+', run_id):
        raise ValueError('run_id must contain only letters and digits')
    cosine = current @ historical.T / np.outer(np.linalg.norm(current, axis=1),
                                                np.linalg.norm(historical, axis=1))
    rows, to_history = linear_sum_assignment(-cosine)
    if not np.array_equal(rows, np.arange(len(current))):
        raise ValueError('Could not match all factor profiles')
    lines = [s.strip() for s in (base_dir / 'iniparams.txt').read_text().splitlines() if s.strip()]
    if len(lines) != 12 or lines[0] != '161110' or lines[-1] != '161110':
        raise ValueError('Unexpected ME-2 iniparams file')
    controls = lines[2].split()
    if len(controls) != 6 or not 1 <= task <= int(controls[4]):
        raise ValueError('Selected base task is unavailable')
    selected = read_q(base_dir)
    selected = selected.loc[selected.task == task]
    if len(selected) != 1 or int(selected.iloc[0].endcode) != 4 or not np.isfinite(selected.iloc[0].q_robust):
        raise ValueError('Selected base task did not converge')
    controls[4], controls[5] = '1', str(task)
    lines[2] = ' '.join(controls)
    continuation = lines[4].split()
    if len(continuation) != 6:
        raise ValueError('Unexpected continuation control')
    continuation[0:2] = ['1', '0']
    lines[4] = ' '.join(continuation)
    normalization = lines[5].split()
    if len(normalization) != 2 or normalization[1] not in ('0', '1'):
        raise ValueError('Unexpected factor normalization control')
    normalization[1] = '1'
    lines[5] = ' '.join(normalization)
    names = re.findall(r"'([^']+)'", lines[9])
    if len(names) != 3:
        raise ValueError('Unexpected ME-2 input filenames')
    lines[9] = f"'pmfdata.txt' 'PMF_prev_res.dat' '{run_id}'"
    final = lines[10].split()
    if len(final) != 5:
        raise ValueError('Unexpected final control')
    final[3:5] = ['0', '0']
    lines[10] = ' '.join(final)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ('pmfdata.txt', 'me2gfP4_1345c4.exe', 'me2key.key', 'PMF_bs_6f8xx_sealed_GUI.ini'):
        shutil.copy2(base_dir / name, output_dir / name)
    shutil.copy2(base_dir / (names[2] + '.dat'), output_dir / 'PMF_prev_res.dat')
    (output_dir / 'iniparams.txt').write_text('\n'.join(lines) + '\n', encoding='ascii')
    keys = np.full((historical.shape[1], historical.shape[0]), -1, dtype=int)
    lower = (historical[to_history] * (1 - a_value)).T
    upper = (historical[to_history] * (1 + a_value)).T
    with (output_dir / 'moreparams.txt').open('w', encoding='ascii') as out:
        out.write('0 0 0 1 0\n')
        for matrix in (keys, lower, upper):
            for row in matrix:
                out.write(' '.join(f'{float(v):.12g}' for v in row) + '\n')
    return {'directory': output_dir, 'current_to_history': to_history, 'lower': lower, 'upper': upper}
