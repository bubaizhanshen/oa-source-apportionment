"""Source-spectrum augmentation and controlled-increment evaluation."""
import numpy as np


def augment_sources(spectra, uncertainties, contributions, source_library,
                    intensity='mixed', seed=2026):
    """Add one randomly selected source to each historical observation.

    ``source_library`` maps factor indices to ordered mappings of spectrum IDs
    to nonnegative, unit-sum profiles. The mixed design alternates the 1x and
    4x intensity multipliers within each factor/profile stratum. Returned
    training arrays contain both the original and augmented observations.
    """
    x, u, y = (np.asarray(a, dtype=float) for a in (spectra, uncertainties, contributions))
    if (x.ndim != 2 or x.shape != u.shape or y.ndim != 2 or len(y) != len(x)
            or not len(x) or not np.isfinite(x).all() or not np.isfinite(u).all()
            or not np.isfinite(y).all() or (u <= 0).any()):
        raise ValueError('Historical spectra, uncertainties, and contributions must be aligned and finite')
    if intensity not in ('narrow', 'wide', 'mixed'):
        raise ValueError('intensity must be narrow, wide, or mixed')
    factors = list(source_library)
    if (not factors or any(not isinstance(k, (int, np.integer)) or k < 0 or k >= y.shape[1]
                           or not source_library[k] for k in factors)):
        raise ValueError('Each source category needs a valid factor and at least one profile')
    profiles = {}
    for factor in factors:
        profiles[factor] = {}
        for sid, values in source_library[factor].items():
            p = np.asarray(values, dtype=float)
            if (p.shape != (x.shape[1],) or not np.isfinite(p).all() or (p < 0).any()
                    or not np.isclose(p.sum(), 1.0, rtol=1e-6, atol=1e-8)):
                raise ValueError('Source profiles must be finite, nonnegative, and sum to one')
            profiles[factor][sid] = p
    signal = np.maximum(x, 0).sum(axis=1)
    if (signal <= 0).any():
        raise ValueError('Every historical spectrum must have positive total signal')
    rng = np.random.default_rng(seed)
    group = rng.integers(0, len(factors), len(x))
    amounts = rng.uniform(0, 1, len(x)) * np.median(signal)
    drawn = []
    for group_index in group:
        ids = list(profiles[factors[group_index]])
        drawn.append(ids[int(rng.integers(0, len(ids)))])
    multiplier = np.ones(len(x))
    if intensity == 'wide':
        multiplier[:] = 4.0
    elif intensity == 'mixed':
        for group_index, factor in enumerate(factors):
            for sid in profiles[factor]:
                indices = [i for i, (k, v) in enumerate(zip(group, drawn))
                           if k == group_index and v == sid]
                multiplier[indices[::2]] = 4.0
    augmented_x, augmented_y = x.copy(), y.copy()
    for i, (group_index, sid, amount) in enumerate(zip(group, drawn, amounts * multiplier)):
        factor = factors[group_index]
        augmented_x[i] += amount * profiles[factor][sid]
        augmented_y[i, factor] += amount
    assignment = {'factor': np.asarray([factors[k] for k in group]), 'profile_id': np.asarray(drawn),
                  'amount': amounts * multiplier, 'multiplier': multiplier}
    return np.vstack((x, augmented_x)), np.vstack((u, u)), np.vstack((y, augmented_y)), assignment


def source_increment(before, after, added_signal, target_factor):
    """Measure the total and source-specific response to a known addition.

    The expected increment is assigned entirely to ``target_factor``. This
    tests the specified source-to-factor mapping, not ambient source truth.
    """
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.ndim == 1:
        before, after = before[None, :], after[None, :]
    if (before.ndim != 2 or before.shape != after.shape or not len(before)
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or not isinstance(target_factor, (int, np.integer))
            or target_factor < 0 or target_factor >= before.shape[1]):
        raise ValueError('Predictions and target factor must be aligned and finite')
    amount = np.asarray(added_signal, dtype=float)
    if amount.ndim == 0:
        amount = np.full(len(before), amount)
    if amount.shape != (len(before),) or not np.isfinite(amount).all() or (amount <= 0).any():
        raise ValueError('Added signals must be positive and aligned with the predictions')
    response = (after - before) / amount[:, None]
    expected = np.zeros(before.shape[1]); expected[target_factor] = 1.0
    return {'response': response, 'total': response.sum(axis=1),
            'target': response[:, target_factor],
            'allocation_error': np.abs(response - expected).sum(axis=1)}


def source_constraint_centers(historical_profiles, source_library):
    """Blend unit-sum historical factors with their category-mean spectra."""
    historical = np.asarray(historical_profiles, dtype=float)
    if (historical.ndim != 2 or not np.isfinite(historical).all() or (historical < 0).any()
            or not np.allclose(historical.sum(axis=1), 1, rtol=1e-6, atol=1e-8)):
        raise ValueError('Historical profiles must be finite, nonnegative, and sum to one')
    centers = historical.copy()
    for factor, profiles in source_library.items():
        if not isinstance(factor, (int, np.integer)) or factor < 0 or factor >= len(centers) or not profiles:
            raise ValueError('Invalid factor or empty source category')
        values = np.asarray(list(profiles.values()), dtype=float)
        if (values.ndim != 2 or values.shape[1] != historical.shape[1]
                or not np.isfinite(values).all() or (values < 0).any()
                or not np.allclose(values.sum(axis=1), 1, rtol=1e-6, atol=1e-8)):
            raise ValueError('Source profiles must be nonnegative and sum to one')
        centers[factor] = 0.5 * historical[factor] + 0.5 * values.mean(axis=0)
    return centers
