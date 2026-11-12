"""Source-spectrum augmentation and controlled-increment evaluation."""
import numpy as np


def replace_source_profiles(factors, profile_ids, wide, additional_ids, seed=6130):
    """Replace half of each existing factor/profile/intensity stratum.

    Additional IDs are supplied by factor after withholding the test study.
    The original factor, intensity, and added amount are unchanged.
    """
    factors, profile_ids, wide = np.asarray(factors), np.asarray(profile_ids).astype(str), np.asarray(wide)
    if factors.ndim != 1 or profile_ids.shape != factors.shape or wide.shape != factors.shape:
        raise ValueError('Factor, profile, and intensity assignments must be aligned')
    if wide.dtype.kind != 'b':
        raise ValueError('wide must be a Boolean array')
    if not additional_ids or any(not ids or len(set(ids)) != len(ids) for ids in additional_ids.values()):
        raise ValueError('Each added factor needs unique profile IDs')
    width = max([64, *(len(str(i)) for ids in additional_ids.values() for i in ids),
                 *(len(i) for i in profile_ids)])
    result = profile_ids.astype(f'U{width}').copy()
    changed = np.zeros(len(factors), dtype=bool)
    rng = np.random.default_rng(seed)
    for factor in sorted(additional_ids):
        offset = 0
        for old_id in sorted(set(profile_ids[factors == factor])):
            for is_wide in (False, True):
                candidates = np.flatnonzero((factors == factor) & (profile_ids == old_id) & (wide == is_wide))
                for index in rng.permutation(candidates)[:len(candidates) // 2]:
                    result[index] = additional_ids[factor][offset % len(additional_ids[factor])]
                    changed[index] = True
                    offset += 1
    return result, changed


def augment_with_assignment(spectra, uncertainties, contributions, source_library, assignment):
    """Reconstruct training additions from an existing assignment of source spectra."""
    x, u, y = (np.asarray(a, dtype=float) for a in (spectra, uncertainties, contributions))
    if (x.ndim != 2 or x.shape != u.shape or y.ndim != 2 or len(x) != len(y) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(u).all() or not np.isfinite(y).all()
            or (u <= 0).any()):
        raise ValueError('Historical observations, uncertainties, and labels must be aligned')
    factors, ids, amounts = (np.asarray(assignment[k]) for k in ('factor', 'profile_id', 'amount'))
    if (factors.shape != (len(x),) or ids.shape != (len(x),) or amounts.shape != (len(x),)
            or not np.isfinite(amounts.astype(float)).all() or (amounts.astype(float) < 0).any()):
        raise ValueError('Invalid assigned source or amount')
    out_x, out_y = x.copy(), y.copy()
    for index, (factor, sid, amount) in enumerate(zip(factors, ids, amounts.astype(float))):
        if not isinstance(factor, (int, np.integer)) or factor < 0 or factor >= y.shape[1]:
            raise ValueError('Invalid assigned factor')
        profile = np.asarray(source_library[factor][str(sid)], dtype=float)
        if (profile.shape != (x.shape[1],) or not np.isfinite(profile).all() or (profile < 0).any()
                or not np.isclose(profile.sum(), 1.0, rtol=1e-6, atol=1e-8)):
            raise ValueError('Assigned profiles must be finite, nonnegative, and sum to one')
        out_x[index] += amount * profile
        out_y[index, factor] += amount
    return np.vstack((x, out_x)), np.vstack((u, u)), np.vstack((y, out_y))


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


def leave_one_study_out_selection(errors, methods):
    """Select a fixed method using source additions from the other studies.

    ``errors`` has dimensions study, source category, addition level, method.
    Each study, category, and level receives equal weight. Supply methods in
    the desired tie-breaking order. The held-out scores are evaluation results,
    not independent measurements of ambient source contributions.
    """
    values = np.asarray(errors, dtype=float)
    methods = tuple(methods)
    if (values.ndim != 4 or values.shape[0] < 2 or not all(values.shape[1:])
            or values.shape[-1] != len(methods) or len(set(methods)) != len(methods)
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Provide finite nonnegative study-by-category-by-level errors for unique methods')
    calibration, heldout, chosen = [], [], []
    for study in range(values.shape[0]):
        scores = np.delete(values, study, axis=0).mean(axis=(0, 1, 2))
        test_scores = values[study].mean(axis=(0, 1))
        calibration.append(scores)
        heldout.append(test_scores)
        chosen.append(int(np.argmin(scores)))
    return {'methods': methods, 'selected_index': np.asarray(chosen),
            'calibration_error': np.asarray(calibration), 'heldout_error': np.asarray(heldout)}


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


def expanded_constraint_centers(historical_profiles, original_library, additional_library):
    """Blend historical factors with existing and added source-category means."""
    if not set(additional_library).issubset(original_library):
        raise ValueError('Added categories must exist in the original library')
    source_constraint_centers(historical_profiles, original_library)
    source_constraint_centers(historical_profiles, additional_library)
    combined = {}
    for factor, original in original_library.items():
        if not original:
            raise ValueError('Original source categories cannot be empty')
        if factor in additional_library:
            additional = additional_library[factor]
            if not additional:
                raise ValueError('Additional source categories cannot be empty')
            combined[factor] = {'original_mean': np.mean(list(original.values()), axis=0),
                                'additional_mean': np.mean(list(additional.values()), axis=0)}
        else:
            combined[factor] = original
    return source_constraint_centers(historical_profiles, combined)
