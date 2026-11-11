"""Historical-profile fitting and PMF-trained regressors."""
import time
import warnings
import numpy as np
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEEDS = (176, 277, 378)

def candidate_list(method, n_features):
    if method == 'ridge':
        return [{'alpha': x} for x in (0.1, 1.0, 10.0, 100.0, 1000.0)]
    if method == 'svr':
        return [{'C': c, 'gamma': multiplier / n_features, 'epsilon': 0.1}
                for c in (0.1, 1.0, 10.0) for multiplier in (0.25, 1.0, 4.0)]
    if method == 'mlp':
        return [{'hidden': list(hidden), 'alpha': a}
                for hidden in ((32,), (64, 32)) for a in (0.001, 0.1)]
    if method == 'knn':
        return [{'neighbors': k, 'weights': w} for k in (3, 8, 20) for w in ('uniform', 'distance')]
    raise ValueError(method)


def estimator(method, params, seed=SEEDS[0]):
    if method == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=params['alpha']))
    if method == 'knn':
        return make_pipeline(StandardScaler(), KNeighborsRegressor(
            n_neighbors=params['neighbors'], weights=params['weights'], n_jobs=1))
    if method == 'svr':
        reg = make_pipeline(StandardScaler(), MultiOutputRegressor(
            SVR(C=params['C'], gamma=params['gamma'], epsilon=params['epsilon']), n_jobs=1))
        return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
    if method == 'mlp':
        reg = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=tuple(params['hidden']), alpha=params['alpha'], activation='relu',
            solver='adam', batch_size='auto', learning_rate_init=0.001, max_iter=2500,
            tol=0.00005, n_iter_no_change=35, random_state=seed))
        return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
    raise ValueError(method)


def select_parameters(method, hx, hy):
    folds = list(TimeSeriesSplit(n_splits=3).split(hx))
    records = []
    start = time.perf_counter()
    for params in candidate_list(method, hx.shape[1]):
        losses, captured = [], []
        for train, valid in folds:
            assert train.max() < valid.min()
            model = estimator(method, params)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                model.fit(hx[train], hy[train])
            captured += [item.category.__name__ for item in w]
            p = np.maximum(np.asarray(model.predict(hx[valid]), dtype=float), 0.0)
            if p.shape != hy[valid].shape or not np.isfinite(p).all():
                raise ValueError('Invalid validation prediction')
            scale = np.maximum(np.std(hy[train], axis=0), 1e-8)
            losses.append(float(np.mean(np.abs(p - hy[valid]) / scale[None, :])))
        records.append({'parameters': params, 'fold_losses': losses, 'mean_loss': float(np.mean(losses)),
                        'warnings': captured})
    winner = min(range(len(records)), key=lambda i: (records[i]['mean_loss'], i))
    return records[winner]['parameters'], records, time.perf_counter() - start


def fit_fixed_profiles(x, u, profiles, groups=None):
    """Fit each new spectrum using historical profiles and its uncertainties."""
    x, u, profiles = map(lambda a: np.asarray(a, dtype=float), (x, u, profiles))
    if x.ndim == 1:
        x, u = x[None, :], u[None, :]
    if x.shape != u.shape or profiles.ndim != 2 or x.shape[1] != profiles.shape[1]:
        raise ValueError('Incompatible spectrum, uncertainty, or profile dimensions')
    if not np.isfinite(x).all() or not np.isfinite(u).all() or (u <= 0).any():
        raise ValueError('Spectra and positive uncertainties must be finite')
    coefficients = np.array([nnls((profiles / sig[None, :]).T, value / sig, maxiter=5000)[0]
                             for value, sig in zip(x, u)])
    if groups is None:
        return coefficients
    return np.column_stack([coefficients[:, indices].sum(axis=1) for indices in groups])


def fit_profile_regularized(x, u, profiles, estimates, weight, groups=None):
    """Fit nonnegative profiles while penalizing deviation from source estimates.

    ``groups`` partitions profile rows into the estimated sources. Profiles
    and estimates must use compatible units. The penalty is ``weight`` times
    the mean diagonal of the uncertainty-weighted profile cross-product.
    Returns one column per profile; zero weight gives ordinary profile fitting.
    """
    x, u, profiles, estimates = (np.asarray(a, dtype=float) for a in (x, u, profiles, estimates))
    if x.ndim == 1:
        x, u = x[None, :], u[None, :]
    if (x.ndim != 2 or x.shape != u.shape or profiles.ndim != 2 or not len(profiles)
            or x.shape[1] != profiles.shape[1]):
        raise ValueError('Incompatible spectrum, uncertainty, or profile dimensions')
    if (not np.isfinite(x).all() or not np.isfinite(u).all() or (u <= 0).any()
            or not np.isfinite(profiles).all() or (profiles < 0).any()
            or (np.linalg.norm(profiles, axis=1) <= 0).any()):
        raise ValueError('Spectra and profiles must be finite and uncertainties positive')
    if not np.isscalar(weight) or not np.isfinite(weight) or weight < 0:
        raise ValueError('weight must be finite and nonnegative')
    if groups is None:
        mapping = np.eye(len(profiles))
    else:
        groups = [list(indices) for indices in groups]
        flat = [index for indices in groups for index in indices]
        if (not groups or any(not indices for indices in groups)
                or any(not isinstance(i, (int, np.integer)) for i in flat)
                or sorted(flat) != list(range(len(profiles)))):
            raise ValueError('groups must partition the profile rows')
        mapping = np.zeros((len(groups), len(profiles)))
        for i, indices in enumerate(groups):
            mapping[i, indices] = 1.0
    if estimates.ndim == 1:
        estimates = estimates[None, :] if len(x) == 1 else estimates[:, None]
    if (estimates.shape != (len(x), len(mapping)) or not np.isfinite(estimates).all()
            or (estimates < 0).any()):
        raise ValueError('Source estimates must be aligned, finite, and nonnegative')
    result = []
    for spectrum, sigma, estimate in zip(x, u, estimates):
        design = (profiles / sigma).T
        target = spectrum / sigma
        regularization = weight * np.sum(design * design) / np.sum(mapping * mapping)
        matrix = np.vstack((design, np.sqrt(regularization) * mapping))
        response = np.concatenate((target, np.sqrt(regularization) * estimate))
        result.append(nnls(matrix, response, maxiter=5000)[0])
    return np.asarray(result)


def fit_weighted_elastic_net(x, u, profiles, mz, groups=None):
    """Fit normalized historical profiles using the CASA-style weighted elastic net.

    Rows of profiles are factors; columns are the m/z channels in x, u, and mz.
    Settings reproduce the earlier CASA fitting function used in the comparison.
    """
    x, u, profiles = map(lambda a: np.asarray(a, dtype=float), (x, u, profiles))
    mz = np.asarray(mz)
    if x.ndim == 1:
        x, u = x[None, :], u[None, :]
    if (x.ndim != 2 or x.shape != u.shape or profiles.ndim != 2 or
            x.shape[1] != profiles.shape[1] or mz.shape != (x.shape[1],)):
        raise ValueError('Incompatible spectrum, uncertainty, profile, or m/z dimensions')
    if (not np.isfinite(x).all() or not np.isfinite(u).all() or
            not np.isfinite(profiles).all() or (u <= 0).any()):
        raise ValueError('Spectra and profiles must be finite and uncertainties positive')
    weights = 1.0 / np.maximum(np.maximum(u, 0.05 * np.abs(x)), np.finfo(float).eps)**2
    for channel, multiplier in {43: 1.5, 44: 2.0, 55: 1.4, 57: 1.8, 60: 2.5, 73: 2.0}.items():
        weights[:, mz == channel] *= multiplier
    solver = ElasticNet(alpha=1e-6, l1_ratio=0.5, positive=True, fit_intercept=False,
                        max_iter=10000, tol=1e-4)
    coefficients = []
    for spectrum, weight in zip(x, weights):
        solver.fit(profiles.T, spectrum, sample_weight=weight)
        coefficients.append(solver.coef_.copy())
    coefficients = np.asarray(coefficients)
    if groups is None:
        return coefficients
    return np.column_stack([coefficients[:, indices].sum(axis=1) for indices in groups])


def fit_tree_models(method, x, y, parameters):
    """Fit one previously selected tree model per source contribution."""
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    builders = {
        'xgboost': lambda p: XGBRegressor(objective='reg:squarederror', tree_method='hist',
                                           n_jobs=1, random_state=2026, verbosity=0, **p),
        'lightgbm': lambda p: LGBMRegressor(objective='regression', n_jobs=1, verbosity=-1,
                                            random_state=2026, subsample_freq=1, **p),
        'random_forest': lambda p: RandomForestRegressor(n_jobs=1, random_state=2026,
                                                        bootstrap=True, **p),
    }
    if method not in builders or len(parameters) != y.shape[1]:
        raise ValueError('Select one parameter dictionary per output')
    models = [builders[method](p) for p in parameters]
    for k, model in enumerate(models):
        model.fit(x, y[:, k])
    return models


def predict_trees(method, models, x):
    x = np.atleast_2d(np.asarray(x))
    output = []
    for model in models:
        if method == 'xgboost':
            values = model.get_booster().inplace_predict(x)
        elif method == 'lightgbm':
            values = model.booster_.predict(x, num_threads=1)
        else:
            values = model.predict(x)
        output.append(np.maximum(np.asarray(values, dtype=float), 0.0))
    return np.column_stack(output)


def fit_regressors(method, x, y, parameters):
    seeds = SEEDS if method == 'mlp' else SEEDS[:1]
    models = [estimator(method, parameters, seed) for seed in seeds]
    for model in models:
        model.fit(x, y)
    return models


def predict_regressors(models, x):
    predictions = np.stack([np.asarray(model.predict(x), dtype=float) for model in models])
    return np.maximum(predictions.mean(axis=0), 0.0)


def normalized_features(x, uncertainty=None):
    """Scale spectra, and optional uncertainties, by each positive signal sum."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Spectra must be a finite matrix')
    scale = np.maximum(x, 0).sum(axis=1)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Each spectrum must have a positive signal sum')
    if uncertainty is None:
        features = x
    else:
        u = np.asarray(uncertainty, dtype=float)
        if u.ndim == 1:
            u = u[None, :]
        if u.shape != x.shape or not np.isfinite(u).all() or (u <= 0).any():
            raise ValueError('Uncertainties must match the spectra and be positive')
        features = np.column_stack((x, u))
    return features / scale[:, None], scale


class SourceScalingRegressor:
    """Compare spectrum and target scaling with a shared per-source estimator."""

    def __init__(self, estimator, input_mode='shape', scale_target=True, weight_signal=False):
        if input_mode not in ('raw', 'shape', 'shape_signal'):
            raise ValueError('input_mode must be raw, shape, or shape_signal')
        if weight_signal and not scale_target:
            raise ValueError('Signal-squared weights require scaled targets')
        self.estimator = estimator
        self.input_mode = input_mode
        self.scale_target = scale_target
        self.weight_signal = weight_signal

    def _features(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        shape, scale = normalized_features(x)
        if self.input_mode == 'raw':
            features = x
        elif self.input_mode == 'shape_signal':
            features = np.column_stack((shape, np.log(scale)))
        else:
            features = shape
        return features, scale

    def fit(self, x, contributions):
        features, scale = self._features(x)
        y = np.asarray(contributions, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != len(scale) or not np.isfinite(y).all():
            raise ValueError('Historical contributions must align with spectra')
        if self.scale_target:
            y = y / scale[:, None]
        if not np.isfinite(y).all():
            raise ValueError('Scaled contributions must be finite')
        weights = None
        if self.weight_signal:
            weights = np.square(scale)
            if not np.isfinite(weights).all() or weights.mean() <= 0:
                raise ValueError('Signal-squared weights must be finite and positive')
            weights = weights / weights.mean()
        self.models_ = []
        for k in range(y.shape[1]):
            model = clone(self.estimator)
            if weights is None:
                model.fit(features, y[:, k])
            else:
                model.fit(features, y[:, k], sample_weight=weights)
            self.models_.append(model)
        self.n_features_in_ = np.atleast_2d(np.asarray(x)).shape[1]
        return self

    def predict(self, x):
        if not hasattr(self, 'models_'):
            raise ValueError('Fit the historical spectra before prediction')
        if np.atleast_2d(np.asarray(x)).shape[1] != self.n_features_in_:
            raise ValueError('Current and historical spectra must use the same channels')
        features, scale = self._features(x)
        prediction = np.column_stack([model.predict(features) for model in self.models_])
        if self.scale_target:
            prediction = prediction * scale[:, None]
        if not np.isfinite(prediction).all():
            raise ValueError('Predicted contributions must be finite')
        return np.maximum(prediction, 0.0)


class NearestShapeRegressor:
    """Estimate factor-to-signal ratios from the nearest historical spectra."""

    def __init__(self, neighbors=5):
        if not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError('neighbors must be a positive integer')
        self.neighbors = neighbors

    def fit(self, x, contributions):
        features, scale = normalized_features(x)
        y = np.asarray(contributions, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if (y.ndim != 2 or y.shape[0] != len(scale) or not np.isfinite(y).all()
                or len(scale) < self.neighbors):
            raise ValueError('Historical contributions or neighbor count are invalid')
        self.shapes_ = features.copy()
        self.ratios_ = y / scale[:, None]
        return self

    def predict(self, x):
        if not hasattr(self, 'shapes_'):
            raise ValueError('Fit the historical spectra before prediction')
        features, scale = normalized_features(x)
        if features.shape[1] != self.shapes_.shape[1]:
            raise ValueError('Current and historical spectra must use the same channels')
        distances = np.maximum(
            np.sum(features * features, axis=1)[:, None]
            + np.sum(self.shapes_ * self.shapes_, axis=1)[None, :]
            - 2 * features @ self.shapes_.T, 0.0)
        nearest = np.argsort(distances, axis=1, kind='stable')[:, :self.neighbors]
        return self.ratios_[nearest].mean(axis=1) * scale[:, None]


def fit_normalized_trees(method, x, contributions, parameters, uncertainty=None):
    """Fit per-source trees to normalized historical spectra and PMF labels."""
    features, scale = normalized_features(x, uncertainty)
    y = np.asarray(contributions, dtype=float)
    if y.ndim != 2 or y.shape[0] != len(scale) or not np.isfinite(y).all():
        raise ValueError('Historical contributions must align with spectra')
    return fit_tree_models(method, features, y / scale[:, None], parameters)


def predict_normalized_trees(method, models, x, uncertainty=None):
    features, scale = normalized_features(x, uncertainty)
    return predict_trees(method, models, features) * scale[:, None]


def fit_normalized_regressors(method, x, contributions, parameters, uncertainty=None):
    """Apply the same representation to the non-tree regressors."""
    features, scale = normalized_features(x, uncertainty)
    y = np.asarray(contributions, dtype=float)
    if y.ndim != 2 or y.shape[0] != len(scale) or not np.isfinite(y).all():
        raise ValueError('Historical contributions must align with spectra')
    return fit_regressors(method, features, y / scale[:, None], parameters)


def predict_normalized_regressors(models, x, uncertainty=None):
    features, scale = normalized_features(x, uncertainty)
    return predict_regressors(models, features) * scale[:, None]
