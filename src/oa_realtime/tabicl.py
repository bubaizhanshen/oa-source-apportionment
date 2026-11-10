"""TabICLv2 estimates with optional signal normalization."""
import numpy as np
from .models import normalized_features


class TabICLSourceRegressor:
    """Fit one pretrained tabular regressor per source factor."""

    def __init__(self, normalized=False, model_path=None, device='cpu', n_jobs=1,
                 n_estimators=8, batch_size=2, cache=False):
        self.normalized = normalized
        self.model_path = model_path
        self.device = device
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.cache = cache

    def _inputs(self, x):
        if self.normalized:
            return normalized_features(x)
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError('Spectra must be a finite matrix')
        return values, np.ones(len(values), dtype=float)

    def fit(self, x, contributions):
        from tabicl import TabICLRegressor
        features, scale = self._inputs(x)
        y = np.asarray(contributions, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != len(scale) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('Historical contributions must be aligned, finite, and nonnegative')
        self.n_features_ = features.shape[1]
        self.models_ = []
        for k in range(y.shape[1]):
            model = TabICLRegressor(n_estimators=self.n_estimators, n_jobs=self.n_jobs,
                batch_size=self.batch_size, device=self.device, use_amp=False, use_fa3=False,
                kv_cache=self.cache, model_path=self.model_path,
                allow_auto_download=self.model_path is None, random_state=42)
            model.fit(features, y[:, k] / scale)
            self.models_.append(model)
        return self

    def predict(self, x, quantiles=None):
        if not hasattr(self, 'models_'):
            raise ValueError('Fit the historical spectra before prediction')
        features, scale = self._inputs(x)
        if features.shape[1] != self.n_features_:
            raise ValueError('Current and historical spectra must use the same channels')
        means, intervals = [], []
        if quantiles is not None:
            quantiles = np.asarray(quantiles, dtype=float)
            if (quantiles.ndim != 1 or len(quantiles) == 0 or
                    not np.isfinite(quantiles).all() or np.any((quantiles <= 0) | (quantiles >= 1))):
                raise ValueError('Quantiles must be a nonempty sequence between zero and one')
        for model in self.models_:
            if quantiles is None:
                mean = model.predict(features, output_type='mean')
            else:
                prediction = model.predict(features, output_type=['mean', 'quantiles'],
                                           alphas=quantiles.tolist())
                mean = prediction['mean']
                intervals.append(np.maximum(np.asarray(prediction['quantiles']) * scale[:, None], 0.0))
            means.append(np.maximum(np.asarray(mean) * scale, 0.0))
        mean = np.column_stack(means)
        if quantiles is None:
            return mean
        return {'mean': mean, 'quantiles': np.stack(intervals, axis=1)}
