"""Sequential adjustment of intervals against completed PMF references."""
import numpy as np


class SequentialInterval:
    """One-factor scalar coverage update; call update only when a reference arrives."""

    def __init__(self, alpha=0.1, step=0.2):
        if not (np.isfinite(alpha) and 0 < alpha < 1 and np.isfinite(step) and step > 0):
            raise ValueError('Expected 0 < alpha < 1 and a positive step')
        self.alpha = float(alpha)
        self.step = float(step)
        self.q = 0.0

    def predict(self, lower, upper, scale):
        if not (np.isfinite([lower, upper, scale]).all() and 0 <= lower <= upper and scale > 0):
            raise ValueError('Expected nonnegative ordered bounds and a positive scale')
        return max(0.0, float(lower) - self.q * scale), float(upper) + self.q * scale

    def update(self, reference, lower, upper):
        if not (np.isfinite([reference, lower, upper]).all() and
                reference >= 0 and 0 <= lower <= upper):
            raise ValueError('Expected a nonnegative reference and ordered interval')
        missed = not lower <= reference <= upper
        self.q = max(0.0, self.q + self.step * (float(missed) - self.alpha))
        return self
