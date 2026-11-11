import copy
import warnings
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Convex piecewise quadratics through instantaneous forward-rate points.

    For monotone ordinates with nondecreasing secant slopes, limit each
    interval's curvature so derivatives have the correct sign and cannot
    jump downward at knots. The curve is C0, not necessarily C1. Incompatible
    data use an explicitly warned piecewise-linear fallback without moving
    any quotes. Shape guarantees apply only inside the node interval.

    This is a custom point interpolator, not the Hagan-West algorithm for
    interval-average forwards. Only flat extrapolation is supported.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        axis = np.array(axis1, dtype=float, copy=True)
        ordinates = np.array(values, dtype=float, copy=True)
        if axis.ndim != 1 or ordinates.ndim != 1 or len(axis) != len(ordinates):
            raise ValueError("axis1 and values must be one-dimensional and equally sized")
        if len(axis) == 0 or not np.all(np.isfinite(axis)) or not np.all(np.isfinite(ordinates)):
            raise ValueError("At least one finite node and finite ordinate are required")
        if np.any(np.diff(axis) <= 0):
            raise ValueError("Node coordinates must be strictly increasing")
        if extrapolation_method != ExtrapMethod.FLAT:
            raise ValueError("MONOTONE_CONVEX supports only FLAT extrapolation")
        super().__init__(axis, ordinates, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)

        self.widths_ = np.diff(axis)
        self.slopes_ = np.diff(ordinates) / self.widths_
        monotone = np.all(self.slopes_ >= 0) or np.all(self.slopes_ <= 0)
        convex = np.all(np.diff(self.slopes_) >= 0)
        self.shape_compatible = bool(monotone and convex)
        self.fallback_reason = None
        self.curvatures_ = np.zeros_like(self.slopes_)
        if not self.shape_compatible:
            failures = []
            if not monotone:
                failures.append("ordinates are not monotone")
            if not convex:
                failures.append("secant slopes are not nondecreasing")
            self.fallback_reason = "; ".join(failures)
            warnings.warn(
                "Using piecewise-linear fallback: " + self.fallback_reason
                + ". Exact node values are retained; global monotonicity and "
                "convexity cannot both be guaranteed.",
                RuntimeWarning, stacklevel=2,
            )
        elif len(self.slopes_) > 1:
            gaps = np.diff(self.slopes_)
            # Each side gets at most half the available slope gap.
            # Thus c[i] + c[i+1] <= s[i+1] - s[i].
            limits = np.full_like(self.slopes_, np.inf)
            limits[:-1] = np.minimum(limits[:-1], 0.5 * gaps)
            limits[1:] = np.minimum(limits[1:], 0.5 * gaps)
            # c <= |s| also keeps both endpoint derivatives the same sign.
            self.curvatures_ = np.minimum(np.abs(self.slopes_), limits)

        areas = self.widths_ * (ordinates[:-1] + ordinates[1:]) / 2.0
        areas -= self.curvatures_ * self.widths_ ** 2 / 6.0
        self.integrals_at_nodes_ = np.concatenate(([0.0], np.cumsum(areas)))

    @staticmethod
    def _finite_scalar(x: float) -> float:
        if np.ndim(x) != 0 or not np.isfinite(x):
            raise ValueError("Evaluation coordinates must be finite scalars")
        return float(x)

    def interpolate(self, x: float) -> float:
        x = self._finite_scalar(x)
        if self.length_ == 1 or x <= self.axis1_[0]:
            return float(self.values_[0])
        if x >= self.axis1_[-1]:
            return float(self.values_[-1])
        i = int(np.searchsorted(self.axis1_, x, side='right') - 1)
        d = x - self.axis1_[i]
        c = self.curvatures_[i]
        return float(self.values_[i] + d * (
            self.slopes_[i] - c + c * d / self.widths_[i]
        ))

    def _primitive(self, x: float) -> float:
        """Signed integral from the first node to x, including flat tails."""
        if self.length_ == 1 or x <= self.axis1_[0]:
            return float((x - self.axis1_[0]) * self.values_[0])
        if x >= self.axis1_[-1]:
            return float(self.integrals_at_nodes_[-1]
                         + (x - self.axis1_[-1]) * self.values_[-1])
        i = int(np.searchsorted(self.axis1_, x, side='right') - 1)
        d = x - self.axis1_[i]
        c = self.curvatures_[i]
        return float(self.integrals_at_nodes_[i] + self.values_[i] * d
                     + (self.slopes_[i] - c) * d ** 2 / 2.0
                     + c * d ** 3 / (3.0 * self.widths_[i]))

    def integrate(self, start_x: float, end_x: float) -> float:
        start = self._finite_scalar(start_x)
        end = self._finite_scalar(end_x)
        return self._primitive(end) - self._primitive(start)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
