from abc import ABC
import numpy as np

__all__ = ['Linear', 'Sigmoid', 'ReLU']


class ActivateFunc(ABC):
    @classmethod
    def activate(cls, *args, **kwargs):
        pass

    @classmethod
    def grad(cls, *args, **kwargs):
        pass


class Linear(ActivateFunc):
    @classmethod
    def activate(cls, x):
        return x

    @classmethod
    def grad(cls, x):
        return 1


class Sigmoid(ActivateFunc):
    @classmethod
    def activate(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def grad(cls, x):
        return cls.activate(x) * (1 - cls.activate(x))


class ReLU(ActivateFunc):
    @classmethod
    def activate(cls, x):
        return (x > 0) * x

    @classmethod
    def grad(cls, x):
        return (x > 0).astype('int64')
