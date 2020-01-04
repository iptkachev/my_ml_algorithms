from abc import ABC
import numpy as np

__all__ = ['Linear', 'Sigmoid', 'ReLU', 'LogLoss', 'MSELoss']


class ActivateFunc(ABC):
    @classmethod
    def activate(cls, *args, **kwargs):
        pass

    @classmethod
    def grad(cls, *args, **kwargs):
        pass


class LogLoss(ActivateFunc):
    @classmethod
    def activate(cls, y_true, y_pred):
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

    @classmethod
    def grad(cls, y_true, y_pred):
        return - (y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


class MSELoss(ActivateFunc):
    @classmethod
    def activate(cls, y_true, y_pred):
        return np.power(y_true - y_pred, 2).mean()

    @classmethod
    def grad(cls, y_true, y_pred):
        return - 2 / y_true.shape[0] * (y_true - y_pred)


class Linear(ActivateFunc):
    @classmethod
    def activate(cls, x):
        return x

    @classmethod
    def grad(cls, x):
        return np.ones(x.shape)


class Sigmoid(ActivateFunc):
    @classmethod
    def activate(cls, x):
        x[x > 100] = 100
        x[x < -100] = -100
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
