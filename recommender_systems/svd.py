import numpy as np
from tqdm import tqdm
from recommender_systems.matrix_factorization_base import MatrixFactorizationBase


class SVD(MatrixFactorizationBase):
    """
    By https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)
    """
    def __init__(self, factors: int = 128, iterations: int = 100, learning_rate: float = 5e-5,
                 l2_regularization: float = 1e-3, compute_loss=False):
        super().__init__(factors, iterations, l2_regularization, compute_loss)
        self.learning_rate = learning_rate

        self._user_bias = None
        self._item_bias = None

    def fit(self, user_items):
        n_users, n_items = user_items.shape
        self._init_matrices(n_items, n_users)

        for _ in tqdm(range(self.iterations), position=0, leave=True):
            residual = self._residual(user_items)
            self.user_factors -= self.learning_rate * self._grad_user(residual)
            self._user_bias -= self.learning_rate * self._grad_user_bias(residual)
            self.item_factors -= self.learning_rate * self._grad_item(residual)
            self._item_bias -= self.learning_rate * self._grad_item_bias(residual)

            if self._compute_loss:
                self._loss_by_iterations.append(self._loss(residual))

        self._compute_factors_norms()

        return self

    def _init_matrices(self, n_items, n_users):
        super(SVD, self)._init_matrices(n_items, n_users)

        self._item_bias = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_items, 1))
        self._user_bias = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_users, 1))

    def _residual(self, user_items):
        return (
            self.user_factors @ self.item_factors.T + self._user_bias + self._item_bias.T - user_items
        )

    def _loss(self, residual):
        return (
            np.power(residual, 2).sum() +
            self.l2_regularization * (
                self._compute_l2_norm(self._user_bias) + self._compute_l2_norm(self._item_bias) +
                self._compute_l2_norm(self.user_factors) + self._compute_l2_norm(self.item_factors)
            )
        )

    def _grad_user(self, residual):
        return 2 * (residual @ self.item_factors + self.l2_regularization * self.user_factors)

    def _grad_user_bias(self, residual):
        return 2 * (residual.mean(axis=1) + self.l2_regularization * self._user_bias)

    def _grad_item(self, residual):
        return 2 * (residual.T @ self.user_factors + self.l2_regularization * self.item_factors)

    def _grad_item_bias(self, residual):
        return 2 * (residual.T.mean(axis=1) + self.l2_regularization * self._item_bias)
