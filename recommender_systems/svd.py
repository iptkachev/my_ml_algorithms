import numpy as np
from tqdm import tqdm
from recommender_systems.matrix_factorization_base import MatrixFactorizationBase


class SVD(MatrixFactorizationBase):
    def __init__(self, factors: int = 128, iterations: int = 100, learning_rate: float = 5e-5,
                 l2_regularization: float = 1e-3):
        super().__init__()
        self.factors = factors
        self.iterations = iterations

        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization

        self._user_biases = None
        self._item_biases = None
        self._common_bias = None

        self._loss_by_iterations = []

    def fit(self, user_items):
        n_users = user_items.shape[0]
        n_items = user_items.shape[1]

        self._init_matrices(n_items, n_users)

        for _ in tqdm(range(self.iterations), position=0, leave=True):
            residual = self._residual(user_items)
            self.user_factors -= self.learning_rate * self._grad_user(residual)
            self._user_biases -= self.learning_rate * self._grad_user_bias(residual)
            self.item_factors -= self.learning_rate * self._grad_item(residual)
            self._item_biases -= self.learning_rate * self._grad_item_bias(residual)

            self._loss_by_iterations.append(self._loss(residual))

        self._compute_norms()

        return self

    def _init_matrices(self, n_items, n_users):
        self.item_factors = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_items, self.factors))
        self.user_factors = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_users, self.factors))
        self._item_biases = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_items, 1))
        self._user_biases = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_users, 1))
        self._common_bias = np.random.uniform(0, 1 / np.sqrt(self.factors))

    def _compute_norms(self):
        self.item_norms = np.linalg.norm(self.item_factors, axis=1)
        self.user_norms = np.linalg.norm(self.user_factors, axis=1)

    def _residual(self, user_items):
        return (
            self.user_factors @ self.item_factors.T + self._user_biases + self._item_biases.T - user_items
        )

    def _loss(self, residual):
        return (
            np.power(residual, 2).sum() +
            self.l2_regularization * (
                np.power(np.linalg.norm(self._user_biases), 2) + np.power(np.linalg.norm(self._item_biases), 2)
            )
        )

    def _grad_user(self, residual):
        return 2 * (residual @ self.item_factors + self.l2_regularization * self.user_factors)

    def _grad_user_bias(self, residual):
        return 2 * (residual.mean(axis=1) + self.l2_regularization * self._user_biases)

    def _grad_item(self, residual):
        return 2 * (residual.T @ self.user_factors + self.l2_regularization * self.item_factors)

    def _grad_item_bias(self, residual):
        return 2 * (residual.T.mean(axis=1) + self.l2_regularization * self._item_biases)
