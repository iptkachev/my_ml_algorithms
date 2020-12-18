import numpy as np
from tqdm import tqdm
from recommender_systems.matrix_factorization_base import MatrixFactorizationBase


class ALS(MatrixFactorizationBase):
    def __init__(self, factors: int = 128, iterations: int = 100, l2_regularization: float = 5e-2):
        super().__init__()
        self.factors = factors
        self.iterations = iterations

        self.l2_regularization = l2_regularization

        self._loss_by_iterations = []

    def fit(self, user_items):
        n_users = user_items.shape[0]
        n_items = user_items.shape[1]

        self._init_matrices(n_items, n_users)

        for _ in tqdm(range(self.iterations), position=0, leave=True):
            self.user_factors = self._grad_user(user_items).T
            self.item_factors = self._grad_item(user_items).T

            self._loss_by_iterations.append(self._loss(user_items))

        self._compute_norms()

    def _init_matrices(self, n_items, n_users):
        self.item_factors = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_items, self.factors))
        self.user_factors = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_users, self.factors))

    def _loss(self, user_items):
        return (
            np.power(user_items - self.user_factors @ self.item_factors.T, 2).sum() +
            self.l2_regularization * np.power(np.linalg.norm(self.user_factors, axis=1), 2).sum() +
            self.l2_regularization * np.power(np.linalg.norm(self.item_factors, axis=1), 2).sum()
        )

    def _grad_user(self, user_items):
        YtY = self.item_factors.T @ self.item_factors

        return (
            np.linalg.inv((YtY + self.l2_regularization * np.eye(self.factors))) @
            (self.item_factors.T @ user_items.T)
        )

    def _grad_item(self, user_items):
        UtU = self.user_factors.T @ self.user_factors

        return (
            np.linalg.inv(UtU + self.l2_regularization * np.eye(self.factors)) @
            (self.user_factors.T @ user_items)
        )
