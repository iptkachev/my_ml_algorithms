import numpy as np
from tqdm import tqdm
from recommender_systems.matrix_factorization_base import MatrixFactorizationBase


class ALS(MatrixFactorizationBase):
    """
    By http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf
    """
    def fit(self, user_items):
        n_users, n_items = user_items.shape
        self._init_matrices(n_items, n_users)

        for _ in tqdm(range(self.iterations), position=0, leave=True):
            self.user_factors = self._grad_user(user_items).T
            self.item_factors = self._grad_item(user_items).T

            if self._compute_loss:
                self._loss_by_iterations.append(self._loss(user_items))

        self._compute_factors_norms()

    def _loss(self, user_items):
        return (
            np.power(user_items - self.user_factors @ self.item_factors.T, 2).sum() +
            self.l2_regularization * (self._compute_l2_norm(self.user_factors) + self._compute_l2_norm(self.item_factors))
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
