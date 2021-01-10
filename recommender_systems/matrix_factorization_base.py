from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from scipy.sparse import csr_matrix


class MatrixFactorizationBase(ABC):
    def __init__(self, factors=128, iterations=100, l2_regularization=5e-2, compute_loss=False):
        self.factors = factors
        self.iterations = iterations
        self.l2_regularization = l2_regularization

        self.item_factors: np.array = None
        self.user_factors: np.array = None

        self.item_norms: np.array = None
        self.user_norms: np.array = None

        self._compute_loss = compute_loss
        self._loss_by_iterations = []

    @abstractmethod
    def fit(self, user_items: csr_matrix) -> MatrixFactorizationBase:
        pass

    def _init_matrices(self, n_items, n_users):
        self.item_factors = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_items, self.factors))
        self.user_factors = np.random.uniform(0, 1 / np.sqrt(self.factors), (n_users, self.factors))

    def _compute_factors_norms(self):
        self.item_norms = np.linalg.norm(self.item_factors, axis=1)
        self.user_norms = np.linalg.norm(self.user_factors, axis=1)

    def _compute_l2_norm(self, x):
        return np.power(np.linalg.norm(x), 2)

    def similar_items(self, item_id: int, top_k: int = 10):
        """
        By cosine similarity
        :param top_k:
        :param item_id:
        :return:
        """
        item_vector = self.item_factors[item_id]
        item_norm = self.item_norms[item_id]
        cos_similar = self.item_factors.dot(item_vector) / (self.item_norms * item_norm + 1e-8)
        top_k_similar = np.argsort(cos_similar)[-top_k:][::-1]

        return list(zip(top_k_similar.tolist(), cos_similar.take(top_k_similar).tolist()))

    def recommend(self, user_id: int, user_items: List[int] = None, top_k: int = 10):
        """
        By top dot product with item_factors
        :param top_k:
        :param user_id:
        :param user_items: items to exclude
        :return:
        """
        if not user_items:
            user_items = []

        user_vector = self.user_factors[user_id]
        user_products = self.item_factors.dot(user_vector)
        top_k_recommend = np.delete(np.argsort(user_products), user_items, axis=0)[-top_k:][::-1]

        return list(zip(top_k_recommend.tolist(), user_products.take(top_k_recommend).tolist()))
