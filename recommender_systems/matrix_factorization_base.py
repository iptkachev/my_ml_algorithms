from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from scipy.sparse import csr_matrix


class MatrixFactorizationBase(ABC):
    def __init__(self):
        self.item_factors: np.array = None
        self.user_factors: np.array = None

        self.item_norms: np.array = None
        self.user_norms: np.array = None

    @abstractmethod
    def fit(self, user_items: csr_matrix) -> MatrixFactorizationBase:
        pass

    def similar_items(self, item_id: int, top_k: int = 10):
        """
        By cosine similarity
        :param top_k:
        :param item_id:
        :return:
        """
        item_vector = self.item_factors[item_id]
        item_norm = self.item_norms[item_id]
        cos_similar = self.item_factors.dot(item_vector) / (self.item_norms * item_norm)
        top_k_similar = np.argsort(cos_similar)[-top_k:][::-1]

        return list(zip(top_k_similar.tolist(), cos_similar.take(top_k_similar).tolist()))

    def recommend(self, user_id: int, user_items: List[int] = [], top_k: int = 10):
        """
        By top dot product with item_factors
        :param top_k:
        :param user_id:
        :param user_items: items to exclude
        :return:
        """
        user_vector = self.user_factors[user_id]
        user_products = self.item_factors.dot(user_vector)
        top_k_similar = np.delete(np.argsort(user_products), user_items, axis=0)[-top_k:][::-1]

        return list(zip(top_k_similar.tolist(), user_products.take(top_k_similar).tolist()))
