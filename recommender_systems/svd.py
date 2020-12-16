import numpy as np
from tqdm import tqdm
from recommender_systems.matrix_factorization_base import MatrixFactorizationBase


class SVD(MatrixFactorizationBase):
    def __init__(self, factors: int, iterations: int, learning_rate: float,
                 l2_regularization: float = 0.005):
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
        n_users = user_items.shape[1]
        n_items = user_items.shape[0]

        self._init_matrices(n_items, n_users)

        for _ in tqdm(range(self.iterations), position=0, leave=True):
            rand_user_id = np.random.randint(0, n_users)
            rand_item_id = np.random.randint(0, n_items)

            self.user_factors -= self.learning_rate * self._grad_user(user_items, rand_item_id)
            self._user_biases -= self.learning_rate * self._grad_user_bias(user_items, rand_item_id)
            self.item_factors -= self.learning_rate * self._grad_item(user_items, rand_user_id)
            self._item_biases -= self.learning_rate * self._grad_item_bias(user_items, rand_user_id)
            self._common_bias -= self.learning_rate * self._grad_common_bias(user_items, rand_user_id, rand_item_id)

            self._loss_by_iterations.append(self._loss(user_items, rand_user_id, rand_item_id)[0])

        self._compute_norms()

        return self

    def _init_matrices(self, n_items, n_users):
        self.item_factors = np.random.uniform(-1, 1, (n_items, self.factors))
        self.user_factors = np.random.uniform(-1, 1, (n_users, self.factors))
        self._item_biases = np.random.uniform(-1, 1, (n_items, 1))
        self._user_biases = np.random.uniform(-1, 1, (n_users, 1))
        self._common_bias = np.random.uniform(-1, 1)

    def _compute_norms(self):
        self.item_norms = np.linalg.norm(self.item_factors, axis=1)
        self.user_norms = np.linalg.norm(self.user_factors, axis=1)

    def _loss(self, user_items, user_id, item_id):
        residual = (
            self.item_factors[item_id] @ self.user_factors[user_id] +
            self._user_biases[user_id] + self._item_biases[item_id] + self._common_bias - user_items[item_id, user_id]
        )

        return (
            np.power(residual, 2) +
            self.l2_regularization * np.power(np.linalg.norm(self._user_biases), 2) +
            self.l2_regularization * np.power(np.linalg.norm(self._item_biases), 2)
        )

    def _grad_user(self, user_items, item_id):
        residual = (
            self.user_factors @ self.item_factors[item_id] +
            self._user_biases.reshape(-1) + self._item_biases[item_id] + self._common_bias -
            user_items[item_id].reshape(1, -1)
        )

        return (
            2 * residual.reshape(-1, 1) @ self.item_factors[item_id].reshape(1, -1) +
            2 * self.l2_regularization * self.user_factors.sum(axis=1, keepdims=1)
        )

    def _grad_user_bias(self, user_items, item_id):
        residual = (
            self.user_factors @ self.item_factors[item_id] +
            self._user_biases.reshape(-1) + self._item_biases[item_id] + self._common_bias -
            user_items[item_id].reshape(1, -1)
        )

        return 2 * residual.reshape(-1, 1) + 2 * self.l2_regularization * self._user_biases.sum()

    def _grad_item(self, user_items, user_id):
        residual = (
            self.item_factors @ self.user_factors[user_id] +
            self._user_biases[user_id] + self._item_biases.reshape(-1) + self._common_bias -
            user_items[:, user_id].reshape(1, -1)
        )

        return (
            2 * residual.reshape(-1, 1) @ self.user_factors[user_id].reshape(1, -1) +
            2 * self.l2_regularization * self.item_factors.sum(axis=1, keepdims=1)
        )

    def _grad_item_bias(self, user_items, user_id):
        residual = (
            self.item_factors @ self.user_factors[user_id] +
            self._user_biases[user_id] + self._item_biases.reshape(-1) + self._common_bias -
            user_items[:, user_id].reshape(1, -1)
        )

        return 2 * residual.reshape(-1, 1) + 2 * self.l2_regularization * self._item_biases.sum()

    def _grad_common_bias(self, user_items, user_id, item_id):
        residual = (
            self.item_factors[item_id] @ self.user_factors[user_id].T +
            self._user_biases[user_id] + self._item_biases[item_id] + self._common_bias -
            user_items[item_id, user_id]
        )

        return 2 * residual
