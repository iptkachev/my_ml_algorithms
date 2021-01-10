import numpy as np
from tqdm import tqdm
from recommender_systems.matrix_factorization_base import MatrixFactorizationBase


class BPR(MatrixFactorizationBase):
    """
    By https://arxiv.org/pdf/1205.2618.pdf
    """
    def __init__(self, factors: int = 128, iterations: int = 100, learning_rate: float = 5e-5,
                 l2_regularization: float = 5e-2, compute_loss=False):
        super().__init__(factors, iterations, l2_regularization, compute_loss)
        self.learning_rate = learning_rate

    def fit(self, user_items):
        n_users, n_items = user_items.shape
        self._init_matrices(n_items, n_users)

        comparison_triplets = self._get_comparison_triplets(user_items)

        for i in tqdm(np.random.choice(range(n_users), self.iterations, True), position=0, leave=True):
            u_id, i_id, j_id = comparison_triplets[i]
            x_uij = self._compute_x_uij(u_id, i_id, j_id)

            self.user_factors[u_id] -= self.learning_rate * self._grad_theta(
                x_uij, self._grad_theta_u(i_id, j_id), self.user_factors[u_id]
            )
            self.item_factors[i_id] -= self.learning_rate * self._grad_theta(
                x_uij, self._grad_theta_i(u_id), self.item_factors[i_id]
            )
            self.item_factors[j_id] -= self.learning_rate * self._grad_theta(
                x_uij, self._grad_theta_j(u_id), self.item_factors[j_id]
            )

            if self._compute_loss:
                self._loss_by_iterations.append(self._loss(comparison_triplets))

        self._compute_factors_norms()

    def _get_comparison_triplets(self, user_items):
        def i2i_comparison(user_row):
            return user_row[None].T > user_row

        pairwise_preference_tensor = np.apply_along_axis(i2i_comparison, 1, user_items)
        comparison_triplets = list(zip(*np.where(pairwise_preference_tensor)))  # (user_id, i_id, j_id) : i_id > j_id

        return np.array(comparison_triplets)

    def _compute_x_uij(self, u_id, i_id, j_id):
        return (
            np.dot(self.user_factors[u_id], self.item_factors[i_id]) -
            np.dot(self.user_factors[u_id], self.item_factors[j_id])
        )

    def _loss(self, comparison_triplets):
        return (
            self._sigmoid(np.apply_along_axis(lambda triplet: self._compute_x_uij(*triplet), 1, comparison_triplets)).sum() +
            self.l2_regularization * (self._compute_l2_norm(self.user_factors) + self._compute_l2_norm(self.item_factors))
        )

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _grad_theta(self, x_uij, grad_x_uij_by_theta, theta):
        return np.exp(-x_uij) / (1 + np.exp(-x_uij)) * grad_x_uij_by_theta + self.l2_regularization * theta

    def _grad_theta_u(self, i_id, j_id):
        return self.item_factors[i_id] - self.item_factors[j_id]

    def _grad_theta_i(self, user_id):
        return self.user_factors[user_id]

    def _grad_theta_j(self, user_id):
        return -self.user_factors[user_id]
