import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities

        # Prevent log(0) and log(1 - 0) / log(1 - 1)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        n = len(y_true)

        loss = -1 / n * np.sum(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        return np.round(loss, 4)

    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        # y_true: one-hot encoded true labels
        # Shape: (n_samples, n_classes)
        #
        # y_pred: predicted probabilities
        # Shape: (n_samples, n_classes)

        # Prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        n = len(y_true)

        loss = -1 / n * np.sum(
            y_true * np.log(y_pred)
        )

        return np.round(loss, 4)