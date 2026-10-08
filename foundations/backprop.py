import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:

    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float
    ) -> Tuple[NDArray[np.float64], float]:

        # Forward pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))

        # dL/dy_hat
        dL_dy_hat = y_hat - y_true

        # dy_hat/dz
        dy_hat_dz = y_hat * (1 - y_hat)

        # dL/dz
        dL_dz = dL_dy_hat * dy_hat_dz

        # dL/dw
        dL_dw = dL_dz * x

        # dL/db
        dL_db = dL_dz

        return np.round(dL_dw, 5), round(float(dL_db), 5)