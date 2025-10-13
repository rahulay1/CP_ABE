# KGC.py
import numpy as np
from util import poly_add
from config import u, N, q, f

def Setup():
    """
    Build E(θ) = u + sum_{i=1}^{N-1} G_i * θ^i  (coeff-wise over R_q),
    with deg E <= N-1 so that sum_i L_i(0) * E(θ_i) = E(0) = u.
    Returns an array [E(1), E(2), ..., E(N)] of length-f polynomials.
    """
    # G has length N-1 (NOT N), each G_i is a polynomial (length f)
    G = np.array(
        [np.random.randint(0, q, size=f, dtype=int) for _ in range(N - 1)],
        dtype=int
    )

    def E(theta: int) -> np.ndarray:
        acc = u.copy()
        power = 1  # θ^0
        for gi in G:            # i = 1..N-1
            power = (power * theta) % q  # θ^i
            acc = poly_add(acc, (gi * power) % q)
        return acc % q

    return np.array([E(i) for i in range(1, N + 1)], dtype=int)
