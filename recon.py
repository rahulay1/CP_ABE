# recon.py
import numpy as np
from config import q

def _modinv(a: int, p: int = q) -> int:
    a %= p
    if a == 0:
        raise ZeroDivisionError("no inverse mod q")
    return pow(a, p - 2, p)  # q is prime

def _rref_solve_rect(A: np.ndarray, b: np.ndarray):
    A = (A % q).astype(np.int64, copy=True)
    b = (b % q).astype(np.int64, copy=True)
    m, n = A.shape
    M = np.concatenate([A, b.reshape(m, 1)], axis=1)

    row = 0
    pivots = [-1] * n
    for col in range(n):
        pivot = -1
        for r in range(row, m):
            if M[r, col] % q != 0:
                pivot = r; break
        if pivot == -1:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        inv = _modinv(int(M[row, col]))
        M[row, :] = (M[row, :] * inv) % q
        for r in range(m):
            if r == row: continue
            f = int(M[r, col] % q)
            if f != 0:
                M[r, :] = (M[r, :] - f * M[row, :]) % q
        pivots[col] = row
        row += 1
        if row == m: break

    for r in range(m):
        if np.all(M[r, :n] % q == 0) and (M[r, n] % q != 0):
            return False, np.zeros(n, dtype=np.int64)

    x = np.zeros(n, dtype=np.int64)
    for col in range(n):
        r = pivots[col]
        if r != -1:
            x[col] = int(M[r, n] % q)
    return True, x

def reconstruct_w(F: np.ndarray, indices: list[int]):
    """
    Find weights w_sub for the subset of rows 'indices' so that:
        sum_i w_i * F[indices[i], :] = e1
    Returns (ok, w_sub) where len(w_sub) == len(indices).
    """
    I = list(indices)
    if len(I) == 0:
        return False, np.zeros(0, dtype=np.int64)

    D = F.shape[1]
    A = (F[I, :].T) % q                   # D x s
    b = np.zeros(D, dtype=np.int64); b[0] = 1

    ok, w = _rref_solve_rect(A, b)
    return ok, (w % q)

def is_authorized(F, rho, user_bits, return_w=False):
    """
    Build the usable row set from 'rho' and 'user_bits', then try to reconstruct e1.
    Returns:
      - if return_w=False: bool
      - if return_w=True: (bool, {"active_set":[...], "w": full_row_aligned_weights})
    """
    # map 'attr12' -> 12; anything else -> None
    row_to_attr = []
    for name in rho:
        name = str(name)
        if name.startswith("attr"):
            try:
                row_to_attr.append(int(name[4:]))
            except Exception:
                row_to_attr.append(None)
        else:
            row_to_attr.append(None)

    indices = []
    for i, a in enumerate(row_to_attr):
        if a is None:
            continue
        if 0 <= a < len(user_bits) and int(user_bits[a]) == 1:
            indices.append(i)

    ok, w_sub = reconstruct_w(F, indices)

    if not return_w:
        return ok

    w_full = np.zeros(F.shape[0], dtype=np.int64)
    if ok:
        for idx, coeff in zip(indices, w_sub):
            w_full[idx] = int(coeff) % q

    info = {
        "active_set": indices,
        "w": w_full
    }
    return ok, info
