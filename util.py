import numpy as np
from config import q, f, sigma, sigma_s, b, k

# ── Modular helpers ───────────────────────────────────────────────────────────

def _modarr(a):
    a = np.array(a, dtype=int)
    a %= q
    return a

def _center_mod_q_scalar(x):
    """Map Z_q to centered (-q/2, q/2] integer reps (as Python int)."""
    x = int(x) % q
    if x > q // 2:
        x -= q
    return x

# ── Ring polynomial ops over R_q = Z_q[x]/(x^f + 1) ───────────────────────────
# Implemented in Python ints to avoid NumPy 32/64-bit overflow for large q.

def poly_add(a, b):
    aa = list(np.array(a).reshape(-1)[:f])
    bb = list(np.array(b).reshape(-1)[:f])
    return np.array([(int(x) + int(y)) % q for x, y in zip(aa, bb)], dtype=int)

def poly_mul_scalar(p, s):
    s = int(s) % q
    if s == 0:
        return np.zeros(f, dtype=int)
    pp = list(np.array(p).reshape(-1)[:f])
    return np.array([(int(x) * s) % q for x in pp], dtype=int)

def poly_mul(a, b):
    """
    Overflow-safe negacyclic convolution in R_q with modulus x^f + 1:
      (a * b)[t] = sum_j a[j] * b[t-j]  with sign flip when index wraps mod f.
    Uses Python big-ints and reduces as it goes.
    """
    aa = [int(x) % q for x in np.array(a).reshape(-1)[:f]]
    bb = [int(x) % q for x in np.array(b).reshape(-1)[:f]]

    # Precompute nonzero terms in b to skip multiplies by zero.
    bb_nz = [(j, bb[j]) for j in range(f) if bb[j] != 0]
    if not bb_nz or all(ai == 0 for ai in aa):
        return np.zeros(f, dtype=int)

    res = [0] * f
    for i in range(f):
        ai = aa[i]
        if ai == 0:
            continue
        for j, bj in bb_nz:
            t = ai * bj  # Python big-int
            idx = i + j
            if idx >= f:
                idx -= f
                t = -t
            res[idx] = (res[idx] + t) % q
    return np.array(res, dtype=int)

def _as_poly(x):
    if isinstance(x, np.ndarray):
        x = x.astype(int)
        if x.shape == (f,):
            return False, x % q
        if x.size == 1:
            return True, np.array([int(x.item())], dtype=int)
    if isinstance(x, (int, np.integer)):
        return True, np.array([int(x)], dtype=int)
    arr = np.array(x, dtype=int).reshape(-1)
    if arr.shape[0] == f:
        return False, arr % q
    raise ValueError("Expected scalar or length-f polynomial.")

def poly_dotprod(vec_a, vec_b):
    A = list(vec_a)
    B = list(vec_b)
    assert len(A) == len(B), "Dotprod length mismatch."
    acc = np.zeros(f, dtype=int)
    for aj, bj in zip(A, B):
        a_is_scal, a = _as_poly(aj)
        b_is_scal, b = _as_poly(bj)
        if a_is_scal and b_is_scal:
            acc[0] = (int(acc[0]) + (int(a.item()) * int(b.item())) % q) % q
        elif a_is_scal and not b_is_scal:
            acc = poly_add(acc, poly_mul_scalar(b, int(a.item())))
        elif not a_is_scal and b_is_scal:
            acc = poly_add(acc, poly_mul_scalar(a, int(b.item())))
        else:
            acc = poly_add(acc, poly_mul(a, b))
    return acc % q

# ── Random/poly generators (toy/demo) ─────────────────────────────────────────

def _disc_gauss(n, std):
    return np.round(np.random.normal(0.0, std, size=n)).astype(int)

def gen_polynomial(gaussian=True, std=sigma):
    v = _disc_gauss(f, std) if gaussian else np.random.randint(0, q, size=f, dtype=int)
    return v % q

def gen_multiple_polynomials(m, gaussian=True, std=sigma):
    return np.array([gen_polynomial(gaussian=gaussian, std=std) for _ in range(m)], dtype=int)

def gen_multiple_polynomials_gauss(k_, std=sigma):
    return gen_multiple_polynomials(k_, gaussian=True, std=std)

# ── Lagrange utilities (at arbitrary x0; decrypt uses x0=0) ───────────────────

def lagrange_coeff_at(xs, i, x0, mod=q):
    """
    L_i(x0) for points xs[0..n-1] over Z_mod, i is 0-based.
    """
    xi = xs[i] % mod
    num, den = 1, 1
    for j, xj in enumerate(xs):
        if j == i:
            continue
        num = (num * ((x0 - xj) % mod)) % mod
        den = (den * ((xi - xj) % mod)) % mod
    return (num * pow(den % mod, mod - 2, mod)) % mod  # mod is prime in this demo

def lagrange_eval(xs, ys, x0, mod=q):
    """
    Evaluate sum_i ys[i] * L_i(x0) over Z_mod.
    Each ys[i] can be:
      - an int (in Z_mod), or
      - a polynomial array (length f), in which case scalar L_i multiplies the poly.
    """
    is_poly = isinstance(ys[0], (np.ndarray, list))
    acc = np.zeros_like(ys[0], dtype=int) if is_poly else 0
    for i in range(len(xs)):
        Li = lagrange_coeff_at(xs, i, x0, mod)
        if is_poly:
            acc = (acc + (np.array(ys[i], dtype=int) * Li)) % mod
        else:
            acc = (acc + (ys[i] * Li)) % mod
    return acc % mod

# Backwards-compatible helper used by AA.Decrypt (1-based i over xs)
def lagrange_coeff(i_one_based, xs_one_based, mod=q):
    return lagrange_coeff_at(xs_one_based, i_one_based - 1, 0, mod)

# ── Modular linear algebra over Z_q (overflow-safe) ───────────────────────────

def solve_linear_mod_q(A, b, mod=q):
    A = (np.array(A, dtype=object)) % mod
    b = (np.array(b, dtype=object).reshape(-1)) % mod
    n_rows, n_cols = A.shape
    row = 0
    pivots = []
    for col in range(n_cols):
        pivot = None
        for r in range(row, n_rows):
            if int(A[r, col]) % mod != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            b[row], b[pivot] = b[pivot], b[row]
        inv = pow(int(A[row, col]) % mod, mod - 2, mod)
        A[row, :] = (A[row, :] * inv) % mod
        b[row] = (b[row] * inv) % mod
        for r in range(n_rows):
            if r == row:
                continue
            factor = int(A[r, col]) % mod
            if factor:
                A[r, :] = (A[r, :] - (factor * A[row, :])) % mod
                b[r] = (b[r] - (factor * b[row])) % mod
        pivots.append(col)
        row += 1
        if row == n_rows:
            break
    for r in range(row, n_rows):
        if all(int(x) % mod == 0 for x in A[r, :]) and (int(b[r]) % mod != 0):
            return None
    x = np.zeros(n_cols, dtype=int)
    for r, c in enumerate(pivots):
        x[c] = int(b[r]) % mod
    return x

def solve_for_g(F, rows, mod=q):
    """
    Given F (h x m) and a list of usable row indices 'rows',
    find G (length h) with F[rows,:]^T * g = e1 (mod q), where e1=[1,0,...,0].
    Returns full-length G with zeros for unused rows, or None if unsatisfiable.
    """
    rows = list(rows)
    if not rows:
        return None
    Fsub = (np.array(F[rows, :], dtype=int)) % mod
    A = (Fsub.T) % mod
    b = np.zeros(A.shape[0], dtype=int)
    b[0] = 1
    g_sub = solve_linear_mod_q(A, b, mod=mod)
    if g_sub is None:
        return None
    G = np.zeros(F.shape[0], dtype=int)
    for idx, gi in zip(rows, g_sub):
        G[idx] = int(gi) % mod
    return G

# ── Paper-faithful preimage sampler (trapdoor-based) ──────────────────────────

def _const_poly(c):
    arr = np.zeros(f, dtype=int)
    arr[0] = int(c) % q
    return arr

def build_J_list():
    """J_i = b^{i-1} as constant polynomials, i=1..k."""
    return [_const_poly(pow(b, i - 1, q)) for i in range(1, k + 1)]

def decompose_poly_base_b(t_poly):
    """
    Balanced base-2 decomposition (NAF-style) for each coefficient of t_poly.
    Produces k polys Y[i] so that sum_i J_i * Y[i] == t_poly (mod q),
    with per-coefficient digits in {-1,0,1}. Works for negative centered coeffs.
    """
    assert b == 2, "This decomposer is specialized for b=2"
    Y = [np.zeros(f, dtype=int) for _ in range(k)]

    for idx in range(f):
        # Center the coefficient in Z around 0
        T = _center_mod_q_scalar(t_poly[idx])  # T ∈ (−q/2, q/2]

        # Generate exactly k signed digits so no wrap vs q occurs
        for i in range(k):
            if T & 1:  # T is odd
                # digit ∈ {+1, −1} chosen by T mod 4 to minimize carry
                d = 2 - (T & 3)   # if T%4==1 -> +1, if T%4==3 -> −1 (works for neg T too)
            else:
                d = 0
            Y[i][idx] = d % q     # store in Z_q
            T = (T - d) // 2      # exact division (Python floor for neg handled by subtracting d first)

        # After k steps T MUST be 0 if k ≥ floor(log2(q))+1
        # If you want to assert in debug:
        # if T != 0: raise RuntimeError(f"Digit overflow at coeff {idx}: remaining {T}")

    return Y

def SamplePre(cc_unused, A, T, v):
    """
    Paper-faithful preimage sampler:
      1) sample ℓ' ∈ R_q^{m} (small)
      2) t = v - A ℓ'
      3) find Y ∈ R_q^k with J Y = t  (base-b digits)
      4) M = ℓ' + N Y, where N = [e'^T; r'^T; I_k]
    Output M ∈ R_q^{m} such that A M = v (mod q).
    """
    from config import m
    r_vec, e_vec = T  # each is length-k list of polys (shape (k,f))

    # 1) ℓ' small
    ell = [gen_polynomial(gaussian=True, std=sigma) for _ in range(m)]  # list length m

    # 2) t = v - A ℓ'
    t = (np.array(v, dtype=int) - poly_dotprod(A, ell)) % q  # poly length f

    # 3) digits Y with J Y = t
    Y = decompose_poly_base_b(t)  # list length k of polys

    # 4) M = ℓ' + N Y; N = [e'^T; r'^T; I_k]
    M = [None] * m
    acc_e = np.zeros(f, dtype=int)
    acc_r = np.zeros(f, dtype=int)
    for i in range(k):
        acc_e = (acc_e + poly_mul(e_vec[i], Y[i])) % q
        acc_r = (acc_r + poly_mul(r_vec[i], Y[i])) % q
    M[0] = (ell[0] + acc_e) % q
    M[1] = (ell[1] + acc_r) % q
    for i in range(k):
        M[2 + i] = (ell[2 + i] + Y[i]) % q

    return np.array(M, dtype=int)  # (m, f)
