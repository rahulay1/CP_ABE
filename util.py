import random
import numpy as np
from config import q, f, sigma  # added sigma import

####################################
# Polynomial generation
####################################

def gen_polynomial():
    return np.array([random.randrange(0, q) for _ in range(f)], dtype=int)

def gen_polynomial_gauss():
    vals = [round(random.gauss(0, sigma)) for _ in range(f)]
    return np.array([int(v) % q for v in vals], dtype=int)

def gen_multiple_polynomials(m):
    return np.array([gen_polynomial() for _ in range(m)], dtype=int)

def gen_multiple_polynomials_gauss(m):
    return np.array([gen_polynomial_gauss() for _ in range(m)], dtype=int)

####################################
# Polynomial arithmetic in R_q[x]/(x^f+1)
####################################

def poly_mod(coeffs):
    arr = np.array(coeffs, dtype=int) % q
    for i in range(len(arr) - f):
        arr[i + f] = (arr[i + f] - arr[i]) % q
    return arr[-f:] % q

def poly_add(a, b):
    s = np.polyadd(a[::-1], b[::-1])
    return poly_mod(s[::-1])

def poly_mul(a, b):
    p = np.convolve(a[::-1], b[::-1])
    return poly_mod(p[::-1])

def poly_dotprod(L1, L2):
    out = np.zeros(f, dtype=int)
    for a, b in zip(L1, L2):
        out = poly_add(out, poly_mul(a, b))
    return out

def poly_mul_scalar(poly, s):
    return np.array([(c * s) % q for c in poly], dtype=int)

####################################
# Modular inverse & Lagrange
####################################

def modinv(a, m_mod):
    t, new_t = 0, 1
    r, new_r = m_mod, a % m_mod
    while new_r:
        qquot = r // new_r
        t, new_t = new_t, t - qquot * new_t
        r, new_r = new_r, r - qquot * new_r
    if r > 1:
        raise ValueError("no inverse")
    if t < 0:
        t += m_mod
    return t

def lagrange_coeff(i, xs, m_mod):
    num, den = 1, 1
    xi = xs[i]
    for j, xj in enumerate(xs):
        if j == i:
            continue
        num = (num * (-xj % m_mod)) % m_mod
        den = (den * ((xi - xj) % m_mod)) % m_mod
    return (num * modinv(den, m_mod)) % m_mod

####################################
# Stubbed GPV preimage sampler
####################################

def SamplePre(cc, A_np, T_tuple, u_np):
    """
    Stub: returns zero‐matrix (m × f) so shapes align.
    """
    from config import m
    return np.zeros((m, f), dtype=int)

def poly_op(a, s1, s2):
    """
    Compute a * s1 + s2 in R_q[x]/(x^f+1).
    Equivalent to poly_add( poly_mul(a,s1), s2 ).
    """
    return poly_add(poly_mul(a, s1), s2)
