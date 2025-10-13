import numpy as np
from config import q, f, b, k, m
from util import gen_polynomial, gen_multiple_polynomials_gauss, poly_add, poly_mul

def _const_poly(c):
    """Constant polynomial c ∈ Z_q as an f-coeff array."""
    arr = np.zeros(f, dtype=int)
    arr[0] = int(c) % q
    return arr

def RTrapGen():
    """
    Paper-faithful ring trapdoor:
      A = [ 1 , a , J1 - (a*r1 + e1) , ... , Jk - (a*rk + ek) ]  ∈ R_q^{m}
      with m = k + 2  and J_i = b^{i-1} (constant polynomial).
    Trapdoor T = (r', e') where r', e' ∈ R_q^k are small (Gaussian).
    """
    # sample public a ∈ R_q uniformly (poly)
    a = gen_polynomial(gaussian=False)

    # trapdoor small r', e' ∈ R_q^k
    r_vec = gen_multiple_polynomials_gauss(k)  # each length-f poly
    e_vec = gen_multiple_polynomials_gauss(k)

    A = np.zeros((m, f), dtype=int)
    A[0] = _const_poly(1)  # 1
    A[1] = a               # a

    # J_i = b^{i-1}, i = 1..k
    for i in range(1, k + 1):
        Ji = _const_poly(pow(b, i - 1, q))
        ari_ei = poly_add(poly_mul(a, r_vec[i - 1]), e_vec[i - 1])  # a*r_i + e_i
        A[1 + i] = (Ji - ari_ei) % q

    T = (r_vec, e_vec)
    return A, T
