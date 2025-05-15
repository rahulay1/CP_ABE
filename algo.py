# algo.py
import numpy as np
from config import q, f, k, m
from util import gen_polynomial, gen_multiple_polynomials_gauss, poly_op, poly_add

def RTrapGen():
    """
    Real trapdoor generator for the GPV/LWE gadget:
      - a ∈ R_q is random
      - for i=0…k-1, sample short r_i,e_i ∼ D_{Z,σ}^f
        and set b_i = 2^i − (a*r_i + e_i) mod q
      - form A ∈ R_q^{(k+2)×f} with rows [1,0,…,0], a, b_0,…,b_{k-1}
      - return (A,(r,e)) as the public matrix and its trapdoor
    """
    # 1) sample the “a” polynomial
    a = gen_polynomial()               # length-f vector in R_q

    # 2) sample your k Gaussian short vectors
    r = gen_multiple_polynomials_gauss(k)  # shape (k,f)
    e = gen_multiple_polynomials_gauss(k)  # shape (k,f)

    # 3) build the first two rows of A
    #    row0 = [1,0,0,…,0], row1 = a
    A_const = np.zeros(f, dtype=int)
    A_const[0] = 1
    A_rows = [A_const, a]

    # 4) for each gadget‐row i, compute b_i = 2^i − (a*r_i + e_i)
    #    where poly_op(a,r_i,e_i) = a*r_i + e_i mod (x^f+1,q)
    for i in range(k):
        two_i = np.full(f, pow(2, i, q), dtype=int)       # constant poly = 2^i
        ai_ri_ei = poly_op(a, r[i], e[i])                  # a*r_i + e_i
        b_i    = poly_add(two_i, -ai_ri_ei)                # 2^i − (a*r_i+e_i)
        A_rows.append(b_i)

    # 5) stack into A ∈ ℤ_q^{m×f} (m = k+2)
    A = np.vstack(A_rows)  # shape (m,f)
    T = (r, e)             # the secret trapdoor

    return A, T
