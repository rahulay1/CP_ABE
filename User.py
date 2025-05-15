import random
import numpy as np

from config import attr, q, f, u, m, N
from util import (
    poly_add, poly_mul, poly_dotprod,
    gen_multiple_polynomials
)

class User:
    def __init__(self):
        self.F        = None
        self.rho      = None
        self.W        = None
        self.W_plus   = None
        self.W_minus  = None

    def AccessControl(self, X):
        mask1 = random.getrandbits(attr)
        mb1   = np.array([(mask1 >> i) & 1 for i in range(attr)],
                         dtype=int)
        self.W = X & mb1

        mask2      = random.getrandbits(attr)
        mb2        = np.array([(mask2 >> i) & 1 for i in range(attr)],
                              dtype=int)
        self.W_plus  = self.W & mb2
        self.W_minus = self.W & (1 - self.W_plus)

    def Encrypt(self, phi, A_list, b_plus_list, b_minus_list):
        if self.F is None:
            raise RuntimeError("LSSS matrix F not set on user.")

        # 1) Sample Σ
        Sigma = np.array([[random.randrange(0, q) for _ in range(f)]
                          for _ in range(m)], dtype=int)
        d  = Sigma[0]
        e  = np.zeros(f, dtype=int)
        ud = poly_mul(u, d)

        # 2) c₀ = 2·u·d + e + (q//2)·φ
        c0 = poly_add(
            poly_add([2*c for c in ud], e),
            [(q//2)*b for b in phi]
        )
        C  = {'c_0': c0}

        # 3) per-authority & per-attribute
        for θ in range(N):
            Aθ, _ = A_list[θ]

            # c_A_θ
            eA = gen_multiple_polynomials(m)
            cA = np.array([
                poly_add(poly_mul(Aθ[i], d), eA[i])
                for i in range(m)
            ], dtype=int)
            C[f'c_A_{θ}'] = cA

            # each attribute i under θ
            for i in range(attr // N):
                row = θ * (attr//N) + i
                Fi  = self.F[row]  # shape (m, f)

                c2 = poly_add(
                    poly_dotprod(Fi[1:], Sigma[1:]),
                    poly_mul(Fi[0], ud)
                )
                C[f'c_{θ}_{i}_2'] = c2

                e1 = gen_multiple_polynomials(m)
                if self.W_plus[row]:
                    bsel = b_plus_list[θ][i]
                    C[f'c_{θ}_{i}_1'] = np.array([
                        poly_add(poly_mul(bsel[j], d), e1[j])
                        for j in range(m)
                    ], dtype=int)

                elif self.W_minus[row]:
                    bsel = b_minus_list[θ][i]
                    C[f'c_{θ}_{i}_1'] = np.array([
                        poly_add(poly_mul(bsel[j], d), e1[j])
                        for j in range(m)
                    ], dtype=int)

                else:
                    bp = b_plus_list[θ][i]
                    bm = b_minus_list[θ][i]
                    C[f'c_plus_{θ}_{i}_1']  = np.array([
                        poly_add(poly_mul(bp[j], d), e1[j])
                        for j in range(m)
                    ], dtype=int)
                    C[f'c_minus_{θ}_{i}_1'] = np.array([
                        poly_add(poly_mul(bm[j], d), e1[j])
                        for j in range(m)
                    ], dtype=int)

        return C

