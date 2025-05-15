import random
import numpy as np

from algo import RTrapGen
from config import attr, q, f, V, m, sigma, sigma_s, N
from util import (
    gen_multiple_polynomials,
    poly_dotprod, poly_add,
    poly_mul_scalar, lagrange_coeff,
    SamplePre
)

class AA:
    def __init__(self, cc=None):
        self.cc = cc
        self.num_attr = attr // N
        # dummy tensor C (unused in stub _P)
        self._C = np.random.randint(0, q, (self.num_attr, m, V), dtype=int)

        self.A_theta_T_theta = []   # will hold N tuples
        self.b_plus          = []   # shape [N][num_attr][m][f]
        self.b_minus         = []   # shape [N][num_attr][m][f]
        self._SK             = None
        self.X               = None

    def AASetup(self):
        self.A_theta_T_theta.clear()
        self.b_plus.clear()
        self.b_minus.clear()

        for _ in range(N):
            Aθ, Tθ = RTrapGen()
            # per-authority b⁺ and b⁻ arrays, shape (num_attr, m, f)
            bp = np.array([gen_multiple_polynomials(m)
                           for _ in range(self.num_attr)],
                          dtype=object)
            bm = np.array([gen_multiple_polynomials(m)
                           for _ in range(self.num_attr)],
                          dtype=object)
            self.A_theta_T_theta.append((Aθ, Tθ))
            self.b_plus.append(bp)
            self.b_minus.append(bm)

    def _P(self, z, i):
        # stub: returns a random m×f vector
        return gen_multiple_polynomials(m)

    def S(self):
        bits = random.getrandbits(attr)
        self.X = np.array([(bits >> i) & 1 for i in range(attr)],
                          dtype=int)
        return self.X

    def SecretKey(self, E):
        # allocate shares: (attr + N) × m × f
        self._SK = np.zeros((attr + N, m, f), dtype=int)

        for θ in range(N):
            Δ = E[θ].copy()  # length-f

            # per-attribute
            for i in range(self.num_attr):
                overall = θ * self.num_attr + i
                z = np.random.randint(0, q, V, dtype=int)
                y = self._P(z, i)           # m×f
                self._SK[overall] = y

                bsel = (self.b_plus[θ][i]
                        if self.X[overall]
                        else self.b_minus[θ][i])
                Δ = poly_add(Δ, -poly_dotprod(bsel, y))

            # trapdoor pre-image sample
            Aθ, Tθ = self.A_theta_T_theta[θ]
            yA = SamplePre(self.cc, Aθ, Tθ, Δ)  # m×f
            self._SK[attr + θ] = yA

    def Decrypt(self, cipher, F, W):
        partials = []
        for θ in range(N):
            # authority‐level share
            share = poly_dotprod(cipher[f'c_A_{θ}'],
                                 self._SK[attr + θ])

            # each attribute share
            for i in range(self.num_attr):
                overall = θ * self.num_attr + i
                share = poly_add(share, cipher[f'c_{θ}_{i}_2'])

                if W[overall]:
                    share = poly_add(
                        share,
                        poly_dotprod(cipher[f'c_{θ}_{i}_1'],
                                     self._SK[overall])
                    )
                elif self.X[overall]:
                    share = poly_add(
                        share,
                        poly_dotprod(cipher[f'c_plus_{θ}_{i}_1'],
                                     self._SK[overall])
                    )
                else:
                    share = poly_add(
                        share,
                        poly_dotprod(cipher[f'c_minus_{θ}_{i}_1'],
                                     self._SK[overall])
                    )
            partials.append(share)

        # Lagrange interpolation
        idxs = list(range(1, N+1))
        Lc   = [lagrange_coeff(i, idxs, q) for i in range(N)]

        total = np.zeros_like(partials[0])
        for θ in range(N):
            total = poly_add(total,
                             poly_mul_scalar(partials[θ], Lc[θ]))

        φ_poly = poly_add(cipher['c_0'], [(-c) % q for c in total])
        return [1 if c >= (q//4) else 0 for c in φ_poly]
