# AA.py
import numpy as np

from algo import RTrapGen
from config import attr, q, m, sigma, sigma_s, N
from util import (
    gen_multiple_polynomials, gen_polynomial,
    poly_dotprod, poly_add, poly_mul_scalar,
    lagrange_coeff,
    SamplePre,
)

class AA:
    """
    Authority aggregator for the demo CP-ABE:
      - AASetup(): generate per-authority trapdoors and per-attribute masks
      - S(bits):   set this user's attribute bitset (authorities' view)
      - SecretKey(E): produce secret key shares bound to X (the bitset)
      - Decrypt(cipher, F, row_of_attr, W, G): recover bits
    Shapes:
      * Aθ: (m, f)    trapdoor public matrix in R_q
      * Tθ: trapdoor (internal structure handled by SamplePre)
      * b_plus/b_minus[θ][i]: (m, f)   per-attribute masking vectors
      * _SK: (attr + N, m, f)   first 'attr' entries per attribute, last N per authority
    """
    def __init__(self, cc=None):
        self.cc = cc
        self.num_attr = attr // N

        # Public/secret per authority
        self.A_theta_T_theta = []   # list of (Aθ, Tθ)
        self.b_plus          = []   # [N][num_attr][m][f]
        self.b_minus         = []   # [N][num_attr][m][f]
        self.P_thetas        = []   # [N][num_attr][m][f] (not critical in this variant)

        # Per-user material
        self._SK = None             # secret key shares
        self.X   = None             # authority's bit-vector for the user (length=attr)

    # -------------------------
    # Setup
    # -------------------------
    def AASetup(self):
        """Generate (Aθ,Tθ) and per-attribute (b⁺, b⁻) for each authority θ."""
        self.A_theta_T_theta.clear()
        self.b_plus.clear()
        self.b_minus.clear()
        self.P_thetas.clear()

        for _ in range(N):
            Aθ, Tθ = RTrapGen()
            # Per-authority b⁺ and b⁻ arrays, shape (num_attr, m, f)
            bp = np.array([gen_multiple_polynomials(m, gaussian=True, std=sigma_s)
                           for _ in range(self.num_attr)], dtype=int)
            bm = np.array([gen_multiple_polynomials(m, gaussian=True, std=sigma_s)
                           for _ in range(self.num_attr)], dtype=int)
            # Optional extra randomness holder (not used in this variant)
            Pθ = np.array([gen_multiple_polynomials(m, gaussian=True, std=sigma)
                           for _ in range(self.num_attr)], dtype=int)

            self.A_theta_T_theta.append((Aθ, Tθ))
            self.b_plus.append(bp)
            self.b_minus.append(bm)
            self.P_thetas.append(Pθ)

    def _P(self, theta_index, z_poly, i_attr):
        """
        Produce a SHORT secret vector y ∈ R_q^m.
        Keep it small so poly_dotprod(e, y) stays in decode window after combining.
        """
        return gen_multiple_polynomials(m, gaussian=True, std=sigma_s) % q

    def S(self, bits=None):
        """
        Set the authority's bit-vector X for THIS user.
        If bits is None, sample uniformly random bits of length 'attr'.
        """
        if bits is None:
            self.X = np.random.randint(0, 2, size=attr, dtype=int)
        else:
            self.X = np.array(bits, dtype=int).reshape(attr)
        return self.X

    # -------------------------
    # Key generation
    # -------------------------
    def SecretKey(self, E, debug_check=True):
        """
        Build secret key shares bound to the user's bitset X.
        E: array of length N (polynomials) from KGC.Setup().
        Fills self._SK with shape (attr+N, m, f).
        """
        f = self.A_theta_T_theta[0][0].shape[1]  # polynomial degree
        self._SK = np.zeros((attr + N, m, f), dtype=int)

        for θ in range(N):
            Δ = E[θ].copy()  # target polynomial for this authority

            # per-attribute contributions
            for i in range(self.num_attr):
                overall = θ * self.num_attr + i

                # short vector y for this attribute
                z = gen_polynomial(gaussian=True, std=sigma_s)
                y = self._P(θ, z, i)                # (m, f), SHORT
                self._SK[overall] = y

                # choose b⁺ or b⁻ depending on X
                bsel = self.b_plus[θ][i] if int(self.X[overall]) else self.b_minus[θ][i]
                # accumulate into Δ
                Δ = poly_add(Δ, (-poly_dotprod(bsel, y)) % q)

            # close the equation with preimage sampler
            Aθ, Tθ = self.A_theta_T_theta[θ]
            yA = SamplePre(self.cc, Aθ, Tθ, Δ)  # (m, f)
            self._SK[attr + θ] = yA % q

            if debug_check:
                resid = (poly_dotprod(Aθ, yA) - Δ) % q
                # report centered infinity-norm to spot mistakes
                centered = ((resid + q // 2) % q) - q // 2
                max_abs = int(np.max(np.abs(centered)))
                if max_abs != 0:
                    print(f"[warn] Preimage residual for authority {θ}: max |coeff| = {max_abs}")

    # -------------------------
    # Decrypt
    # -------------------------
    def Decrypt(self, cipher, F, row_of_attr, W, G):
        """
        Combine authority partials via Lagrange at 0, add u-part (from G and c_*_2),
        and threshold to recover bits.

        Args:
          cipher: dict from User.Encrypt
          F:      LSSS matrix (unused here; kept for signature symmetry)
          row_of_attr: mapping attr_index -> row index (unused; User applied it already)
          W:      per-row choice mask from User (1 when the policy row mapped; else 0)
          G:      reconstruction weights per row (integers mod q)
        """
        E_partials = []
        u_part = np.zeros_like(cipher['c_0'], dtype=int)

        for θ in range(N):
            # authority linear part
            e_share = poly_dotprod(cipher[f'c_A_{θ}'], self._SK[attr + θ])

            for i in range(self.num_attr):
                overall = θ * self.num_attr + i

                # u-part from satisfied rows (gi ≠ 0)
                gi = int(G[overall]) if G is not None else 0
                if gi:
                    u_part = poly_add(u_part, poly_mul_scalar(cipher[f'c_{θ}_{i}_2'], gi))

                # pick the proper branch of c_*_1
                if int(W[overall]) == 1:
                    # policy row was mapped for this attribute → single branch provided
                    e_share = poly_add(e_share, poly_dotprod(cipher[f'c_{θ}_{i}_1'], self._SK[overall]))
                else:
                    # row not mapped: both branches present; select by X
                    if int(self.X[overall]) == 1:
                        e_share = poly_add(e_share, poly_dotprod(cipher[f'c_plus_{θ}_{i}_1'], self._SK[overall]))
                    else:
                        e_share = poly_add(e_share, poly_dotprod(cipher[f'c_minus_{θ}_{i}_1'], self._SK[overall]))

            E_partials.append(e_share % q)

        # Lagrange combine at x=0 using points θ=1..N
        idxs = list(range(1, N + 1))
        Lc   = [lagrange_coeff(i, idxs, q) for i in range(1, N + 1)]

        E_sum = np.zeros_like(E_partials[0], dtype=int)
        for θ in range(N):
            E_sum = poly_add(E_sum, poly_mul_scalar(E_partials[θ], Lc[θ]))

        total = poly_add(E_sum, u_part)

        # Recover bits by thresholding around q/2
        diff = (cipher['c_0'] - total) % q
        q4  = q // 4
        q34 = (3 * q) // 4
        bits = [1 if (q4 <= int(c) % q < q34) else 0 for c in diff]
        return bits

