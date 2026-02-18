# User.py
import numpy as np
from config import attr, q, f, u, m, N, sigma, sigma_s
from util import (
    poly_add, poly_mul, poly_dotprod, poly_mul_scalar,
    gen_multiple_polynomials, gen_polynomial, solve_for_g
)

# Tunable demo noise (keep consistent with AA.py expectations)
# Row-0 noise of c_A,θ should stay 0 (matches trapdoor structure).
SIGMA_CA = 0.0   # rows 1..m-1 in c_A,θ; row 0 fixed 0
SIGMA_C1 = 2.8   # per-branch c_{θ,i,1} noise
SIGMA_C0 = 2.8   # c_0 noise


class User:
    """
    Keeps only the per-user state needed for:
      - Mapping attributes -> LSSS rows
      - Computing reconstruction vector G
      - Producing an encryption that AA.Decrypt can consume
    """
    def __init__(self):
        # policy + mapping
        self.F = None                 # LSSS matrix, shape (#rows, D)
        self.rho = None               # labels per row
        self.row_of_attr = None       # length=attr, maps attr j -> row index or -1

        # user attributes
        self.X = None                 # length=attr bit-vector (0/1)

        # reconstruction
        self.G_row = None             # length=#rows, weights aligned to F rows
        self.G_attr = None            # length=attr, weights at attribute index
        self.W_attr = None            # length=attr, 1 if attribute appears in policy else 0
        self.satisfiable = False

    # ------------------------------------------------------------------ #
    # Phase 1: Build mapping + reconstruction weights
    # ------------------------------------------------------------------ #
    def AccessControl(self, X_bits, F, rho):
        """
        X_bits: list/array of 0/1 of length 'attr'
        F, rho: from LSSS
        Sets:
          - row_of_attr[j] = row index in F that corresponds to attr j, else -1
          - G_row          = reconstruction weights aligned with F rows
          - G_attr         = same weights projected onto attribute indices
          - W_attr         = 1 if attr j appears in the policy (mapped), else 0
          - satisfiable    = True if e1 reconstructed using user’s available rows
        """
        self.F = np.array(F, dtype=int)
        self.rho = list(rho)
        self.X = np.array(X_bits, dtype=int)

        # Build attr->row mapping
        row_of_attr = [-1] * attr
        for r, name in enumerate(self.rho):
            # Expect labels like "attr12"
            if isinstance(name, str) and name.startswith("attr"):
                try:
                    j = int(name[4:])
                except Exception:
                    continue
                if 0 <= j < attr:
                    # Note: if an attribute appears multiple times in the policy,
                    # the last occurrence will overwrite the earlier one.
                    row_of_attr[j] = r
        self.row_of_attr = np.array(row_of_attr, dtype=int)

        # Determine which rows the user actually has (attr present AND bit=1)
        usable_rows = []
        for j in range(attr):
            r = self.row_of_attr[j]
            if r != -1 and self.X[j] == 1:
                usable_rows.append(r)

        # Solve F_I^T * g = e1 over Z_q
        G_row = solve_for_g(self.F, usable_rows, mod=q)
        self.satisfiable = (G_row is not None)
        if G_row is None:
            # Make zero-shaped defaults so later code won’t crash
            self.G_row = np.zeros(self.F.shape[0], dtype=int)
            self.G_attr = np.zeros(attr, dtype=int)
            self.W_attr = (self.row_of_attr != -1).astype(int)
            return

        self.G_row = np.array(G_row, dtype=int) % q

        # Project weights to attribute index (so AA.Decrypt can use per-attr gi)
        G_attr = np.zeros(attr, dtype=int)
        for j in range(attr):
            r = self.row_of_attr[j]
            if r != -1:
                G_attr[j] = int(self.G_row[r]) % q
        self.G_attr = G_attr

        # W_attr indicates which attributes are in the policy (mapped)
        self.W_attr = (self.row_of_attr != -1).astype(int)

    # ------------------------------------------------------------------ #
    # Phase 2: Encrypt
    # ------------------------------------------------------------------ #
    def Encrypt(self, phi, A_list, b_plus_list, b_minus_list):
        """
        Encrypt a bit-vector phi (length f) under this user's policy state.
        Emits dictionary C with:
          - c_0
          - c_A_θ, θ=0..N-1
          - For each authority θ and its attributes i:
             * if attribute i is in policy:     c_{θ}_{i}_1
               (single branch matching the user's bit)
             * otherwise (not in policy):       c_plus_{θ}_{i}_1 and c_minus_{θ}_{i}_1
             * always:                           c_{θ}_{i}_2 (uses Fi if in-policy; zeros otherwise)
        This matches AA.Decrypt(C, F, row_of_attr, W_attr, G_attr).
        """
        if self.F is None or self.rho is None or self.row_of_attr is None or self.X is None:
            raise RuntimeError("AccessControl() must be called before Encrypt().")

        D = self.F.shape[1]  # LSSS dimension
        # Track rows that actually contribute to reconstruction (diagnostic)
        active_rows_mask = (np.array(self.G_row, dtype=int) % q) != 0
        num_active = int(active_rows_mask.sum())

        # 1) Sample Σ: m polynomials; take d = Σ[0]
        Sigma = gen_multiple_polynomials(m, gaussian=True, std=sigma_s)   # (m, f)
        d = Sigma[0]
        ud = poly_mul(u, d)

        # Build Sigma_tail of EXACT length D-1 to match Fi[1:]
        tail_len = max(0, D - 1)
        reuse_len = min(tail_len, max(0, m - 1))
        Sigma_tail_parts = []
        if reuse_len > 0:
            Sigma_tail_parts.append(Sigma[1:1 + reuse_len])
        if tail_len > reuse_len:
            extra = gen_multiple_polynomials(tail_len - reuse_len, gaussian=True, std=sigma_s)
            Sigma_tail_parts.append(extra)
        if tail_len > 0:
            Sigma_tail = np.vstack(Sigma_tail_parts)
        else:
            Sigma_tail = np.zeros((0, f), dtype=int)  # harmless; matches Fi[1:] length

        # 2) c0 = 2*u*d + e0 + (q//2)*phi
        e0 = gen_polynomial(gaussian=True, std=SIGMA_C0)
        C = {
            'c_0': (poly_add(poly_add((2 * ud) % q, e0),
                             ((q // 2) * np.array(phi, dtype=int)) % q)) % q
        }

        # 3) Per authority
        for θ in range(N):
            Aθ, _ = A_list[θ]

            # c_A_θ = Aθ * d + eA  (row 0 noise == 0)
            eA = np.zeros((m, f), dtype=int)
            if m > 1 and SIGMA_CA > 0:
                eA[1:, :] = gen_multiple_polynomials(m - 1, gaussian=True, std=SIGMA_CA)
            C[f'c_A_{θ}'] = np.array(
                [poly_add(poly_mul(Aθ[i], d), eA[i]) for i in range(m)],
                dtype=int
            ) % q

            # Each authority manages exactly (attr // N) attributes
            for i in range(attr // N):
                overall = θ * (attr // N) + i
                r = int(self.row_of_attr[overall])  # mapped row or -1
                in_policy = (r != -1)

                # c2 term
                if in_policy:
                    Fi = self.F[r]  # length D vector over Z_q
                    # Fi[1:] length == D-1; Sigma_tail length == D-1
                    c2 = poly_add(
                        poly_dotprod(Fi[1:], Sigma_tail),
                        poly_mul_scalar(ud, int(Fi[0]))
                    ) % q
                else:
                    c2 = np.zeros(f, dtype=int)
                C[f'c_{θ}_{i}_2'] = c2

                # per-branch noise: only add noise when in-policy
                std1 = SIGMA_C1 if in_policy else 0.0
                if std1 > 0:
                    e1_plus = gen_multiple_polynomials(m, gaussian=True, std=std1)
                    e1_minus = gen_multiple_polynomials(m, gaussian=True, std=std1)
                else:
                    e1_plus = np.zeros((m, f), dtype=int)
                    e1_minus = np.zeros((m, f), dtype=int)

                # c1 branch(es)
                # if in_policy:
                #     # Emit the single branch that matches the user's bit
                #     bsel = b_plus_list[θ][i] if int(self.X[overall]) else b_minus_list[θ][i]
                #     C[f'c_{θ}_{i}_1'] = np.array(
                #         [poly_add(poly_mul(bsel[j], d), e1_plus[j]) for j in range(m)],
                #         dtype=int
                #     ) % q
                # else:
                #     # Attribute not in policy: emit both branches so AA can pick
                #     bp = b_plus_list[θ][i]
                #     bm = b_minus_list[θ][i]
                #     C[f'c_plus_{θ}_{i}_1'] = np.array(
                #         [poly_add(poly_mul(bp[j], d), e1_plus[j]) for j in range(m)],
                #         dtype=int
                #     ) % q
                #     C[f'c_minus_{θ}_{i}_1'] = np.array(
                #         [poly_add(poly_mul(bm[j], d), e1_minus[j]) for j in range(m)],
                #         dtype=int
                #     ) % q
                # Always emit BOTH branches (ciphertext must be user-independent)
                bp = b_plus_list[θ][i]
                bm = b_minus_list[θ][i]

                # You can keep noise only when in_policy if you want, but simplest: always add noise
                std1 = SIGMA_C1 if in_policy else 0.0
                if std1 > 0:
                    e1_plus = gen_multiple_polynomials(m, gaussian=True, std=std1)
                    e1_minus = gen_multiple_polynomials(m, gaussian=True, std=std1)
                else:
                    e1_plus = np.zeros((m, f), dtype=int)
                    e1_minus = np.zeros((m, f), dtype=int)

                C[f'c_plus_{θ}_{i}_1'] = np.array(
                    [poly_add(poly_mul(bp[j], d), e1_plus[j]) for j in range(m)],
                    dtype=int
                ) % q

                C[f'c_minus_{θ}_{i}_1'] = np.array(
                    [poly_add(poly_mul(bm[j], d), e1_minus[j]) for j in range(m)],
                    dtype=int
                ) % q

        # small diagnostic to see how many rows actually reconstruct
        print(f"[enc] active rows used by reconstruction: {num_active} (<= {m})")
        return C


