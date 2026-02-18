# decrypt_user.py
import numpy as np
from config import q, N, attr
from util import poly_add, poly_mul_scalar, poly_dotprod, lagrange_coeff
from User import User

def decrypt_with_userkey(ct_bundle: dict, sk_bundle: dict):
    C   = ct_bundle["C"]
    F   = np.array(ct_bundle["F"], dtype=int)
    rho = ct_bundle["rho"]

    X   = np.array(sk_bundle["X"], dtype=int)
    SK  = np.array(sk_bundle["_SK"], dtype=int)

    # compute G from policy + user attributes
    usr = User()
    usr.AccessControl(X.tolist(), F, rho)
    if not usr.satisfiable:
        return {"ok": False, "error": "User attributes do not satisfy policy."}

    G = usr.G_attr  # length attr

    E_partials = []
    u_part = np.zeros_like(C["c_0"], dtype=int)

    for θ in range(N):
        e_share = poly_dotprod(C[f"c_A_{θ}"], SK[attr + θ])

        for i in range(attr // N):
            overall = θ * (attr // N) + i

            gi = int(G[overall]) % q
            if gi:
                u_part = poly_add(u_part, poly_mul_scalar(C[f"c_{θ}_{i}_2"], gi))

            # select by X
            if int(X[overall]) == 1:
                e_share = poly_add(e_share, poly_dotprod(C[f"c_plus_{θ}_{i}_1"], SK[overall]))
            else:
                e_share = poly_add(e_share, poly_dotprod(C[f"c_minus_{θ}_{i}_1"], SK[overall]))

        E_partials.append(e_share % q)

    # Lagrange combine at x=0 using points 1..N
    idxs = list(range(1, N + 1))
    Lc   = [lagrange_coeff(i, idxs, q) for i in range(1, N + 1)]

    E_sum = np.zeros_like(E_partials[0], dtype=int)
    for θ in range(N):
        E_sum = poly_add(E_sum, poly_mul_scalar(E_partials[θ], Lc[θ]))

    total = poly_add(E_sum, u_part)

    diff = (C["c_0"] - total) % q
    q4  = q // 4
    q34 = (3 * q) // 4
    bits = [1 if (q4 <= int(c) % q < q34) else 0 for c in diff]

    # return {"ok": True, "phi_dec": bits, "phi_dec_first_128": "".join(map(str, bits[:128]))}
    return {
        "success": True,
        "satisfiable": True,
        "length": len(bits),
        "hamming_weight": int(sum(bits)),
        "phi_dec_first_128": "".join(map(str, bits[:128])),
        "phi_dec": bits
    }