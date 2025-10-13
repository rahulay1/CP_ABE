# scheme.py
import random
import numpy as np
from config import q, f, DT
from recon import is_authorized

# ── Deterministic PRG from secret s -> bit pad of length L ───────────────────
def _prg_bits(s, L):
    rnd = random.Random(int(s) % q)
    return [rnd.getrandbits(1) for _ in range(L)]

# ── Demo CP-ABE interface (toy; NOT cryptographically secure) ────────────────
def setup():
    return {"q": q, "f": f}

def aa_setup(GP):
    return {"authorities": 3}

def keygen(GP, AA, F, rho, user_bits):
    ok, info = is_authorized(F, rho, user_bits, return_w=True)
    if not ok:
        raise ValueError("Policy not satisfied by the user attributes.")
    return {
        "F_shape": F.shape,
        "rho": list(rho),
        "w": info["w"],              # weights aligned with F rows
        "active_set": info["active_set"],
        "user_bits": list(user_bits)
    }

def encrypt(GP, F, rho, phi_bits):
    """
    Choose secret s = y[0]; random y[1..D-1].
    Shares: share_i = <F[i,:], y> mod q.
    Mask the bitstring with PRG(s).
    """
    D = F.shape[1]
    y = np.random.randint(0, q, size=D, dtype=DT)
    s = int(y[0] % q)
    shares = (F.dot(y) % q).astype(DT)  # shape (#rows,)

    pad = _prg_bits(s, len(phi_bits))
    c_bits = [int(phi_bits[i]) ^ pad[i] for i in range(len(phi_bits))]

    C = {
        "D": D,
        "rho": list(rho),
        "shares": shares.tolist(),
        "c": c_bits
    }
    return C

def decrypt(GP, sk, C):
    """
    Reconstruct s = sum_i w_i * share_i (mod q), then unmask.
    """
    shares = C["shares"]
    w = sk["w"]
    s = 0
    for i, wi in enumerate(w):
        wi = int(wi) % q
        if wi != 0:
            s = (s + wi * (int(shares[i]) % q)) % q

    pad = _prg_bits(s, len(C["c"]))
    phi_dec = [int(C["c"][i]) ^ pad[i] for i in range(len(pad))]
    return phi_dec
