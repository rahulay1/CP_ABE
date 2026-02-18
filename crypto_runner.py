# crypto_runner.py
import re
import time
import numpy as np

from config import m, attr
from lsss import convert_policy_to_lsss
from KGC import Setup as KGC_Setup
from AA import AA
from User import User

from typing import Optional
def _row_attr_map(rho):
    idxs = []
    for name in rho:
        mobj = re.search(r"(\d+)$", str(name))
        idxs.append(int(mobj.group(1)) if mobj else -1)
    return idxs


def _parse_user_bits(bitstring: str):
    s = re.sub(r"\s+", "", (bitstring or ""))
    if not s:
        raise ValueError("Attribute bitstring required.")
    if len(s) != attr:
        raise ValueError(f"Expected bitstring of length {attr}, got {len(s)}.")
    if not re.fullmatch(r"[01]+", s):
        raise ValueError("Bitstring must contain only 0/1.")
    return [int(c) for c in s]


def _parse_message_bits(bitstring: str, L=512, random_if_empty=True):
    s = re.sub(r"\s+", "", (bitstring or ""))
    if not s and random_if_empty:
        return np.random.randint(0, 2, size=L, dtype=np.int64).tolist(), True
    if len(s) != L or not re.fullmatch(r"[01]+", s):
        raise ValueError(f"Expected a {L}-bit string of 0/1.")
    return [int(c) for c in s], False


def run_pipeline(policy_str: str, user_attr_bits: str, message_bits: Optional[str]):
    if not policy_str or not policy_str.strip():
        policy_str = "or(and(attr0,attr1), attr2)"

    # 1) Build LSSS
    t0 = time.perf_counter()
    F, rho = convert_policy_to_lsss(policy_str.strip())
    t1 = time.perf_counter()

    # 2) Parse inputs
    user_bits = _parse_user_bits(user_attr_bits)
    phi_bits, used_random = _parse_message_bits(message_bits, L=512, random_if_empty=True)

    # 3) Setup
    s0 = time.perf_counter()
    E = KGC_Setup()
    s1 = time.perf_counter()

    aa = AA()
    aa.AASetup()
    s2 = time.perf_counter()

    # 4) Fix authorities to THIS user's bitset
    aa.S(user_bits)

    # 5) Access control
    usr = User()
    usr.AccessControl(user_bits, F, rho)

    # 6) KeyGen
    k0 = time.perf_counter()
    aa.SecretKey(E)
    k1 = time.perf_counter()

    # 7) Encrypt
    e0 = time.perf_counter()
    C = usr.Encrypt(phi_bits, aa.A_theta_T_theta, aa.b_plus, aa.b_minus)
    e1 = time.perf_counter()

    # 8) Decrypt
    d0 = time.perf_counter()
    phi_dec = aa.Decrypt(C, F, usr.row_of_attr, usr.W_attr, usr.G_attr)
    d1 = time.perf_counter()

    ok = (phi_dec == phi_bits)

    return {
        "policy": policy_str.strip(),
        "lsss": {
            "D": int(F.shape[1]),
            "m": int(m),
            "row_attr_map": _row_attr_map(rho),
            "build_time_s": round(t1 - t0, 6),
        },
        "inputs": {
            "attr_bits": "".join(str(b) for b in user_bits),
            "message_used_random": bool(used_random),
        },
        "timings_s": {
            "kgc_setup": round(s1 - s0, 6),
            "aa_setup": round(s2 - s1, 6),
            "keygen": round(k1 - k0, 6),
            "encrypt": round(e1 - e0, 6),
            "decrypt": round(d1 - d0, 6),
        },
        "access": {
            "satisfiable": bool(getattr(usr, "satisfiable", False)),
        },
        "result": {
            "ok": bool(ok),
            "phi_first_128": "".join(str(b) for b in phi_bits[:128]),
            "phi_dec_first_128": "".join(str(b) for b in phi_dec[:128]),
        },
        # Optional: keep these only if you really want them in UI/debug
        # "phi_bits": phi_bits,
        # "phi_dec": phi_dec,
        # "ciphertext_repr": str(type(C)),
    }