# main.py
import re
import time
import numpy as np

from config import q, m, attr
from lsss import convert_policy_to_lsss
from KGC import Setup as KGC_Setup
from AA import AA
from User import User

def _read_policy():
    default = "or(and(attr0,attr1), attr2)"
    s = input(f"What is the policy? [default: {default}]\n> ").strip()
    return s if s else default

def _read_user_bits():
    s = input(f"\nWhat is the user's attribute set? (bitstring length={attr}, e.g. 1100)\n> ").strip()
    s = re.sub(r"\s+", "", s)
    if not s:
        raise ValueError("Attribute bitstring required.")
    if len(s) != attr:
        raise ValueError(f"Expected bitstring of length {attr}, got {len(s)}.")
    if not re.fullmatch(r"[01]+", s):
        raise ValueError("Bitstring must contain only 0/1.")
    return [int(c) for c in s]

def _read_message(L=512):
    s = input(f"\nWhat is the message? (bitstring length={L}, Enter for random)\n> ").strip()
    s = re.sub(r"\s+", "", s)
    if not s:
        phi = np.random.randint(0, 2, size=L, dtype=np.int64).tolist()
        print(f"[random] Using φ = {phi}")
        return phi
    if len(s) != L or not re.fullmatch(r"[01]+", s):
        raise ValueError(f"Expected a {L}-bit string of 0/1.")
    return [int(c) for c in s]

def _row_attr_map(rho):
    idxs = []
    for name in rho:
        mobj = re.search(r'(\d+)$', str(name))
        idxs.append(int(mobj.group(1)) if mobj else -1)
    return idxs

def main():
    # 1) Read inputs
    policy_str = _read_policy()

    # 2) Build LSSS
    t0 = time.perf_counter()
    F, rho = convert_policy_to_lsss(policy_str)
    t1 = time.perf_counter()

    D = F.shape[1]
    print(f"\n[policy] LSSS dimension D = {D}, trapdoor m = {m}\n")
    print(f"Policy: {policy_str}")
    map_idxs = _row_attr_map(rho)
    print("Policy rows mapped to attributes:", map_idxs)

    # 3) Read user attributes & message
    user_bits = _read_user_bits()
    phi_bits = _read_message(512)

    # 4) System setup (KGC and authorities)
    s0 = time.perf_counter()
    E = KGC_Setup()              # public E(θ) shares (length N)
    s1 = time.perf_counter()

    aa = AA()
    aa.AASetup()                 # generate (Aθ,Tθ), b⁺/b⁻
    s2 = time.perf_counter()

    print(f"\nKGC.Setup – {s1 - s0:.3f}s")
    print(f"AA.AASetup – {s2 - s1:.3f}s")

    # 5) Make the authorities use THIS user's bitset (no randomness)
    aa.S(user_bits)

    # 6) Build user's access control (W/G) and print satisfiability
    usr = User()
    usr.AccessControl(user_bits, F, rho)
    print(f"\nSatisfiable? {usr.satisfiable}")

    # 7) Key generation (heavy)
    k0 = time.perf_counter()
    aa.SecretKey(E)
    k1 = time.perf_counter()
    print(f"KeyGen – {k1 - k0:.3f}s")

    # 8) Encrypt (heavy)
    e0 = time.perf_counter()
    C = usr.Encrypt(phi_bits, aa.A_theta_T_theta, aa.b_plus, aa.b_minus)
    e1 = time.perf_counter()
    print(f"Encrypt – {e1 - e0:.3f}s")

    # 9) Decrypt
    d0 = time.perf_counter()
    phi_dec = aa.Decrypt(C, F, usr.row_of_attr, usr.W_attr, usr.G_attr)
    d1 = time.perf_counter()
    print("Original  φ (first 128):", ''.join(str(b) for b in phi_bits[:128]))
    print("Decrypted φ (first 128):", ''.join(str(b) for b in phi_dec[:128]))


    # 10) Check
    ok = (phi_dec == phi_bits)
    print("OK?", ok)

if __name__ == "__main__":
    main()
