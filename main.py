import random
import time
import numpy as np

from lsss import convert_tree_to_lsss
from KGC import Setup
from AA import AA
from User import User
from config import attr, f

def timeit(fn, title):
    st = time.time()
    res = fn()
    print(f"{title} – {time.time() - st:.3f}s")
    return res

# Instantiate authority and user
aa   = AA(None)
user = User()

# System setup
E = timeit(lambda: Setup(),    "Setup")
timeit(lambda: aa.AASetup(),   "AASetup")

# Three toy users
test_users = {
    "User1": np.array([1,1,0,0], dtype=int),
    "User2": np.array([0,0,1,0], dtype=int),
    "User3": np.array([0,0,0,1], dtype=int),
}

# Build & pad LSSS
attributes = [f"attr{i}" for i in range(attr)]
policy     = "(attr0 and attr1) or attr2"
F, rho     = convert_tree_to_lsss(policy, attributes)

# pad so we have exactly `attr` rows
if F.shape[0] < attr:
    pad = attr - F.shape[0]
    F = np.vstack([F,
                   np.zeros((pad, F.shape[1]), dtype=int)])
    rho += [None] * pad

# **NEW**: convert numeric LSSS F into polynomial form (h × m × f)
h, m_mat = F.shape
F_poly = np.zeros((h, m_mat, f), dtype=int)
for r in range(h):
    for j in range(m_mat):
        F_poly[r, j, 0] = F[r, j]       # constant‐term polynomial

user.F   = F_poly
user.rho = rho

# Run the three tests
for name, X in test_users.items():
    print(f"\n=== {name} with X = {X} ===")
    aa.X = X
    timeit(lambda: aa.SecretKey(E), "KeyGen")
    user.AccessControl(X)

    phi = np.array([random.getrandbits(1) for _ in range(f)], dtype=int)
    print("Plain:", phi)

    C = timeit(
        lambda: user.Encrypt(phi,
                             aa.A_theta_T_theta,
                             aa.b_plus,
                             aa.b_minus),
        "Encrypt"
    )
    dec = timeit(lambda: aa.Decrypt(C, user.F, user.W),
                 "Decrypt")
    print("Got:", dec, "OK?", np.array_equal(phi, dec))
