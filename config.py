import math
import numpy as np
import random

# ── Scheme parameters ─────────────────────────────────────────────────────────

# number of authorities
N = 3

# polynomial degree parameter: f = 2^v
v = 2
f = 1 << v  # = 4

# small modulus for toy
q = 12289

# trapdoor digit‐expansion params
b = 17
k = math.floor(math.log(q, b)) + 1
m = k + 2

# Gaussian widths (toy)
sigma   = 3.2
sigma_s = 8.2

# vector dimension for RLWE
V = f

# total number of attributes (must be divisible by N)
attr = 4

# fixed public “u” polynomial
u = np.array([random.randrange(0, q) for _ in range(f)], dtype=int)
