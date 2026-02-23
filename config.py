# config.py
import math
import numpy as np
import random

# ── Scheme parameters (demo) ──────────────────────────────────────────────────
N = 1

# polynomial degree parameter: f = 2^v
v = 5
f = 1 << v  # 512

# prime modulus (NTT-friendly)
# 2013265921 = 15 * 2^27 + 1 (prime)
q = 2013265921

# gadget base and digit count
b = 2
k = math.floor(math.log(q, b)) + 1  # 31
m = k + 2                           # 33

# Gaussian widths (kept for compatibility)
sigma   = 2.8
sigma_s = 3.2

# total number of attributes (must be divisible by N)
attr = 33

# “u” polynomial (public) – kept for compatibility
u = np.array([random.randrange(0, q) for _ in range(f)], dtype=np.int64)

# numeric dtype
DT = np.int64

# RLWE vector dimension alias (used by older code)
V = f

