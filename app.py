# app.py
import re
import uuid
from typing import Any, Dict, Optional

import numpy as np
from flask import Flask, jsonify, request, render_template

from config import q, m, f, u, N, attr, sigma_s
from KGC import Setup as KGC_Setup
from AA import AA
from User import User
from lsss import convert_policy_to_lsss
from util import (
    poly_add, poly_mul, poly_dotprod, poly_mul_scalar,
    gen_multiple_polynomials, gen_polynomial,
    lagrange_coeff,
)

app = Flask(__name__)

# Demo-only in-memory store.
# For real deployment: store setup secrets securely (DB/HSM) and rotate IDs.
_SETUPS: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Helpers: parsing + serialization
# -----------------------------
def _parse_bits(bitstring: str, expected_len: int) -> np.ndarray:
    s = re.sub(r"\s+", "", (bitstring or ""))
    if len(s) != expected_len or not re.fullmatch(r"[01]+", s):
        raise ValueError(f"Expected a {expected_len}-bit string of 0/1.")
    return np.array([int(c) for c in s], dtype=int)


def _to_jsonable(x: Any) -> Any:
    """Recursively convert numpy objects into pure Python (lists/ints)."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _public_from_A_list(A_theta_T_theta):
    """
    AA.A_theta_T_theta is list of (Aθ, Tθ). Return list of Aθ only.
    """
    A_pub = []
    for item in A_theta_T_theta:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            A_pub.append(item[0])
        elif isinstance(item, dict) and "A" in item:
            A_pub.append(item["A"])
        else:
            # Fallback: include as-is (may break json)
            A_pub.append(item)
    return A_pub


# -----------------------------
# Core: PUBLIC encryption (A-mode)
# -----------------------------
def encrypt_public(phi_bits: np.ndarray,
                   F: np.ndarray,
                   rho: list,
                   A_theta_T_theta,
                   b_plus_list,
                   b_minus_list) -> Dict[str, Any]:
    """
    Encrypt using ONLY public material:
      - A_theta_T_theta : list of (Aθ, Tθ) but we only use Aθ
      - b_plus_list / b_minus_list
    Ciphertext is USER-INDEPENDENT:
      - always emits both branches c_plus and c_minus for every attribute.
    """
    D = F.shape[1]

    # Build mapping row_of_attr from rho (like User.AccessControl does)
    row_of_attr = np.full(attr, -1, dtype=int)
    for r, name in enumerate(rho):
        if isinstance(name, str) and name.startswith("attr"):
            try:
                j = int(name[4:])
            except Exception:
                continue
            if 0 <= j < attr:
                row_of_attr[j] = r

    # 1) Sample Σ: m polynomials; take d = Σ[0]
    Sigma = gen_multiple_polynomials(m, gaussian=True, std=sigma_s)  # (m, f)
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
    Sigma_tail = np.vstack(Sigma_tail_parts) if tail_len > 0 else np.zeros((0, f), dtype=int)

    # 2) c0 = 2*u*d + e0 + (q//2)*phi
    # Use the same noise knob as User.py did (SIGMA_C0=2.8). Keep it explicit here.
    SIGMA_C0 = 2.8
    e0 = gen_polynomial(gaussian=True, std=SIGMA_C0)
    C: Dict[str, Any] = {
        "c_0": (poly_add(poly_add((2 * ud) % q, e0),
                         ((q // 2) * phi_bits) % q)) % q
    }

    # 3) Per authority
    num_attr_per_auth = attr // N

    SIGMA_CA = 0.0  # keep row-0 noise = 0
    SIGMA_C1 = 2.8  # branch noise (only for in-policy c2/c1 typically; keep consistent)

    for theta in range(N):
        A_theta, _T = A_theta_T_theta[theta]
        # c_A_theta = A_theta * d + eA  (row 0 noise == 0)
        eA = np.zeros((m, f), dtype=int)
        if m > 1 and SIGMA_CA > 0:
            eA[1:, :] = gen_multiple_polynomials(m - 1, gaussian=True, std=SIGMA_CA)

        C[f"c_A_{theta}"] = np.array(
            [poly_add(poly_mul(A_theta[i], d), eA[i]) for i in range(m)],
            dtype=int
        ) % q

        for i in range(num_attr_per_auth):
            overall = theta * num_attr_per_auth + i
            r = int(row_of_attr[overall])
            in_policy = (r != -1)

            # c2 term (policy dependent)
            if in_policy:
                Fi = F[r]  # length D
                c2 = poly_add(
                    poly_dotprod(Fi[1:], Sigma_tail),
                    poly_mul_scalar(ud, int(Fi[0]))
                ) % q
            else:
                c2 = np.zeros(f, dtype=int)

            C[f"c_{theta}_{i}_2"] = c2

            # Always emit BOTH c1 branches (user-independent ciphertext)
            std1 = SIGMA_C1 if in_policy else 0.0
            if std1 > 0:
                e1_plus = gen_multiple_polynomials(m, gaussian=True, std=std1)
                e1_minus = gen_multiple_polynomials(m, gaussian=True, std=std1)
            else:
                e1_plus = np.zeros((m, f), dtype=int)
                e1_minus = np.zeros((m, f), dtype=int)

            bp = b_plus_list[theta][i]
            bm = b_minus_list[theta][i]

            C[f"c_plus_{theta}_{i}_1"] = np.array(
                [poly_add(poly_mul(bp[j], d), e1_plus[j]) for j in range(m)],
                dtype=int
            ) % q

            C[f"c_minus_{theta}_{i}_1"] = np.array(
                [poly_add(poly_mul(bm[j], d), e1_minus[j]) for j in range(m)],
                dtype=int
            ) % q

    return C


# -----------------------------
# Core: USER decrypt (stateless)
# -----------------------------
def decrypt_with_userkey(ct_bundle: dict, sk_bundle: dict) -> dict:
    C = ct_bundle["C"]
    F = np.array(ct_bundle["F"], dtype=int)
    rho = list(ct_bundle["rho"])

    X = _parse_bits(sk_bundle["X"], attr) if isinstance(sk_bundle["X"], str) else np.array(sk_bundle["X"], dtype=int)
    SK = np.array(sk_bundle["_SK"], dtype=int)

    # Compute reconstruction weights from policy + user attributes
    usr = User()
    usr.AccessControl(X.tolist(), F, rho)
    if not usr.satisfiable:
        return {"ok": False, "error": "User attributes do not satisfy policy."}

    G = np.array(usr.G_attr, dtype=int) % q
    num_attr_per_auth = attr // N

    E_partials = []
    u_part = np.zeros_like(np.array(C["c_0"], dtype=int), dtype=int)

    for theta in range(N):
        # e_share starts with c_A dot SK authority share
        e_share = poly_dotprod(np.array(C[f"c_A_{theta}"], dtype=int), SK[attr + theta])

        for i in range(num_attr_per_auth):
            overall = theta * num_attr_per_auth + i
            gi = int(G[overall]) % q

            if gi != 0:
                u_part = poly_add(u_part, poly_mul_scalar(np.array(C[f"c_{theta}_{i}_2"], dtype=int), gi))

            # Select branch by X (A-mode)
            if int(X[overall]) == 1:
                e_share = poly_add(e_share, poly_dotprod(np.array(C[f"c_plus_{theta}_{i}_1"], dtype=int), SK[overall]))
            else:
                e_share = poly_add(e_share, poly_dotprod(np.array(C[f"c_minus_{theta}_{i}_1"], dtype=int), SK[overall]))

        E_partials.append(e_share % q)

    # Lagrange combine at x=0 using points 1..N
    idxs = list(range(1, N + 1))
    Lc = [lagrange_coeff(i, idxs, q) for i in range(1, N + 1)]

    E_sum = np.zeros_like(E_partials[0], dtype=int)
    for theta in range(N):
        E_sum = poly_add(E_sum, poly_mul_scalar(E_partials[theta], Lc[theta]))

    total = poly_add(E_sum, u_part)

    diff = (np.array(C["c_0"], dtype=int) - total) % q
    q4 = q // 4
    q34 = (3 * q) // 4
    bits = [1 if (q4 <= int(c) % q < q34) else 0 for c in diff]

    return {
        "ok": True,
        "satisfiable": True,
        "phi_dec_first_128": "".join(str(b) for b in bits[:128]),
        "phi_dec": bits,
    }


# -----------------------------
# Pages (optional)
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    # Keep your existing index.html if you have it
    # If not, this will error; you can remove this route if you only want APIs.
    return render_template("index.html", attr=attr)


# -----------------------------
# API 1: Setup
# -----------------------------
@app.route("/api/setup", methods=["POST"])
def api_setup():
    # Create fresh setup
    setup_id = uuid.uuid4().hex

    E = KGC_Setup()
    aa = AA()
    aa.AASetup()

    # Store full AA object (contains trapdoors) server-side
    _SETUPS[setup_id] = {
        "E": E,
        "aa": aa,
    }

    public_setup = {
        "setup_id": setup_id,
        "E": _to_jsonable(E),
        "A": _to_jsonable(_public_from_A_list(aa.A_theta_T_theta)),
        "b_plus": _to_jsonable(aa.b_plus),
        "b_minus": _to_jsonable(aa.b_minus),
        "meta": {
            "q": int(q),
            "m": int(m),
            "f": int(f),
            "N": int(N),
            "attr": int(attr),
        },
    }
    return jsonify(public_setup)


# -----------------------------
# API 2: Encrypt (Data Owner)
# -----------------------------
@app.route("/api/encrypt", methods=["POST"])
def api_encrypt():
    j = request.get_json(force=True, silent=True) or {}

    policy = (j.get("policy") or "").strip()
    if not policy:
        policy = "or(and(attr0,attr1), attr2)"

    # Either provide setup_id (convenience) OR provide public_setup bundle (true independence)
    setup_id = (j.get("setup_id") or "").strip()
    public_setup = j.get("public_setup")

    if setup_id:
        st = _SETUPS.get(setup_id)
        if not st:
            return jsonify({"error": "Invalid setup_id"}), 400
        aa = st["aa"]
        A_theta_T_theta = aa.A_theta_T_theta
        b_plus = aa.b_plus
        b_minus = aa.b_minus
    elif public_setup:
        # For public_setup, we only have A (not trapdoors). Rebuild A_theta_T_theta-like list (A, None)
        A_list = public_setup.get("A")
        b_plus = public_setup.get("b_plus")
        b_minus = public_setup.get("b_minus")
        if A_list is None or b_plus is None or b_minus is None:
            return jsonify({"error": "public_setup must include A, b_plus, b_minus"}), 400
        A_theta_T_theta = [(np.array(A_list[theta], dtype=int), None) for theta in range(N)]
        b_plus = np.array(b_plus, dtype=int)
        b_minus = np.array(b_minus, dtype=int)
    else:
        return jsonify({"error": "Provide setup_id or public_setup"}), 400

    # Parse/generate message bits
    msg_bits = (j.get("message_bits") or "").strip()
    if msg_bits:
        phi_bits = _parse_bits(msg_bits, f)
        used_random = False
    else:
        phi_bits = np.random.randint(0, 2, size=f, dtype=int)
        used_random = True

    # DEBUG: print plaintext being encrypted (terminal)
    phi_str = "".join(str(int(b)) for b in phi_bits.tolist())
    print(f"[ENCRYPT] used_random={used_random} len={len(phi_str)} phi={phi_str}", flush=True)

    # Build LSSS from policy
    F_mat, rho = convert_policy_to_lsss(policy)
    F_mat = np.array(F_mat, dtype=int)

    # Encrypt with public parameters only
    C = encrypt_public(phi_bits, F_mat, rho, A_theta_T_theta, b_plus, b_minus)

    ct_bundle = {
        "policy": policy,
        "F": _to_jsonable(F_mat),
        "rho": _to_jsonable(rho),
        "C": _to_jsonable(C),
        "meta": {
            "used_random_message": used_random,
            "phi_first_128": "".join(str(b) for b in phi_bits[:128].tolist()),
        }
    }
    return jsonify(ct_bundle)


# -----------------------------
# API 3: KeyGen (User precompute)
# -----------------------------
@app.route("/api/keygen", methods=["POST"])
def api_keygen():
    j = request.get_json(force=True, silent=True) or {}

    setup_id = (j.get("setup_id") or "").strip()
    if not setup_id or setup_id not in _SETUPS:
        return jsonify({"error": "Valid setup_id is required for keygen"}), 400

    user_bits_str = (j.get("user_bits") or "").strip()
    X = _parse_bits(user_bits_str, attr)

    st = _SETUPS[setup_id]
    E = st["E"]
    aa: AA = st["aa"]

    # Bind user + generate key
    aa.S(X.tolist())
    aa.SecretKey(E)

    sk_bundle = {
        "X": user_bits_str,
        "_SK": _to_jsonable(aa._SK),
        "meta": {"setup_id": setup_id, "attr": int(attr), "N": int(N), "m": int(m), "f": int(f)}
    }
    return jsonify(sk_bundle)


# -----------------------------
# API 4: Decrypt (User)
# -----------------------------
@app.route("/api/decrypt", methods=["POST"])
def api_decrypt():
    j = request.get_json(force=True, silent=True) or {}
    ct_bundle = j.get("ciphertext")
    sk_bundle = j.get("userkey")

    if not isinstance(ct_bundle, dict) or not isinstance(sk_bundle, dict):
        return jsonify({"error": "Provide ciphertext and userkey objects"}), 400

    try:
        res = decrypt_with_userkey(ct_bundle, sk_bundle)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# if __name__ == "__main__":
#     app.run(debug=True, port=8000)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)