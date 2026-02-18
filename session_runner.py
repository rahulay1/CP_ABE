# session_runner.py
import re
import uuid
import time
import numpy as np
from typing import Optional

from config import m, attr
from lsss import convert_policy_to_lsss
from KGC import Setup as KGC_Setup
from AA import AA
from User import User

# Demo-only in-memory store
_SESSIONS = {}

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

def _parse_message_bits(bitstring: Optional[str], L=512, random_if_empty=True):
    s = re.sub(r"\s+", "", (bitstring or ""))
    if not s and random_if_empty:
        return np.random.randint(0, 2, size=L, dtype=np.int64).tolist(), True
    if len(s) != L or not re.fullmatch(r"[01]+", s):
        raise ValueError(f"Expected a {L}-bit string of 0/1.")
    return [int(c) for c in s], False


def create_session(policy_str: str, user_attr_bits: str, message_bits: Optional[str]):
    if not policy_str or not policy_str.strip():
        policy_str = "or(and(attr0,attr1), attr2)"
    policy_str = policy_str.strip()

    # Steps 1–3: Build LSSS + parse inputs (done once)
    t0 = time.perf_counter()
    F, rho = convert_policy_to_lsss(policy_str)
    t1 = time.perf_counter()

    user_bits = _parse_user_bits(user_attr_bits)
    phi_bits, used_random = _parse_message_bits(message_bits, L=512, random_if_empty=True)

    sid = uuid.uuid4().hex

    _SESSIONS[sid] = {
        "sid": sid,
        "policy": policy_str,
        "F": F,
        "rho": rho,
        "user_bits": user_bits,
        "phi_bits": phi_bits,
        "used_random": used_random,

        # will be filled later
        "E": None,
        "aa": None,
        "usr": None,
        "C": None,
        "phi_dec": None,

        "timings_s": {
            "lsss_build": round(t1 - t0, 6),
        },
        "done_steps": set(),  # {4,5,6,7,8,9,10}
    }

    return {
        "sid": sid,
        "policy": policy_str,
        "lsss": {
            "D": int(F.shape[1]),
            "m": int(m),
            "row_attr_map": _row_attr_map(rho),
            "build_time_s": round(t1 - t0, 6),
        },
        "inputs": {
            "attr_bits": "".join(str(b) for b in user_bits),
            "message_used_random": bool(used_random),
            "phi_first_128": "".join(str(b) for b in phi_bits[:128]),
        }
    }


def _get(sid: str):
    if sid not in _SESSIONS:
        raise ValueError("Invalid or expired session id.")
    return _SESSIONS[sid]


def step4_setup(sid: str):
    st = _get(sid)
    if 4 in st["done_steps"]:
        return {"ok": True, "note": "Step 4 already done."}

    s0 = time.perf_counter()
    E = KGC_Setup()
    s1 = time.perf_counter()

    aa = AA()
    aa.AASetup()
    s2 = time.perf_counter()

    st["E"] = E
    st["aa"] = aa
    st["timings_s"]["kgc_setup"] = round(s1 - s0, 6)
    st["timings_s"]["aa_setup"] = round(s2 - s1, 6)
    st["done_steps"].add(4)

    return {"ok": True, "timings_s": st["timings_s"]}


def step5_bind_user(sid: str):
    st = _get(sid)
    if 4 not in st["done_steps"]:
        raise ValueError("Run Step 4 first.")
    if 5 in st["done_steps"]:
        return {"ok": True, "note": "Step 5 already done."}

    st["aa"].S(st["user_bits"])
    st["done_steps"].add(5)
    return {"ok": True}


def step6_access_control(sid: str):
    st = _get(sid)
    if 5 not in st["done_steps"]:
        raise ValueError("Run Step 5 first.")
    if 6 in st["done_steps"]:
        return {"ok": True, "note": "Step 6 already done.", "satisfiable": st["usr"].satisfiable}

    usr = User()
    usr.AccessControl(st["user_bits"], st["F"], st["rho"])
    st["usr"] = usr
    st["done_steps"].add(6)

    return {"ok": True, "satisfiable": bool(getattr(usr, "satisfiable", False))}


def step7_keygen(sid: str):
    st = _get(sid)
    if 6 not in st["done_steps"]:
        raise ValueError("Run Step 6 first.")
    if 7 in st["done_steps"]:
        return {"ok": True, "note": "Step 7 already done.", "timings_s": st["timings_s"]}

    k0 = time.perf_counter()
    st["aa"].SecretKey(st["E"])
    k1 = time.perf_counter()

    st["timings_s"]["keygen"] = round(k1 - k0, 6)
    st["done_steps"].add(7)
    return {"ok": True, "timings_s": st["timings_s"]}


def step8_encrypt(sid: str):
    st = _get(sid)
    if 7 not in st["done_steps"]:
        raise ValueError("Run Step 7 first.")
    if 8 in st["done_steps"]:
        return {"ok": True, "note": "Step 8 already done.", "timings_s": st["timings_s"]}

    e0 = time.perf_counter()
    C = st["usr"].Encrypt(
        st["phi_bits"],
        st["aa"].A_theta_T_theta,
        st["aa"].b_plus,
        st["aa"].b_minus
    )
    e1 = time.perf_counter()

    st["C"] = C
    st["timings_s"]["encrypt"] = round(e1 - e0, 6)
    st["done_steps"].add(8)
    return {"ok": True, "timings_s": st["timings_s"]}


def step9_decrypt(sid: str):
    st = _get(sid)
    if 8 not in st["done_steps"]:
        raise ValueError("Run Step 8 first.")
    if 9 in st["done_steps"]:
        return {"ok": True, "note": "Step 9 already done.", "timings_s": st["timings_s"]}

    d0 = time.perf_counter()
    phi_dec = st["aa"].Decrypt(st["C"], st["F"], st["usr"].row_of_attr, st["usr"].W_attr, st["usr"].G_attr)
    d1 = time.perf_counter()

    st["phi_dec"] = phi_dec
    st["timings_s"]["decrypt"] = round(d1 - d0, 6)
    st["done_steps"].add(9)

    return {
        "ok": True,
        "timings_s": st["timings_s"],
        "phi_dec_first_128": "".join(str(b) for b in phi_dec[:128]),
    }


def step10_check(sid: str):
    st = _get(sid)
    if 9 not in st["done_steps"]:
        raise ValueError("Run Step 9 first.")
    if 10 in st["done_steps"]:
        return {"ok": True, "note": "Step 10 already done."}

    ok = (st["phi_dec"] == st["phi_bits"])
    st["done_steps"].add(10)
    return {"ok": True, "match": bool(ok)}