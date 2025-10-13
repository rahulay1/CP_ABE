# lsss.py
import re
import numpy as np
from config import q

# ──────────────────────────────────────────────────────────────────────────────
# Shorthand expansion: attrA..attrB  or  attrA..B  → attrA,attrA+1,...,attrB
# ──────────────────────────────────────────────────────────────────────────────

def _expand_attr_ranges(s: str) -> str:
    def repl(m: re.Match) -> str:
        a = int(m.group(1)); b = int(m.group(2))
        if a > b: a, b = b, a
        return ",".join(f"attr{i}" for i in range(a, b + 1))
    s = re.sub(r"attr(\d+)\.\.\s*attr(\d+)", repl, s)
    s = re.sub(r"attr(\d+)\.\.\s*(\d+)", repl, s)
    return s

# ──────────────────────────────────────────────────────────────────────────────
# Parser → AST
#   ("LEAF", name) | ("OR", [kids]) | ("AND", [kids]) | ("KOF", k, [kids])
# ──────────────────────────────────────────────────────────────────────────────

def parse_policy(s: str):
    s = re.sub(r"\s+", "", s)
    s = _expand_attr_ranges(s)

    def parse_expr(i):
        if s.startswith("and(", i):
            kids, j = parse_args(i + 4)
            return ("AND", kids), j
        if s.startswith("or(", i):
            kids, j = parse_args(i + 3)
            return ("OR", kids), j
        if s.startswith("kof(", i):
            k_end = s.find(",", i + 4)
            if k_end == -1:
                raise ValueError("kof: expected comma after k")
            kreq = int(s[i + 4:k_end])
            kids, j = parse_args(k_end + 1)
            return ("KOF", kreq, kids), j

        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
        if not m:
            raise ValueError(f"Parse error at {i}: '{s[i:i+20]}'")
        name = m.group(0)
        return ("LEAF", name), i + len(name)

    def parse_args(i):
        if i >= len(s):
            raise ValueError("Unexpected end while parsing args")
        if s[i] == ')':
            return [], i + 1
        parts = []
        start = i
        depth = 0
        j = i
        while j < len(s):
            c = s[j]
            if c == '(':
                depth += 1
            elif c == ')':
                if depth == 0:
                    parts.append(s[start:j])
                    kids = [parse_policy(x) for x in parts]
                    return kids, j + 1
                depth -= 1
            elif c == ',' and depth == 0:
                parts.append(s[start:j]); start = j + 1
            j += 1
        raise ValueError("Unbalanced parentheses in policy")

    node, j = parse_expr(0)
    if j != len(s):
        raise ValueError("Trailing input in policy")
    return node

# ──────────────────────────────────────────────────────────────────────────────
# LSSS with:
#   * Root: e1 = (1,0,...,0)
#   * OR:   copy parent to each child
#   * AND:  allocate (c-1) fresh coords; enforce "all children required"
#   * KOF:  allocate (k-1) fresh coords; Vandermonde block
# Global dimension D = 1 + Σ_AND (c-1) + Σ_KOF (k-1)
# ──────────────────────────────────────────────────────────────────────────────

def _sum_extra_dims(node):
    t = node[0]
    if t == "LEAF":
        return 0
    if t == "OR":
        return sum(_sum_extra_dims(c) for c in node[1])
    if t == "AND":
        kids = node[1]
        return (max(len(kids), 1) - 1) + sum(_sum_extra_dims(c) for c in kids)
    if t == "KOF":
        kreq, kids = node[1], node[2]
        return (kreq - 1) + sum(_sum_extra_dims(c) for c in kids)
    raise ValueError("bad node")

def convert_policy_to_lsss(policy_str):
    node = parse_policy(policy_str)

    D_extra = _sum_extra_dims(node)
    D = 1 + D_extra
    v_root = np.zeros(D, dtype=np.int64); v_root[0] = 1

    rows, rho = [], []
    ctx = {"D": D, "ptr": 1}  # next fresh coord index for new blocks

    def assign(node, v_parent):
        typ = node[0]

        if typ == "LEAF":
            rho.append(node[1])
            rows.append((v_parent % q).astype(np.int64))
            return

        if typ == "OR":
            for c in node[1]:
                assign(c, v_parent)  # same parent to each child
            return

        if typ == "AND":
            kids = node[1]
            c = len(kids)
            if c == 0:
                return
            # Fresh block for this AND: size (c-1)
            block_start = ctx["ptr"]
            block_end   = block_start + (c - 1)
            if block_end > ctx["D"]:
                raise ValueError("Internal error: ran out of AND dimensions.")
            coords = list(range(block_start, block_end))
            ctx["ptr"] = block_end

            inv_c = pow(c % q, q - 2, q)  # modular inverse of c
            # First (c-1) children:  (1/c)*v_parent + e_{coords[j]}
            for j, child in enumerate(kids[:-1]):
                v_child = (inv_c * v_parent) % q
                v_child = v_child.copy()
                v_child[coords[j]] = (v_child[coords[j]] + 1) % q
                assign(child, v_child)
            # Last child: (1/c)*v_parent - sum e_{coords}
            v_last = (inv_c * v_parent) % q
            v_last = v_last.copy()
            for idx in coords:
                v_last[idx] = (v_last[idx] - 1) % q
            assign(kids[-1], v_last)
            return

        if typ == "KOF":
            kreq, children = node[1], node[2]
            n = len(children)
            if not (1 <= kreq <= n):
                raise ValueError("kof: k must be in [1..n]")

            # allocate (k-1) fresh coordinates for THIS KOF (disjoint block)
            block_start = ctx["ptr"]
            block_end   = block_start + (kreq - 1)
            if block_end > ctx["D"]:
                raise ValueError("Internal error: ran out of KOF dimensions.")
            coords = list(range(block_start, block_end))
            ctx["ptr"] = block_end

            # secret degree-(k-1) p(x) with p(0) = 1
            coeffs = np.random.randint(0, q, size=kreq, dtype=np.int64)
            coeffs[0] = 1

            def p_eval(jval: int) -> int:
                acc = 0
                for a in reversed(coeffs):
                    acc = (acc * jval + int(a)) % q
                return acc

            # child j: p(j)·v_parent + Vandermonde fresh block
            for j_idx, child in enumerate(children, start=1):
                pj = p_eval(j_idx)
                v_child = (pj * v_parent) % q
                v_child = v_child.copy()
                power = j_idx % q
                for _r, idx in enumerate(coords, start=1):
                    v_child[idx] = (v_child[idx] + power) % q
                    power = (power * j_idx) % q
                assign(child, v_child)
            return

        raise ValueError("bad node")

    assign(node, v_root)
    F = np.array(rows, dtype=np.int64) % q
    return F, rho

