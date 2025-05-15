import numpy as np
from config import q

def parse_policy(p):
    p = p.strip()
    # strip outer ( )
    if p.startswith('(') and p.endswith(')'):
        depth = 0
        for i,ch in enumerate(p):
            if ch=='(': depth+=1
            elif ch==')': depth-=1
            if depth==0 and i< len(p)-1:
                break
        else:
            return parse_policy(p[1:-1])
    # split OR
    depth=0
    for i in range(len(p)-3):
        if p[i]=='(': depth+=1
        elif p[i]==')': depth-=1
        elif depth==0 and p[i:i+3]=='or ':
            return ('or', parse_policy(p[:i].strip()), parse_policy(p[i+3:].strip()))
    # split AND
    depth=0
    for i in range(len(p)-4):
        if p[i]=='(': depth+=1
        elif p[i]==')': depth-=1
        elif depth==0 and p[i:i+4]=='and ':
            return ('and', parse_policy(p[:i].strip()), parse_policy(p[i+4:].strip()))
    return p

def _lsss_and(F1, ρ1, F2, ρ2):
    m1,m2 = F1.shape[1], F2.shape[1]
    top = np.hstack([F1, np.zeros((F1.shape[0],m2),int)])
    bot = np.hstack([np.zeros((F2.shape[0],m1),int), F2])
    return np.vstack([top,bot])%q, ρ1+ρ2

def _lsss_or(F1, ρ1, F2, ρ2):
    h1,m1 = F1.shape; h2,m2=F2.shape
    mnew = max(m1,m2)+1
    M = np.zeros((h1+h2,mnew),int)
    M[:h1,:m1]=F1; M[h1:, :m2]=F2; M[:, -1]=1
    return M%q, ρ1+ρ2

def convert_tree_to_lsss(policy, attributes):
    tree = parse_policy(policy)
    def rec(node):
        if isinstance(node,str):
            if node not in attributes:
                raise ValueError(f"Unknown attribute '{node}'")
            return np.array([[1]],int), [node]
        op,L,R = node
        F1,ρ1 = rec(L)
        F2,ρ2 = rec(R)
        if op=='and': return _lsss_and(F1,ρ1,F2,ρ2)
        else:         return _lsss_or(F1,ρ1,F2,ρ2)
    F,ρ = rec(tree)
    return F%q, ρ
