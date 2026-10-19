"""Fig. 1 in ChemDraw 'ACS Document 1996' style:
bond length 14.4 pt, line width 0.6 pt, double-bond spacing 18 % of bond length,
Arial(-metric) 10 pt atom labels, bonds trimmed around heteroatom labels."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Liberation Sans'
rcParams['pdf.fonttype'] = 42; rcParams['svg.fonttype'] = 'none'

PT_PER_UNIT = 14.4          # 1 bond = 14.4 pt
LW = 0.6                    # pt
FS = 10                     # pt
TRIM = 0.42                 # bond units cleared around a label
DBL = 0.18                  # double bond spacing (fraction of bond length)

def ring_on_edge(p1, p2, n, away):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    mid = (p1 + p2) / 2; e = p2 - p1; L = np.linalg.norm(e)
    nrm = np.array([-e[1], e[0]]) / L
    if np.dot(nrm, np.asarray(away) - mid) > 0: nrm = -nrm
    c = mid + nrm * L / (2 * np.tan(np.pi / n)); R = L / (2 * np.sin(np.pi / n))
    a1 = np.arctan2(*(p1 - c)[::-1]); a2 = np.arctan2(*(p2 - c)[::-1])
    s = np.sign((a2 - a1 + np.pi) % (2 * np.pi) - np.pi); st = 2 * np.pi / n
    return [c + R * np.array([np.cos(a1 + s * k * st), np.sin(a1 + s * k * st)]) for k in range(n)], c

class Mol:
    def __init__(s, ax, off=(0, 0)):
        s.ax = ax; s.off = np.asarray(off, float); s.labels = []; s.bonds = []
    def lab(s, p, t): s.labels.append((np.asarray(p, float), t))
    def bond(s, a, b, double=False, center=None, inner=True):
        s.bonds.append((np.asarray(a, float), np.asarray(b, float), double, center, inner))
    def _is_lab(s, p): return any(np.linalg.norm(p - q) < 1e-6 for q, _ in s.labels)
    def render(s):
        ax = s.ax; o = s.off
        for a, b, dbl, cen, inner in s.bonds:
            u = (b - a) / np.linalg.norm(b - a)
            ta = TRIM if s._is_lab(a) else 0; tb = TRIM if s._is_lab(b) else 0
            A, B = a + u * ta, b - u * tb
            ax.plot(*zip(A + o, B + o), color='k', lw=LW, solid_capstyle='butt')
            if dbl:
                n = np.array([-u[1], u[0]])
                if cen is not None and np.dot(n, np.asarray(cen) - (a + b) / 2) < 0: n = -n
                sh = 0.12  # shorten inner line at each end
                A2 = a + n * DBL + u * max(sh, ta); B2 = b + n * DBL - u * max(sh, tb)
                ax.plot(*zip(A2 + o, B2 + o), color='k', lw=LW, solid_capstyle='butt')
        for p, t in s.labels:
            ax.text(*(p + o), t, ha='center', va='center_baseline', fontsize=FS, family='Liberation Sans')

def build(m, bridged, quinox=False):
    B = [np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) for a in range(0, 360, 60)]
    cB = np.zeros(2)
    # benzothiadiazole Kekule: C4=C5 (B3=B4), C6=C7 (B5=B0); C3a=N, C7a=N
    for k in range(6): m.bond(B[k], B[(k + 1) % 6], double=k in (3, 5), center=cB)
    D, cD = ring_on_edge(B[1], B[2], 5, cB)
    m.lab(D[2], 'N'); m.lab(D[4], 'N'); m.lab(D[3], 'Y')
    m.bond(D[1], D[2], True, cD); m.bond(D[2], D[3]); m.bond(D[3], D[4]); m.bond(D[4], D[0], True, cD)
    if quinox:
        Q, cQ = ring_on_edge(B[4], B[5], 6, cB)   # C, C, N, CH, CH, N
        m.lab(Q[2], 'N'); m.lab(Q[5], 'N')
        m.bond(Q[1], Q[2]); m.bond(Q[2], Q[3], True, cQ); m.bond(Q[3], Q[4]); m.bond(Q[4], Q[5], True, cQ); m.bond(Q[5], Q[0])
    for side, vi in [(1, 0), (-1, 3)]:
        a = B[vi]
        if quinox:   # donors point upwards, away from the pyrazine ring
            ang = np.radians(30) if side == 1 else np.radians(150)
            ang2 = np.radians(-48) if side == 1 else np.radians(228)
        elif bridged:
            ang = np.radians(-30) if side == 1 else np.radians(210)
            ang2 = np.radians(-108) if side == 1 else np.radians(288)
        else:        # system A: horizontal inter-ring bond, X drawn up (flipped below)
            ang = np.radians(0) if side == 1 else np.radians(180)
            ang2 = np.radians(-36) if side == 1 else np.radians(216)
        c2 = a + np.array([np.cos(ang), np.sin(ang)])
        m.bond(a, c2)
        xpos = c2 + np.array([np.cos(ang2), np.sin(ang2)])
        # system A: flip the donor about the inter-ring bond so that X points up (syn to the acceptor N),
        # leaving the alpha carbon next to X free for chain extension
        if not bridged and not quinox:
            u = (c2 - a) / np.linalg.norm(c2 - a); v = xpos - a
            xpos = a + 2 * np.dot(v, u) * u - v
        R, cR = ring_on_edge(c2, xpos, 5, a)       # C2, X, C5, C4, C3
        m.lab(R[1], 'X')
        m.bond(R[0], R[1]); m.bond(R[1], R[2]); m.bond(R[2], R[3], True, cR); m.bond(R[3], R[4]); m.bond(R[4], R[0], True, cR)
        if bridged:
            H, cH = ring_on_edge(R[4], R[3], 6, cR)  # C3, C4, O, CH2, CH2, O
            m.lab(H[2], 'O'); m.lab(H[5], 'O')
            m.bond(H[1], H[2]); m.bond(H[2], H[3]); m.bond(H[3], H[4]); m.bond(H[4], H[5]); m.bond(H[5], H[0])

def extent(bridged, quinox=False):
    m = Mol(None); build(m, bridged, quinox)
    pts = np.array([p for b in m.bonds for p in b[:2]]); return pts.min(0), pts.max(0)


specs = [(False, False, '(a)'), (True, False, '(b)'), (False, True, '(c)')]
ext = [extent(b, q) for b, q, _ in specs]
gap = 2.2; pad = 0.6
offs = []; x = pad
for lo, hi in ext:
    offs.append(np.array([x - lo[0], 0])); x += hi[0] - lo[0] + gap
W = x - gap + pad
ymin = min(lo[1] for lo, _ in ext) - pad; ymax = max(hi[1] for _, hi in ext) + pad + 0.9
Hh = ymax - ymin
fig = plt.figure(figsize=(W * PT_PER_UNIT / 72, Hh * PT_PER_UNIT / 72))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(ymin, ymax); ax.axis('off')
for (bridged, quinox, lab), off, (lo, hi) in zip(specs, offs, ext):
    m = Mol(ax, off); build(m, bridged, quinox); m.render()
    ax.text(off[0] + lo[0], ymax - 0.45, lab, fontsize=FS, family='Liberation Sans', ha='left', va='center')
print('Fig1 size (in):', round(W * PT_PER_UNIT / 72, 2), round(Hh * PT_PER_UNIT / 72, 2))
import sys, os
out = sys.argv[1] if len(sys.argv) > 1 else '.'
base = os.path.join(out, 'Fig1')
fig.savefig(base + '.pdf'); fig.savefig(base + '.svg')
fig.savefig(base + '.tiff', dpi=1200, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig(base + '.png', dpi=600)

