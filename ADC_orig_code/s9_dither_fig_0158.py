import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = 'SPT-CLJ0158-5500'
c2 = json.load(open(f'{scr_dir(t)}/c2.json'))['rows']
pt = {r['exp']: r for r in json.load(open(f'{scr_dir(t)}/pointing.json'))}
e0 = c2[0]['exp']; cosd = np.cos(np.radians(pt[e0]['hdec']))
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
for k, r in enumerate(c2):
    cmd = (-(pt[r['exp']]['hra'] - pt[e0]['hra']) * cosd * 3600, -(pt[r['exp']]['hdec'] - pt[e0]['hdec']) * 3600)
    mv = (r['bundle'][0] - c2[0]['bundle'][0], r['bundle'][1] - c2[0]['bundle'][1])
    ax[0].plot(*cmd, 's', mfc='none', mec=f'C{k}', ms=10); ax[0].plot(*mv, 'o', color=f'C{k}', ms=6, label=r['exp'])
    ax[0].plot([cmd[0], mv[0]], [cmd[1], mv[1]], color=f'C{k}', lw=0.6)
    ax[1].plot(*r['sky'], 'o', color=f'C{k}', ms=7, label=r['exp'])
ax[0].set_title('C2 line: motion ACROSS THE FIBRE BUNDLE (●) vs commanded dither (□)\n→ the telescope really dithered', fontsize=10)
ax[1].set_title('C2 line: position ON THE SKY per exposure (fibre maps as combined)\n→ fixed to ±0.4": the combine aligns the dithers', fontsize=10)
for a in ax:
    a.axhline(0, color='gray', lw=0.4); a.axvline(0, color='gray', lw=0.4); a.set_aspect('equal'); a.set_xlabel('ΔE-W ["]'); a.set_ylabel('ΔN-S ["]')
ax[0].set_xlim(-4.5, 1); ax[0].set_ylim(-1, 4); ax[1].set_xlim(-2.5, 2.5); ax[1].set_ylim(-2.5, 2.5)
ax[1].add_patch(plt.Circle((0, 0), 0.5, fill=False, ls=':', color='gray')); ax[0].legend(fontsize=7)
fig.suptitle(f'{t}: dither check using the emission-line source C2 (no bright continuum source in this field)', fontsize=11)
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/03_dither_check_with_C2.png', dpi=100)
