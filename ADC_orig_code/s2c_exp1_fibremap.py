"""0111 exposure 1 vs 2: fibre-level continuum maps in the instrument frame (fibre map RA/Dec
relative to each frame's own centre), raw COUNTS and SKYSUB, to see if exposure 1 contains any source."""
import sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = 'SPT-CLJ0111-5637'
ps = rss_paths(t, 'red')[:3]
fig, ax = plt.subplots(3, 3, figsize=(15, 14))
for j, p in enumerate(ps):
    with fits.open(p) as h:
        ss = h['SKYSUB'].data; cn = h['COUNTS'].data; sky = h['SKY'].data; w = h['WAVE'].data; fm = h['FIBERMAP'].data
        hd = h[0].header
    b = (w > 7000) & (w < 9000)
    lv = np.nanmedian(np.where(b, ss, np.nan), axis=1); lc = np.nanmedian(np.where(b, cn, np.nan), axis=1)
    lsky = np.nanmedian(np.where(b, sky, np.nan), axis=1)
    ra, dec = np.array(fm['RA'], float), np.array(fm['DEC'], float)
    x = (ra - np.median(ra)) * np.cos(np.radians(np.median(dec))) * 3600; y = (dec - np.median(dec)) * 3600
    for i, (v, nm) in enumerate([(lv, 'SKYSUB'), (lc, 'COUNTS'), (lsky, 'SKY model')]):
        lo, hi = np.nanpercentile(v, [2, 99.5])
        sc = ax[i, j].scatter(x, y, c=v, s=9, cmap='gray', vmin=lo, vmax=hi, marker='h')
        ax[i, j].set_aspect('equal'); plt.colorbar(sc, ax=ax[i, j], fraction=0.046)
        ax[i, j].set_title(f'{exp_id(p)}  red 7000–9000 Å {nm}', fontsize=9)
    print(exp_id(p), 'median COUNTS %.1f  SKY %.1f  SKYSUB %.2f  | 99.9th pct SKYSUB %.1f | EXPTIME %s  TEL RA/DEC %s %s  GUIDE? %s' % (
        np.nanmedian(lc), np.nanmedian(lsky), np.nanmedian(lv), np.nanpercentile(lv, 99.9), hd.get('SEXPTIME'),
        hd.get('TEL RA'), hd.get('TEL DEC'), {k: hd[k] for k in hd if 'GUID' in k.upper()}))
ax[2, 0].set_xlabel('ΔRA within frame ["]')
fig.suptitle('SPT-CLJ0111-5637: exposure 1 vs 2 & 3, fibre maps (each frame in its own frame)', fontsize=12)
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/03_exposure1_problem.png', dpi=90)
