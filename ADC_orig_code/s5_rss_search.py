"""Fibre-level line search in every blue RSS of a target, cross-matched between exposures.
Hit: per-fibre continuum-subtracted (running median 101 px), robust per-fibre noise, 5-px box;
>=5σ peak with its 3 central pixels each >2σ. Group: same sky position (<1.5") and λ (<3 Å).
Keep groups seen in >=3 exposures by >=2 distinct fibres, not at field-wide systematic
wavelengths, not on continuum sources, not on the bundle-edge artefact fibres."""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter, uniform_filter1d
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = sys.argv[1]; sd = scr_dir(t)
FIX = exposure_shifts(t)   # exp1 re-registration (arcsec)
ARTEFACT_BUNDLE = [(-15.2, 2.0), (-15.6, -14.5), (-15.8, 6.9), (-13.7, 13.2)]     # same in both fields
hits = []
for p in rss_paths(t, 'blue'):
    e = exp_id(p)
    with fits.open(p) as h:
        ss = h['SKYSUB'].data.astype(float); wave = h['WAVE'].data.astype(float)
        mk = h['MASK'].data != 0; fm = h['FIBERMAP'].data
    ra, dec = np.array(fm['RA'], float), np.array(fm['DEC'], float)
    c = np.cos(np.radians(np.nanmedian(dec)))
    bx = (ra - np.nanmedian(ra)) * c * 3600; by = (dec - np.nanmedian(dec)) * 3600
    art = np.zeros(len(ra), bool)
    for ax_, ay_ in ARTEFACT_BUNDLE:
        art |= np.hypot(bx - ax_, by - ay_) < 2.0
    art[135] = True
    if e in FIX:
        ra = ra + FIX[e][0] / 3600 / c; dec = dec + FIX[e][1] / 3600
    ss[mk | ~np.isfinite(ss)] = np.nan
    ok = (np.isfinite(ss).sum(axis=1) > 1000) & ~art & np.isfinite(ra) & (ra > 0)
    band = (wave > 3500) & (wave < 4780)
    filled = np.where(np.isfinite(ss), ss, np.nanmedian(ss, axis=1, keepdims=True))
    r = filled - median_filter(filled, size=(1, 101), mode='nearest')
    noise = 1.4826 * np.nanmedian(np.abs(np.where(band, r, np.nan)), axis=1, keepdims=True)
    z1 = r / noise; z1[z1 < -10] = 0
    zs = uniform_filter1d(z1, 5, axis=1) * np.sqrt(5)
    zs[~band | ~ok[:, None]] = 0
    n = 0
    for i in np.where(ok)[0]:
        cand = np.where(zs[i] > 5)[0]
        while cand.size:
            k = cand[np.argmax(zs[i, cand])]
            if k > 1 and np.all(z1[i, k - 1:k + 2] > 2):
                hits.append(dict(exp=e, fib=int(i), ra=float(ra[i]), dec=float(dec[i]), lam=float(wave[i, k]), z=float(zs[i, k])))
                n += 1
            cand = cand[np.abs(cand - k) > 8]
    print(f'  {e}: {n} fibre-level hits')
lam_all = np.array([h['lam'] for h in hits])
hist, edges = np.histogram(lam_all, bins=np.arange(3500, 4790, 5))
busy = hist > max(5 * np.median(hist[hist > 0]), 8)
def is_sys(l):
    i = np.searchsorted(edges, l) - 1
    return 0 <= i < len(busy) and (busy[max(i - 1, 0):i + 2].any())
print('  field-wide systematic wavelength bins:', [int(edges[i]) for i in np.where(busy)[0]])
groups = []
for h in sorted(hits, key=lambda q: -q['z']):
    for g in groups:
        g0 = g[0]
        if abs(h['lam'] - g0['lam']) < 3 and np.hypot((h['ra'] - g0['ra']) * np.cos(np.radians(g0['dec'])), h['dec'] - g0['dec']) * 3600 < 1.5:
            g.append(h); break
    else:
        groups.append([h])
with fits.open(cube_path(t, 'blue')) as hh:
    cw = WCS(hh[0].header).celestial
srcpk = json.load(open(f'{sd}/search.json'))['srcpk'] if os.path.exists(f'{sd}/search.json') else []
keep = []
for g in groups:
    exps = {h['exp'] for h in g}; fibs = {h['fib'] for h in g}
    if len(exps) < 3 or len(fibs) < 2: continue
    lam = float(np.mean([h['lam'] for h in g]))
    x, y = [float(v) for v in cw.world_to_pixel_values(g[0]['ra'], g[0]['dec'])]
    on_src = any(np.hypot(x - sx, y - sy) < 6 for sy, sx in srcpk)
    keep.append(dict(lam=lam, ra=g[0]['ra'], dec=g[0]['dec'], x=x, y=y, nexp=len(exps), nfib=len(fibs),
                     snr=sorted([round(h['z'], 1) for h in g], reverse=True), sys=bool(is_sys(lam)), on_src=bool(on_src)))
keep.sort(key=lambda k: (-k['nexp'], -sum(k['snr'])))
clean = [k for k in keep if not k['sys'] and not k['on_src']]
print(f'  groups in >=3 exposures with >=2 fibres: {len(keep)}  (systematic λ: {sum(k["sys"] for k in keep)}, on continuum source: {sum(k["on_src"] for k in keep)}) -> clean candidates: {len(clean)}')
for k in clean[:20]:
    print('    %7.1f Å (Lyα z=%.3f) cube (%5.1f,%5.1f) seen in %d/8 exposures, %d fibres, S/N %s' % (
        k['lam'], k['lam'] / 1215.67 - 1, k['x'], k['y'], k['nexp'], k['nfib'], k['snr'][:6]))
json.dump(dict(hits=len(hits), keep=keep, clean=clean, busy=[int(edges[i]) for i in np.where(busy)[0]]), open(f'{sd}/rss_search.json', 'w'), indent=1)
fig, ax = plt.subplots(figsize=(12, 3.5))
for e in sorted({h['exp'] for h in hits}):
    ax.hist([h['lam'] for h in hits if h['exp'] == e], bins=edges, histtype='step', label=e, lw=0.8)
for k in clean: ax.axvline(k['lam'], color='k', ls=':', lw=0.8)
ax.set_yscale('log'); ax.legend(fontsize=6, ncol=4); ax.set_xlabel('λ [Å]')
ax.set_title(f'{t}: single-exposure fibre line hits vs λ (dotted = surviving multi-exposure candidates)', fontsize=10)
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/07_rss_line_hits.png', dpi=95)
