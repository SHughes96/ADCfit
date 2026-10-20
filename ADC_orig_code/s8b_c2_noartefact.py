"""C2 with the camera-3A defect fibres removed: per-exposure line flux vs empirical noise from random
apertures (also with those fibres removed), plus the same artefact census for C1 and C4."""
import sys, json, numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = 'SPT-CLJ0158-5500'; other = 'SPT-CLJ0111-5637'
vet = {c['c']: c for c in json.load(open(f'{scr_dir(t)}/vet.json'))}
with fits.open(cube_path(t, 'blue')) as h:
    cw = WCS(h[0].header).celestial
def load(p):
    with fits.open(p) as h:
        fm = h['FIBERMAP'].data
        fl = h['FLAM'].data.astype(float); w = h['WAVE'].data.astype(float)
    fl = np.where(np.isfinite(fl), fl, 0)
    return dict(res=fl - median_filter(fl, size=(1, 101), mode='nearest'), w=w, dw=np.nanmedian(np.diff(w, axis=1), axis=1),
                ra=np.array(fm['RA'], float), dec=np.array(fm['DEC'], float))
data = {exp_id(p): load(p) for p in rss_paths(t, 'blue')}
odata = [load(p) for p in rss_paths(other, 'blue')]
rng = np.random.default_rng(3)
for cid in [2, 1, 4]:
    c = vet[cid]; lam0 = c['lam']
    ra0, dec0 = [float(q) for q in cw.pixel_to_world_values(c['x'], c['y'])]
    cosd = np.cos(np.radians(dec0))
    # artefact census: fibres whose line-window S/N in the OTHER field averages > 1.5
    Lo = []
    for d in odata:
        L = np.nansum(np.where(np.abs(d['w'] - lam0) < 4, d['res'], 0), axis=1) * d['dw']
        ok = L != 0; s = 1.4826 * np.median(np.abs(L[ok] - np.median(L[ok]))); Lo.append(L / s)
    Lo = np.mean(Lo, axis=0)
    bad = set(np.where(Lo > 1.5)[0].tolist())
    for variant, excl in [('all fibres', set()), ('defect fibres removed', bad)]:
        vals, errs = [], []
        for e, d in data.items():
            L = np.nansum(np.where(np.abs(d['w'] - lam0) < 4, d['res'], 0), axis=1) * d['dw']
            keep = np.ones(len(L), bool); keep[list(excl)] = False
            keep &= np.isfinite(d['ra']) & (d['ra'] > 0)
            def ap(r0, d0):
                nn = keep & (np.hypot((d['ra'] - r0) * cosd, d['dec'] - d0) * 3600 < 1.2)
                return np.nansum(L[nn]) if nn.sum() >= 3 else np.nan
            dc = np.hypot((d['ra'] - ra0) * cosd, d['dec'] - dec0) * 3600
            pool = np.where(keep & (dc > 5))[0]
            rnd = np.array([ap(d['ra'][j], d['dec'][j]) for j in rng.choice(pool, 200, replace=False)]); rnd = rnd[np.isfinite(rnd)]
            m = np.median(rnd); s = 1.4826 * np.median(np.abs(rnd - m))
            vals.append(ap(ra0, dec0) - m); errs.append(s)
        vals, errs = np.array(vals), np.array(errs)
        wm = np.nansum(vals / errs ** 2) / np.nansum(1 / errs ** 2); se = np.sqrt(1 / np.nansum(1 / errs ** 2))
        print('C%d %.1f Å  %-22s per-exposure S/N %s  -> combined %.1fσ, flux %.1e, χ²=%.1f/%d' % (
            cid, lam0, variant + ':', np.round(vals / errs, 1).tolist(), wm / se, wm, np.nansum(((vals - wm) / errs) ** 2), np.isfinite(vals).sum() - 1))
    near_bad = [k for k in sorted(bad) if k in range(1380, 1480) or Lo[k] > 2.5]
    print(f'     defect fibres at {lam0:.0f} Å (mean S/N>1.5 in the other field): {len(bad)} total; strongest: {sorted(bad, key=lambda k: -Lo[k])[:6]} (S/N {np.round(sorted(Lo[list(bad)], reverse=True)[:6], 1).tolist()})')
