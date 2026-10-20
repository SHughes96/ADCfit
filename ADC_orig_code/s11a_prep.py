"""Inputs for s11_source_figure: sky spectrum near the source and per-exposure line fluxes with
detector-defect fibres removed (fibres with an excess at the same λ in the OTHER field)."""
import sys, json, numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t, x0, y0, lam0, label = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
other = [q for q in TARGETS if q != t][0]
sd = scr_dir(t)
cw = WCS(fits.getheader(cube_path(t, 'blue'))).celestial
ra0, dec0 = [float(q) for q in cw.pixel_to_world_values(x0, y0)]; cosd = np.cos(np.radians(dec0))
FIX = exposure_shifts(t)
def load(p, fix=None):
    with fits.open(p) as h:
        fm = h['FIBERMAP'].data; fl = h['FLAM'].data.astype(float); w = h['WAVE'].data.astype(float); sky = h['SKY'].data.astype(float)
    fl = np.where(np.isfinite(fl), fl, 0)
    d = dict(res=fl - median_filter(fl, size=(1, 101), mode='nearest'), w=w, dw=np.nanmedian(np.diff(w, axis=1), axis=1),
             ra=np.array(fm['RA'], float), dec=np.array(fm['DEC'], float), sky=sky)
    if fix: d['ra'] += fix[0] / 3600 / cosd; d['dec'] += fix[1] / 3600
    return d
def Lline(d):
    return np.nansum(np.where(np.abs(d['w'] - lam0) < 4, d['res'], 0), axis=1) * d['dw']
Lo = []
for p in rss_paths(other, 'blue'):
    d = load(p); L = Lline(d); ok = L != 0; s = 1.4826 * np.median(np.abs(L[ok] - np.median(L[ok]))); Lo.append(L / s)
bad = np.mean(Lo, axis=0) > 1.5
rng = np.random.default_rng(9)
out = dict(exp=[], val=[], err=[]); skies = []
for p in rss_paths(t, 'blue'):
    e = exp_id(p); d = load(p, FIX.get(e)); L = Lline(d)
    keep = ~bad & np.isfinite(d['ra']) & (d['ra'] > 0)
    def ap(r0, d0):
        nn = keep & (np.hypot((d['ra'] - r0) * cosd, d['dec'] - d0) * 3600 < 1.2)
        return np.nansum(L[nn]) if nn.sum() >= 3 else np.nan
    dc = np.hypot((d['ra'] - ra0) * cosd, d['dec'] - dec0) * 3600
    pool = np.where(keep & (dc > 5))[0]
    rnd = np.array([ap(d['ra'][j], d['dec'][j]) for j in rng.choice(pool, 200, replace=False)]); rnd = rnd[np.isfinite(rnd)]
    m = np.median(rnd); s = 1.4826 * np.median(np.abs(rnd - m))
    out['exp'].append(e); out['val'].append(float(ap(ra0, dec0) - m)); out['err'].append(float(s))
    near = dc < 3
    skies.append((np.nanmedian(d['w'][near], axis=0), np.nanmedian(d['sky'][near], axis=0)))
json.dump(out, open(f'{sd}/perexp_{label}.json', 'w'))
w0 = skies[0][0]; sky = np.nanmean([np.interp(w0, w, s) for w, s in skies], axis=0)
json.dump(dict(w=w0.tolist(), v=sky.tolist()), open(f'{sd}/sky_{label}.json', 'w'))
v, e = np.array(out['val']), np.array(out['err'])
print(f'{label}: per-exposure S/N {np.round(v / e, 1).tolist()}; combined {np.nansum(v / e**2) / np.sqrt(np.nansum(1 / e**2)):.1f}σ; defect fibres excluded: {bad.sum()}')
