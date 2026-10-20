"""Deep-dive on SPT-CLJ0158-5500 C2 (4193 Å): which fibres carry the line in each exposure, where
it sits on the bundle, whether those fibres show the same 'line' in the other field (detector
artefact test), the line profile, and any continuum counterpart."""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Utils.centroid import fibre_centroid
t = 'SPT-CLJ0158-5500'; other_t = 'SPT-CLJ0111-5637'
v = [c for c in json.load(open(f'{scr_dir(t)}/vet.json')) if c['c'] == 2][0]
lam0, x0, y0 = v['lam'], v['x'], v['y']
with fits.open(cube_path(t, 'blue')) as h:
    cw = WCS(h[0].header).celestial
ra0, dec0 = [float(q) for q in cw.pixel_to_world_values(x0, y0)]
pt = {r['exp']: r for r in json.load(open(f'{scr_dir(t)}/pointing.json'))}
def lineL(d):
    fl = np.where(np.isfinite(d['fl']), d['fl'], 0)
    res = fl - median_filter(fl, size=(1, 101), mode='nearest')
    win = np.abs(d['w'] - lam0) < 4
    dw = np.nanmedian(np.diff(d['w'], axis=1), axis=1)
    return np.nansum(np.where(win, res, 0), axis=1) * dw, res
def load(p):
    with fits.open(p) as h:
        fm = h['FIBERMAP'].data
        return dict(fl=h['FLAM'].data.astype(float), w=h['WAVE'].data.astype(float),
                    ra=np.array(fm['RA'], float), dec=np.array(fm['DEC'], float), id=np.array(fm['FIBER_ID']),
                    bs=np.array(fm['BENCHSIDE']).astype(str))
rows = []; specs = []
allfib = set()
for p in rss_paths(t, 'blue'):
    d = load(p); e = exp_id(p)
    L, res = lineL(d)
    cosd = np.cos(np.radians(dec0))
    sky_d = np.hypot((d['ra'] - ra0) * cosd, d['dec'] - dec0) * 3600
    near = np.where(sky_d < 2.5)[0]
    # line centroid on the SKY (fibre positions as combined) and on the BUNDLE (instrument frame)
    wts = np.clip(L[near], 0, None)
    sx = np.sum(wts * (d['ra'][near] - ra0) * cosd * 3600) / wts.sum(); sy = np.sum(wts * (d['dec'][near] - dec0) * 3600) / wts.sum()
    c_ = np.cos(np.radians(np.nanmedian(d['dec'])))
    bx = (d['ra'] - np.nanmedian(d['ra'])) * c_ * 3600; by = (d['dec'] - np.nanmedian(d['dec'])) * 3600
    ibx = np.sum(wts * bx[near]) / wts.sum(); iby = np.sum(wts * by[near]) / wts.sum()
    top = near[np.argsort(L[near])[::-1][:3]]
    allfib |= set(top.tolist())
    rows.append(dict(exp=e, sky=(sx, sy), bundle=(ibx, iby), top=[(int(k), str(d['bs'][k]), float(L[k])) for k in top]))
    ap = near[sky_d[near] < 1.2]
    w_ = np.nanmedian(d['w'][ap], axis=0); specs.append((e, w_, np.nansum(res[ap], axis=0)))
    print('%s  line centroid on SKY (rel. to cube pos) %+.2f %+.2f"   on BUNDLE %+6.2f %+6.2f"   top fibres %s' % (
        e, sx, sy, ibx, iby, [(k, b, '%.1e' % l) for k, b, l in rows[-1]['top']]))
b0 = np.array(rows[0]['bundle'])
print('\nbundle motion vs commanded dither (relative to exposure 1):')
for r in rows:
    print('  %s moved on bundle %+5.2f %+5.2f"   commanded (−offset) %+5.2f %+5.2f"' % (
        r['exp'], r['bundle'][0] - b0[0], r['bundle'][1] - b0[1],
        -(pt[r['exp']]['hra'] - pt[rows[0]['exp']]['hra']) * np.cos(np.radians(dec0)) * 3600,
        -(pt[r['exp']]['hdec'] - pt[rows[0]['exp']]['hdec']) * 3600))
# detector-artefact test: the same fibre indices in the OTHER field's exposures, same λ window
print(f'\nartefact test: fibres {sorted(allfib)} in {other_t} exposures at {lam0:.1f} Å')
zs = []
for p in rss_paths(other_t, 'blue'):
    d = load(p); L, _ = lineL(d)
    ok = np.isfinite(L) & (L != 0)
    s = 1.4826 * np.median(np.abs(L[ok] - np.median(L[ok])))
    zs.append([L[k] / s for k in sorted(allfib)])
zs = np.array(zs)
print('  per-fibre line S/N in the other field (rows = exposures):'); print(np.round(zs, 1))
print('  mean over exposures per fibre:', np.round(zs.mean(axis=0), 2))
# same test in THIS field at a wavelength-shifted control and at random fibres
d = load(rss_paths(t, 'blue')[3]); L, _ = lineL(d); ok = np.isfinite(L) & (L != 0)
s = 1.4826 * np.median(np.abs(L[ok] - np.median(L[ok])))
print('  for scale: in this field (exp 4) those fibres give', np.round([L[k] / s for k in sorted(allfib)], 1))
# continuum counterpart (green/red white light) near the position, expecting red ~1.4" E of blue (ADC)
for ch in ['green', 'red']:
    wli = np.load(f'{scr_dir(t)}/wl_{ch}.npy')
    yy, xx = np.mgrid[:wli.shape[0], :wli.shape[1]]
    vals = wli[np.isfinite(wli)]; m = np.median(vals); sg = 1.4826 * np.median(np.abs(vals - m))
    for dx_ in [0, -2.8]:                      # 0 and 1.4" East (East = -x pixels)
        a_ = np.hypot(xx - (x0 + dx_), yy - y0) <= 2.4
        print(f'  {ch} white light at C2{" +1.4\\\" E" if dx_ else ""}: mean {np.nanmean(wli[a_]):.2e} ({(np.nanmean(wli[a_]) - m) / (sg / np.sqrt(a_.sum())):+.1f}σ of aperture mean)')
# figure: profile per exposure + stacked
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
stack_w = specs[0][1]; acc = np.zeros_like(stack_w)
for e, w_, s_ in specs:
    sel = np.abs(w_ - lam0) < 40
    ax[0].plot(w_[sel], s_[sel], lw=0.7, label=e)
    acc += np.interp(stack_w, w_, s_)
ax[0].axvline(lam0, color='k', ls=':'); ax[0].legend(fontsize=6); ax[0].set_title('C2: per-exposure aperture spectra (fibres < 1.2")', fontsize=10)
sel = np.abs(stack_w - lam0) < 40
ax[1].step(stack_w[sel], acc[sel] / len(specs), where='mid', color='k')
ax[1].axvline(lam0, color='b', ls=':'); ax[1].axhline(0, color='gray', lw=0.5)
ax[1].set_title(f'C2: mean of 8 exposures — Lyα z={lam0/1215.67-1:.3f} / [O II] z={lam0/3727.4-1:.3f}', fontsize=10)
for a_ in ax: a_.set_xlabel('λ [Å]')
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/09_C2_line_profile.png', dpi=95)
json.dump(dict(rows=rows, artefact_z=zs.tolist(), fibres=sorted(allfib)), open(f'{scr_dir(t)}/c2.json', 'w'), indent=1, default=float)
