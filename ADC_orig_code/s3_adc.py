"""ADC / differential-refraction test: centroid of a source vs wavelength in each exposure (2-D
Gaussian fit to fibre values, instrument frame), against the UNCORRECTED atmospheric dispersion
predicted from airmass, P, T and the parallactic angle (blue displaced toward the zenith)."""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Combine.spectrum import _refractivity
t = sys.argv[1]; sx0, sy0 = float(sys.argv[2]), float(sys.argv[3]); label = sys.argv[4] if len(sys.argv) > 4 else 'star'
bins = {'blue': np.linspace(3600, 4700, 6), 'green': np.linspace(4800, 6900, 5), 'red': np.linspace(7000, 9600, 5)}
LREF = 6000.0

SIG = None
def fit(x, y, v):
    def model(p):
        a, xc, yc, s, b = p if SIG is None else (p[0], p[1], p[2], SIG, p[3])
        return a * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * s ** 2)) + b - v
    p0 = [np.nanmax(v) - np.nanmedian(v), x[np.nanargmax(v)], y[np.nanargmax(v)]]
    if SIG is None:
        p0 += [0.7, np.nanmedian(v)]; lo = [0, x.min(), y.min(), 0.25, -np.inf]; hi = [np.inf, x.max(), y.max(), 3, np.inf]
    else:
        p0 += [np.nanmedian(v)]; lo = [0, x.min(), y.min(), -np.inf]; hi = [np.inf, x.max(), y.max(), np.inf]
    r = least_squares(model, p0, bounds=(lo, hi))
    J = r.jac; cov = np.linalg.pinv(J.T @ J) * np.sum(r.fun ** 2) / max(len(v) - len(p0), 1)
    out = r.x if SIG is None else np.r_[r.x[:3], SIG, r.x[3]]
    err = np.sqrt(np.diag(cov)); err = err if SIG is None else np.r_[err[:3], 0, err[3]]
    return out, err

results = []
paths = {ch: rss_paths(t, ch) for ch in bins}
for i in range(len(paths['red'])):
    e = exp_id(paths['red'][i])
    SIG = None
    for _pass in range(2):
      rows = []
      for ch, edges in (bins.items() if _pass else [('red', bins['red'])]):
          with fits.open(paths[ch][i]) as h:
              ss = h['SKYSUB'].data; w = h['WAVE'].data; fm = h['FIBERMAP'].data; hd = h[0].header
          ra, dec = np.array(fm['RA'], float), np.array(fm['DEC'], float)
          c = np.cos(np.radians(np.median(dec)))
          x = (ra - np.median(ra)) * c * 3600; y = (dec - np.median(dec)) * 3600   # +x = East
          sel = np.hypot(x - sx0, y - sy0) < 3.0
          sel[135] = False
          for lo, hi in zip(edges[:-1], edges[1:]):
              v = np.nanmedian(np.where((w[sel] > lo) & (w[sel] < hi), ss[sel], np.nan), axis=1)
              ok = np.isfinite(v)
              if ok.sum() < 8: continue
              p, e_ = fit(x[sel][ok], y[sel][ok], v[ok])
              if p[0] <= 0 or e_[1] > 1 or e_[2] > 1: continue
              rows.append(dict(lam=0.5 * (lo + hi), x=p[1], y=p[2], ex=e_[1], ey=e_[2], fwhm=2.355 * p[3], ch=ch))
      if _pass == 0:
          SIG = np.median([r['fwhm'] for r in rows]) / 2.355
    X = float(hd['AIRMASS']); q = float(hd['TEL PARANGL']); P = float(hd.get('WTH-PRES', 765)); T = float(hd.get('WTH-TEMP', 10))
    z = np.arccos(1 / X)
    scale = (P / 1013.25) * (288.15 / (273.15 + T))
    lam = np.array([r['lam'] for r in rows])
    mag = 206265 * (_refractivity(lam) - _refractivity(LREF)) * scale * np.tan(z)   # arcsec, + toward zenith
    # zenith direction on sky has position angle q (E of N): +x(East) = sin q, +y(North) = cos q
    pred_x, pred_y = mag * np.sin(np.radians(q)), mag * np.cos(np.radians(q))
    xs = np.array([r['x'] for r in rows]); ys = np.array([r['y'] for r in rows])
    ref = np.argmin(abs(lam - LREF))
    k = (lam > 7000)
    xr, yr = np.average(xs[k]), np.average(ys[k])            # anchor at red mean
    pxr, pyr = np.average(pred_x[k]), np.average(pred_y[k])
    obs_dx, obs_dy = xs - xr, ys - yr
    prd_dx, prd_dy = pred_x - pxr, pred_y - pyr
    b = lam < 4300
    obs_b = (obs_dx[b].mean(), obs_dy[b].mean()); prd_b = (prd_dx[b].mean(), prd_dy[b].mean())
    results.append(dict(exp=e, X=X, q=q, rows=rows, obs=list(zip(lam, obs_dx, obs_dy)), prd=list(zip(lam, prd_dx, prd_dy)),
                        obs_blue=obs_b, prd_blue=prd_b))
    print('%s X %.3f parang %+6.1f | blue(<4300) minus red centroid: observed (%+.2f, %+.2f)" |%.2f"|  vs uncorrected DAR (%+.2f, %+.2f)" |%.2f"|  FWHM blue %.2f red %.2f' % (
        e, X, q, *obs_b, np.hypot(*obs_b), *prd_b, np.hypot(*prd_b),
        np.median([r['fwhm'] for r in rows if r['lam'] < 4300]), np.median([r['fwhm'] for r in rows if r['lam'] > 7000])))
json.dump(results, open(f'{scr_dir(t)}/adc_{label}.json', 'w'), indent=1, default=float)
# figure: one panel per exposure (dx, dy vs lambda) + vector summary
n = len(results)
fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 6.2), sharey='row')
for j, r in enumerate(results):
    lam = np.array([o[0] for o in r['obs']])
    for row, idx, nm in [(0, 1, 'ΔE-W (RA)'), (1, 2, 'ΔN-S (Dec)')]:
        ax[row, j].errorbar(lam, [o[idx] for o in r['obs']], yerr=[rr['ex' if idx == 1 else 'ey'] for rr in r['rows']], fmt='o', ms=3, color='k', label='measured')
        ax[row, j].plot(lam, [p[idx] for p in r['prd']], 'r-', label='uncorrected DAR')
        ax[row, j].axhline(0, color='gray', lw=0.5)
        if j == 0: ax[row, j].set_ylabel(nm + ' ["]')
        ax[row, j].set_ylim(-2.5, 2.5)
    ax[0, j].set_title(f"{r['exp']}\nX={r['X']:.2f} q={r['q']:+.0f}°", fontsize=8)
    ax[1, j].set_xlabel('λ [Å]', fontsize=8)
ax[0, 0].legend(fontsize=7)
fig.suptitle(f'{t}: {label} centroid vs wavelength (relative to red) — flat = ADC working', fontsize=12)
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/05_ADC_dispersion_{label}.png', dpi=95)
