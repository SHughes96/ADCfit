"""C2 identity: line width (Gaussian fit to the cube aperture spectrum), continuum & EW limit,
spatial size (2-D Gaussian on the narrowband image), and measured fluxes/limits of the lines that
would accompany it as [O II] (z=0.125) or Lyα (z=2.449), each with empirical errors from random
apertures at the same wavelength."""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Combine.spectrum import fit_gaussian_image
t = 'SPT-CLJ0158-5500'; sd = scr_dir(t)
c = [q for q in json.load(open(f'{sd}/vet.json')) if q['c'] == 2][0]
x0, y0, lam0 = c['x'], c['y'], c['lam']
use = np.load(f'{sd}/use_mask.npy')
cubes = {}
for ch in ['blue', 'green', 'red']:
    with fits.open(cube_path(t, ch)) as h:
        cubes[ch] = (h[0].data.astype(float), h['WAVELENGTH'].data.field(0))
ny, nx = cubes['blue'][0].shape[1:]; yy, xx = np.mgrid[:ny, :nx]
R = 3.0                                                  # aperture radius in pixels (1.5")
def apmask(px, py): return np.hypot(xx - px, yy - py) <= R
rng = np.random.default_rng(11); rnd_pts = np.argwhere(use & (np.hypot(xx - x0, yy - y0) > 10))[rng.integers(0, 10**6, 150) % (use & (np.hypot(xx - x0, yy - y0) > 10)).sum()]
def line_flux(ch, lw, half=4.0):
    f, wl = cubes[ch]
    if not (wl[0] + 60 < lw < wl[-1] - 60): return None
    sel = np.abs(wl - lw) < 80
    sub = f[sel]; w_ = wl[sel]
    win = np.abs(w_ - lw) < half; side = (np.abs(w_ - lw) > 15) & (np.abs(w_ - lw) < 70)
    dw = np.median(np.diff(wl))
    def meas(px, py):
        a = apmask(px, py)
        s = np.nansum(sub[:, a], axis=1) * 0.25              # SB -> flux/Å
        return (np.nansum(s[win]) - np.nanmedian(s[side]) * win.sum()) * dw, np.nanmedian(s[side])
    val, cont = meas(x0, y0)
    rv = np.array([meas(px, py)[0] for py, px in rnd_pts])
    m = np.median(rv); sg = 1.4826 * np.median(np.abs(rv - m))
    return val - m, sg, cont
F, eF, cont = line_flux('blue', lam0)
print('C2 Lyα/[O II] line: flux %.2e ± %.1e erg/s/cm² (%.1fσ, cube aperture r=1.5")' % (F, eF, F / eF))
# continuum at the line (blue, 4100–4300 excluding line) and EW limit
f, wl = cubes['blue']; a = apmask(x0, y0)
side = ((wl > 4100) & (wl < 4300)) & (np.abs(wl - lam0) > 15)
cs = np.nansum(f[side][:, a], axis=1) * 0.25
crnd = np.array([np.nanmedian(np.nansum(f[side][:, apmask(px, py)], axis=1) * 0.25) for py, px in rnd_pts])
cm, cmad = np.median(crnd), 1.4826 * np.median(np.abs(crnd - np.median(crnd)))
cval = np.nanmedian(cs) - cm
ew_obs_lim = F / max(cval + 2 * cmad, 2 * cmad)
print('   blue continuum at line: %.1e ± %.1e erg/s/cm²/Å  -> observed EW > %.0f Å (2σ); rest-frame EW: Lyα > %.0f Å, [O II] > %.0f Å' % (
    cval, cmad, ew_obs_lim, ew_obs_lim / (lam0 / 1215.67), ew_obs_lim / (lam0 / 3727.4)))
# line width: Gaussian fit to aperture spectrum
sel = np.abs(wl - lam0) < 30
s = np.nansum(f[sel][:, a], axis=1) * 0.25; s = s - np.nanmedian(s[np.abs(wl[sel] - lam0) > 10])
g = lambda x, A, mu, sg: A * np.exp(-(x - mu) ** 2 / (2 * sg ** 2))
p, cv = curve_fit(g, wl[sel], np.nan_to_num(s), p0=[np.nanmax(s), lam0, 2.0])
fwhm = 2.355 * abs(p[2]); efwhm = 2.355 * np.sqrt(cv[2, 2])
res_fwhm = lam0 / 1738.0
print('   line centre %.2f Å, FWHM %.1f ± %.1f Å (instrumental ~%.1f Å at R≈1738) -> intrinsic ~%.0f km/s' % (
    p[1], fwhm, efwhm, res_fwhm, 3e5 * np.sqrt(max(fwhm ** 2 - res_fwhm ** 2, 0)) / p[1]))
# spatial size
k = np.argmin(abs(wl - lam0)); nb = np.nanmean(f[k - 4:k + 5], axis=0)
gg = fit_gaussian_image(np.where(np.isfinite(nb), nb, 0), x0, y0, 5)
if gg is not None:
    print('   narrowband 2-D Gaussian FWHM %.2f" × %.2f" (blue PSF in this run ~1.0–1.3"; kernel 0.9")' % (2.355 * abs(gg.x_stddev.value) * 0.5, 2.355 * abs(gg.y_stddev.value) * 0.5))
# companion lines
for nm, rest, z in [('C IV 1549', 1549.0, lam0 / 1215.67 - 1), ('He II 1640', 1640.4, lam0 / 1215.67 - 1), ('C III] 1909', 1908.7, lam0 / 1215.67 - 1),
                    ('Hβ 4861', 4861.3, lam0 / 3727.4 - 1), ('[O III] 5007', 5006.8, lam0 / 3727.4 - 1), ('Hα 6563', 6562.8, lam0 / 3727.4 - 1),
                    ('[N II] 6583', 6583.4, lam0 / 3727.4 - 1), ('[S II] 6716/31', 6724.0, lam0 / 3727.4 - 1)]:
    lw = rest * (1 + z)
    for ch in ['blue', 'green', 'red']:
        r = line_flux(ch, lw, half=6.0)
        if r is not None:
            print('   %-15s @ %6.0f Å (%-5s, z=%.3f): %+.1e ± %.1e  (%+.1fσ)  ratio to C2 line: %+.2f (2σ upper %.2f)' % (
                nm, lw, ch, z, r[0], r[1], r[0] / r[1], r[0] / F, (max(r[0], 0) + 2 * r[1]) / F))
            break
