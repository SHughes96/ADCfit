"""Matched-filter emission-line search in the blue cube.

Continuum = running median along wavelength; residual is smoothed with a 3-D kernel (Gaussian PSF
spatially, box spectrally); S/N is normalised EMPIRICALLY per wavelength slice (robust MAD over
fully-covered spaxels), because the cube's resampled VAR ignores the covariance the smoothing adds.
Optional --inject adds fake Gaussian lines first, to measure recovery.
"""
import sys, json, argparse, numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter1d, label, find_objects

p = argparse.ArgumentParser()
p.add_argument('scratch'); p.add_argument('--inject', type=float, default=0.0,
               help='total line flux erg/s/cm2 of each fake line (0 = none)')
p.add_argument('--tag', default='real')
a = p.parse_args()
S = a.scratch
with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_blue.fits') as h:
    f = h[0].data.astype(float); nexp = h['NEXP'].data
    wl = h['WAVELENGTH'].data.field(0); pix_as = h[0].header['PIXSCALE']
nz, ny, nx = f.shape
dw = np.median(np.diff(wl))

fakes = []
if a.inject > 0:
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:ny, :nx]
    sig_pix = 1.2 / 2.355 / pix_as                     # 1.2" FWHM seeing
    for k in range(12):
        lam = rng.uniform(3400, 4700)
        y0, x0 = rng.uniform(15, 70), rng.uniform(15, 70)
        sig_l = 2.0 / 2.355 * 2                         # ~4.7 A FWHM (unresolved + ~200 km/s)
        spat = np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sig_pix**2))
        spat /= spat.sum() * pix_as**2                  # per arcsec2, integrates to 1
        spec = np.exp(-(wl - lam)**2 / (2 * sig_l**2)); spec /= spec.sum() * dw   # per A
        f += a.inject * spec[:, None, None] * spat[None]
        fakes.append(dict(lam=lam, y=y0, x=x0))

good = nexp == 3
fz = np.where(np.isfinite(f), f, np.nan)
# continuum: running median over 101 px (~77 A) along wavelength
cont = median_filter(np.nan_to_num(fz), size=(101, 1, 1), mode='nearest')
res = np.nan_to_num(fz - cont)
valid = np.isfinite(fz).astype(float)
# matched filter: spectral box 5 px (3.8 A), spatial Gaussian sigma ~1 px (1.2" FWHM / 0.5"pix)
def smooth(x):
    x = uniform_filter1d(x, 5, axis=0)
    return gaussian_filter(x, sigma=(0, 1.0, 1.0))
w = smooth(valid)
sm = np.where(w > 0.5, smooth(res) / np.maximum(w, 1e-6), 0.0)
# empirical per-slice noise over fully covered spaxels, edges eroded by 3 px
from scipy.ndimage import binary_erosion
core = binary_erosion(good, iterations=3)
mad = np.array([np.nanmedian(np.abs(s[core] - np.nanmedian(s[core]))) for s in sm]) * 1.4826
snr = sm / mad[:, None, None]
snr[:, ~core] = 0
np.save(f'{S}/snr_{a.tag}.npy', snr.astype(np.float32)); np.save(f'{S}/mad_{a.tag}.npy', mad)
np.save(f'{S}/res_{a.tag}.npy', res.astype(np.float32))
print('slice-noise (MAD) range', np.nanpercentile(mad, [5, 50, 95]))
print('global snr std over core (should be ~1):', np.nanstd(snr[:, core]))

lab, n = label(snr > 5)
cands = []
for i, sl in enumerate(find_objects(lab), 1):
    m = lab[sl] == i
    pk = np.unravel_index(np.argmax(np.where(m, snr[sl], -1)), m.shape)
    z0, y0, x0 = [s.start + q for s, q in zip(sl, pk)]
    cands.append(dict(id=i, lam=float(wl[z0]), y=int(y0), x=int(x0), snr=float(snr[z0, y0, x0]),
                      nvox=int(m.sum()), dz=int(sl[0].stop - sl[0].start),
                      area=int(m.any(axis=0).sum())))
cands.sort(key=lambda c: -c['snr'])
print(f'{len(cands)} connected regions above 5 sigma (and {int((snr < -5).sum())} voxels below -5)')
for c in cands[:40]:
    print('  lam %7.1f  (x,y)=(%2d,%2d)  peak S/N %5.1f  nvox %4d  dlam_px %3d  area %3d' %
          (c['lam'], c['x'], c['y'], c['snr'], c['nvox'], c['dz'], c['area']))
if fakes:
    for fk in fakes:
        hit = [c for c in cands if abs(c['lam'] - fk['lam']) < 4 and np.hypot(c['x'] - fk['x'], c['y'] - fk['y']) < 2.5]
        print('fake lam %.1f (x,y)=(%.0f,%.0f) -> %s' % (fk['lam'], fk['x'], fk['y'],
              'RECOVERED S/N %.1f' % hit[0]['snr'] if hit else 'missed (snr at pos %.1f)' %
              snr[np.argmin(abs(wl - fk['lam'])), int(round(fk['y'])), int(round(fk['x']))]))
json.dump(dict(cands=cands, fakes=fakes), open(f'{S}/cands_{a.tag}.json', 'w'), indent=1)
