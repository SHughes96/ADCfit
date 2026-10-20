"""Blind emission-line (Lyα) search in a blue cube, plus fake-line injection to calibrate limits.
Continuum = running median (101 px) along λ; residual matched-filtered (Gaussian σ=1 px spatially,
5-px box spectrally); S/N normalised empirically per slice then per spaxel. Searches spaxels with
NEXP >= nmin (eroded 3 px), excluding compact continuum sources (discs of 3" around peaks)."""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import (median_filter, gaussian_filter, uniform_filter1d, label, find_objects,
                           binary_erosion, maximum_filter)
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = sys.argv[1]; sd = scr_dir(t)
with fits.open(cube_path(t, 'blue')) as h:
    f = h[0].data.astype(float); nexp = h['NEXP'].data; wl = h['WAVELENGTH'].data.field(0)
    pix = float(h[0].header['PIXSCALE']); ntot = int(h[0].header['NEXPTOT'])
nz, ny, nx = f.shape; dw = np.median(np.diff(wl))
nmin = 5 if ntot >= 8 else ntot
core = binary_erosion(nexp >= nmin, iterations=3)
# compact continuum sources: peaks in green+red white light S/N, 3" discs
def on_blue_grid(ch):
    # white light of another channel, resampled onto the blue cube's grid by WCS
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates
    img = np.load(f'{sd}/wl_{ch}.npy')
    if img.shape == (ny, nx): return img
    wb = WCS(fits.getheader(cube_path(t, 'blue'))).celestial; wc = WCS(fits.getheader(cube_path(t, ch))).celestial
    yy_, xx_ = np.mgrid[:ny, :nx]; r_, d_ = wb.pixel_to_world_values(xx_, yy_); px_, py_ = wc.world_to_pixel_values(r_, d_)
    return map_coordinates(img, [py_, px_], order=1, cval=np.nan)
wlg, wlr = on_blue_grid('green'), on_blue_grid('red')
def z(im):
    v = im[core & np.isfinite(im)]; m = np.median(v); s = 1.4826 * np.median(np.abs(v - m))
    return np.nan_to_num((im - m) / s)
det = gaussian_filter(z(wlg) + z(wlr), 1)
yy, xx = np.mgrid[:ny, :nx]
pk = (det == maximum_filter(det, 7)) & (det > 12) & (nexp > 0)
srcmask = np.zeros((ny, nx), bool)
for y0, x0 in zip(*np.where(pk)):
    srcmask |= np.hypot(xx - x0, yy - y0) < 3.0 / pix
use = core & ~srcmask
print(f'{t}: searching {use.sum()} spaxels = {use.sum() * pix**2:.0f} arcsec² (NEXP>={nmin}), masked {len(list(zip(*np.where(pk))))} continuum peaks')
np.save(f'{sd}/use_mask.npy', use)
zg = (wl > 3500) & (wl < 4780)
fin = np.isfinite(f)
cont = median_filter(np.nan_to_num(f), size=(101, 1, 1), mode='nearest')
valid = fin.astype(float)
def smooth(x):
    return gaussian_filter(uniform_filter1d(x, 5, axis=0), sigma=(0, 1.0, 1.0))
W = smooth(valid)

def run(extra=None):
    r = np.where(fin, f - cont, 0.0)
    if extra is not None: r = r + extra * valid
    sm = np.where(W > 0.5, smooth(r) / np.maximum(W, 1e-6), 0.0)
    mad = np.array([1.4826 * np.median(np.abs(s[use] - np.median(s[use]))) for s in sm])
    mad[mad == 0] = np.inf
    snr = sm / mad[:, None, None]
    snr[~zg] = 0; snr[:, ~use] = 0
    q = np.where(snr != 0, snr, np.nan)
    sp = 1.4826 * np.nanmedian(np.abs(q - np.nanmedian(q, axis=0)), axis=0)
    sp[~np.isfinite(sp) | (sp == 0)] = 1
    return snr / sp[None], mad

def regions(cube, thr=5, minvox=10):
    lab, n = label(cube > thr); out = []
    for i, sl in enumerate(find_objects(lab), 1):
        m = lab[sl] == i
        if m.sum() < minvox: continue
        pk_ = np.unravel_index(np.argmax(np.where(m, cube[sl], -1e9)), m.shape)
        z0, y0, x0 = [a.start + b for a, b in zip(sl, pk_)]
        out.append(dict(lam=float(wl[z0]), y=int(y0), x=int(x0), snr=float(cube[z0, y0, x0]), nvox=int(m.sum()),
                        area=int(m.any(axis=0).sum()), dz=int(sl[0].stop - sl[0].start)))
    return sorted(out, key=lambda c: -c['snr'])

snr, mad = run()
np.save(f'{sd}/snr_real.npy', snr.astype(np.float32)); np.save(f'{sd}/mad_real.npy', mad)
pos, neg = regions(snr), regions(-snr)
print(f'  S/N std over searched voxels {np.std(snr[zg][:, use]):.2f}; regions >5σ (>=10 vox): positive {len(pos)}, negative {len(neg)}')
for c in pos[:15]:
    print('    + %7.1f Å (Lyα z=%.3f) (x,y)=(%3d,%3d) S/N %5.1f nvox %4d area %3d dλpx %3d' % (c['lam'], c['lam'] / 1215.67 - 1, c['x'], c['y'], c['snr'], c['nvox'], c['area'], c['dz']))
for c in neg[:5]:
    print('    − %7.1f Å (x,y)=(%3d,%3d) S/N %5.1f nvox %4d' % (c['lam'], c['x'], c['y'], c['snr'], c['nvox']))
# injections (cont reused: a 101-px median is insensitive to a ~6-px line)
rng = np.random.default_rng(7)
cand_xy = np.argwhere(use)
fakes = [dict(lam=float(rng.uniform(3550, 4750)), yx=cand_xy[rng.integers(len(cand_xy))].tolist()) for _ in range(30)]
sig_pix = 1.2 / 2.355 / pix; sig_l = 4.7 / 2.355
inj = {}
for F in [2e-17, 4e-17, 8e-17, 1.6e-16]:
    extra = np.zeros_like(f)
    for fk in fakes:
        y0, x0 = fk['yx']
        spat = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig_pix ** 2)); spat /= spat.sum() * pix ** 2
        spec = np.exp(-(wl - fk['lam']) ** 2 / (2 * sig_l ** 2)); spec /= spec.sum() * dw
        extra += F * spec[:, None, None] * spat[None]
    s_inj, _ = run(extra)
    p_inj = regions(s_inj)
    rec = []
    for fk in fakes:
        y0, x0 = fk['yx']
        hit = [c for c in p_inj if abs(c['lam'] - fk['lam']) < 4 and np.hypot(c['x'] - x0, c['y'] - y0) < 2.5]
        rec.append(dict(lam=fk['lam'], hit=bool(hit), snr=hit[0]['snr'] if hit else float(s_inj[np.argmin(abs(wl - fk['lam'])), y0, x0])))
    inj[F] = rec
    print(f'  injected {F:.1e}: recovered {sum(r["hit"] for r in rec)}/{len(rec)}')
# 5σ limit vs λ: calibrate K = S/N * mad(λ) / F from recovered fakes
K = [r['snr'] * mad[np.argmin(abs(wl - r['lam']))] / F for F, rr in inj.items() for r in rr if r['hit']]
Kmed = np.median(K)
lim = 5 * median_filter(np.where(np.isfinite(mad), mad, np.nan), 41, mode='nearest') / Kmed
for l in [3550, 3650, 3800, 4000, 4300, 4600]:
    print('  5σ limit at %d Å (Lyα z=%.2f): %.1e erg/s/cm²' % (l, l / 1215.67 - 1, lim[np.argmin(abs(wl - l))]))
json.dump(dict(pos=pos, neg=neg, inj={str(k): v for k, v in inj.items()}, area=float(use.sum() * pix ** 2),
               lim=[[float(a), float(b)] for a, b in zip(wl[zg], lim[zg])], srcpk=[list(map(int, p)) for p in zip(*np.where(pk))]),
          open(f'{sd}/search.json', 'w'), indent=1)
# figure
fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
mx = np.where(use, snr.max(axis=0), np.nan); mn = np.where(use, -snr.min(axis=0), np.nan)
for a_, im, ttl in [(ax[0, 0], mx, 'max POSITIVE matched-filter S/N per spaxel'), (ax[0, 1], mn, 'max NEGATIVE S/N (false-positive benchmark)')]:
    p_ = a_.imshow(im, origin='lower', vmin=0, vmax=10, cmap='inferno'); plt.colorbar(p_, ax=a_, fraction=0.046)
    a_.contour(srcmask, levels=[0.5], colors='c', linewidths=0.7); a_.set_title(ttl, fontsize=10)
for c in pos[:10]:
    ax[0, 0].plot(c['x'], c['y'], 'o', mfc='none', mec='lime', ms=12)
v = snr[zg][:, use].ravel(); v = v[v != 0]
bins = np.linspace(-10, 10, 101)
ax[1, 0].hist(v, bins=bins, histtype='step', color='k', label='data'); ax[1, 0].hist(-v, bins=bins, histtype='step', color='r', ls='--', label='mirror (negative side)')
g = np.exp(-bins ** 2 / 2) * len(v) * (bins[1] - bins[0]) / np.sqrt(2 * np.pi); ax[1, 0].plot(bins, g, 'b:', label='unit Gaussian')
ax[1, 0].set_yscale('log'); ax[1, 0].set_ylim(0.5, None); ax[1, 0].axvline(5, color='g', lw=0.6); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title(f'S/N distribution ({len(pos)} positive vs {len(neg)} negative regions >5σ)', fontsize=10); ax[1, 0].set_xlabel('S/N')
ax[1, 1].semilogy(wl[zg], lim[zg], 'k')
for F, rr, mk in zip(inj, inj.values(), 'v o s D'.split()):
    for r in rr: ax[1, 1].plot(r['lam'], F, mk, color='g' if r['hit'] else 'r', ms=4, alpha=0.7)
ax[1, 1].plot([], [], 'go', label='injected, recovered'); ax[1, 1].plot([], [], 'ro', label='injected, missed'); ax[1, 1].legend(fontsize=8)
ax[1, 1].axvline(3647, color='b', ls='--', lw=0.6)
ax[1, 1].set_xlabel('observed λ [Å]'); ax[1, 1].set_ylabel('5σ point-source line limit [erg/s/cm²]')
sec = ax[1, 1].secondary_xaxis('top', functions=(lambda l: l / 1215.67 - 1, lambda zz: (zz + 1) * 1215.67)); sec.set_xlabel('Lyα redshift')
ax[1, 1].set_title(f'detection limit (calibrated with {len(K)} recovered fakes)', fontsize=10)
fig.suptitle(f'{t}: blind Lyα search in the blue cube', fontsize=13)
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/06_lya_cube_search.png', dpi=95)
