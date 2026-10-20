import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import median_filter
S = sys.argv[1]
with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_blue.fits') as h:
    wl = h['WAVELENGTH'].data.field(0); cube = h[0].data; nexp = h['NEXP'].data
snr2 = np.load(f'{S}/snr2_real.npy'); mad = np.load(f'{S}/mad_real.npy')
src = np.load(f'{S}/srcmask.npy')
used = (snr2 != 0).any(axis=0)

# ---- Figure 2: search summary
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
mx = np.where(used, snr2.max(axis=0), np.nan); mn = np.where(used, -snr2.min(axis=0), np.nan)
for a, im, t in [(ax[0, 0], mx, 'max positive matched-filter S/N per spaxel'),
                 (ax[0, 1], mn, 'max NEGATIVE S/N per spaxel (false-positive benchmark)')]:
    p = a.imshow(im, origin='lower', vmin=0, vmax=10, cmap='inferno'); plt.colorbar(p, ax=a, fraction=0.046)
    a.contour(src, levels=[0.5], colors='c', linewidths=0.8); a.set_title(t, fontsize=10)
v = snr2[:, used].ravel(); v = v[v != 0]
bins = np.linspace(-12, 12, 121)
ax[1, 0].hist(v, bins=bins, histtype='step', color='k', label='data (all voxels searched)')
ax[1, 0].hist(-v, bins=bins, histtype='step', color='r', ls='--', label='mirror of data (negative side)')
g = np.exp(-bins**2 / 2); g *= len(v) * (bins[1] - bins[0]) / np.sqrt(2 * np.pi)
ax[1, 0].plot(bins, g, 'b:', label='unit Gaussian'); ax[1, 0].set_yscale('log'); ax[1, 0].set_ylim(0.5, None)
ax[1, 0].axvline(5, color='g', lw=0.6); ax[1, 0].set_xlabel('S/N'); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title('S/N distribution: the tail is NEGATIVE (defects), not positive', fontsize=10)
# limit curve calibrated from injections
K = []
for F in ['8e-17', '1.6e-16', '3.2e-16']:
    fk = json.load(open(f'{S}/cands_inj{F}.json'))['fakes']
    pos = json.load(open(f'{S}/regions_inj{F}.json'))['pos']
    for f in fk:
        hit = [c for c in pos if abs(c['lam'] - f['lam']) < 4 and np.hypot(c['x'] - f['x'], c['y'] - f['y']) < 2.5]
        if hit:
            k = np.argmin(abs(wl - f['lam']))
            K.append(hit[0]['snr'] * mad[k] / float(F))
Kmed = np.median(K)
madsm = median_filter(np.where(mad > 0, mad, np.nan), 41, mode='nearest')
lim = 5 * madsm / Kmed
ok = (wl > 3300) & (wl < 4780)
ax[1, 1].semilogy(wl[ok], lim[ok], 'k')
ax[1, 1].set_xlabel('observed wavelength [Å]'); ax[1, 1].set_ylabel('5σ line flux limit [erg/s/cm²]')
ax[1, 1].set_title('point-source line detection limit (calibrated with %d injected lines)' % len(K), fontsize=10)
for F, mk in [('4e-17', 'v'), ('8e-17', 'o'), ('1.6e-16', 's'), ('3.2e-16', 'D')]:
    fk = json.load(open(f'{S}/cands_inj{F}.json'))['fakes']
    pos = json.load(open(f'{S}/regions_inj{F}.json'))['pos']
    for f in fk:
        if src[int(round(f['y'])), int(round(f['x']))]: continue
        hit = any(abs(c['lam'] - f['lam']) < 4 and np.hypot(c['x'] - f['x'], c['y'] - f['y']) < 2.5 for c in pos)
        ax[1, 1].plot(f['lam'], float(F), mk, color='g' if hit else 'r', ms=5, alpha=0.7)
ax[1, 1].plot([], [], 'go', label='injected: recovered'); ax[1, 1].plot([], [], 'ro', label='injected: missed')
ax[1, 1].axvline(1215.67 * 3, color='b', lw=0.6, ls='--'); ax[1, 1].text(1215.67 * 3 + 5, 2e-15, 'Lyα z=2', color='b', fontsize=8)
ax[1, 1].legend(fontsize=8)
sec = ax[1, 1].secondary_xaxis('top', functions=(lambda l: l / 1215.67 - 1, lambda z: (z + 1) * 1215.67))
sec.set_xlabel('Lyα redshift')
plt.tight_layout(); plt.savefig(f'{S}/plots/02_search_summary.png', dpi=95)
for l in [3500, 3650, 3800, 4000, 4300, 4600]:
    print('5-sigma limit at %d A (z=%.2f): %.1e' % (l, l / 1215.67 - 1, lim[np.argmin(abs(wl - l))]))

# ---- Figure 3: the negative-hole defect
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
k = np.argmin(abs(wl - 3852.3))
nb = np.nanmean(cube[k - 3:k + 4], axis=0)
p = ax[0].imshow(nb, origin='lower', cmap='RdBu_r', vmin=-np.nanpercentile(abs(nb), 99.5), vmax=np.nanpercentile(abs(nb), 99.5))
ax[0].plot(24, 43, 'ko', mfc='none', ms=18); ax[0].set_title('blue cube, 3848–3857 Å: hole at (24,43)')
plt.colorbar(p, ax=ax[0], fraction=0.046)
with fits.open(f'{S}/data/LLAMAS_2026-10-15_08-57-45.9_RSS_blue.fits') as h:
    w = h['WAVE'].data[1148]; c = h['COUNTS'].data[1148]; e = h['ERROR'].data[1148]; m = h['MASK'].data[1148]
sel = abs(w - 3852) < 40
ax[1].plot(w[sel], c[sel], 'k', lw=0.8, label='COUNTS fibre 1148 (2B)')
ax[1].plot(w[sel], e[sel], 'r', lw=0.8, label='ERROR'); ax[1].plot(w[sel], m[sel] * 100, 'g', lw=0.8, label='MASK×100')
ax[1].set_title('exp 08-57-45.9, fibre 1148: unmasked −3000/−5000 count pixels'); ax[1].legend(fontsize=8)
ax[1].set_xlabel('wavelength [Å]')
# all-exposure census of unmasked |SKYSUB/ERROR| > 50 outliers
import glob
neg, posn = [], []
for f in sorted(glob.glob(f'{S}/data/*RSS_blue.fits')):
    with fits.open(f) as h:
        z = h['SKYSUB'].data / h['ERROR'].data; mk = h['MASK'].data != 0
    z = np.where(mk, np.nan, z)
    neg.append(np.sum(z < -50)); posn.append(np.sum(z > 50))
x = np.arange(3); lab = ['08-33-54.7', '08-57-45.9', '09-21-37.8']
ax[2].bar(x - 0.2, neg, 0.4, color='r', label='SKYSUB/ERROR < −50'); ax[2].bar(x + 0.2, posn, 0.4, color='b', label='> +50')
ax[2].set_xticks(x); ax[2].set_xticklabels(lab); ax[2].legend(); ax[2].set_title('unmasked extreme pixels per blue RSS')
plt.tight_layout(); plt.savefig(f'{S}/plots/03_negative_defects.png', dpi=95)
print('extreme negative/positive per exposure', neg, posn, 'Kmed', Kmed, 'nK', len(K))
print('ERROR at the defect pixels:', e[np.argsort(c)[:2]], 'COUNTS', np.sort(c)[:2])
