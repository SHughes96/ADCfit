"""Search the top band that the continuum mask removed, masking only a 3" disc on the star; and test
the diffuse blue blob at (32,72) for line vs continuum."""
import sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import label, find_objects, binary_erosion
S = sys.argv[1]
snr = np.load(f'{S}/snr_real.npy').astype(float); snr[~np.isfinite(snr)] = 0
with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_blue.fits') as h:
    wl = h['WAVELENGTH'].data.field(0); nexp = h['NEXP'].data; cube = h[0].data; var = h['VAR'].data
ny, nx = nexp.shape
yy, xx = np.mgrid[:ny, :nx]
core = binary_erosion(nexp == 3, iterations=3)
top = core & (yy >= 58) & (np.hypot(xx - 60, yy - 67) > 6) & (np.hypot(xx - 68, yy - 52) > 6)
zgood = (wl > 3300) & (wl < 4780)
q = np.where(snr != 0, snr, np.nan)
sp = 1.4826 * np.nanmedian(np.abs(q - np.nanmedian(q, axis=0)), axis=0); sp[~np.isfinite(sp) | (sp == 0)] = 1
s2 = snr / sp[None]; s2[~zgood] = 0; s2[:, ~top] = 0
print('top-band spaxels searched:', int(top.sum()), ' (%.0f arcsec2)' % (top.sum() * 0.25))
for sign, nm in [(1, 'positive'), (-1, 'negative')]:
    lab, n = label(sign * s2 > 5)
    big = [(i, (lab[sl] == i).sum(), sl) for i, sl in enumerate(find_objects(lab), 1) if (lab[sl] == i).sum() >= 10]
    print(f'  {nm} regions >5 sigma, >=10 voxels: {len(big)}')
    for i, nv, sl in sorted(big, key=lambda t: -t[1])[:8]:
        m = lab[sl] == i
        pk = np.unravel_index(np.argmax(np.where(m, sign * s2[sl], -1)), m.shape)
        z0, y0, x0 = [a.start + b for a, b in zip(sl, pk)]
        print('     lam %.1f (x,y)=(%d,%d) S/N %.1f nvox %d' % (wl[z0], x0, y0, sign * s2[z0, y0, x0], nv))
# blob: aperture spectrum r=3 px at (32,72) vs a same-row control at (45,72)
fig, ax = plt.subplots(2, 1, figsize=(11, 7))
for (x0, y0), c, lab_ in [((32, 72), 'C0', 'blob (32,72)'), ((45, 72), 'C1', 'control (45,72), same rows'), ((40, 30), 'C2', 'control (40,30), lower field')]:
    ap = np.hypot(xx - x0, yy - y0) <= 3
    sp_ = np.nanmean(cube[:, ap], axis=1); er = np.sqrt(np.nansum(var[:, ap], axis=1)) / ap.sum()
    k = np.ones(15) / 15
    ax[0].plot(wl, np.convolve(sp_, k, 'same'), color=c, lw=0.8, label=lab_)
    r = sp_ - np.convolve(sp_, np.ones(101) / 101, 'same')
    ax[1].plot(wl, r / er, color=c, lw=0.6, label=lab_)
ax[0].set_ylabel('SB (15-px boxcar)'); ax[0].legend(fontsize=8); ax[0].set_xlim(3300, 4780)
ax[0].set_title('diffuse blue blob: continuum, not a line? (aperture r=1.5")')
ax[1].set_ylabel('(SB − running mean)/σ_VAR'); ax[1].set_xlim(3300, 4780); ax[1].set_xlabel('wavelength [Å]')
ax[1].set_ylim(-15, 15)
plt.tight_layout(); plt.savefig(f'{S}/plots/06_blue_blob.png', dpi=95)
for x0, y0 in [(32, 72), (45, 72), (40, 30)]:
    ap = np.hypot(xx - x0, yy - y0) <= 3
    sp_ = np.nanmean(cube[:, ap], axis=1)
    print('(%d,%d) mean SB 3600-4000: %.2e  4000-4500: %.2e' % (x0, y0, np.nanmean(sp_[(wl > 3600) & (wl < 4000)]), np.nanmean(sp_[(wl > 4000) & (wl < 4500)])))
for c in ['green', 'red']:
    wli = np.load(f'{S}/wl_{c}.npy')
    print(c, 'white light at blob %.2e, at control (45,72) %.2e, lower field (40,30) %.2e' % (np.nanmean(wli[69:76, 29:36]), np.nanmean(wli[69:76, 42:49]), np.nanmean(wli[27:34, 37:44])))
wli = np.load(f'{S}/wl_blue.npy')
print('blue white light at blob %.2e, at control %.2e, lower %.2e' % (np.nanmean(wli[69:76, 29:36]), np.nanmean(wli[69:76, 42:49]), np.nanmean(wli[27:34, 37:44])))
