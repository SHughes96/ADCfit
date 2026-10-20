import sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
S = sys.argv[1]
fig, ax = plt.subplots(2, 3, figsize=(14, 9))
for i, c in enumerate(['blue', 'green', 'red']):
    with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_{c}.fits') as h:
        f = h[0].data; nexp = h['NEXP'].data
    wl_img = np.nanmedian(f, axis=0)
    wl_img[nexp == 0] = np.nan
    lo, hi = ZScaleInterval().get_limits(wl_img[np.isfinite(wl_img)])
    im = ax[0, i].imshow(wl_img, origin='lower', vmin=lo, vmax=hi, cmap='gray')
    ax[0, i].set_title(f'{c} cube: median over wavelength')
    plt.colorbar(im, ax=ax[0, i], fraction=0.046)
    np.save(f'{S}/wl_{c}.npy', wl_img)
im = ax[1, 0].imshow(nexp, origin='lower', cmap='viridis'); ax[1, 0].set_title('NEXP (exposures per spaxel)')
plt.colorbar(im, ax=ax[1, 0], fraction=0.046)
# Sky + residual spectra from one blue RSS
with fits.open(f'{S}/data/LLAMAS_2026-10-15_08-33-54.7_RSS_blue.fits') as h:
    wave = h['WAVE'].data; sky = h['SKY'].data; res = h['SKYRESID'].data; ss = h['SKYSUB'].data
w0 = np.nanmedian(wave, axis=0)
ax[1, 1].plot(w0, np.nanmedian(sky, axis=0), lw=0.6, label='SKY (median fibre)')
ax[1, 1].plot(w0, np.nanmedian(ss, axis=0), lw=0.6, label='SKYSUB (median fibre)')
ax[1, 1].set_xlabel('wavelength [Å]'); ax[1, 1].set_ylabel('counts'); ax[1, 1].legend(fontsize=8)
ax[1, 1].set_title('blue RSS 08-33-54.7: sky vs sky-subtracted')
ax[1, 2].plot(w0, np.nanstd(ss, axis=0), lw=0.6, label='std over fibres SKYSUB')
ax[1, 2].plot(w0, np.nanmedian(res, axis=0), lw=0.6, label='SKYRESID median')
ax[1, 2].legend(fontsize=8); ax[1, 2].set_xlabel('wavelength [Å]'); ax[1, 2].set_title('blue RSS scatter')
plt.tight_layout(); plt.savefig(f'{S}/plots/01_overview.png', dpi=110)
