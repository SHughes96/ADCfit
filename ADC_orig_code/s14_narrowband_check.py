"""Mark C2 on the user's CubeViewer narrowband (mean 4150-4200 Å, no continuum subtraction) and compare
with better-targeted narrowbands. S/N of C2 in each = r=1.5" aperture vs 200 random apertures.
Also writes a continuum-subtracted narrowband FITS and a DS9 region file for C2."""
import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
t = 'SPT-CLJ0158-5500'
user = fits.getdata(os.path.join(ROOT, 'narrowband.fits')).astype(float)
with fits.open(cube_path(t, 'blue')) as h:
    f = h[0].data.astype(float); wl = h['WAVELENGTH'].data.field(0); w = WCS(h[0].header).celestial; nexp = h['NEXP'].data
c2 = SkyCoord('01h58m56.12s', '-55d00m28.26s')
x0, y0 = [float(q) for q in w.world_to_pixel_values(c2.ra.deg, c2.dec.deg)]
ny, nx = user.shape; yy, xx = np.mgrid[:ny, :nx]
use = np.load(os.path.join(SCR, t, 'use_mask.npy'))
def mean_band(lo, hi): return np.nanmean(f[(wl >= lo) & (wl <= hi)], axis=0)
def contsub(lo, hi):                       # line band minus side bands (15-60 Å either side)
    side = ((wl > lo - 60) & (wl < lo - 15)) | ((wl > hi + 15) & (wl < hi + 60))
    return mean_band(lo, hi) - np.nanmedian(f[side], axis=0)
def snr(img):
    a = np.hypot(xx - x0, yy - y0) <= 3
    v = np.nanmean(img[a])
    rng = np.random.default_rng(1); pts = np.argwhere(use & (np.hypot(xx - x0, yy - y0) > 8))
    r = np.array([np.nanmean(img[np.hypot(xx - px, yy - py) <= 3]) for py, px in pts[rng.integers(0, len(pts), 200)]])
    return (v - np.median(r)) / (1.4826 * np.median(np.abs(r - np.median(r))))
panels = [('Your image: mean 4150–4200 Å\n(no continuum subtraction)', user),
          ('Same window, continuum-subtracted', contsub(4150, 4200)),
          ('Line window 4187–4201 Å\n(no continuum subtraction)', mean_band(4187, 4201)),
          ('Line window 4187–4201 Å,\ncontinuum-subtracted (recommended)', contsub(4187, 4201))]
fig, ax = plt.subplots(1, 4, figsize=(19, 5.6))
for a_, (title, img) in zip(ax, panels):
    s = snr(img)
    disp = gaussian_filter(np.nan_to_num(img), 0.8); disp[nexp == 0] = np.nan
    v = disp[use]; lo, hi = np.nanpercentile(v, [2, 99.7])
    a_.imshow(disp, origin='lower', cmap='viridis', vmin=lo, vmax=hi)
    a_.add_patch(plt.Circle((x0, y0), 6, fill=False, color='r', lw=1.8))
    a_.annotate('C2 (Lyα z=2.45)', (x0 + 6, y0 + 6), color='r', fontsize=9, weight='bold')
    a_.set_title(f'{title}\nC2 S/N in r=1.5" aperture: {s:.1f}', fontsize=10)
    a_.set_xlabel('x pixel (0-based)'); a_.set_ylabel('y pixel')
fig.suptitle(f'SPT-CLJ0158-5500 blue cube: where C2 is (pixel x={x0 + 1:.1f}, y={y0 + 1:.1f} in DS9 1-based coordinates; '
             'RA 01:58:56.12 Dec −55:00:28.3)', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(out_dir(t), '10_narrowband_where_is_C2.png'), dpi=110)
for (title, img) in panels: print('%-62s C2 S/N %.1f' % (title.replace('\n', ' '), snr(img)))
# products for CubeViewer / DS9
hdr = w.to_header(); hdr['WAVEMIN'] = (4187.0, 'A'); hdr['WAVEMAX'] = (4201.0, 'A')
hdr['CONTSUB'] = ('4127-4172,4216-4261 A median', 'continuum side bands subtracted'); hdr['OBJECT'] = t
hdr['BUNIT'] = 'erg/s/cm2/Angstrom/arcsec2'; hdr['COMMENT'] = 'Lya narrowband for C2 (z=2.450); mean over line window minus side bands'
fits.PrimaryHDU(contsub(4187, 4201).astype(np.float32), header=hdr).writeto(os.path.join(out_dir(t), 'narrowband_C2_Lya_4187-4201_contsub.fits'), overwrite=True)
with open(os.path.join(out_dir(t), 'C2.reg'), 'w') as fh:
    fh.write('# Region file format: DS9\nfk5\ncircle(01:58:56.12,-55:00:28.26,3") # color=red width=2 text={C2 Lya z=2.450}\n')
print('C2 pixel 0-based (%.2f, %.2f); DS9 1-based (%.2f, %.2f)' % (x0, y0, x0 + 1, y0 + 1))
