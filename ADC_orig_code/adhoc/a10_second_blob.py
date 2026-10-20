"""Is the bright blob near (x≈17, y≈60) in the 4187–4201 Å narrowband of SPT-CLJ0158-5500 real line emission?
S/N in the same r=1.5" aperture method as s14, in the line window and in neighbouring line-free windows."""
import sys, os, json, numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
t = 'SPT-CLJ0158-5500'
with fits.open(cube_path(t, 'blue')) as h:
    f = h[0].data.astype(float); wl = h['WAVELENGTH'].data.field(0); nexp = h['NEXP'].data
ny, nx = nexp.shape; yy, xx = np.mgrid[:ny, :nx]
use = np.load(os.path.join(SCR, t, 'use_mask.npy'))
def mean_band(lo, hi): return np.nanmean(f[(wl >= lo) & (wl <= hi)], axis=0)
def contsub(lo, hi):
    side = ((wl > lo - 60) & (wl < lo - 15)) | ((wl > hi + 15) & (wl < hi + 60))
    return mean_band(lo, hi) - np.nanmedian(f[side], axis=0)
def snr(img, x0, y0):
    a = np.hypot(xx - x0, yy - y0) <= 3; v = np.nanmean(img[a])
    rng = np.random.default_rng(1); pts = np.argwhere(use & (np.hypot(xx - x0, yy - y0) > 8))
    r = np.array([np.nanmean(img[np.hypot(xx - px, yy - py) <= 3]) for py, px in pts[rng.integers(0, len(pts), 200)]])
    return (v - np.median(r)) / (1.4826 * np.median(np.abs(r - np.median(r))))
img = contsub(4187, 4201)
sub = np.nan_to_num(img)[50:70, 10:25]; j = np.unravel_index(np.argmax(sub), sub.shape); bx, by = 10 + j[1], 50 + j[0]
print('blob peak at pixel (0-based) x=%d y=%d; NEXP there = %d of 8; inside searched area: %s' % (bx, by, nexp[by, bx], bool(use[by, bx])))
for lo, hi in [(4130, 4144), (4160, 4174), (4187, 4201), (4215, 4229), (4240, 4254)]:
    print('   window %d-%d Å, continuum-subtracted: blob S/N %+.1f   | C2 S/N %+.1f' % (lo, hi, snr(contsub(lo, hi), bx, by), snr(contsub(lo, hi), 48.92, 45.0)))
s = json.load(open(os.path.join(SCR, t, 'search.json')))
print('   blind-search regions within 5 px of the blob:', [(round(c['lam'], 1), round(c['snr'], 1)) for c in s['pos'] + s['neg'] if abs(c['x'] - bx) < 5 and abs(c['y'] - by) < 5])
