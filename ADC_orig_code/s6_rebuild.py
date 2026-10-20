import sys, numpy as np
from astropy.io import fits
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.cube import combine_cube
t = sys.argv[1]
with fits.open(cube_path(t, 'blue')) as h:
    box = h[0].data.astype(float); bh = h[0].header; bwl = h['WAVELENGTH'].data.field(0)
sr = build_super_rss(rss_paths(t, 'blue'), channels=['blue'])
cube = combine_cube(sr, 'blue', pixscale=bh['PIXSCALE'], kernel_fwhm=bh['KERNFWHM'], weighting=bh['WGHT'].strip(), units=bh['UNITS'].strip())
print(t, 'rebuilt', cube.data.shape, 'delivered', box.shape, 'λ match', cube.data.shape[0] == box.shape[0] and np.allclose(cube.wave, bwl))
if cube.data.shape == box.shape:
    d = cube.data - box
    print('  max |diff| %.2e vs median |cube| %.2e ; identical NaN pattern: %s' % (np.nanmax(abs(d)), np.nanmedian(abs(box)), np.array_equal(np.isnan(cube.data), np.isnan(box))))
    rel = np.abs(d) / np.sqrt(np.abs(fits.getdata(cube_path(t, 'blue'), 'VAR')) + 1e-60)
    print('  |diff|/sigma percentiles 50/99/99.9/max: %.3f %.3f %.3f %.1f' % tuple(np.nanpercentile(rel, [50, 99, 99.9, 100])))
    big = rel > 1
    print('  voxels with |diff| > 1 sigma: %d of %d (%.4f%%)' % (big.sum(), np.isfinite(rel).sum(), 100 * big.sum() / np.isfinite(rel).sum()))
    zz, yy_, xx_ = np.where(big)
    if big.any():
        print('  where (λ range, x range, y range):', (round(float(bwl[zz.min()])), round(float(bwl[zz.max()]))), (xx_.min(), xx_.max()), (yy_.min(), yy_.max()))
        u, cnt = np.unique(zz, return_counts=True); print('  most-affected slices λ:', [round(float(bwl[k])) for k in u[np.argsort(cnt)[::-1][:8]]])
    ratio = np.nanmedian(box[np.abs(box) > 5e-18] / cube.data[np.abs(box) > 5e-18])
    print('  median delivered/rebuilt ratio on bright voxels: %.4f' % ratio)
    print('  delivered header keys:', {k: bh[k] for k in bh if k not in ('COMMENT', 'HISTORY') and not k.startswith(('NAXIS', 'C', 'RSSFIL', 'WCS', 'LON', 'LAT', 'MJD', 'RADE', 'SIMPLE', 'BITPIX', 'EXTEND'))})
