import sys, glob, numpy as np
from astropy.io import fits
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.cube import combine_cube
S = sys.argv[1]
paths = sorted(glob.glob(f'{S}/data/*RSS_blue.fits'))
ref = None
print('Dither offsets of the stored fibre positions (same fibre IDs), relative to exposure 1:')
for p in paths:
    with fits.open(p) as h:
        fm = h['FIBERMAP'].data; hd = h[0].header
        ra, dec = np.array(fm['RA'], float), np.array(fm['DEC'], float)
        wh = h['FIBERWCS'].header
    ok = np.isfinite(ra) & (ra > 0)
    if ref is None:
        ref = (ra, dec, ok, hd['RA'], hd['DEC'])
    m = ok & ref[2]
    d_ra = (ra[m] - ref[0][m]) * np.cos(np.radians(dec[m])) * 3600
    d_de = (dec[m] - ref[1][m]) * 3600
    h_ra = (hd['RA'] - ref[3]) * np.cos(np.radians(hd['DEC'])) * 3600
    h_de = (hd['DEC'] - ref[4]) * 3600
    # rotation between solutions: fit fibre offset field as translation + rotation
    x0 = (ref[0][m] - ref[0][m].mean()) * np.cos(np.radians(dec[m])) * 3600
    y0 = (ref[1][m] - ref[1][m].mean()) * 3600
    A = np.c_[np.ones_like(x0), -y0]; th = np.linalg.lstsq(A, d_ra - d_ra.mean(), rcond=None)[0][1]
    print('  %s  fibre-WCS shift dRA %+.2f" dDec %+.2f" (spread %.3f")  rel. rotation %.3f deg | header-commanded dRA %+.2f" dDec %+.2f"' % (
        p.split('/')[-1][18:28], np.median(d_ra), np.median(d_de), np.std(d_ra - np.median(d_ra)),
        np.degrees(th), h_ra, h_de))
# independent rebuild of the blue cube with the default settings recorded in the Box cube header
with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_blue.fits') as h:
    box = h[0].data.astype(float); bh = h[0].header; bwl = h['WAVELENGTH'].data.field(0)
sr = build_super_rss(paths, channels=['blue'])
cube = combine_cube(sr, 'blue', pixscale=bh['PIXSCALE'], kernel_fwhm=bh['KERNFWHM'], weighting=bh['WGHT'].strip(),
                    units=bh['UNITS'].strip())
cube.write(f'{S}/rebuilt_cube_blue.fits')
print('rebuilt shape', cube.data.shape, 'box shape', box.shape, 'wl match', np.allclose(cube.wave, bwl))
if cube.data.shape == box.shape:
    d = cube.data - box
    fin = np.isfinite(d)
    print('rebuilt - box: finite voxels %d, max |diff| %.3e, median |box| %.3e, NaN pattern identical: %s' % (
        fin.sum(), np.nanmax(abs(d)), np.nanmedian(abs(box)), np.array_equal(np.isnan(cube.data), np.isnan(box))))
