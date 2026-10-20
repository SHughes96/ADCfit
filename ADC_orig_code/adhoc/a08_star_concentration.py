"""Encircled-flux concentration (r<1"/r<3") of the J0111 field star: header combine vs star-aligned, per channel."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from llamas_pyjamas.Combine.spectrum import fit_gaussian_image
from astropy.wcs import WCS
t = 'SPT-CLJ0111-5637'
ra0, dec0 = json.load(open(os.path.join(SCR, t, 'staralign_shifts.json')))['anchor']
for ch, (lo, hi) in [('red', (7000, 9500)), ('green', (5000, 6800)), ('blue', (3800, 4600))]:
    for lab, p in [('star-aligned', os.path.join(OUT, t, 'star_aligned', f'{t}_staraligned_cube_{ch}.fits')), ('header combine', os.path.join(ROOT, 'combined', f'{t}_cube_{ch}.fits'))]:
        with fits.open(p) as h:
            wl = h['WAVELENGTH'].data.field(0); f = h[0].data; w = WCS(h[0].header).celestial
        img = np.nanmedian(f[(wl > lo) & (wl < hi)], axis=0)
        x0, y0 = [float(q) for q in w.world_to_pixel_values(ra0, dec0)]
        g = fit_gaussian_image(np.nan_to_num(img), x0, y0, 8)
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]; r = np.hypot(xx - g.x_mean.value, yy - g.y_mean.value) * 0.5
        bg = np.nanmedian(img[(r > 4) & (r < 7)])
        print('  %-5s %-15s conc r<1/r<3 = %.3f' % (ch, lab, np.nansum(img[r <= 1] - bg) / np.nansum(img[r <= 3] - bg)))
