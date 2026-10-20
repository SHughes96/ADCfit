"""For a cube voxel (x, y, lam): find the RSS fibres within r arcsec in each exposure and print
their SKYSUB / SKY / COUNTS / MASK around lam."""
import sys, glob, numpy as np
from astropy.io import fits
from astropy.wcs import WCS
S = sys.argv[1]; x, y, lam = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
r = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_blue.fits') as h:
    w = WCS(h[0].header).celestial
ra0, dec0 = w.pixel_to_world_values(x, y)
print('voxel -> RA %.6f Dec %.6f' % (ra0, dec0))
for f in sorted(glob.glob(f'{S}/data/*RSS_blue.fits')):
    with fits.open(f) as h:
        fm = h['FIBERMAP'].data; wave = h['WAVE'].data
        d = np.hypot((fm['RA'] - ra0) * np.cos(np.radians(dec0)), fm['DEC'] - dec0) * 3600
        near = np.where(d < r)[0]
        print(f.split('/')[-1][:30], 'fibres within %.1f":' % r, len(near))
        for i in near:
            k = np.argmin(abs(wave[i] - lam)); sl = slice(max(k - 3, 0), k + 4)
            print('   fib %4d %s d=%.2f"  SKYSUB %s' % (i, fm['BENCHSIDE'][i], d[i], np.round(h['SKYSUB'].data[i, sl], 0)))
            print('              SKY    %s' % np.round(h['SKY'].data[i, sl], 0))
            print('              COUNTS %s' % np.round(h['COUNTS'].data[i, sl], 0))
            print('              MASK   %s' % h['MASK'].data[i, sl])
