import sys, json, numpy as np
from astropy.io import fits
from astropy.wcs import WCS
S = sys.argv[1]
multi = json.load(open(f'{S}/rss_multi.json'))
hits = json.load(open(f'{S}/rss_hits.json'))
with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_blue.fits') as h:
    w = WCS(h[0].header).celestial
src = np.load(f'{S}/srcmask.npy')
lam_all = np.array([h['lam'] for h in hits])
hist, edges = np.histogram(lam_all, bins=np.arange(3300, 4790, 10))
print('Busiest 10-A bins of ALL single-exposure hits (field-wide systematic wavelengths):')
for i in np.argsort(hist)[::-1][:8]:
    print('   %d-%d A: %d hits' % (edges[i], edges[i + 1], hist[i]))
sys_bands = [(3325, 3365), (4640, 4740)]
keep = []
for g in multi:
    lam = np.mean([h['lam'] for h in g]); ra = g[0]['ra']; dec = g[0]['dec']
    x, y = w.world_to_pixel_values(ra, dec)
    xi, yi = int(round(float(x))), int(round(float(y)))
    on_src = 0 <= yi < src.shape[0] and 0 <= xi < src.shape[1] and src[yi, xi]
    in_sys = any(a <= lam <= b for a, b in sys_bands)
    if on_src or in_sys:
        continue
    keep.append(dict(lam=lam, ra=ra, dec=dec, x=xi, y=yi, nexp=len({h['exp'] for h in g}),
                     exps=sorted({h['exp'] for h in g}), bs=sorted({h['bs'] for h in g}),
                     snr=[round(h['z'], 1) for h in g]))
print('\nAfter removing continuum-source positions and the two systematic bands: %d groups' % len(keep))
for k in keep:
    print('  lam %7.1f z_Lya %.3f cube(x,y)=(%2d,%2d)  nexp %d %s  bench %s  S/N %s' % (
        k['lam'], k['lam'] / 1215.67 - 1, k['x'], k['y'], k['nexp'], k['exps'], k['bs'], k['snr']))
x, y = w.world_to_pixel_values(45.4513, -51.7182)
print('\nlocalised many-line group at cube pixel (%.1f, %.1f); srcmask there: %s' % (x, y, src[int(round(float(y))), int(round(float(x)))]))
json.dump(keep, open(f'{S}/rss_keep.json', 'w'), indent=1)
