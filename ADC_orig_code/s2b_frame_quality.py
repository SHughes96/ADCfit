"""Per-exposure quality: aperture flux of the field star (fibres within 1.5" of its true position,
exp 1 taken at its header pointing) and the bench-side median background per channel."""
import sys, json, numpy as np
from astropy.io import fits
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = sys.argv[1]
star = json.load(open(f'{scr_dir(t)}/dithers_exp1_at_header.json'))['res']['red'] if t.endswith('5637') else None
pt = {r['exp']: r for r in json.load(open(f'{scr_dir(t)}/pointing.json'))}
if star:
    s = list(star.values())[0]['combined']; ra0, dec0 = s['ra'], s['dec']
for ch, (lo, hi) in [('green', (5000, 6800)), ('red', (7000, 9500)), ('blue', (3800, 4600))]:
    print(f'== {ch}')
    for p in rss_paths(t, ch):
        e = exp_id(p)
        with fits.open(p) as h:
            ss = h['SKYSUB'].data; w = h['WAVE'].data; fm = h['FIBERMAP'].data
            bs = np.array(fm['BENCHSIDE']).astype(str); ra = np.array(fm['RA'], float); dec = np.array(fm['DEC'], float)
        band = (w > lo) & (w < hi)
        lvl = np.nanmedian(np.where(band, ss, np.nan), axis=1)
        per = {b: np.nanmedian(lvl[bs == b]) for b in sorted(set(bs))}
        line = '  %s  bench-side median SKYSUB: %s' % (e, ' '.join('%s %+6.1f' % (b, v) for b, v in per.items()))
        if star:
            r = pt[e]
            if e == '00-31-08.4':      # undo the bad registration: back to header frame
                ra = ra - (r['fra'] - r['hra']); dec = dec - (r['fdec'] - r['hdec'])
            d = np.hypot((ra - ra0) * np.cos(np.radians(dec0)), dec - dec0) * 3600
            near = d < 1.5; far = (d > 4) & (d < 8)
            flux = np.nansum(lvl[near] - np.nanmedian(lvl[far]))
            line += '  | star aperture (%d fibres) %.0f' % (near.sum(), flux)
        print(line)
