"""J0111 first attempt: is the field star present in exposure 1 at its STORED (registered) position?
Run against the first-attempt fibre maps (now overwritten by the failed auto-register, see LOG)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
t = 'SPT-CLJ0111-5637'
s = list(json.load(open(scr_dir(t) + '/dithers.json'))['res']['red'].values())[0]['combined']; ra0, dec0 = s['ra'], s['dec']
for ch, (lo, hi) in [('green', (5000, 6800)), ('red', (7000, 9500))]:
    for p in rss_paths(t, ch)[:3]:
        with fits.open(p) as h:
            ss = h['SKYSUB'].data; w = h['WAVE'].data; fm = h['FIBERMAP'].data
        lvl = np.nanmedian(np.where((w > lo) & (w < hi), ss, np.nan), axis=1)
        ra = np.array(fm['RA'], float); dec = np.array(fm['DEC'], float)
        d = np.hypot((ra - ra0) * np.cos(np.radians(dec0)), dec - dec0) * 3600
        near = d < 1.5; far = (d > 4) & (d < 8); top = np.argsort(np.nan_to_num(lvl))[::-1][:5]
        print(ch, exp_id(p), 'star at STORED position: %d fibres, flux %.0f' % (near.sum(), np.nansum(lvl[near] - np.nanmedian(lvl[far]))),
              '| brightest fibre at %.1f" from star, fibre %d' % (d[top[0]], top[0]))
