"""Where does a bright source land on the fibre bundle in each exposure? Positions in the
INSTRUMENT frame = fibre-map offsets relative to that frame's own fibre-map centre. If the telescope
truly dithered by the header offsets, the source moves across the bundle by (minus) those offsets."""
import sys, json, numpy as np
from astropy.io import fits
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Utils.centroid import fibre_centroid
t = sys.argv[1]
pt = {r['exp']: r for r in json.load(open(f'{scr_dir(t)}/pointing.json'))}
rows = []
for p in rss_paths(t, 'red'):
    e = exp_id(p)
    with fits.open(p) as h:
        ss = h['SKYSUB'].data; w = h['WAVE'].data; fm = h['FIBERMAP'].data
    lv = np.nanmedian(np.where((w > 7000) & (w < 9500), ss, np.nan), axis=1)
    ra, dec = np.array(fm['RA'], float), np.array(fm['DEC'], float)
    c = np.cos(np.radians(np.median(dec)))
    x = (ra - np.median(ra)) * c * 3600; y = (dec - np.median(dec)) * 3600   # instrument frame
    lv[135] = np.nan                                                        # hot fibre
    good = np.isfinite(lv)
    # brightest compact peak: fibre with highest 3-fibre neighbourhood sum
    nb = np.array([np.nansum(lv[(np.hypot(x - x[i], y - y[i]) < 1.0)]) if good[i] else -1e9 for i in range(len(lv))])
    k = int(np.argmax(nb))
    sel = np.hypot(x - x[k], y - y[k]) < 2.5
    bg = np.nanmedian(lv[(np.hypot(x - x[k], y - y[k]) > 4) & (np.hypot(x - x[k], y - y[k]) < 7)])
    wgt = np.clip(lv[sel] - bg, 0, None)
    cx, cy = np.sum(wgt * x[sel]) / wgt.sum(), np.sum(wgt * y[sel]) / wgt.sum()
    hdx = (pt[e]['hra'] - pt[exp_id(rss_paths(t)[0])]['hra']) * c * 3600
    hdy = (pt[e]['hdec'] - pt[exp_id(rss_paths(t)[0])]['hdec']) * 3600
    fdx = (pt[e]['fra'] - pt[exp_id(rss_paths(t)[0])]['fra']) * c * 3600
    fdy = (pt[e]['fdec'] - pt[exp_id(rss_paths(t)[0])]['fdec']) * 3600
    rows.append(dict(exp=e, ix=cx, iy=cy, hdx=hdx, hdy=hdy, fdx=fdx, fdy=fdy, peak=float(nb[k])))
r0 = rows[0]
print(f'{t}: brightest compact source, position on the bundle (instrument frame) relative to exposure 1')
print('   exp          moved on bundle      expected from header (=-offset)   expected from fibre maps (=-offset)')
for r in rows:
    print('   %s   %+6.2f %+6.2f        %+6.2f %+6.2f                    %+6.2f %+6.2f      (peak %.0f)' % (
        r['exp'], r['ix'] - r0['ix'], r['iy'] - r0['iy'], -r['hdx'], -r['hdy'], -(r['fdx']), -(r['fdy']), r['peak']))
json.dump(rows, open(f'{scr_dir(t)}/instrument_frame.json', 'w'), indent=1)
