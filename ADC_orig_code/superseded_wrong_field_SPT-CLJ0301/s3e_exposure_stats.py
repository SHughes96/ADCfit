import sys, glob, json, numpy as np
from astropy.io import fits
S = sys.argv[1]
out = {}
for f in sorted(glob.glob(f'{S}/data/*RSS_blue.fits')):
    eid = f.split('/')[-1][7:28]
    with fits.open(f) as h:
        p = h[0].header
        ss = h['SKYSUB'].data.astype(float); er = h['ERROR'].data.astype(float)
        sky = h['SKY'].data.astype(float); mk = h['MASK'].data != 0; wave = h['WAVE'].data
        bs = np.array(h['FIBERMAP'].data['BENCHSIDE']).astype(str)
        live = np.nanmedian(np.abs(h['COUNTS'].data), axis=1) > 0
    band = (wave > 3600) & (wave < 4500)
    z = np.where(band & ~mk & (er > 0), ss / er, np.nan)
    print(f'{eid}  UT start {p.get("UTC-OBS", p.get("UT", "?"))}  airmass {p.get("AIRMASS")}  '
          f'moon? {p.get("MOONSEP", p.get("MOON ALT", "-"))}')
    print('   median SKY counts (3600-4500): %.1f   median ERROR %.1f' %
          (np.nanmedian(np.where(band, sky, np.nan)), np.nanmedian(np.where(band, er, np.nan))))
    print('   unmasked pixels with SKYSUB/ERROR < -10: %d   > +10: %d   (< -50: %d, > +50: %d)' %
          ((z < -10).sum(), (z > 10).sum(), (z < -50).sum(), (z > 50).sum()))
    per_bs = {}
    for b in sorted(set(bs)):
        sel = (bs == b) & live
        if sel.sum() == 0: continue
        per_bs[b] = float(np.nanmedian(np.where(band[sel], sky[sel], np.nan)))
    print('   median SKY by bench-side:', {k: round(v, 1) for k, v in per_bs.items()})
    out[eid] = per_bs
json.dump(out, open(f'{S}/sky_by_bs.json', 'w'))
