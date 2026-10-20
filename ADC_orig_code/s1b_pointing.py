import sys, json, numpy as np
from astropy.io import fits
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = sys.argv[1]
rows = []
for p in rss_paths(t, 'blue'):
    with fits.open(p) as h:
        hd = h[0].header; fm = h['FIBERMAP'].data
        names = [x.name for x in h]
        wh = dict(h['FIBERWCS'].header) if 'FIBERWCS' in names else {}
    ra, dec = np.array(fm['RA'], float), np.array(fm['DEC'], float)
    ok = np.isfinite(ra) & (ra > 0)
    rows.append(dict(exp=exp_id(p), fra=float(np.median(ra[ok])), fdec=float(np.median(dec[ok])),
                     hra=float(hd['RA']), hdec=float(hd['DEC']), rot=hd.get('TEL ROT'),
                     meth=wh.get('WCSMETH'), tier=wh.get('WCSTIER'), nstar=wh.get('WCSNSTAR'),
                     rms=wh.get('WCSRMS'), extra={k: wh[k] for k in wh if k.startswith(('WCS', 'PA', 'ANCH', 'REL'))}))
r0 = rows[0]; c = np.cos(np.radians(r0['fdec']))
print(f'{t}: offsets relative to exposure 1 (arcsec)')
for r in rows:
    print('  %s  fibre-centre dRA %+7.2f dDec %+7.2f | header dRA %+7.2f dDec %+7.2f | fibre-header diff %+6.2f %+6.2f | rot %s | %s/%s n=%s' % (
        r['exp'], (r['fra'] - r0['fra']) * c * 3600, (r['fdec'] - r0['fdec']) * 3600,
        (r['hra'] - r0['hra']) * c * 3600, (r['hdec'] - r0['hdec']) * 3600,
        ((r['fra'] - r['hra']) * c * 3600), (r['fdec'] - r['hdec']) * 3600, r['rot'], r['meth'], r['tier'], r['nstar']))
print('  FIBERWCS keys (first frame):', rows[0]['extra'])
json.dump(rows, open(f'{scr_dir(t)}/pointing.json', 'w'), indent=1)
