"""After the user's 'Auto-register block' + re-combine: cube provenance, NEXP histogram and the
FIBERWCS method now stored in each J0111 RSS (all rough-header)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from llamas_pyjamas.Utils.wcsLlamas import pointing_from_header
t = 'SPT-CLJ0111-5637'
h = fits.getheader(cube_path(t, 'blue'))
print('cube', h['NAXIS1'], h['NAXIS2'], 'NRSS', h.get('NRSSFILE'), {k: h[k] for k in ['KERNFWHM', 'PIXSCALE', 'WGHT', 'UNITS', 'NEXPTOT'] if k in h})
print('NEXP histogram', np.unique(fits.getdata(cube_path(t, 'blue'), 'NEXP'), return_counts=True))
for p in rss_paths(t, 'green'):
    with fits.open(p) as hd:
        names = [x.name for x in hd]; fm = hd['FIBERMAP'].data; hh = hd[0].header
        wh = dict(hd['FIBERWCS'].header) if 'FIBERWCS' in names else {}
    ra, dec, pa = pointing_from_header(hh); ok = np.isfinite(fm['RA']) & (fm['RA'] > 0); c = np.cos(np.radians(dec))
    print(exp_id(p), 'fibre-map vs TEL pointing %+6.1f %+6.1f"' % ((np.median(fm['RA'][ok]) - ra) * c * 3600, (np.median(fm['DEC'][ok]) - dec) * 3600),
          {k: wh.get(k) for k in ['WCSMETH', 'WCSTIER', 'WCSNSTAR', 'WCSRMS']})
