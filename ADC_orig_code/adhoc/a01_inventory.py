"""Inventory: which RSS files built each delivered cube, plus per-exposure header/WCS provenance.
(Inline check run at the start of the michael_reduction analysis.)"""
import sys, glob, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
for f in sorted(glob.glob(f'{ROOT}/combined/*.fits')):
    h = fits.getheader(f); n = h.get('NRSSFILE', 0)
    print(os.path.basename(f), h['NAXIS1'], h['NAXIS2'], h['NAXIS3'], 'CRVAL3 %.0f' % (h['CRVAL3'] * 1e10), 'NEXPTOT', h.get('NEXPTOT'), h.get('BUNIT'))
    if 'blue' in f:
        for i in range(1, n + 1): print('    ', os.path.basename(h['RSSFIL%d' % i]))
for f in sorted(glob.glob(f'{ROOT}/extractions/*RSS_blue.fits')):
    with fits.open(f) as hd:
        p = hd[0].header; names = [x.name for x in hd]
        wh = hd['FIBERWCS'].header if 'FIBERWCS' in names else {}
    print(os.path.basename(f)[7:28], '%-18s' % p.get('OBJECT'), 'exp %7.1f' % float(p.get('SEXPTIME', 0)), 'X %.3f' % float(p.get('AIRMASS', 0)),
          'open', p.get('INS SHUTTER OPEN TIMESTAMP'), 'FLAM' if 'FLAM' in names else '----',
          'WCS:%s/%s n=%s' % (wh.get('WCSMETH'), wh.get('WCSTIER'), wh.get('WCSNSTAR')))
