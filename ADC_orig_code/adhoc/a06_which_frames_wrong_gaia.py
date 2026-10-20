"""J0111 first attempt: fibre-map centre vs telescope pointing per exposure, and the star vs Gaia,
to establish that exposure 1 (not frames 2-8) was the misplaced one."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from llamas_pyjamas.Utils.wcsLlamas import pointing_from_header
from llamas_pyjamas.Utils.register import query_gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
t = 'SPT-CLJ0111-5637'
for p in rss_paths(t, 'green'):
    h = fits.getheader(p); fm = fits.getdata(p, 'FIBERMAP'); ra, dec, pa = pointing_from_header(h)
    ok = np.isfinite(fm['RA']) & (fm['RA'] > 0); c = np.cos(np.radians(dec))
    print(exp_id(p), 'fibre-map centre offset from TEL pointing: %+6.1f %+6.1f"' % ((np.median(fm['RA'][ok]) - ra) * c * 3600, (np.median(fm['DEC'][ok]) - dec) * 3600))
d = json.load(open(scr_dir(t) + '/dithers.json'))['res']['red']; s = list(d.values())[0]['combined']
g = query_gaia(s['ra'], s['dec'], radius_arcsec=60, mag_limit=21); sc = SkyCoord(s['ra'] * u.deg, s['dec'] * u.deg)
print('star:', sc.to_string('hmsdms', precision=2), '| Gaia sources within 60":', len(g))
if len(g):
    i, dd, _ = sc.match_to_catalog_sky(g); print('  nearest Gaia: %.1f"' % dd.arcsec)
