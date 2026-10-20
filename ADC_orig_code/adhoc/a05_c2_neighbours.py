"""Coordinates of 0158 candidates C1, C2, C4 and their separations / velocity offsets from C2."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
t = 'SPT-CLJ0158-5500'
w = WCS(fits.getheader(cube_path(t, 'blue'))).celestial
v = {c['c']: c for c in json.load(open(scr_dir(t) + '/vet.json'))}
sc = {}
for k in [2, 1, 4]:
    ra, dec = w.pixel_to_world_values(v[k]['x'], v[k]['y']); sc[k] = SkyCoord(ra * u.deg, dec * u.deg)
    print('C%d  %s  λ %.1f  z_Lya %.3f' % (k, sc[k].to_string('hmsdms', precision=2), v[k]['lam'], v[k]['lam'] / 1215.67 - 1))
for k in [1, 4]:
    print('C%d vs C2: separation %.1f", Δv %+.0f km/s' % (k, sc[k].separation(sc[2]).arcsec, 3e5 * (v[k]['lam'] - v[2]['lam']) / v[2]['lam']))
