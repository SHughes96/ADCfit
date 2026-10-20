"""DATE-OBS vs MJD-OBS: DATE-OBS holds the LOCAL date for frames taken before local midnight."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from astropy.time import Time
for t in TARGETS:
    for p in rss_paths(t):
        h = fits.getheader(p)
        print(exp_id(p), h['DATE-OBS'], h['INS SHUTTER OPEN TIMESTAMP'], 'MJD-OBS', h.get('MJD-OBS'), Time(h['MJD-OBS'], format='mjd').isot)
