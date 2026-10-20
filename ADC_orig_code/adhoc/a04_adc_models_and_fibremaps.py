"""(1) Fit simple ADC models to the per-exposure blue-minus-red star offsets (from s3_adc.py).
(2) Confirm the three channels share identical fibre maps (so chromatic offsets arise before the fibres)."""
import sys, os, json, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
r = json.load(open(os.path.join(SCR, 'SPT-CLJ0111-5637', 'adc_star.json')))
q = np.radians([x['q'] for x in r]); m = np.array([np.hypot(*x['prd_blue']) for x in r]); obs = np.array([x['obs_blue'] for x in r])
u = lambda a: np.array([np.sin(a), np.cos(a)]).T
models = [('no ADC correction (pure atmosphere)', lambda p: (m[:, None] * u(q) - obs).ravel(), []),
          ('ADC tracks q, rotated by phi, amplitude A', lambda p: (m[:, None] * (u(q) - p[0] * u(q + p[1])) - obs).ravel(), [1, 0.5]),
          ('ADC stuck at fixed orientation phi, amplitude A', lambda p: (m[:, None] * u(q) - p[0] * m[:, None] * u(np.full_like(q, p[1])) - obs).ravel(), [1, 0.]),
          ('constant chromatic offset (dE, dN arcsec)', lambda p: (np.array(p)[None, :] - obs).ravel(), [-1.5, 0.3])]
for name, f, p0 in models:
    if p0:
        s = least_squares(f, p0); res, par = s.fun, s.x
    else:
        res, par = f(None), []
    print('%-50s rms resid %.2f"  params %s' % (name, np.sqrt(np.mean(res ** 2)), np.round(par, 3).tolist()))
print('perfect ADC (residual 0) rms: %.2f"' % np.sqrt(np.mean(obs ** 2)))
t = 'SPT-CLJ0111-5637'
fm = {ch: fits.open(rss_paths(t, ch)[3])['FIBERMAP'].data for ch in ['blue', 'green', 'red']}
for a in ['blue', 'green']:
    d = np.hypot((fm[a]['RA'] - fm['red']['RA']) * np.cos(np.radians(fm['red']['DEC'])), fm[a]['DEC'] - fm['red']['DEC']) * 3600
    print(a, 'vs red FIBERMAP: max position difference %.4f"' % np.nanmax(d), ' same fibre IDs:', np.array_equal(fm[a]['FIBER_ID'], fm['red']['FIBER_ID']))
