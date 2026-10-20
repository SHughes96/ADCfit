"""Star-aligned SPT-CLJ0111-5637 cubes. For every channel and exposure, fit the field star on the
fibres (2-D Gaussian; width free in red, then fixed per exposure for green/blue) and shift that
exposure's fibre positions so the star lands on its Gaia position. Per-channel alignment also
removes the blue-vs-red ADC offset. Writes NEW cubes; inputs are not modified."""
import sys, json, os, numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import least_squares
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.cube import combine_cube
from llamas_pyjamas.Utils.register import query_gaia
t = 'SPT-CLJ0111-5637'
odir = os.path.join(out_dir(t), 'star_aligned')
BANDS = {'red': (7000, 9500), 'green': (5000, 6800), 'blue': (3800, 4600)}
# anchor: Gaia position of the star (search near the current combined position)
prev = list(json.load(open(f'{scr_dir(t)}/dithers_header_combine.json'))['res']['red'].values())[0]['combined']
g = query_gaia(prev['ra'], prev['dec'], radius_arcsec=30, mag_limit=21)
sc = SkyCoord(prev['ra'] * u.deg, prev['dec'] * u.deg)
i_, d_, _ = sc.match_to_catalog_sky(g)
anchor = g[int(i_)]
print('anchor = Gaia source %.2f" from the header-combine star position: %s' % (d_.arcsec, anchor.to_string('hmsdms', precision=2)))
ra0, dec0 = anchor.ra.deg, anchor.dec.deg; cosd = np.cos(np.radians(dec0))

def fit_star(x, y, v, sig=None):
    def model(p):
        a, xc, yc = p[:3]; s = p[3] if sig is None else sig; b = p[-1]
        return a * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * s ** 2)) + b - v
    k = np.nanargmax(v)
    p0 = [v[k] - np.nanmedian(v), x[k], y[k]] + ([0.6] if sig is None else []) + [np.nanmedian(v)]
    lo = [0, x.min(), y.min()] + ([0.25] if sig is None else []) + [-np.inf]
    hi = [np.inf, x.max(), y.max()] + ([3] if sig is None else []) + [np.inf]
    r = least_squares(model, p0, bounds=(lo, hi))
    return r.x[1], r.x[2], (r.x[3] if sig is None else sig)

sigmas, shifts, written = {}, {}, {}
hdr0 = fits.getheader(cube_path(t, 'blue'))
for ch in ['red', 'green', 'blue']:
    sr = build_super_rss(rss_paths(t, ch), channels=[ch])
    st = sr.channels[ch]; lo, hi = BANDS[ch]
    ids = [e.exposure_id[18:] for e in sr.exposures]
    for i, e in enumerate(ids):
        sel = np.where(st.exposure == i)[0]
        x = (st.ra[sel] - ra0) * cosd * 3600; y = (st.dec[sel] - dec0) * 3600
        near = np.hypot(x, y) < 8
        w = st.wave[sel][near]; fl = np.where(st.mask[sel][near], np.nan, st.flux[sel][near])
        v = np.nanmedian(np.where((w > lo) & (w < hi), fl, np.nan), axis=1)
        ok = np.isfinite(v)
        # first locate the peak, then fit within 3" of it
        xs, ys, vs = x[near][ok], y[near][ok], v[ok]
        k = np.argmax(np.convolve(vs, [1], 'same'))
        sm = np.array([np.nansum(vs[np.hypot(xs - xs[j], ys - ys[j]) < 1.0]) for j in range(len(vs))])
        k = np.argmax(sm)
        m = np.hypot(xs - xs[k], ys - ys[k]) < 3.0
        scale = np.nanmax(np.abs(vs[m]))                     # FLAM ~1e-17: rescale to O(1) for the fit
        xc, yc, s = fit_star(xs[m], ys[m], vs[m] / scale, None if ch == 'red' else sigmas[e])
        if not (abs(xc - xs[k]) > 1e-6 or abs(yc - ys[k]) > 1e-6):
            print('    WARNING: fit did not move from the starting fibre')
        if ch == 'red': sigmas[e] = s
        st.ra[sel] -= xc / 3600 / cosd; st.dec[sel] -= yc / 3600
        shifts.setdefault(e, {})[ch] = (-float(xc), -float(yc))
        print('  %-5s %s: star at (%+5.2f, %+5.2f)" from Gaia -> shifted; FWHM %.2f"' % (ch, e, xc, yc, 2.355 * s))
    cube = combine_cube(sr, ch, pixscale=hdr0['PIXSCALE'], kernel_fwhm=hdr0['KERNFWHM'], weighting=hdr0['WGHT'].strip(), units=hdr0['UNITS'].strip())
    cube.meta['ALIGNMTH'] = 'star-aligned per exposure+channel (Gaia anchor)'
    cube.meta['ANCHRA'] = float(ra0); cube.meta['ANCHDEC'] = float(dec0)
    for j, e in enumerate(ids, 1):
        cube.meta[f'SHFT{j}'] = '%s %+.2f %+.2f' % (e, *shifts[e][ch])
    path = os.path.join(odir, f'{t}_staraligned_cube_{ch}.fits')
    cube.write(path); written[ch] = path
    print(f'  wrote {path}  shape {cube.data.shape}')
    del sr, st, cube
json.dump(dict(anchor=[ra0, dec0], shifts=shifts, sigmas=sigmas), open(f'{scr_dir(t)}/staralign_shifts.json', 'w'), indent=1)
