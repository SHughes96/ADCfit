"""Choose the blue alignment: compare the combined blue star image for three shift recipes."""
import sys, json, copy, numpy as np
from astropy.io import fits
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.coadd import combine_image
from llamas_pyjamas.Combine.spectrum import fit_gaussian_image
t = 'SPT-CLJ0111-5637'
J = json.load(open(f'{scr_dir(t)}/staralign_shifts.json')); sh = J['shifts']; ra0, dec0 = J['anchor']
adc = json.load(open(f'{scr_dir(t)}/adc_star.json'))
# smooth ADC model: blue(3800-4600 mean) minus red, constant fit over exposures, from the counts-space fits
blue_minus_red = []
for r in adc:
    o = np.array(r['obs']); sel = (o[:, 0] > 3800) & (o[:, 0] < 4600)
    blue_minus_red.append(o[sel, 1:].mean(axis=0))
const = np.median(blue_minus_red, axis=0)
print('ADC blue(3800-4600)-red offset per exposure:', np.round(blue_minus_red, 2).tolist(), '-> median', np.round(const, 2))
recipes = {
    'a_blue_fit': {e: sh[e]['blue'] for e in sh},
    'b_red_plus_adc': {e: (sh[e]['red'][0] - const[0], sh[e]['red'][1] - const[1]) for e in sh},
    'c_red_only': {e: tuple(sh[e]['red']) for e in sh},
    'd_red_plus_adc_per_exp': {r['exp']: (sh[r['exp']]['red'][0] - bmr[0], sh[r['exp']]['red'][1] - bmr[1]) for r, bmr in zip(adc, blue_minus_red)},
}
base = build_super_rss(rss_paths(t, 'blue'), channels=['blue'])
ids = [e.exposure_id[18:] for e in base.exposures]
ra_orig, dec_orig = base.channels['blue'].ra.copy(), base.channels['blue'].dec.copy()
res = {}
for name, rec in recipes.items():
    st = base.channels['blue']; st.ra[:] = ra_orig; st.dec[:] = dec_orig
    for k, e in enumerate(ids):
        sel = st.exposure == k
        st.ra[sel] += rec[e][0] / 3600 / np.cos(np.radians(dec0)); st.dec[sel] += rec[e][1] / 3600
    img = combine_image(base, 3800, 4600, channels=['blue'], center=(ra0, dec0))
    x0, y0 = [float(q) for q in img.wcs.world_to_pixel_values(ra0, dec0)]
    g = fit_gaussian_image(img.data, x0, y0, 6)
    fw = 2.355 * np.sqrt(abs(g.x_stddev.value * g.y_stddev.value)) * 0.5
    ell = abs(g.x_stddev.value - g.y_stddev.value) / max(g.x_stddev.value, g.y_stddev.value)
    pra, pdec = img.wcs.pixel_to_world_values(g.x_mean.value, g.y_mean.value)
    yy, xx = np.mgrid[:img.data.shape[0], :img.data.shape[1]]
    cx, cy = g.x_mean.value, g.y_mean.value
    r = np.hypot(xx - cx, yy - cy) * 0.5
    bg = np.nanmedian(img.data[(r > 4) & (r < 7)])
    f1 = np.nansum(img.data[r <= 1.0] - bg); f3 = np.nansum(img.data[r <= 3.0] - bg)
    res[name] = dict(fwhm=float(fw), peak=float(g.amplitude.value), ell=float(ell), conc=float(f1 / f3))
    print('     encircled flux r<1" / r<3": %.3f' % (f1 / f3))
    print('%-16s combined blue star FWHM %.2f"  peak %.3e  ellipticity %.2f  offset from Gaia anchor (%+.2f, %+.2f)"' % (
        name, fw, g.amplitude.value, ell, (pra - ra0) * np.cos(np.radians(dec0)) * 3600, (pdec - dec0) * 3600))
best = max(res, key=lambda k: res[k]['conc'])
print('best (most concentrated):', best)
json.dump(dict(recipes={k: {e: list(v) for e, v in r.items()} for k, r in recipes.items()}, res=res, best=best, adc_const=const.tolist()),
          open(f'{scr_dir(t)}/blue_choice.json', 'w'), indent=1)
