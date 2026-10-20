"""Per-exposure vs combined images; centroid + FWHM of compact sources in each, in sky coords."""
import sys, glob, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.coadd import combine_image
from llamas_pyjamas.Combine.spectrum import fit_gaussian_image
S = sys.argv[1]
paths = sorted(glob.glob(f'{S}/data/*RSS_green.fits'))
with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_blue.fits') as h:
    cw = WCS(h[0].header).celestial
    center = (h[0].header['CRVAL1'], h[0].header['CRVAL2'])
# compact sources (cube pixels from the overview): star, red galaxy, plus a fainter one
seeds = {'star (60,67)': (60, 67), 'red gal (68,52)': (68, 52), 'NW src (10,72)': (10, 72)}
seed_sky = {k: cw.pixel_to_world_values(*v) for k, v in seeds.items()}
bands = {'green': (5000, 6800), 'red': (7000, 9500), 'blue': (4000, 4600)}
results = {}
imgs = {}
for band, (lo, hi) in bands.items():
    ch = [band]
    sets = {'combined': paths, **{p.split('/')[-1][18:28]: [p] for p in paths}}
    for name, ps in sets.items():
        sr = build_super_rss(ps, channels=ch)
        img = combine_image(sr, lo, hi, channels=ch, center=center)
        imgs[(band, name)] = img
        for src, (ra, dec) in seed_sky.items():
            x0, y0 = img.wcs.world_to_pixel_values(ra, dec)
            g = fit_gaussian_image(img.data, float(x0), float(y0), 5)
            if g is None:
                continue
            fra, fdec = img.wcs.pixel_to_world_values(g.x_mean.value, g.y_mean.value)
            fwhm = 2.355 * np.sqrt(abs(g.x_stddev.value * g.y_stddev.value)) * 0.5
            results.setdefault(band, {}).setdefault(src, {})[name] = dict(ra=float(fra), dec=float(fdec), fwhm=float(fwhm))
json.dump(results, open(f'{S}/dither_fits.json', 'w'), indent=1)
for band in results:
    print(f'== {band}')
    for src, d in results[band].items():
        if 'combined' not in d: continue
        r0, d0 = d['combined']['ra'], d['combined']['dec']
        print(f'  {src}: combined FWHM {d["combined"]["fwhm"]:.2f}"')
        for name, v in d.items():
            if name == 'combined': continue
            dx = (v['ra'] - r0) * np.cos(np.radians(d0)) * 3600; dy = (v['dec'] - d0) * 3600
            print(f'     {name}: offset from combined  dRA {dx:+.2f}"  dDec {dy:+.2f}"   FWHM {v["fwhm"]:.2f}"')
# figure: green images per exposure + combined, with the star zoom
fig, ax = plt.subplots(2, 4, figsize=(17, 8.5))
names = ['08-33-54.7', '08-57-45.9', '09-21-37.8', 'combined']
for j, name in enumerate(names):
    img = imgs[('green', name)]
    d = img.data
    lo, hi = ZScaleInterval().get_limits(d[np.isfinite(d)])
    ax[0, j].imshow(d, origin='lower', vmin=lo, vmax=hi, cmap='gray')
    ax[0, j].set_title(f'green 5000–6800 Å: {name}')
    ra, dec = seed_sky['star (60,67)']
    x0, y0 = img.wcs.world_to_pixel_values(ra, dec)
    x0, y0 = int(round(float(x0))), int(round(float(y0)))
    st = d[y0 - 10:y0 + 11, x0 - 10:x0 + 11]
    ax[1, j].imshow(st, origin='lower', cmap='magma', extent=(-5.25, 5.25, -5.25, 5.25))
    for src in results['green']:
        v = results['green'][src].get(name)
        if v is None or src != 'star (60,67)': continue
        px, py = img.wcs.world_to_pixel_values(v['ra'], v['dec'])
        ax[1, j].plot((px - x0) * 0.5, (py - y0) * 0.5, 'c+', ms=14)
        c = results['green'][src]['combined']
        cx, cy = img.wcs.world_to_pixel_values(c['ra'], c['dec'])
        ax[1, j].plot((cx - x0) * 0.5, (cy - y0) * 0.5, 'wx', ms=10)
        ax[1, j].set_title(f'star zoom (arcsec): FWHM {v["fwhm"]:.2f}"', fontsize=10)
    for (k, (sx, sy)) in seeds.items():
        px, py = img.wcs.world_to_pixel_values(*seed_sky[k])
        ax[0, j].add_patch(plt.Circle((px, py), 4, fill=False, color='y', lw=0.8))
ax[1, 0].set_ylabel('+ = this frame centroid, x = combined')
plt.tight_layout(); plt.savefig(f'{S}/plots/05_dither_images.png', dpi=95)
