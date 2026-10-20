"""Dither-combine quality: per-exposure vs combined images, compact-source centroids and FWHM.
Optional --shift EXP DRA DDEC moves one exposure's fibres (arcsec) to test a registration fix."""
import sys, json, argparse, dataclasses, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.ndimage import gaussian_filter, maximum_filter
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.coadd import combine_image
from llamas_pyjamas.Combine.spectrum import fit_gaussian_image, _subselect

ap = argparse.ArgumentParser()
ap.add_argument('target'); ap.add_argument('--shift', nargs=3, default=None)
ap.add_argument('--tag', default=''); ap.add_argument('--nsig', type=float, default=10); ap.add_argument('--shifts', default=None); ap.add_argument('--blue', action='store_true')
a = ap.parse_args(); t = a.target
with fits.open(cube_path(t, 'green')) as h:
    center = (h[0].header['CRVAL1'], h[0].header['CRVAL2'])
sr = build_super_rss(rss_paths(t, 'green'), channels=['green', 'red'] + (['blue'] if a.blue else []))
ids = [e.exposure_id[7:] for e in sr.exposures]
if a.shifts:
    sh = json.load(open(a.shifts))['shifts']
    for c, st in sr.channels.items():
        for k, e in enumerate(ids):
            dx, dy = sh[e[11:]][c]; sel = st.exposure == k
            st.ra[sel] += dx / 3600 / np.cos(np.radians(st.dec[sel].mean())); st.dec[sel] += dy / 3600
    print('applied per-exposure, per-channel star-alignment shifts')
if a.shift:
    k = [i for i, e in enumerate(ids) if a.shift[0] in e][0]
    for st in sr.channels.values():
        sel = st.exposure == k
        st.ra[sel] += float(a.shift[1]) / 3600 / np.cos(np.radians(st.dec[sel].mean()))
        st.dec[sel] += float(a.shift[2]) / 3600
    print(f'shifted {ids[k]} by {a.shift[1]}", {a.shift[2]}"')

def sub(keep):
    return dataclasses.replace(sr, exposures=sr.exposures,
                               channels={c: _subselect(st, np.isin(st.exposure, keep)) for c, st in sr.channels.items()})

bands = {'green': (5000, 6800), 'red': (7000, 9500)}
if a.blue: bands['blue'] = (3800, 4600)
imgs = {}
for b, (lo, hi) in bands.items():
    imgs[(b, 'combined')] = combine_image(sr, lo, hi, channels=[b], center=center)
    for i, e in enumerate(ids):
        imgs[(b, e)] = combine_image(sub([i]), lo, hi, channels=[b], center=center)
# compact sources in the combined green+red image (well covered, >10 robust sigma, separated)
cg, cr = imgs[('green', 'combined')], imgs[('red', 'combined')]
det = np.nan_to_num(cg.snr()) + (np.nan_to_num(cr.snr()) if cr.data.shape == cg.data.shape else np.nan_to_num(cg.snr()))
sm = gaussian_filter(det, 1.0)
v = sm[(cg.nexp >= max(2, len(ids) // 2))]
med, mad = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
pk = (sm == maximum_filter(sm, 9)) & (sm > med + a.nsig * mad) & (cg.nexp >= 2)
ys, xs = np.where(pk)
order = np.argsort(sm[ys, xs])[::-1][:8]
srcs = [(int(xs[i]), int(ys[i]), float(sm[ys[i], xs[i]])) for i in order]
print('compact sources (x, y, combined S/N):', [(x, y, round(s)) for x, y, s in srcs])
res = {}
for b in bands:
    cw = imgs[(b, 'combined')].wcs
    for j, (x0, y0, _) in enumerate(srcs):
        g = fit_gaussian_image(imgs[(b, 'combined')].data, x0, y0, 5)
        if g is None: continue
        cra, cdec = cw.pixel_to_world_values(g.x_mean.value, g.y_mean.value)
        fw = 2.355 * np.sqrt(abs(g.x_stddev.value * g.y_stddev.value)) * 0.5
        entry = {'combined': dict(dra=0.0, ddec=0.0, fwhm=float(fw), ra=float(cra), dec=float(cdec))}
        for e in ids:
            im = imgs[(b, e)]
            ex, ey = [float(q) for q in im.wcs.world_to_pixel_values(cra, cdec)]
            xi, yi = int(round(ex)), int(round(ey))
            if not (0 <= yi < im.data.shape[0] and 0 <= xi < im.data.shape[1]): continue
            if not np.isfinite(im.data[yi, xi]) or im.nexp[yi, xi] == 0: continue
            ge = fit_gaussian_image(im.data, ex, ey, 5)
            if ge is None: continue
            era, edec = im.wcs.pixel_to_world_values(ge.x_mean.value, ge.y_mean.value)
            entry[e] = dict(dra=float((era - cra) * np.cos(np.radians(cdec)) * 3600), ddec=float((edec - cdec) * 3600),
                            fwhm=float(2.355 * np.sqrt(abs(ge.x_stddev.value * ge.y_stddev.value)) * 0.5))
        res.setdefault(b, {})[f'S{j+1} ({x0},{y0})'] = entry
json.dump(dict(srcs=srcs, res=res, ids=ids), open(f'{scr_dir(t)}/dithers{a.tag}.json', 'w'), indent=1)
for b in res:
    print(f'== {b}')
    for s, entry in res[b].items():
        offs = [(e, d) for e, d in entry.items() if e != 'combined']
        r = [np.hypot(d['dra'], d['ddec']) for _, d in offs]
        fw = [d['fwhm'] for _, d in offs]
        print('  %-12s combined FWHM %.2f"  | %d frames: median single FWHM %.2f"  centroid offsets rms %.2f" max %.2f"' % (
            s, entry['combined']['fwhm'], len(offs), np.median(fw) if fw else np.nan,
            np.sqrt(np.mean(np.square(r))) if r else np.nan, max(r) if r else np.nan))
        for e, d in offs:
            if np.hypot(d['dra'], d['ddec']) > 0.6:
                print('      outlier %s: dRA %+.2f" dDec %+.2f" FWHM %.2f"' % (e, d['dra'], d['ddec'], d['fwhm']))
# figure
n = len(ids) + 1
fig = plt.figure(figsize=(3.2 * min(n, 5), 3.4 * (int(np.ceil(n / 5)) + 1)))
for i, e in enumerate(['combined'] + ids):
    axi = fig.add_subplot(int(np.ceil(n / 5)) + 1, 5, i + 1)
    d = imgs[('green', e)].data
    fin = d[np.isfinite(d)]
    if fin.size:
        lo, hi = ZScaleInterval().get_limits(fin)
        axi.imshow(d, origin='lower', vmin=lo, vmax=hi, cmap='gray')
    for j, (x0, y0, _) in enumerate(srcs):
        sra, sdec = cg.wcs.pixel_to_world_values(x0, y0)
        px, py = imgs[('green', e)].wcs.world_to_pixel_values(sra, sdec)
        axi.add_patch(plt.Circle((px, py), 4, fill=False, color='y', lw=0.7))
        axi.text(px + 5, py + 5, f'S{j+1}', color='y', fontsize=7)
    axi.set_title(e if e != 'combined' else 'COMBINED (green)', fontsize=9); axi.set_xticks([]); axi.set_yticks([])
axo = fig.add_subplot(int(np.ceil(n / 5)) + 1, 2, 2 * (int(np.ceil(n / 5)) + 1) - 1)
cols = plt.cm.tab10(np.arange(len(ids)) % 10)
for s, entry in res.get('green', {}).items():
    for k, e in enumerate(ids):
        if e in entry:
            axo.plot(entry[e]['dra'], entry[e]['ddec'], 'o', color=cols[k], ms=5, label=e if s.startswith('S1 ') else None)
axo.axhline(0, color='k', lw=0.4); axo.axvline(0, color='k', lw=0.4)
axo.add_patch(plt.Circle((0, 0), 0.5, fill=False, ls=':', color='gray'))
lim = 1.5 if 'shift' not in a.tag else 1.5
axo.set_xlim(-lim, lim); axo.set_ylim(-lim, lim); axo.set_aspect('equal')
axo.set_xlabel('ΔRA vs combined ["]'); axo.set_ylabel('ΔDec ["]'); axo.legend(fontsize=6, ncol=2)
axo.set_title('per-frame centroid offsets, all sources (green); dotted = 0.5"', fontsize=9)
axf = fig.add_subplot(int(np.ceil(n / 5)) + 1, 2, 2 * (int(np.ceil(n / 5)) + 1))
for s, entry in res.get('green', {}).items():
    axf.plot(range(len(ids)), [entry.get(e, {}).get('fwhm', np.nan) for e in ids], 'o-', ms=4, label=s)
    axf.axhline(entry['combined']['fwhm'], ls=':', lw=0.8)
axf.set_xticks(range(len(ids))); axf.set_xticklabels(ids, rotation=60, fontsize=7)
axf.set_ylabel('FWHM ["]'); axf.legend(fontsize=6); axf.set_title('single-frame FWHM (dotted = combined)', fontsize=9)
fig.suptitle(f'{t}: dither combine check {a.tag}', fontsize=13)
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/02_dither_combine{a.tag}.png', dpi=95)
