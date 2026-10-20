"""Track the N brightest compact sources on the fibre bundle across exposures (instrument frame),
and save each one's spectrum so we can tell real objects from detector artefacts."""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = sys.argv[1]; N = 5
# reference frame: median-stack of all exposures in the INSTRUMENT frame (same fibre ids)
lvs, xs, ys = [], None, None
for p in rss_paths(t, 'red'):
    with fits.open(p) as h:
        ss = h['SKYSUB'].data; w = h['WAVE'].data; fm = h['FIBERMAP'].data
    lvs.append(np.nanmedian(np.where((w > 7000) & (w < 9500), ss, np.nan), axis=1))
    if xs is None:
        ra, dec = np.array(fm['RA'], float), np.array(fm['DEC'], float); c = np.cos(np.radians(np.median(dec)))
        xs = (ra - np.median(ra)) * c * 3600; ys = (dec - np.median(dec)) * 3600
lvs = np.array(lvs); lvs[:, 135] = np.nan
med = np.nanmedian(lvs, axis=0)
nbr = [np.where(np.hypot(xs - xs[i], ys - ys[i]) < 1.0)[0] for i in range(len(xs))]
score = np.array([np.nansum(med[n]) for n in nbr])
peaks = []
for k in np.argsort(np.nan_to_num(score, nan=-1e9))[::-1]:
    if all(np.hypot(xs[k] - xs[q], ys[k] - ys[q]) > 4 for q in peaks) and abs(xs[k]) < 16 and abs(ys[k]) < 16:
        peaks.append(k)
    if len(peaks) == N: break
print(f'{t}: {N} brightest compact peaks in the instrument-frame median; per-exposure centroid scatter')
out = []
for j, k in enumerate(peaks):
    pos = []
    for i in range(len(lvs)):
        sel = np.hypot(xs - xs[k], ys - ys[k]) < 2.5
        ring = (np.hypot(xs - xs[k], ys - ys[k]) > 4) & (np.hypot(xs - xs[k], ys - ys[k]) < 7)
        wgt = np.clip(lvs[i, sel] - np.nanmedian(lvs[i, ring]), 0, None)
        wgt = np.nan_to_num(wgt)
        pos.append((np.sum(wgt * xs[sel]) / wgt.sum(), np.sum(wgt * ys[sel]) / wgt.sum(), float(wgt.sum())))
    pos = np.array(pos)
    print('  P%d at bundle (%+5.1f, %+5.1f)": flux %.0f  centroid scatter across exposures x %.2f" y %.2f"  (range %.2f", %.2f")' % (
        j + 1, xs[k], ys[k], np.median(pos[:, 2]), pos[:, 0].std(), pos[:, 1].std(), np.ptp(pos[:, 0]), np.ptp(pos[:, 1])))
    out.append(dict(fib=int(k), x=float(xs[k]), y=float(ys[k]), pos=pos.tolist()))
json.dump(out, open(f'{scr_dir(t)}/bundle_sources.json', 'w'), indent=1)
# spectra of the peaks (exposure 1, fibres within 1.2"), all three channels
fig, ax = plt.subplots(len(peaks), 1, figsize=(12, 2.4 * len(peaks)))
for j, k in enumerate(peaks):
    for ch, col in [('blue', 'C0'), ('green', 'C2'), ('red', 'C3')]:
        p = rss_paths(t, ch)[0]
        with fits.open(p) as h:
            fl = h['FLAM'].data if 'FLAM' in h else h['SKYSUB'].data; w = h['WAVE'].data
        sel = np.hypot(xs - xs[k], ys - ys[k]) < 1.2
        wl = np.nanmedian(w[sel], axis=0); sp = np.nansum(fl[sel], axis=0)
        kk = np.ones(15) / 15
        ax[j].plot(wl, np.convolve(np.nan_to_num(sp), kk, 'same'), color=col, lw=0.7)
    ax[j].set_title(f'P{j+1} at bundle ({xs[k]:+.1f}, {ys[k]:+.1f})" — exposure 1, FLAM summed over fibres < 1.2"', fontsize=9)
    ax[j].set_ylim(np.nanpercentile(np.convolve(np.nan_to_num(sp), kk, 'same'), 1) * 1.5 if False else None)
ax[-1].set_xlabel('wavelength [Å]')
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/04_bundle_sources_spectra.png', dpi=90)
