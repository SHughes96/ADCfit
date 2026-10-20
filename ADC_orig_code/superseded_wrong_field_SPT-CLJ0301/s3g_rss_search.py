"""Fibre-level line search per exposure, then cross-match between exposures.
Per fibre: subtract a running-median continuum (101 px), normalise by a robust per-fibre noise
(MAD of the residual, so it is empirical, not ERROR), and smooth with a 5-px box. A 'hit' is a
>= 5 sigma peak whose 3 central pixels are each > 2 sigma (rejects single-pixel CR / defects).
Cross-match: same sky position (< 1.0") and wavelength (< 3 A) in >= 2 of 3 exposures."""
import sys, glob, json, numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, uniform_filter1d
S = sys.argv[1]
hits = []
for f in sorted(glob.glob(f'{S}/data/*RSS_blue.fits')):
    eid = f.split('/')[-1][18:28]
    with fits.open(f) as h:
        ss = h['SKYSUB'].data.astype(float); wave = h['WAVE'].data.astype(float)
        mk = h['MASK'].data != 0; fm = h['FIBERMAP'].data
    ss[mk | ~np.isfinite(ss)] = np.nan
    ok = np.isfinite(ss).sum(axis=1) > 1000
    band = (wave > 3300) & (wave < 4780)
    filled = np.where(np.isfinite(ss), ss, np.nanmedian(ss, axis=1, keepdims=True))
    cont = median_filter(filled, size=(1, 101), mode='nearest')
    r = filled - cont
    # clip the huge negative defects before estimating noise and smoothing
    noise = 1.4826 * np.nanmedian(np.abs(np.where(band, r, np.nan)), axis=1, keepdims=True)
    z1 = r / noise
    z1[z1 < -10] = 0
    zs = uniform_filter1d(z1, 5, axis=1) * np.sqrt(5)
    zs[~band | ~ok[:, None]] = 0
    n = 0
    for i in np.where(ok)[0]:
        if not np.isfinite(fm['RA'][i]) or fm['RA'][i] < 0: continue
        cand = np.where(zs[i] > 5)[0]
        while cand.size:
            k = cand[np.argmax(zs[i, cand])]
            if k > 1 and np.all(z1[i, k - 1:k + 2] > 2):
                hits.append(dict(exp=eid, fib=int(i), bs=str(fm['BENCHSIDE'][i]), ra=float(fm['RA'][i]),
                                 dec=float(fm['DEC'][i]), lam=float(wave[i, k]), z=float(zs[i, k])))
                n += 1
            cand = cand[np.abs(cand - k) > 8]
    print(eid, 'fibre-level line hits:', n)
json.dump(hits, open(f'{S}/rss_hits.json', 'w'))
# cross-match
exps = sorted({h['exp'] for h in hits})
groups = []
for h in hits:
    for g in groups:
        g0 = g[0]
        dd = np.hypot((h['ra'] - g0['ra']) * np.cos(np.radians(g0['dec'])), h['dec'] - g0['dec']) * 3600
        if dd < 1.0 and abs(h['lam'] - g0['lam']) < 3:
            g.append(h); break
    else:
        groups.append([h])
multi = [g for g in groups if len({h['exp'] for h in g}) >= 2]
print('groups:', len(groups), ' seen in >=2 exposures:', len(multi))
for g in sorted(multi, key=lambda g: -sum(h['z'] for h in g)):
    print('  lam %.1f (z_Lya %.3f) RA %.5f Dec %.5f  exps %s  fibres %s  S/N %s' % (
        np.mean([h['lam'] for h in g]), np.mean([h['lam'] for h in g]) / 1215.67 - 1, g[0]['ra'], g[0]['dec'],
        sorted({h['exp'] for h in g}), [(h['fib'], h['bs']) for h in g], [round(h['z'], 1) for h in g]))
json.dump(multi, open(f'{S}/rss_multi.json', 'w'))
