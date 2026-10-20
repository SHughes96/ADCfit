"""Per-candidate figure: per-exposure spectra of the fibres within 1" (SKYSUB, continuum-
subtracted), the scaled SKY for comparison, and the field-wide hit histogram for context."""
import sys, glob, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import median_filter
S = sys.argv[1]
keep = json.load(open(f'{S}/rss_keep.json'))
hits = json.load(open(f'{S}/rss_hits.json'))
exps = {}
for f in sorted(glob.glob(f'{S}/data/*RSS_blue.fits')):
    with fits.open(f) as h:
        exps[f.split('/')[-1][18:28]] = dict(ss=h['SKYSUB'].data.astype(float), sky=h['SKY'].data.astype(float),
                                            wave=h['WAVE'].data.astype(float), fm=h['FIBERMAP'].data)
cols = {'08-33-54.7': 'C0', '08-57-45.9': 'C1', '09-21-37.8': 'C2'}
fig, axes = plt.subplots(len(keep) + 1, 1, figsize=(11, 3.0 * (len(keep) + 1)))
lam_all = np.array([h['lam'] for h in hits])
for e, c in cols.items():
    axes[0].hist([h['lam'] for h in hits if h['exp'] == e], bins=np.arange(3300, 4790, 5), histtype='step', color=c, label=e)
axes[0].set_title('single-exposure fibre "line" hits vs wavelength (whole field) — spikes = instrumental')
axes[0].set_xlabel('wavelength [Å]'); axes[0].legend(fontsize=8); axes[0].set_yscale('log')
for ax, k in zip(axes[1:], keep):
    for e, d in exps.items():
        fm = d['fm']
        dd = np.hypot((fm['RA'] - k['ra']) * np.cos(np.radians(k['dec'])), fm['DEC'] - k['dec']) * 3600
        near = np.where(dd < 1.0)[0]
        if not len(near): continue
        w = np.nanmedian(d['wave'][near], axis=0)
        sp = np.nanmean(d['ss'][near], axis=0)
        sp = sp - median_filter(np.nan_to_num(sp), 101, mode='nearest')
        sel = abs(w - k['lam']) < 60
        ax.plot(w[sel], sp[sel], color=cols[e], lw=0.9, label=f'{e} fib {list(near)}')
        sky = np.nanmean(d['sky'][near], axis=0)
        sky = sky - np.nanmedian(sky[sel])
        ax.plot(w[sel], sky[sel] / max(np.nanmax(abs(sky[sel])), 1) * np.nanstd(sp[sel]) * 3, color=cols[e], lw=0.5, ls=':')
    ax.axvline(k['lam'], color='k', lw=0.5, ls='--')
    ax.set_title('cand %.1f Å (Lyα z=%.3f) at cube (%d,%d), bench %s  — dotted = scaled SKY' %
                 (k['lam'], k['lam'] / 1215.67 - 1, k['x'], k['y'], k['bs']), fontsize=9)
    ax.legend(fontsize=6, loc='upper right'); ax.set_ylabel('counts − cont')
axes[-1].set_xlabel('wavelength [Å]')
plt.tight_layout(); plt.savefig(f'{S}/plots/04_rss_candidates.png', dpi=100)
