"""Vet cube candidates: cube aperture spectrum + narrowband, field-average residual at the same λ,
per-exposure aperture line fluxes (RSS fibres within 1.2"), SKY model at the line, and the
interloper check ([O II] 3727 -> Hβ/[O III]/Hα; Lyα -> C IV 1549, He II 1640) in green/red."""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = sys.argv[1]; sd = scr_dir(t)
cands = [c for c in json.load(open(f'{sd}/search.json'))['pos']]
use = np.load(f'{sd}/use_mask.npy')
with fits.open(cube_path(t, 'blue')) as h:
    f = h[0].data.astype(float); v = h['VAR'].data.astype(float); wl = h['WAVELENGTH'].data.field(0)
    cw = WCS(h[0].header).celestial
ny, nx = f.shape[1:]; yy, xx = np.mgrid[:ny, :nx]
cont = median_filter(np.nan_to_num(f), size=(101, 1, 1), mode='nearest')
res = np.where(np.isfinite(f), f - cont, np.nan)
field = np.nanmedian(res[:, use], axis=1)                       # field-average residual
field_sd = 1.4826 * np.nanmedian(np.abs(res[:, use] - field[:, None]), axis=1)
FIX = exposure_shifts(t)
rss = []
for p in rss_paths(t, 'blue'):
    with fits.open(p) as h:
        fm = h['FIBERMAP'].data
        d = dict(exp=exp_id(p), fl=h['FLAM'].data.astype(float), er=h['FLAM_ERR'].data.astype(float),
                 sky=h['SKY'].data.astype(float), w=h['WAVE'].data.astype(float),
                 ra=np.array(fm['RA'], float), dec=np.array(fm['DEC'], float))
    if d['exp'] in FIX:
        c_ = np.cos(np.radians(np.nanmedian(d['dec']))); d['ra'] += FIX[d['exp']][0] / 3600 / c_; d['dec'] += FIX[d['exp']][1] / 3600
    fl_ = np.where(np.isfinite(d['fl']), d['fl'], np.nanmedian(d['fl'], axis=1, keepdims=True))
    d['res'] = fl_ - median_filter(np.nan_to_num(fl_), size=(1, 101), mode='nearest')
    d['dw'] = np.nanmedian(np.diff(d['w'], axis=1), axis=1)
    d.pop('fl')
    rss.append(d)
other = {}
for ch in ['green', 'red']:
    with fits.open(cube_path(t, ch)) as h:
        other[ch] = (h[0].data.astype(float), h['WAVELENGTH'].data.field(0))
n = len(cands)
fig, ax = plt.subplots(n, 4, figsize=(19, 3.3 * n), squeeze=False)
summary = []
for i, c in enumerate(cands):
    lam0, x0, y0 = c['lam'], c['x'], c['y']
    k = np.argmin(abs(wl - lam0))
    ap = (np.hypot(xx - x0, yy - y0) <= 2.4) & np.isfinite(f[k])
    sp = np.nansum(res[:, ap], axis=1) * 0.25                      # SB*arcsec² -> flux per Å
    er = np.sqrt(np.nansum(v[:, ap], axis=1)) * 0.25
    win = np.abs(wl - lam0) < 4
    lineflux = np.nansum(sp[win]) * np.median(np.diff(wl))
    # empirical error: same aperture at 200 random searched positions
    rng = np.random.default_rng(i); pts = np.argwhere(use)[rng.integers(0, use.sum(), 200)]
    rnd = [np.nansum(res[win][:, (np.hypot(xx - px, yy - py) <= 2.4)]) * 0.25 * np.median(np.diff(wl)) for py, px in pts]
    emp = 1.4826 * np.median(np.abs(np.array(rnd) - np.median(rnd)))
    k = np.argmin(abs(wl - lam0))
    nb = np.nanmean(res[k - 4:k + 5], axis=0)
    a0 = ax[i, 0]; m = np.nanpercentile(np.abs(nb[use]), 99)
    a0.imshow(nb, origin='lower', cmap='RdBu_r', vmin=-m, vmax=m); a0.plot(x0, y0, 'ko', mfc='none', ms=14)
    a0.set_title(f'C{i+1} narrowband {lam0 - 3.4:.0f}–{lam0 + 3.4:.0f} Å', fontsize=9)
    a1 = ax[i, 1]; sel = np.abs(wl - lam0) < 60
    a1.fill_between(wl[sel], -er[sel], er[sel], color='0.85', label='±1σ (VAR)'); a1.plot(wl[sel], sp[sel], 'k', lw=0.8, label='aperture r=1.2"')
    fsc = field[sel] * ap.sum() * 0.25
    a1.plot(wl[sel], fsc, 'm', lw=0.8, label='field-average residual (same aperture)')
    a1.axvline(lam0, color='b', ls=':'); a1.legend(fontsize=6)
    a1.set_title(f'C{i+1}: {lam0:.1f} Å  Lyα z={lam0/1215.67-1:.3f}  F={lineflux:.1e}  S/N_emp={lineflux/emp:.1f}', fontsize=9)
    # per-exposure
    ra0, dec0 = [float(q) for q in cw.pixel_to_world_values(x0, y0)]
    per = []
    rng2 = np.random.default_rng(100 + i)
    for d in rss:
        L = np.array([np.nansum(d['res'][j][np.abs(d['w'][j] - lam0) < 4]) * d['dw'][j] for j in range(len(d['ra']))])
        L[~np.isfinite(d['ra']) | (d['ra'] <= 0)] = np.nan
        cosd = np.cos(np.radians(dec0))
        def apflux(r0, d0):
            nn = np.hypot((d['ra'] - r0) * cosd, d['dec'] - d0) * 3600 < 1.2
            return np.nansum(L[nn]) if nn.sum() >= 3 else np.nan
        val = apflux(ra0, dec0)
        # random blank positions: fibre centres > 5" from the candidate and > 4" from continuum peaks
        ok = np.isfinite(L)
        dc = np.hypot((d['ra'] - ra0) * cosd, d['dec'] - dec0) * 3600
        cand_i = np.where(ok & (dc > 5))[0]
        rndv = np.array([apflux(d['ra'][j], d['dec'][j]) for j in rng2.choice(cand_i, 150, replace=False)])
        rndv = rndv[np.isfinite(rndv)]
        med_r = np.median(rndv); sig_r = 1.4826 * np.median(np.abs(rndv - med_r))
        per.append((d['exp'], val - med_r, sig_r, med_r))
    a2 = ax[i, 2]
    xs_ = np.arange(len(per)); vals = np.array([q[1] for q in per]); errs = np.array([q[2] for q in per])
    a2.errorbar(xs_, vals, yerr=errs, fmt='o'); a2.axhline(0, color='gray', lw=0.5)
    wmean = np.nansum(vals / errs ** 2) / np.nansum(1 / errs ** 2); chi2 = np.nansum(((vals - wmean) / errs) ** 2)
    a2.axhline(wmean, color='r', ls='--', lw=0.8)
    a2.set_xticks(xs_); a2.set_xticklabels([q[0] for q in per], rotation=60, fontsize=6)
    a2.set_title(f'per-exposure line flux (fibres <1.2"): mean {wmean:.1e}, χ²/{np.isfinite(vals).sum()-1} = {chi2:.1f}', fontsize=8)
    # interlopers
    a3 = ax[i, 3]; z_l = lam0 / 1215.67 - 1; z_o = lam0 / 3727.4 - 1
    checks = [('C IV 1549 (Lyα z)', 1549.0 * (1 + z_l)), ('He II 1640 (Lyα z)', 1640.4 * (1 + z_l)),
              ('Hβ ([O II] z)', 4861.3 * (1 + z_o)), ('[O III] 5007 ([O II] z)', 5006.8 * (1 + z_o)), ('Hα ([O II] z)', 6562.8 * (1 + z_o))]
    txt = []
    for nm, lw_ in checks:
        for ch, (cube, w2) in other.items():
            if w2[0] + 30 < lw_ < w2[-1] - 30:
                cc = median_filter(np.nan_to_num(np.nansum(cube[:, ap], axis=1)), 101, mode='nearest')
                s2 = (np.nansum(cube[:, ap], axis=1) - cc) * 0.25
                wn = np.abs(w2 - lw_) < 5
                rn = [np.nansum(s2[np.abs(w2 - lw_ - off) < 5]) for off in np.arange(-150, 151, 12) if abs(off) > 20]
                txt.append(f'{nm} @ {lw_:.0f} Å ({ch}): {np.nansum(s2[wn]) / (1.4826 * np.median(np.abs(np.array(rn) - np.median(rn))) + 1e-40):+.1f}σ')
    a3.axis('off'); a3.text(0, 0.95, '\n'.join(txt) if txt else 'no interloper lines in range', va='top', fontsize=8, family='monospace')
    a3.set_title('counterpart lines (same aperture)', fontsize=9)
    summary.append(dict(c=i + 1, lam=lam0, x=x0, y=y0, flux=lineflux, snr_emp=lineflux / emp, chi2=chi2, nexp=int(np.isfinite(vals).sum()),
                        per=[(q[0], q[1], q[2]) for q in per], field_bump=float(np.nansum(field[win]) / (np.sqrt(np.nansum(field_sd[win] ** 2)) / np.sqrt(use.sum()) + 1e-40)),
                        interlopers=txt))
    print('C%d %.1f Å (x,y)=(%d,%d) flux %.1e  S/N(empirical) %.1f | per-exp mean %.1e χ²=%.1f (%d dof) | per-exp S/N %s | %s' % (
        i + 1, lam0, x0, y0, lineflux, lineflux / emp, wmean, chi2, np.isfinite(vals).sum() - 1,
        [round(float(q[1] / q[2]), 1) for q in per], '; '.join(txt)))
    print('     combined empirical S/N from 8 exposures: %.1f ; field-wide systematic at this λ (median of random apertures / their σ): %s' % (
        wmean / np.sqrt(1 / np.nansum(1 / errs ** 2)), [round(float(q[3] / q[2]), 2) for q in per]))
print('field-average residual around 4150–4220 Å (×1e-20):', np.round(field[(wl > 4170) & (wl < 4200)] * 1e20, 1))
json.dump(summary, open(f'{sd}/vet.json', 'w'), indent=1, default=float)
fig.suptitle(f'{t}: vetting of blind-search candidates (blue cube)', fontsize=13)
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/08_candidate_vetting.png', dpi=85)
