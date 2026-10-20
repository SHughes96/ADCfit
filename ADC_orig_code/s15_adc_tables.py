"""Printable tables + figures of the ADC (chromatic) star offsets vs wavelength for SPT-CLJ0111-5637,
and of the per-channel shifts that were applied when building the star-aligned cubes.

Reads only saved results: work/<t>/adc_star.json (s3_adc.py) and work/<t>/staralign_shifts.json
(s10_staralign.py); cross-checks the shifts against the SHFTn cards of the star-aligned cubes.
Paths are taken from this script's location, so it runs wherever lya_analysis/ has been moved.

Sign conventions (arcsec on the sky, +E = East, +N = North):
  measured offset  = star centroid at wavelength lam minus the red-channel mean centroid (same exposure)
  applied shift    = shift added to that exposure+channel's fibre positions (= minus the star's offset
                     from its Gaia position); the offset REMOVED relative to red is -(shift_ch - shift_red)
  residual         = measured offset - offset removed for that channel (what the rigid shifts left behind)
"""
import os, csv, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits

T = 'SPT-CLJ0111-5637'
LA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(LA, 'work', T)
ODIR = os.path.join(LA, T, 'ADC'); os.makedirs(ODIR, exist_ok=True)
CH = ['blue', 'green', 'red']
FITBAND = {'blue': (3800, 4600), 'green': (5000, 6800), 'red': (7000, 9500)}   # s10_staralign.py bands
COL = {'blue': '#1f5fbf', 'green': '#2a9d3a', 'red': '#c0392b'}
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})

adc = json.load(open(os.path.join(WORK, 'adc_star.json')))
sa = json.load(open(os.path.join(WORK, 'staralign_shifts.json')))
shifts = sa['shifts']
exps = [r['exp'] for r in adc]


def removed(e, ch):
    """Offset (relative to red) that the applied per-channel shift removed for exposure e."""
    s, r = np.array(shifts[e][ch]), np.array(shifts[e]['red'])
    return -(s - r)


# ---- 1. per exposure x wavelength bin
rows = []
for r in adc:
    e = r['exp']
    for fit, ob, pr in zip(r['rows'], r['obs'], r['prd']):
        rm = removed(e, fit['ch'])
        rows.append(dict(exp=e, utc_end='2026-10-16T' + e.replace('-', ':', 2), airmass=r['X'], parang=r['q'],
                         channel=fit['ch'], lam=ob[0], dE=ob[1], dN=ob[2], err_E=fit['ex'], err_N=fit['ey'],
                         pred_dE=pr[1], pred_dN=pr[2], meas_minus_pred_E=ob[1] - pr[1], meas_minus_pred_N=ob[2] - pr[2],
                         removed_E=rm[0], removed_N=rm[1], resid_E=ob[1] - rm[0], resid_N=ob[2] - rm[1]))


def write_csv(name, recs, fmt='%.3f'):
    with open(os.path.join(ODIR, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader()
        for rec in recs:
            w.writerow({k: (fmt % v if isinstance(v, float) else v) for k, v in rec.items()})


write_csv('adc_offsets_vs_wavelength.csv', rows)

# ---- 2. per-wavelength summary over the exposures
lams = sorted({r['lam'] for r in rows})
summ = []
for l in lams:
    s = [r for r in rows if r['lam'] == l]
    dE, dN = np.array([r['dE'] for r in s]), np.array([r['dN'] for r in s])
    pE, pN = np.array([r['pred_dE'] for r in s]), np.array([r['pred_dN'] for r in s])
    rE, rN = np.array([r['resid_E'] for r in s]), np.array([r['resid_N'] for r in s])
    summ.append(dict(lam=l, channel=s[0]['channel'], n_exp=len(s),
                     mean_dE=dE.mean(), rms_dE=dE.std(), min_dE=dE.min(), max_dE=dE.max(),
                     mean_dN=dN.mean(), rms_dN=dN.std(), min_dN=dN.min(), max_dN=dN.max(),
                     mean_pred_dE=pE.mean(), mean_pred_dN=pN.mean(),
                     mean_resid_E=rE.mean(), mean_resid_N=rN.mean(), median_err=float(np.median([r['err_E'] for r in s]))))
write_csv('adc_offsets_vs_wavelength_summary.csv', summ)

# ---- 3. applied shifts per exposure
app = []
for r in adc:
    e = r['exp']; sh = {ch: shifts[e][ch] for ch in CH}
    lo, hi = FITBAND['blue']                                   # s3 offsets averaged over the s10 blue fit band
    ob_b = np.array([o for o in r['obs'] if lo <= o[0] <= hi])
    app.append(dict(exp=e, airmass=r['X'], parang=r['q'],
                    red_E=sh['red'][0], red_N=sh['red'][1], green_E=sh['green'][0], green_N=sh['green'][1],
                    blue_E=sh['blue'][0], blue_N=sh['blue'][1],
                    green_minus_red_E=sh['green'][0] - sh['red'][0], green_minus_red_N=sh['green'][1] - sh['red'][1],
                    blue_minus_red_E=sh['blue'][0] - sh['red'][0], blue_minus_red_N=sh['blue'][1] - sh['red'][1],
                    s3_meas_blue_E=float(np.mean(ob_b[:, 1])), s3_meas_blue_N=float(np.mean(ob_b[:, 2]))))
write_csv('adc_applied_shifts.csv', app)

# ---- cross-check against the star-aligned cube headers
hdr_ok = True
for ch in CH:
    h = fits.getheader(os.path.join(LA, T, 'star_aligned', f'{T}_staraligned_cube_{ch}.fits'))
    for j in range(1, int(h.get('NRSSFILE', len(exps))) + 1):
        if f'SHFT{j}' not in h: continue
        e, sE, sN = h[f'SHFT{j}'].split()
        if abs(float(sE) - shifts[e][ch][0]) > 0.006 or abs(float(sN) - shifts[e][ch][1]) > 0.006:
            hdr_ok = False; print(f'MISMATCH {ch} {e}: header {sE} {sN} vs json {shifts[e][ch]}')
print('applied shifts match cube SHFTn header cards:', hdr_ok)

# ---- plain-text tables (stdout + .txt) for printing
SUMM_HDR = ['λ [Å]', 'chan', 'ΔE mean', 'rms', 'min', 'max', 'ΔN mean', 'rms', 'min', 'max', 'DAR ΔE', 'DAR ΔN', 'resid ΔE', 'resid ΔN', 'σ fit']
summ_cells = [['%.0f' % s['lam'], s['channel']] + ['%+.2f' % s[k] for k in
              ['mean_dE', 'rms_dE', 'min_dE', 'max_dE', 'mean_dN', 'rms_dN', 'min_dN', 'max_dN',
               'mean_pred_dE', 'mean_pred_dN', 'mean_resid_E', 'mean_resid_N']] + ['%.2f' % s['median_err']] for s in summ]
for c in summ_cells:
    for i in (3, 7):
        c[i] = c[i].lstrip('+')                                         # rms columns are unsigned
APP_HDR = ['UT end', 'airm', 'parang', 'red ΔE', 'red ΔN', 'grn ΔE', 'grn ΔN', 'blue ΔE', 'blue ΔN',
           'g−r ΔE', 'g−r ΔN', 'b−r ΔE', 'b−r ΔN', 'meas b ΔE', 'meas b ΔN']
app_cells = [[a['exp'], '%.2f' % a['airmass'], '%+.0f°' % a['parang']] + ['%+.2f' % a[k] for k in
             ['red_E', 'red_N', 'green_E', 'green_N', 'blue_E', 'blue_N', 'green_minus_red_E', 'green_minus_red_N',
              'blue_minus_red_E', 'blue_minus_red_N', 's3_meas_blue_E', 's3_meas_blue_N']] for a in app]
bmr = np.array([[a['blue_minus_red_E'], a['blue_minus_red_N']] for a in app])
gmr = np.array([[a['green_minus_red_E'], a['green_minus_red_N']] for a in app])
app_cells.append(['mean', '', '', '', '', '', '', '', '', '%+.2f' % gmr[:, 0].mean(), '%+.2f' % gmr[:, 1].mean(),
                  '%+.2f' % bmr[:, 0].mean(), '%+.2f' % bmr[:, 1].mean(),
                  '%+.2f' % np.mean([a['s3_meas_blue_E'] for a in app]), '%+.2f' % np.mean([a['s3_meas_blue_N'] for a in app])])


def text_table(hdr, cells):
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *cells)]
    line = lambda r: '  '.join(str(x).rjust(n) for x, n in zip(r, w))
    return '\n'.join([line(hdr), '  '.join('-' * n for n in w)] + [line(c) for c in cells])


txt = ('ADC star offsets vs wavelength, %s (8 exposures, arcsec; relative to the red-channel mean)\n\n' % T
       + text_table(SUMM_HDR, summ_cells)
       + '\n\nApplied per-exposure, per-channel shifts in the star-aligned cubes (arcsec; b−r = effective ADC correction)\n\n'
       + text_table(APP_HDR, app_cells) + '\n')
open(os.path.join(ODIR, 'adc_tables.txt'), 'w').write(txt)
print(txt)

# ---- figure A: all exposures overlaid
cmap = plt.get_cmap('viridis')
fig, ax = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
for k, (comp, pk, nm) in enumerate([('dE', 'pred_dE', 'ΔE-W  (+ = East)'), ('dN', 'pred_dN', 'ΔN-S  (+ = North)')]):
    a = ax[k]
    for ch, (lo, hi) in FITBAND.items():
        a.axvspan(lo, hi, color=COL[ch], alpha=0.06, lw=0)
    for i, e in enumerate(exps):
        s = [r for r in rows if r['exp'] == e]
        a.errorbar([r['lam'] for r in s], [r[comp] for r in s], yerr=[r['err_E' if k == 0 else 'err_N'] for r in s],
                   color=cmap(i / (len(exps) - 1)), lw=0.8, ms=2.5, fmt='o-', alpha=0.75, capsize=0, label=e)
    a.plot(lams, [s['mean_' + comp] for s in summ], 'k-', lw=2.2, label='mean measured')
    a.plot(lams, [s['mean_' + pk] for s in summ], 'k--', lw=1.3, label='uncorrected DAR (no ADC), mean')
    for ch, (lo, hi) in FITBAND.items():
        m = np.mean([removed(e, ch)[k] for e in exps])
        a.plot([lo, hi], [m, m], color=COL[ch], lw=3.5, solid_capstyle='butt',
               label='applied correction, %s (mean)' % ch)
    a.axhline(0, color='0.5', lw=0.5)
    a.set_ylabel(nm + '  ["]'); a.set_ylim(-3, 3)
ax[0].legend(fontsize=6.5, ncol=3, loc='lower right')
ax[1].set_xlabel('wavelength [Å]')
fig.suptitle(f'{T}: field-star centroid vs wavelength, relative to red (ADC test)\n'
             'flat = ADC working;  coloured bars = rigid offset removed per channel in the star-aligned cubes', fontsize=10)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(ODIR, f'ADC_shift_vs_wavelength.{ext}'), dpi=130)
figA = fig

# ---- figure B: one panel pair per exposure, with that exposure's applied steps and residuals
fig, ax = plt.subplots(4, 4, figsize=(11.0, 9.6), sharex=True, sharey=True)
for i, e in enumerate(exps):
    s = [r for r in rows if r['exp'] == e]; r0 = adc[i]
    for k, (comp, pk, rk) in enumerate([('dE', 'pred_dE', 'resid_E'), ('dN', 'pred_dN', 'resid_N')]):
        a = ax[2 * (i // 4) + k, i % 4]
        a.errorbar([r['lam'] for r in s], [r[comp] for r in s], yerr=[r['err_E' if k == 0 else 'err_N'] for r in s],
                   fmt='o', ms=3, color='k', capsize=0, label='measured')
        a.plot([r['lam'] for r in s], [r[pk] for r in s], 'k--', lw=1, label='no-ADC DAR')
        a.plot([r['lam'] for r in s], [r[rk] for r in s], 'x', color='0.55', ms=4, label='residual after shift')
        for ch, (lo, hi) in FITBAND.items():
            a.plot([lo, hi], [removed(e, ch)[k]] * 2, color=COL[ch], lw=3, solid_capstyle='butt')
        a.axhline(0, color='0.6', lw=0.5); a.set_ylim(-3, 3)
        if i % 4 == 0: a.set_ylabel(('ΔE' if k == 0 else 'ΔN') + ' ["]')
        if k == 0: a.set_title(f"{e}  X={r0['X']:.2f}  q={r0['q']:+.0f}°", fontsize=8)
        if 2 * (i // 4) + k == 3: a.set_xlabel('λ [Å]')
ax[0, 0].legend(fontsize=6, loc='lower right')
fig.suptitle(f'{T}: per-exposure star offset vs wavelength (rows: ΔE then ΔN), applied per-channel shift (bars), residual (×)', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(ODIR, 'ADC_shift_vs_wavelength_per_exposure.png'), dpi=110)
figB = fig


# ---- printable PDF: figures + tables on A4
def table_page(pdf, title, hdr, cells, note, landscape=True):
    W, H = (11.69, 8.27) if landscape else (8.27, 11.69)
    f = plt.figure(figsize=(W, H)); f.text(0.04, 0.95, title, fontsize=12, weight='bold', va='top')
    a = f.add_axes([0.03, 0.12, 0.94, 0.78]); a.axis('off')
    tb = a.table(cellText=cells, colLabels=hdr, loc='upper center', cellLoc='right')
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.35)
    for (rr, cc), cell in tb.get_celld().items():
        if rr == 0: cell.set_facecolor('#e8e8e8'); cell.set_text_props(weight='bold')
    f.text(0.04, 0.03, note, fontsize=7.5, va='bottom', wrap=True)
    pdf.savefig(f); plt.close(f)


with PdfPages(os.path.join(ODIR, 'ADC_tables_printable.pdf')) as pdf:
    pdf.savefig(figA); pdf.savefig(figB)
    table_page(pdf, f'{T}: star offset vs wavelength, statistics over 8 exposures (arcsec)', SUMM_HDR, summ_cells,
               'Offsets: star centroid in each wavelength bin minus the mean of the red bins, same exposure (s3_adc.py; 2-D Gaussian on the fibres, '
               'SKYSUB counts, PSF width fixed from red). DAR = predicted uncorrected atmospheric dispersion (no ADC), re-referenced to red the same way. '
               'resid = measured minus the rigid per-channel offset removed in the star-aligned cubes (s10_staralign.py). σ fit = median centroid error.')
    table_page(pdf, f'{T}: shifts applied to each exposure and channel in the star-aligned cubes (arcsec)', APP_HDR, app_cells,
               'Shifts are added to the fibre RA/Dec so the field star lands on its Gaia DR3 position (01:11:53.42 −56:37:45.6); stored as SHFTn in each '
               'star-aligned cube header. b−r and g−r are the chromatic part of the correction. meas b = s3 blue offset relative to red, averaged over '
               'the 3930–4590 Å bins (inside the blue fit band); it should be ≈ −(b−r). Red/green/blue fit bands 7000–9500 / 5000–6800 / 3800–4600 Å. SPT-CLJ0158-5500 has no star: no ADC correction was applied there.')
print('wrote', ODIR)
