"""QA tables and figures. Every figure is saved as qa/NN_name.png and appended to adc_qa.pdf."""

import csv
import json
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from adcfit.io import tangent_offsets                 # noqa: E402

logger = logging.getLogger(__name__)

CH_COL = {'blue': '#1f5fbf', 'green': '#2a9d3a', 'red': '#c0392b'}
plt.rcParams.update({'font.size': 8.5, 'axes.linewidth': 0.8})


# ----------------------------------------------------------------------------- tables
def short_id(eid):
    """'LLAMAS_2026-10-16_00-31-08.4_SCI24_..._extract' -> '00-31-08.4' (else the last 12 chars)."""
    import re
    m = re.search(r'\d{2}-\d{2}-\d{2}\.\d', str(eid))
    return m.group(0) if m else str(eid)[-12:]


def _fmt(v, f):
    if isinstance(v, (float, np.floating)):
        return '-' if not np.isfinite(v) else (f % v)
    return str(v)


def text_table(rows, cols, fmts=None, title=''):
    fmts = fmts or {}
    cells = [[short_id(r.get(c, '')) if c == 'exp' else _fmt(r.get(c, ''), fmts.get(c, '%.3f'))
              for c in cols] for r in rows]
    w = [max([len(c)] + [len(x[i]) for x in cells]) for i, c in enumerate(cols)]
    line = lambda xs: '  '.join(str(x).rjust(n) for x, n in zip(xs, w))   # noqa: E731
    body = [line(cols), '  '.join('-' * n for n in w)] + [line(c) for c in cells]
    return (title + '\n' if title else '') + '\n'.join(body)


def emit_table(path, rows, cols, fmts=None, title='', echo=True):
    """Write a .txt table (and print it)."""
    txt = text_table(rows, cols, fmts, title)
    with open(path, 'w') as fh:
        fh.write(txt + '\n')
    if echo:
        print('\n' + txt)
    return txt


def write_csv(path, rows):
    if not rows:
        open(path, 'w').close()
        return path
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (('%.6g' % v) if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()})
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def write_json(path, obj):
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1, default=_json_default)
    return path


class QAReport:
    def __init__(self, outdir, pdf=True):
        self.dir = os.path.join(outdir, 'qa')
        os.makedirs(self.dir, exist_ok=True)
        self.pdf = PdfPages(os.path.join(outdir, 'adc_qa.pdf')) if pdf else None
        self.n = 0
        self.files = []

    def save(self, fig, name):
        self.n += 1
        p = os.path.join(self.dir, f'{self.n:02d}_{name}.png')
        fig.savefig(p, dpi=110)
        if self.pdf is not None:
            self.pdf.savefig(fig)
        plt.close(fig)
        self.files.append(p)
        return p

    def text(self, title, text, name):
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.96, title, fontsize=13, weight='bold', va='top')
        fig.text(0.04, 0.91, text, fontsize=7.5, family='monospace', va='top')
        return self.save(fig, name)

    def close(self):
        if self.pdf is not None:
            self.pdf.close()


def _shade_channels(ax, ranges):
    for c, (lo, hi) in ranges.items():
        ax.axvspan(lo, hi, color=CH_COL.get(c, '0.5'), alpha=0.06, lw=0)


def _ecol(n):
    cm = plt.get_cmap('viridis')
    return [cm(i / max(n - 1, 1)) for i in range(n)]


# ----------------------------------------------------------------------------- stage 1
def fig_inventory(infos, geoms):
    x = np.arange(len(infos))
    lab = [short_id(i.exp_id) for i in infos]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(x, [g.airmass_start for g in geoms], 's-', label='computed start')
    ax[0].plot(x, [g.airmass_mean for g in geoms], 'o-', label='exposure mean')
    ax[0].plot(x, [g.airmass_end for g in geoms], '^-', label='computed end')
    ax[0].plot(x, [i.airmass_start for i in infos], 'k+', ms=10, label='AIRMASS (hdr)')
    ax[0].plot(x, [i.airmass_end for i in infos], 'kx', ms=7, label='AIRMASSE (hdr)')
    ax[0].set_ylabel('airmass')
    ax[1].plot(x, [g.q_start for g in geoms], 's-', label='computed start')
    ax[1].plot(x, [g.q_mean for g in geoms], 'o-', label='mean dispersion direction')
    ax[1].plot(x, [g.q_end for g in geoms], '^-', label='computed end')
    ax[1].plot(x, [i.parang_end for i in infos], 'kx', ms=7, label='TEL PARANGL (hdr, end)')
    ax[1].set_ylabel('parallactic angle q [deg]')
    ax[2].plot(x, [g.H_start for g in geoms], 's-', label='HA start')
    ax[2].plot(x, [g.H_end for g in geoms], '^-', label='HA end')
    ax[2].plot(x, [((i.ha_end + 12) % 24 - 12) for i in infos], 'kx', ms=7, label='TEL HA (hdr)')
    ax[2].set_ylabel('hour angle [h]')
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(lab, rotation=45, fontsize=7); a.legend(fontsize=6.5)
        a.grid(alpha=0.3)
    fig.suptitle('Stage 1: exposure geometry (computed vs header)')
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- stage 2/3
def _stamp(img, w2, ra, dec, half_pix):
    px, py = w2.world_to_pixel_values(ra, dec)
    x0, y0 = int(round(float(px))), int(round(float(py)))
    ny, nx = img.shape
    out = np.full((2 * half_pix + 1, 2 * half_pix + 1), np.nan)
    ys, xs = slice(max(0, y0 - half_pix), min(ny, y0 + half_pix + 1)), slice(max(0, x0 - half_pix), min(nx, x0 + half_pix + 1))
    oy, ox = ys.start - (y0 - half_pix), xs.start - (x0 - half_pix)
    sub = img[ys, xs]
    out[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
    return out


def fig_registration(reg, images, combined, grid, pixscale, star_radec):
    """Per-exposure broad-band stamps on a common grid + per-channel centroid offsets + FWHM."""
    ids = reg.exp_ids
    n = len(ids)
    half = int(round(7.0 / pixscale))
    ext = (half + 0.5) * pixscale
    ncol = min(n + 1, 6)
    nrow_img = int(np.ceil((n + 1) / ncol))
    fig = plt.figure(figsize=(2.4 * ncol, 2.5 * nrow_img + 4.2))
    gs = fig.add_gridspec(nrow_img + 1, ncol, height_ratios=[1] * nrow_img + [1.7])
    w2 = grid[0]
    for k, (lab, img) in enumerate([('combined', combined)] + list(zip(ids, images))):
        a = fig.add_subplot(gs[k // ncol, k % ncol])
        st = _stamp(img, w2, *star_radec, half)
        if np.isfinite(st).any():
            lo, hi = np.nanpercentile(st, [5, 99.7])
            a.imshow(st, origin='lower', cmap='magma', extent=(-ext, ext, -ext, ext), vmin=lo, vmax=hi)
        a.plot(0, 0, 'c+', ms=10)
        a.set_title(short_id(lab), fontsize=7); a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(gs[nrow_img, :ncol // 2])
    cols = _ecol(n)
    for j, b in enumerate(reg.bins):
        ok = (reg.flags[:, j] & 1024) == 0
        if not ok.any():
            continue
        dE = reg.E[:, j] - np.nanmedian(reg.E[ok, j])
        dN = reg.N[:, j] - np.nanmedian(reg.N[ok, j])
        mk = {'blue': 's', 'green': 'o', 'red': '^'}[b.channel]
        for e in range(n):
            if ok[e]:
                a.plot(dE[e], dN[e], mk, color=cols[e], mec=CH_COL[b.channel], ms=6,
                       label=(short_id(ids[e]) if b.channel == 'red' else None))
    a.add_patch(plt.Circle((0, 0), 0.25, fill=False, ls=':', color='0.5'))
    a.axhline(0, color='0.6', lw=0.5); a.axvline(0, color='0.6', lw=0.5); a.set_aspect('equal', 'datalim')
    a.set_xlabel('dE ["] (vs median over exposures)'); a.set_ylabel('dN ["]')
    a.set_title('star position per exposure (square blue, circle green, triangle red)', fontsize=8)
    a.legend(fontsize=6, ncol=2)
    a = fig.add_subplot(gs[nrow_img, ncol // 2:])
    for j, b in enumerate(reg.bins):
        a.plot(range(n), reg.fwhm[:, j], 'o-', color=CH_COL[b.channel], label=b.channel)
    a.set_xticks(range(n)); a.set_xticklabels([short_id(i) for i in ids], rotation=45, fontsize=7)
    a.set_ylabel('FWHM ["]'); a.legend(fontsize=7); a.grid(alpha=0.3)
    a.set_title('broad-band FWHM per exposure', fontsize=8)
    fig.suptitle('Stage 2: registration of the exposures (uncorrected, fibre positions as delivered)')
    fig.tight_layout()
    return fig


def fig_star(img, grid, cands, chosen, star_radec):
    w2 = grid[0]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    if np.isfinite(img).any():
        lo, hi = np.nanpercentile(img, [5, 99.7])
        ax.imshow(img, origin='lower', cmap='gray', vmin=lo, vmax=hi)
    for c in cands:
        px, py = w2.world_to_pixel_values(c['ra'], c['dec'])
        col = 'lime' if c['id'] == chosen else ('y' if c['qualifies'] else 'r')
        ax.add_patch(plt.Circle((px, py), 5, fill=False, color=col, lw=1.2))
        lab = f"{c['id']}: S/N {c['snr']:.0f} FWHM {c['fwhm']:.2f}\" n={c['n_exp']}"
        if c.get('reason'):
            lab += '\n' + c['reason']
        ax.text(px + 6, py + 6, lab, color=col, fontsize=7)
    if star_radec is not None:
        px, py = w2.world_to_pixel_values(*star_radec)
        ax.plot(px, py, 'c+', ms=18, mew=1.5)
    head = 'Stage 3: NO STAR SELECTED' if chosen is None and star_radec is None else 'Stage 3: reference star'
    ax.set_title(head + ' (cyan +; green = chosen, yellow = qualifying, red = rejected)', fontsize=9)
    ax.set_xlabel('x [pix] (East left)'); ax.set_ylabel('y [pix] (North up)')
    fig.tight_layout()
    return fig


def fig_star_channels(imgs, grid, cands, chosen):
    """Per-channel broad-band images with each candidate at its own per-channel fitted position
    (green circle = passes the per-channel check, red = fails it)."""
    w2 = grid[0]
    chans = [c for c in ('blue', 'green', 'red') if c in imgs]
    fig, axs = plt.subplots(1, len(chans), figsize=(5.2 * len(chans), 5.4), squeeze=False)
    for ax, ch in zip(axs[0], chans):
        img = imgs[ch]
        if np.isfinite(img).any():
            lo, hi = np.nanpercentile(img, [5, 99.7])
            ax.imshow(img, origin='lower', cmap='gray', vmin=lo, vmax=hi)
        for c in cands:
            ra, dec = c.get(f'ra_{ch}', np.nan), c.get(f'dec_{ch}', np.nan)
            if not (np.isfinite(ra) and np.isfinite(dec)):
                ra, dec = c['ra'], c['dec']
            px, py = w2.world_to_pixel_values(ra, dec)
            col = 'lime' if c.get('chan_ok') else 'r'
            lab = f"{c['id']}: S/N {c.get(f'snr_{ch}', np.nan):.0f}"
            if c['id'] == chosen:
                ax.add_patch(plt.Circle((px, py), 6.5, fill=False, color='c', lw=2.5))
                lab = 'SELECTED ' + lab
                col = 'c'
            else:
                ax.add_patch(plt.Circle((px, py), 5, fill=False, color=col, lw=0.8))
            ax.text(px + 6, py + 6, lab, color=col, fontsize=7, weight='bold' if c['id'] == chosen else 'normal')
        ax.set_title(f'{ch} broad band', fontsize=9)
        ax.set_xlabel('x [pix] (East left)')
    axs[0][0].set_ylabel('y [pix] (North up)')
    sel = 'the star used: cyan circle "SELECTED"' if chosen is not None else 'NO STAR SELECTED'
    fig.suptitle(f'Stage 3: per-channel check of each candidate -- {sel}. Other candidates: green = '
                 'passes the per-channel check, red = fails (pass/fail does not mean selected)', fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- stage 4
def fig_centroids(cents, fatal, ranges, frame='sky'):
    lam = cents.lam
    n = len(cents.exp_ids)
    cols = _ecol(n)
    good = cents.good(fatal)
    A, B = (cents.E, cents.N) if frame == 'sky' else (cents.xi, cents.yi)
    labs = ('dE (East) ["]', 'dN (North) ["]') if frame == 'sky' else ('x_inst ["]', 'y_inst ["]')
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for k, (arr, lab) in enumerate(zip((A, B), labs)):
        a = ax[k]
        _shade_channels(a, ranges)
        for e in range(n):
            ref = np.nanmedian(arr[e][good[e] & (lam > 7000)]) if np.any(good[e] & (lam > 7000)) else np.nanmedian(arr[e])
            y = arr[e] - ref
            err = cents.sE[e] if k == 0 else cents.sN[e]
            a.errorbar(lam[good[e]], y[good[e]], yerr=(err[good[e]] if frame == 'sky' else None),
                       fmt='o-', ms=3, lw=0.7, color=cols[e], label=short_id(cents.exp_ids[e]))
            bad = ~good[e] & np.isfinite(y)
            a.plot(lam[bad], y[bad], 'x', color=cols[e], ms=6)
        a.axhline(0, color='0.5', lw=0.5)
        a.set_ylabel(lab + ' rel. red'); a.grid(alpha=0.3)
    ax[0].legend(fontsize=6.5, ncol=4)
    ax[1].set_xlabel('wavelength [A]')
    fig.suptitle(f'Stage 4: star centroid vs wavelength per exposure ({"sky E/N" if frame == "sky" else "instrument x/y"} '
                 'frame; x = flagged bins)')
    fig.tight_layout()
    return fig


def fig_fwhm(cents, fatal, ranges):
    lam = cents.lam
    good = cents.good(fatal)
    cols = _ecol(len(cents.exp_ids))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    _shade_channels(ax[0], ranges)
    for e in range(len(cents.exp_ids)):
        m = good[e]
        ax[0].plot(lam[m], cents.fwhm[e][m], 'o-', ms=3, color=cols[e], label=short_id(cents.exp_ids[e]))
        if np.any(m & (lam > 7000)):
            f0 = np.nanmedian(cents.fwhm[e][m & (lam > 7000)])
            l0 = np.nanmedian(lam[m & (lam > 7000)])
            g = np.linspace(3500, 9800, 50)
            ax[0].plot(g, f0 * (g / l0) ** -0.2, ':', color=cols[e], lw=0.8)
    ax[0].set_xlabel('wavelength [A]'); ax[0].set_ylabel('FWHM ["]'); ax[0].grid(alpha=0.3)
    ax[0].set_title('PSF FWHM vs wavelength (dotted: lambda^-0.2 from red)', fontsize=9)
    ax[0].legend(fontsize=6.5, ncol=2)
    _shade_channels(ax[1], ranges)
    for e in range(len(cents.exp_ids)):
        ax[1].plot(lam, cents.snr[e], 'o', ms=3, color=cols[e])
    ax[1].set_yscale('log'); ax[1].set_xlabel('wavelength [A]'); ax[1].set_ylabel('fit S/N (amplitude)')
    ax[1].grid(alpha=0.3, which='both'); ax[1].set_title('per-bin S/N', fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- stage 5
def fig_fit_exposure(e, cents, fit, poly, info, geom, fatal, ranges, holdout, lam_grid):
    from adcfit.dar import predicted_dar
    lam = cents.lam
    good = cents.good(fatal)[e]
    roles = np.array([b.role for b in cents.bins])
    ch = np.array([b.channel for b in cents.bins])
    off = np.c_[cents.E[e], cents.N[e]]
    err = np.c_[cents.sE[e], cents.sN[e]]
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1, 2.2, 1]))
    ok = fit.status != 'failed'
    m = fit.offset(lam_grid) if ok else None
    s = fit.offset_err(lam_grid) if ok else None
    pr = None
    if geom.ok and ok:
        pr = predicted_dar(info, geom, lam_grid, fit.lam_ref) + fit.offset([fit.lam_ref])[0][None, :]
    used = fit.used if fit.used is not None else np.zeros(lam.size, bool)
    clipped = fit.clipped if fit.clipped is not None else np.zeros(lam.size, bool)
    for k, nm in enumerate(('E', 'N')):
        a, r = ax[2 * k], ax[2 * k + 1]
        _shade_channels(a, ranges); _shade_channels(r, ranges)
        if ok:
            a.plot(lam_grid, m[:, k], 'k-', lw=1.4, label=f'refractivity model (order {fit.order}, {fit.status}{", c1 GATE FAILED" if fit.gate_failed else ""})')
            a.fill_between(lam_grid, m[:, k] - s[:, k], m[:, k] + s[:, k], color='0.6', alpha=0.35, lw=0)
            if poly is not None:
                a.plot(lam_grid, poly.offset(lam_grid)[:, k], '--', color='0.35', lw=0.9,
                       label='polynomial in lambda (comparison)')
            if pr is not None:
                a.plot(lam_grid, pr[:, k], ':', color='m', lw=1.2, label='no-ADC atmosphere (predicted)')
        for j in range(lam.size):
            if not np.isfinite(off[j, k]):
                continue
            col = CH_COL[ch[j]]
            if used[j]:
                kw = dict(fmt='o', color=col, ms=5)
            elif clipped[j]:
                kw = dict(fmt='x', color='k', ms=7)
            elif roles[j] == 'fit':
                kw = dict(fmt='o', color='0.7', mfc='none', ms=4)
            elif roles[j] == 'overlap':
                kw = dict(fmt='D', color=col, mfc='none', ms=4)
            else:
                kw = dict(fmt='o', color=col, mfc='none' if good[j] else '0.85', ms=5)
            a.errorbar(lam[j], off[j, k], yerr=err[j, k], **kw)
            if ok:
                r.errorbar(lam[j], off[j, k] - fit.offset([lam[j]])[0, k], yerr=err[j, k], **kw)
        a.set_ylabel(f'd{nm} ["]'); a.grid(alpha=0.3)
        r.axhline(0, color='k', lw=0.6); r.set_ylabel('resid ["]'); r.grid(alpha=0.3)
        if ok:
            rr = np.nanmax(np.abs([off[j, k] - fit.offset([lam[j]])[0, k] for j in range(lam.size)
                                   if np.isfinite(off[j, k]) and good[j]] + [0.05]))
            r.set_ylim(-1.3 * rr, 1.3 * rr)
    ax[0].legend(fontsize=7, loc='best')
    ax[3].set_xlabel('wavelength [A]  (filled = fitted green/red; hollow = held-out blue; x = clipped; '
                     'grey = flagged; diamonds = overlap)')
    hd = holdout or {}
    c1 = fit.c1 if ok else [np.nan, np.nan]
    fig.suptitle(f'{fit.exp_id}  X={geom.airmass_mean:.3f}  q={geom.q_mean:+.1f} deg  '
                 f'c1=({c1[0]:+.3f}, {c1[1]:+.3f}) "/1e-6  chi2r=({fit.chi2r[0]:.2f}, {fit.chi2r[1]:.2f})\n'
                 f'blue hold-out rms E {hd.get("rms_E", np.nan):.3f}"  N {hd.get("rms_N", np.nan):.3f}"  '
                 f'(chi2/n {hd.get("chi2n_E", np.nan):.2f}, {hd.get("chi2n_N", np.nan):.2f})', fontsize=9)
    fig.tight_layout()
    return fig


def fig_fit_summary(fits, geoms, infos, chrom, lam_grid, ranges):
    from adcfit.dar import predicted_dar
    n = len(fits)
    cols = _ecol(n)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for k, nm in enumerate(('E', 'N')):
        a = ax[0, k]
        _shade_channels(a, ranges)
        for e, f in enumerate(fits):
            if f.status == 'failed':
                continue
            m = f.offset(lam_grid) - f.offset([f.lam_ref])[0][None, :]
            a.plot(lam_grid, m[:, k], '-', color=cols[e], lw=1.2, label=f'{short_id(f.exp_id)} ({f.status}{", GATE" if f.gate_failed else ""})')
            if geoms[e].ok:
                a.plot(lam_grid, predicted_dar(infos[e], geoms[e], lam_grid, f.lam_ref)[:, k], ':',
                       color=cols[e], lw=0.9)
        a.axhline(0, color='0.5', lw=0.5); a.grid(alpha=0.3)
        a.set_xlabel('wavelength [A]'); a.set_ylabel(f'fitted d{nm}(lambda) - d{nm}(lambda_ref) ["]')
        a.set_title(f'fitted chromatic shift, {nm} (solid) vs no-ADC prediction (dotted)', fontsize=9)
    ax[0, 0].legend(fontsize=6.5)
    a = ax[1, 0]
    for e, r in enumerate(chrom):
        if 'c1_E' not in r:
            continue
        a.annotate('', xy=(r['c1_E'], r['c1_N']), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=cols[e], lw=1.6))
        if 'p_E' in r:
            a.annotate('', xy=(r['p_E'], r['p_N']), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color=cols[e], lw=0.9, ls='--'))
    lim = max([abs(r.get(k, 0)) for r in chrom for k in ('c1_E', 'c1_N', 'p_E', 'p_N')] + [0.05]) * 1.2
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
    a.axhline(0, color='0.6', lw=0.5); a.axvline(0, color='0.6', lw=0.5); a.grid(alpha=0.3)
    a.set_xlabel('c1 East ["/1e-6]'); a.set_ylabel('c1 North ["/1e-6]')
    a.set_title('dispersion vectors: measured (solid) vs no-ADC prediction (dashed)', fontsize=9)
    a = ax[1, 1]
    x = [r.get('airmass', np.nan) for r in chrom]
    a.plot(x, [r.get('amp_ratio', np.nan) for r in chrom], 'ko', label='|c1| / |predicted| (ADC residual fraction)')
    a.plot(x, [r.get('along_frac', np.nan) for r in chrom], 'bs', mfc='none', label='component along prediction')
    a.set_xlabel('mean airmass'); a.set_ylabel('fraction of uncorrected dispersion'); a.grid(alpha=0.3)
    a2 = a.twinx()
    a2.plot(x, [r.get('dir_resid_deg', np.nan) for r in chrom], 'r^', label='direction residual')
    a2.set_ylabel('measured - predicted direction [deg]', color='r')
    a.legend(fontsize=7, loc='upper left'); a2.legend(fontsize=7, loc='upper right')
    a.set_title('how much of the atmospheric dispersion remains', fontsize=9)
    fig.suptitle('Stage 5: per-exposure ADC fits (green+red fitted, extrapolated to blue)')
    fig.tight_layout()
    return fig


def fig_blue_relative(rows, common, exp_ids):
    """Relative held-out test: measured minus the median over exposures vs the model treated the same
    way (broad band = large markers, single bins = small), per exposure; residuals below."""
    cols = _ecol(len(exp_ids))
    idx = {e: i for i, e in enumerate(exp_ids)}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for k, nm in enumerate(('E', 'N')):
        a, r = ax[0, k], ax[1, k]
        for row in rows:
            i = idx.get(row['exp'], 0)
            broad = row['source'] == 'broad'
            kw = dict(color=cols[i], ms=8 if broad else 4, mfc=cols[i] if broad else 'none')
            a.plot(row[f'model_{nm}'], row[f'meas_{nm}'], 'o', **kw)
            r.plot(i, row[f'resid_{nm}'], 'o', **kw)
        lim = max([abs(row[f'{p}_{nm}']) for row in rows for p in ('model', 'meas')] + [0.1]) * 1.15
        a.plot([-lim, lim], [-lim, lim], 'k-', lw=0.6)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal'); a.grid(alpha=0.3)
        a.set_xlabel(f'model d{nm} - median over exposures ["]'); a.set_ylabel(f'measured d{nm} - median ["]')
        a.set_title(f'{nm}: relative held-out position (on the line = predicted)', fontsize=9)
        r.axhline(0, color='k', lw=0.6); r.grid(alpha=0.3)
        r.set_xticks(range(len(exp_ids))); r.set_xticklabels([short_id(e) for e in exp_ids], rotation=45, fontsize=7)
        r.set_ylabel(f'relative residual d{nm} ["]')
    txt = '; '.join(f"common {c} offset E {cm['E']:+.3f}\" N {cm['N']:+.3f}\"" for c, cm in (common or {}).items())
    fig.suptitle('Stage 5: held-out channel vs model, relative to the median over exposures (large = broad band, '
                 f'small = single bins)\n{txt}', fontsize=10)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- stage 6/7
def fig_slabs(corr_by_channel, exp_ids):
    """corr_by_channel: {channel: (wl, corr (nexp, nw, 2), slabs)}"""
    cols = _ecol(len(exp_ids))
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for c, (wl, corr, slabs) in corr_by_channel.items():
        for k in (0, 1):
            for e in range(corr.shape[0]):
                ax[k].plot(wl, corr[e, :, k], '-', color=cols[e], lw=0.8,
                           label=(short_id(exp_ids[e]) if (k == 0 and c == list(corr_by_channel)[0]) else None))
                stair = np.concatenate([np.full(s.i1 - s.i0, s.shift[e, k]) for s in slabs])
                ax[k].plot(wl, stair, '-', color=cols[e], lw=0.5, alpha=0.6, drawstyle='steps-mid')
            for s in slabs:
                ax[k].axvline(wl[s.i0], color=CH_COL[c], lw=0.3, alpha=0.5)
    ax[0].set_ylabel('applied shift E ["]'); ax[1].set_ylabel('applied shift N ["]')
    ax[1].set_xlabel('wavelength [A]  (thin vertical lines = slab boundaries)')
    ax[0].legend(fontsize=6.5, ncol=4)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle('Stage 6: correction applied to the fibre positions (smooth) and its slab approximation')
    fig.tight_layout()
    return fig


def cube_star_metrics(cube, ra, dec, bins):
    """Centroid offset (E, N) from (ra, dec), FWHM and r<1"/r<3" concentration of the star per
    wavelength bin [(lo, hi, lam_eff, channel)], from the median image of the bin's planes."""
    from llamas_pyjamas.Combine.spectrum import fit_gaussian_image
    w2 = cube.wcs.celestial
    pix = float(cube.meta.get('PIXSCALE', 0.5))
    ny, nx = cube.data.shape[1:]
    yy, xx = np.mgrid[:ny, :nx]
    px, py = [float(v) for v in w2.world_to_pixel_values(ra, dec)]
    out = []
    for lo, hi, lam, ch in bins:
        sel = (cube.wave >= lo) & (cube.wave < hi)
        row = dict(channel=ch, lam=lam, E=np.nan, N=np.nan, fwhm=np.nan, conc=np.nan)
        if sel.any():
            with np.errstate(all='ignore'):
                img = np.nanmedian(cube.data[sel], axis=0)
            g = fit_gaussian_image(np.nan_to_num(img), px, py, max(4.0, 2.5 / pix))
            if g is not None:
                gr, gd = w2.pixel_to_world_values(g.x_mean.value, g.y_mean.value)
                E, N = tangent_offsets(gr, gd, ra, dec)
                cx, cy = g.x_mean.value, g.y_mean.value
                r = np.hypot(xx - cx, yy - cy) * pix
                bg = np.nanmedian(img[(r > 4) & (r < 6)])
                f1, f3 = np.nansum((img - bg)[r <= 1.0]), np.nansum((img - bg)[r <= 3.0])
                row.update(E=float(E), N=float(N), conc=float(f1 / f3) if f3 > 0 else np.nan,
                           fwhm=float(2.354820045 * np.sqrt(abs(g.x_stddev.value * g.y_stddev.value)) * pix))
        out.append(row)
    return out


def fig_validation(metrics, resid_rows, anchor, ranges):
    """metrics: {variant: [rows]} from cube_star_metrics over all channels; resid_rows: fibre-space
    residuals after correction (long table)."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    sty = {'noadc': ('o', '0.4', 'uncorrected'), 'adc': ('s', 'C3', 'ADC-corrected'),
           'regonly': ('^', 'C0', 'registration only')}
    for v, rows in metrics.items():
        mk, col, lab = sty.get(v, ('d', 'k', v))
        lam = np.array([r['lam'] for r in rows])
        for k, key in enumerate(('E', 'N')):
            y = np.array([r[key] for r in rows]) - (anchor[k] if anchor is not None else 0.0)
            ax[0, k].plot(lam, y, mk + '-', color=col, ms=4, lw=0.8, label=lab)
        ax[1, 0].plot(lam, [r['fwhm'] for r in rows], mk + '-', color=col, ms=4, lw=0.8, label=lab)
        ax[1, 1].plot(lam, [r['conc'] for r in rows], mk + '-', color=col, ms=4, lw=0.8, label=lab)
    for a, t in zip(ax.ravel(), ('stacked-cube star dE vs anchor ["]', 'stacked-cube star dN vs anchor ["]',
                                 'stacked-cube FWHM ["]', 'concentration f(r<1")/f(r<3")')):
        _shade_channels(a, ranges)
        a.set_title(t, fontsize=9); a.grid(alpha=0.3); a.set_xlabel('wavelength [A]'); a.legend(fontsize=7)
    fig.suptitle('Stage 7: validation in the stacked cubes (flat centroid / smaller FWHM = success)')
    fig.tight_layout()
    return fig


def fig_resid_after(after, anchor, fatal, ranges):
    lam = after.lam
    good = after.good(fatal)
    cols = _ecol(len(after.exp_ids))
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for k, arr in enumerate((after.E, after.N)):
        _shade_channels(ax[k], ranges)
        for e in range(len(after.exp_ids)):
            y = arr[e] - anchor[k]
            ax[k].plot(lam[good[e]], y[good[e]], 'o-', ms=3, lw=0.7, color=cols[e],
                       label=short_id(after.exp_ids[e]))
        ax[k].axhline(0, color='k', lw=0.6); ax[k].grid(alpha=0.3)
        ax[k].set_ylabel(('dE' if k == 0 else 'dN') + ' after correction ["]')
    ax[0].legend(fontsize=6.5, ncol=4); ax[1].set_xlabel('wavelength [A]')
    fig.suptitle('Stage 7: per-exposure star position after applying the correction (fibre space); '
                 'blue is the held-out test')
    fig.tight_layout()
    return fig


def fig_stamps(stamps, pixscale):
    """stamps: list of (label, before, after) 2-D arrays centred on the star."""
    n = len(stamps)
    fig, ax = plt.subplots(3, n, figsize=(2.6 * n, 7.6), squeeze=False)
    for j, (lab, b, a) in enumerate(stamps):
        half = (b.shape[0] // 2 + 0.5) * pixscale
        ext = (-half, half, -half, half)
        vmax = np.nanmax([np.nanmax(b), np.nanmax(a)]) if np.isfinite(b).any() else 1
        for i, (img, t) in enumerate(((b, 'uncorrected'), (a, 'corrected'), (a - b, 'corrected - uncorr.'))):
            if i < 2:
                ax[i, j].imshow(img, origin='lower', cmap='magma', extent=ext, vmin=0, vmax=vmax)
            else:
                m = np.nanmax(np.abs(img)) if np.isfinite(img).any() else 1
                ax[i, j].imshow(img, origin='lower', cmap='RdBu_r', extent=ext, vmin=-m, vmax=m)
            ax[i, j].plot(0, 0, 'c+', ms=9)
            ax[i, j].set_title(f'{lab}\n{t}', fontsize=7); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle('Stage 7: star stamps in the stacked cubes (cyan + = anchor position)')
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- standard / sensfunc
def fig_sensfunc_diag(capture, r_tot, lam_ref, verdict, sens=None):
    """Captured fraction f = C_ap / C_tot behind the sensfunc, and the colour error 1/f it implies."""
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.6, 1.4]))
    for c in ('blue', 'green', 'red'):
        rr = [r for r in capture if r['channel'] == c]
        if not rr:
            continue
        lam = np.array([r['lam'] for r in rr])
        for k, R in enumerate(r_tot):
            ax[0].plot(lam, [r[f'f_R{R:g}'] for r in rr], '-o', ms=2.5, color=CH_COL[c], alpha=0.4 + 0.6 * k / max(len(r_tot) - 1, 1),
                       lw=1, label=f'{c}: total within {R:g}"' if c == 'green' else None)
        if 'fmod_track' in rr[0]:
            ax[0].plot(lam, [r['fmod_track'] for r in rr], 'k--', lw=0.9, label='PSF model, star on its track' if c == 'green' else None)
            ax[0].plot(lam, [r['fmod_noADC'] for r in rr], 'k:', lw=0.9, label='PSF model, star held at lambda_ref' if c == 'green' else None)
        ax[1].plot(lam, 100 * np.array([r['colour_err_rel_ref'] for r in rr]), '-o', ms=2.5, color=CH_COL[c])
        ax[2].plot(lam, [r['track_E'] for r in rr], '-', color=CH_COL[c])
        ax[2].plot(lam, [r['track_N'] for r in rr], '--', color=CH_COL[c])
    ax[0].axhline(1, color='k', lw=0.6)
    ax[0].set_ylabel('f = C_ap / C_total'); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color='k', lw=0.6); ax[1].axhline(2, color='0.5', lw=0.6, ls=':'); ax[1].axhline(-2, color='0.5', lw=0.6, ls=':')
    ax[1].set_ylabel(f'S_used/S_true - 1 [%]\n(rel. to {lam_ref:.0f} A)'); ax[1].grid(alpha=0.3)
    ax[2].set_ylabel('star track ["]\n(solid E, dashed N)'); ax[2].grid(alpha=0.3)
    ax[2].set_xlabel('wavelength [A]')
    fig.suptitle('sensitivity function: fraction of the standard captured by its aperture\n' + (verdict or ''), fontsize=9)
    return fig


def fig_std_aperture(EN, rows_ap, cents, good, note):
    """Fibre map of the standard (green positions) with the best-matching picker aperture and the
    measured star track coloured by wavelength."""
    fig, ax = plt.subplots(figsize=(7, 7))
    c = 'green' if 'green' in EN else next(iter(EN))
    xy = EN[c]
    near = np.hypot(xy[:, 0], xy[:, 1]) < 8
    ax.scatter(xy[near, 0], xy[near, 1], s=60, facecolor='none', edgecolor='0.75', lw=0.6)
    if rows_ap is not None and len(rows_ap):
        ax.scatter(xy[rows_ap, 0], xy[rows_ap, 1], s=60, color='orange', alpha=0.5, label='best-matching circular aperture')
    m = good[0] & np.isfinite(cents.E[0])
    sc = ax.scatter(cents.E[0][m], cents.N[0][m], c=cents.lam[m], cmap='turbo', s=18, zorder=3)
    fig.colorbar(sc, ax=ax, label='wavelength [A]', shrink=0.8)
    ax.set_xlim(8, -8); ax.set_ylim(-8, 8); ax.set_aspect('equal')
    ax.set_xlabel('E ["]'); ax.set_ylabel('N ["]'); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title(f'standard: star centroid vs wavelength\n{note}', fontsize=8)
    return fig


def fig_std_rss_corrected(before, after, fatal, capture, radii):
    """Standard star: centroid vs wavelength in the input and the ADC-corrected RSS, and the
    fraction of the star in fixed picker apertures before/after."""
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for cc, mk, lab in ((before, 'o', 'input RSS'), (after, 's', 'corrected RSS')):
        g = cc.good(fatal)[0]
        first = True
        for j, b in enumerate(cc.bins):
            if b.role == 'overlap' or not g[j]:
                continue
            for k, v in enumerate((cc.E, cc.N)):
                ax[k].plot(b.lam_eff, v[0, j], mk, color=CH_COL[b.channel], ms=4,
                           mfc='none' if mk == 'o' else CH_COL[b.channel], label=lab if first and k == 0 else None)
            first = False
    for k, nm in enumerate('EN'):
        ax[k].set_ylabel(f'd{nm} ["]'); ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    ls = {'before': ':', 'after': '-'}
    for c in ('blue', 'green', 'red'):
        for when in ('before', 'after'):
            rr = [r for r in capture if r['channel'] == c and r['when'] == when]
            if not rr:
                continue
            lam = [r['lam'] for r in rr]
            for k, R in enumerate(radii):
                ax[2].plot(lam, [r[f'f_R{R:g}'] for r in rr], ls[when], color=CH_COL[c], lw=0.7 + 0.5 * k,
                           label=f'R={R:g}" {when}' if c == 'green' else None)
    ax[2].set_ylabel('fraction in fixed aperture'); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=7, ncol=2)
    ax[2].set_xlabel('wavelength [A]  (hollow / dotted = input, filled / solid = corrected)')
    fig.suptitle('standard star: ADC-corrected RSS', fontsize=9)
    return fig
