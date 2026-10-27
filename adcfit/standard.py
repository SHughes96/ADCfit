"""Standard star + sensitivity function diagnostic (measure and report; changes nothing).

The LLAMAS sensitivity function (Flux/sensFunc.py, built in the CubeViewer picker) is
S(lambda) = F_ref / (C_ap / t), with C_ap the SKYSUB counts summed over ONE fixed set of fibres at
every wavelength. If the standard's centroid moves with wavelength (ADC error + DAR at its airmass),
or its PSF widens to the blue, that set captures a wavelength-dependent fraction f(lambda) of the
star, and S carries a 1/f(lambda) colour error into every FLAM spectrum calibrated with it.

Here, for the standard exposure the sensfunc was built from:

* the star's centroid track vs wavelength is measured with the same machinery as the science
  (``star.measure_star``) and fitted on green+red with blue held out (a high-S/N test of the
  refractivity-basis extrapolation);
* the counts the sensfunc implies, C_ap = F_ref * EXPTIME / RAW, are compared with the TOTAL counts
  of the star in the same RSS (fibres within ``r_tot`` of the measured track at each wavelength,
  interpolated onto the sensfunc's wavelength grid exactly as the picker does). f = C_ap / C_tot is
  the captured fraction baked into S, whatever aperture was used;
* the picker aperture is searched for (circles about the sensfunc's reference fibre, which is
  identified exactly from its wavelength grid), so it can be drawn and the ADC part of f modelled.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

CHANNELS = ('blue', 'green', 'red')
LATTICE_TOL = 1e-3          #: fibre spacings; the picker's tolerance on "R fibre spacings"


def std_siblings(path) -> Dict[str, str]:
    from llamas_pyjamas.CubeViewer.cubeViewRSS import channel_siblings
    sib = channel_siblings(path)
    if not sib:
        raise FileNotFoundError(f'{path}: not an _RSS_<channel> file')
    return sib


def std_plane(sib: Dict[str, str]) -> str:
    """'flam' if every channel has FLAM, else 'skysub'. A standard is usually not yet
    flux-calibrated (the sensfunc is built from it), and plane='auto' refuses that."""
    for p in sib.values():
        with fits.open(p) as h:
            if 'FLAM' not in h:
                return 'skysub'
    return 'flam'


def load_reference(header):
    """(name, wave, flux) of the catalogue standard matching the RSS pointing, else None."""
    try:
        from llamas_pyjamas.Flux.fluxStandards import load_catalog
        m = load_catalog().match_header(header)
    except Exception as exc:                              # noqa: BLE001
        logger.warning('standard crossmatch failed: %s', exc)
        return None
    if m is None or not m.standard.has_spectrum:
        return None
    w, f = m.standard.load_spectrum()
    return m.name, np.asarray(w, float), np.asarray(f, float)


# ----------------------------------------------------------------------------- checks
def consistency_checks(sens, sens_path, std_hdr, sci_paths) -> List[dict]:
    """Sensfunc vs standard RSS header, and every science exposure's FLAM provenance."""
    from llamas_pyjamas.Utils.utils import exposure_time
    rows = []

    def add(item, value, expected, ok, note=''):
        rows.append(dict(check=item, value=value, expected=expected, ok=bool(ok), note=note))

    meta = sens.meta if sens is not None else {}
    if std_hdr is not None and sens is not None:
        obj = str(std_hdr.get('OBJECT', '')).strip()
        name = str(meta.get('standard', '')).strip()
        add('standard name', name, obj, name.replace(' ', '').lower() in obj.replace(' ', '').lower()
            or obj.replace(' ', '').lower() in name.replace(' ', '').lower())
        try:
            t = float(exposure_time(std_hdr))
        except (TypeError, ValueError):
            t = np.nan
        add('exptime', meta.get('exptime'), t, np.isclose(float(meta.get('exptime', np.nan)), t, rtol=1e-3))
        x = std_hdr.get('AIRMASS', std_hdr.get('TEL AIRMASS'))
        add('airmass', meta.get('airmass'), x,
            x is not None and np.isclose(float(meta.get('airmass', np.nan)), float(x), atol=2e-3))
    for p in sci_paths or []:
        for c, cp in std_siblings(p).items():
            try:
                h = fits.getheader(cp, 'FLAM')
            except (KeyError, OSError):
                add(f'{os.path.basename(cp)} FLAM', 'missing', 'FLUXCAL', False, 'no FLAM plane')
                continue
            sf = str(h.get('SENSFILE', ''))
            ok = sens_path is None or sf == os.path.basename(sens_path)[:68]
            add(f'{os.path.basename(cp)} SENSFILE', sf, os.path.basename(sens_path or '') or '-', ok,
                f"STDNAME={h.get('STDNAME', '')} STDAIR={h.get('STDAIR', '')} SCIAIR={h.get('SCIAIR', '')} "
                f"EXTCORR={h.get('EXTCORR', '')}")
    return rows


# ----------------------------------------------------------------------------- star track
def track_function(cents, good, info=None, lam_ref=None):
    """(E, N) of the star vs wavelength from the measured bins (all channels, good bins only): an
    order-1 refractivity fit to all of them plus the linearly interpolated residuals (held at the end
    values beyond the measured range, so the extrapolation follows the dispersion law). Without
    ``info`` plain interpolation. Returns (fn, (lam_min, lam_max)) or (None, None)."""
    from adcfit.dar import basis
    lam = cents.lam
    m = good[0] & np.isfinite(cents.E[0]) & np.isfinite(cents.N[0])
    if m.sum() < 2:
        return None, None
    o = np.argsort(lam[m])
    L, Y = lam[m][o], np.c_[cents.E[0][m][o], cents.N[0][m][o]]
    coef = None
    if info is not None and m.sum() >= 3:
        lr = lam_ref or float(np.median(L))
        B = basis(L, info, lr, 1)
        coef = np.linalg.lstsq(B, Y, rcond=None)[0]
        R = Y - B @ coef

    def f(w):
        w = np.atleast_1d(np.asarray(w, float))
        if coef is None:
            return np.c_[np.interp(w, L, Y[:, 0]), np.interp(w, L, Y[:, 1])]
        return basis(w, info, lr, 1) @ coef + np.c_[np.interp(w, L, R[:, 0]), np.interp(w, L, R[:, 1])]
    return f, (float(L[0]), float(L[-1]))


# ----------------------------------------------------------------------------- RSS on the sensfunc grid
@dataclass
class ChannelGrid:
    channel: str
    wave: np.ndarray                    #: sensfunc WAVE (the picker's reference-fibre grid)
    counts: np.ndarray                  #: (nsel, nw) SKYSUB interpolated, NaN -> 0 (as the picker)
    rows: np.ndarray                    #: RSS rows of ``counts``
    ref_rows: np.ndarray = None         #: RSS rows (among ``rows``) whose WAVE equals the sensfunc grid


def rss_on_sens_grid(path, sens_ch, rows) -> ChannelGrid:
    with fits.open(path) as h:
        W = np.asarray(h['WAVE'].data, float)
        F = np.asarray(h[next(n for n in ('SKYSUB', 'FLUX', 'FLAM') if n in h)].data, float)
    w = sens_ch.wave
    rows = np.asarray(rows)
    with np.errstate(invalid='ignore'):
        d = np.nanmax(np.abs(W[rows] - w[None, :]), axis=1)
    C = np.array([np.nan_to_num(np.interp(w, W[r], F[r], left=np.nan, right=np.nan)) for r in rows])
    return ChannelGrid(sens_ch.channel, w, C.reshape(len(rows), w.size), rows, rows[d < 1e-6])


def implied_counts(sens_ch, exptime, ref_wave, ref_flux):
    """C_ap(lambda) = F_ref * t / RAW on the sensfunc grid; NaN where RAW is not usable."""
    F = np.interp(sens_ch.wave, ref_wave, ref_flux, left=np.nan, right=np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        C = F * float(exptime) / sens_ch.raw
    ok = sens_ch.good & np.isfinite(C) & (C > 0)
    return np.where(ok, C, np.nan)


def _bins(wave, ok, width):
    w = wave[ok]
    if w.size == 0:
        return []
    edges = np.arange(w.min(), w.max() + width, width)
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if ((wave >= a) & (wave < b) & ok).sum() >= 5]


def capture_fraction(grids, implied, EN, track, r_tot=(4.0, 5.0, 6.0), width=100.0):
    """Per channel and wavelength bin: f = C_ap / C_tot for each total-aperture radius (arcsec,
    centred on the measured track at the bin centre). Returns rows."""
    out = []
    for c, g in grids.items():
        C = implied.get(c)
        if C is None:
            continue
        E, N = EN[c][g.rows, 0], EN[c][g.rows, 1]
        for a, b in _bins(g.wave, np.isfinite(C), width):
            s = (g.wave >= a) & (g.wave < b) & np.isfinite(C)
            lam = 0.5 * (a + b)
            cE, cN = track(lam)[0]
            d = np.hypot(E - cE, N - cN)
            row = dict(channel=c, lam=lam, C_ap=float(np.mean(C[s])), track_E=float(cE), track_N=float(cN))
            for R in r_tot:
                tot = float(np.mean(g.counts[d <= R][:, s].sum(axis=0)))
                row[f'f_R{R:g}'] = row['C_ap'] / tot if tot > 0 else np.nan
            out.append(row)
    return out


# ----------------------------------------------------------------------------- aperture recovery
def recover_aperture(grids, implied, xy_lat, star_dist=None, radii=np.arange(0.5, 10.01, 0.5), width=100.0,
                     max_refs=12):
    """Circles of R fibre spacings about centres inside the reference fibre's hex cell (the picker
    centres on the DS9 crosshair, whose nearest fibre supplies the wavelength grid). If several fibres
    share that grid, the ``max_refs`` nearest the star (``star_dist`` per row of the grids) are tried.
    Score: rms over channels and bins of log(sum / C_ap). Returns dict(best..., candidates)."""
    rows = next(iter(grids.values())).rows
    refs = None
    for g in grids.values():
        refs = set(g.ref_rows) if refs is None else refs & set(g.ref_rows)
    refs = np.array(sorted(refs or []), int)
    if refs.size == 0:
        return dict(ok=False, ref_row=-1, note='no RSS fibre has the sensfunc wavelength grid (different '
                                                'exposure, or re-extracted data?)')
    if refs.size > 1 and star_dist is not None:
        dd = np.array([star_dist[np.flatnonzero(rows == r)[0]] for r in refs])
        refs = refs[np.argsort(dd)][:max_refs]
    pos = xy_lat[rows]
    binned = {}
    for c, g in grids.items():
        C = implied.get(c)
        if C is None:
            continue
        bb = _bins(g.wave, np.isfinite(C), width)
        if not bb:
            continue
        sel = [(g.wave >= a) & (g.wave < b) & np.isfinite(C) for a, b in bb]
        binned[c] = (np.array([g.counts[:, s].mean(axis=1) for s in sel]).T,       # (nfib, nbin)
                     np.array([C[s].mean() for s in sel]))
    seen, cands = set(), []
    for ref in refs:
        iref = int(np.flatnonzero(rows == ref)[0])
        c0 = xy_lat[ref]
        _search(pos, c0, iref, int(ref), radii, binned, rows, seen, cands)
    if not cands:
        return dict(ok=False, ref_row=int(refs[0]), note='no candidate apertures')
    cands.sort(key=lambda r: r['rms'])
    best = cands[0]
    return dict(ok=best['rms'] < 0.02, ref_row=best['ref_row'], n_ref=int(refs.size), best=best,
                candidates=cands[:10],
                note='reproduced' if best['rms'] < 0.02 else
                'no circular aperture about the reference fibre reproduces the sensfunc (hand-grown '
                'aperture, or the RSS changed since the sensfunc was built)')


def _search(pos, c0, iref, ref, radii, binned, rows, seen, cands):
    for dx in np.linspace(-0.5, 0.5, 11):
        for dy in np.linspace(-0.5, 0.5, 11):
            d = np.hypot(pos[:, 0] - c0[0] - dx, pos[:, 1] - c0[1] - dy)
            if np.nanargmin(d) != iref:
                continue
            for R in radii:
                sel = np.flatnonzero(d <= R + LATTICE_TOL)
                key = tuple(sel)
                if key in seen or sel.size == 0:
                    continue
                seen.add(key)
                lr, per = [], {}
                for c, (M, Cb) in binned.items():
                    with np.errstate(invalid='ignore', divide='ignore'):
                        q = np.log(M[sel].sum(axis=0) / Cb)
                    q = q[np.isfinite(q)]
                    per[c] = float(np.exp(np.median(q))) if q.size else np.nan
                    lr.append(q)
                q = np.concatenate(lr) if lr else np.array([])
                cands.append(dict(R=float(R), dx=float(dx), dy=float(dy), nfib=int(sel.size), ref_row=ref,
                                  rms=float(np.sqrt(np.mean(q ** 2))) if q.size else np.inf,
                                  rows=rows[sel], **{f'ratio_{c}': v for c, v in per.items()}))


def model_fraction(rows_ap, EN_c, track, fwhm_fn, lam, pitch, held=None, profile='gaussian', beta=3.5, ns=5):
    """Fraction of a unit-flux PSF (FWHM fwhm_fn(lam), centre track(lam) or ``held``) landing on the
    fibres ``rows_ap`` (hexagonal cells of side pitch/sqrt3, supersampled ns x ns)."""
    from adcfit.star import _profile
    prof = _profile(profile, beta)
    # supersample a hexagon by a square grid clipped to the hex (flat-topped/pointy invariance is
    # irrelevant at this accuracy); cell area = sqrt(3)/2 pitch^2
    u = (np.arange(ns) + 0.5) / ns - 0.5
    gx, gy = np.meshgrid(u * pitch * 1.1547, u * pitch * 1.1547)
    inside = _in_hex(gx, gy, pitch)
    ox, oy = gx[inside], gy[inside]
    area = np.sqrt(3) / 2 * pitch ** 2
    out = []
    for w in np.atleast_1d(lam):
        cE, cN = (held if held is not None else track(w)[0])
        fw = fwhm_fn(w)
        x = EN_c[rows_ap, 0][:, None] + ox[None, :] - cE
        y = EN_c[rows_ap, 1][:, None] + oy[None, :] - cN
        val = prof(x ** 2 + y ** 2, fw) / _psf_integral(profile, beta, fw)   # unit-flux density
        out.append(float(np.nansum(val.mean(axis=1) * area)))
    return np.array(out)


def _psf_integral(profile, beta, fwhm):
    """Integral over the plane of star._profile (peak 1)."""
    if profile == 'moffat':
        alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        return np.pi * alpha ** 2 / (beta - 1.0)
    return 2.0 * np.pi * (fwhm / 2.354820045) ** 2


def _in_hex(x, y, pitch):
    """Point-in-hexagon for a hex cell of inradius pitch/2 (pointy-top)."""
    r = pitch / 2.0
    ax, ay = np.abs(x), np.abs(y)
    return (ax <= r) & (ax * 0.5 + ay * np.sqrt(3) / 2 <= r)


@dataclass
class StandardResult:
    checks: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    capture: List[dict] = field(default_factory=list)
    aperture: dict = field(default_factory=dict)
    holdout: List[dict] = field(default_factory=list)
    cents: object = None
    fit: object = None
    poly: object = None
    info: object = None
    geom: object = None
    good: Optional[np.ndarray] = None
    EN: dict = field(default_factory=dict)
    sr: object = None
    track: object = None
    lam_ref: float = np.nan
    star: tuple = None


# ----------------------------------------------------------------------------- driver
def run_standard(std_path, sens_path, sci_paths, opts, fit_channels=('green', 'red'), target_snr=30.0,
                 nmax=20, excl=None, star=None, min_snr=5.0, max_fwhm=3.0, err_floor=0.01, clip=3.0,
                 r_tot=(4.0, 5.0, 6.0), fatal=None, lam_ref=None) -> StandardResult:
    """Everything above for one standard exposure. ``std_path`` (any channel's RSS) and/or
    ``sens_path``; with only the sensfunc, only the provenance checks run."""
    from llamas_pyjamas.Flux.sensFunc import SensFunc
    from llamas_pyjamas.Utils.utils import exposure_time
    from adcfit import io as aio, dar, star as astar

    res = StandardResult()
    sens = SensFunc.load(sens_path) if sens_path else None
    sib = std_siblings(std_path) if std_path else {}
    hdr = fits.getheader(sib.get('green') or next(iter(sib.values()))) if sib else None
    res.checks = consistency_checks(sens, sens_path, hdr, sci_paths)
    res.summary = dict(sensfunc=sens_path, standard_rss=sib, sens_meta={k: (v if isinstance(v, (int, float, str, bool))
                                                                           else str(v)) for k, v in
                                                                       (sens.meta if sens else {}).items()})
    if not sib:
        return res

    # --- the star's track in the standard exposure
    sr = aio.load_field([sib.get('green') or next(iter(sib.values()))], plane=std_plane(sib))
    info = aio.read_exposure_info(sr)[0]
    geom = dar.exposure_geometry(info)
    try:
        inst = aio.instrument_xy(sr)
    except Exception as exc:                              # noqa: BLE001
        logger.warning('standard: instrument positions unavailable (%s)', exc)
        inst = {}
    bins = astar.make_bins(sr, {c: 8 for c in CHANNELS}, fit_channels, None, excl)
    choice = astar.find_star(sr, bins, star=star, inst=inst or None, min_snr=min_snr, max_fwhm=max_fwhm,
                             excl=excl)
    ra0, dec0 = choice.ra, choice.dec
    mkw = dict(inst=inst or None, guesses=choice.exp_pos)
    broad = astar.measure_star(sr, (ra0, dec0), astar.broad_bins(bins), opts, **mkw)
    nb = {}
    for j, b in enumerate(broad.bins):
        s_b = float(broad.apsnr[0, j]) if np.isfinite(broad.apsnr[0, j]) else np.nan
        nb[b.channel] = astar.bins_for_snr(s_b, nmax, target_snr, nmin=2 if b.channel in fit_channels else 1)
    bins = astar.make_bins(sr, nb, fit_channels, None, excl)
    cents = astar.measure_star(sr, (ra0, dec0), bins, opts, **mkw)
    fatal = astar.FATAL_DEFAULT if fatal is None else fatal
    good = cents.good(fatal)
    lam = cents.lam
    roles = np.array([b.role for b in bins])
    red_fit = (roles == 'fit') & np.array([b.channel == 'red' for b in bins])
    lam_ref = float(lam_ref or (np.mean(lam[red_fit]) if red_fit.any() else np.mean(lam[roles == 'fit'])))
    off, err = np.c_[cents.E[0], cents.N[0]], np.c_[cents.sE[0], cents.sN[0]]
    fit = dar.fit_exposure(lam, off, err, (roles == 'fit') & good[0], info, lam_ref, order=1,
                           err_floor=err_floor, clip=clip, exp_id=cents.exp_ids[0], index=0)
    hm = (roles == 'holdout') & good[0]
    hold = dar.holdout_stats(fit, lam, off, err, hm, err_floor)
    fit2 = dar.fit_exposure(lam, off, err, (roles == 'fit') & good[0], info, lam_ref, order=2,
                            err_floor=err_floor, clip=clip, exp_id=cents.exp_ids[0], index=0)
    hold2 = dar.holdout_stats(fit2, lam, off, err, hm, err_floor)
    poly = dar.fit_poly_lambda(lam, off, err, (roles == 'fit') & good[0], 2, lam_ref, err_floor)
    chrom = dar.chromatic_summary([fit], [geom])
    res.summary.update(object=sr.field, exp=cents.exp_ids[0], star=dict(ra=ra0, dec=dec0, source=choice.source),
                       airmass=geom.airmass_mean, q=geom.q_mean, lam_ref=lam_ref, bins_per_channel=nb,
                       fit=dict(status=fit.status, coef_E=list(fit.coef[0]), coef_N=list(fit.coef[1]),
                                chi2r=list(fit.chi2r), note=fit.note),
                       blue_holdout=hold, blue_holdout_order2=hold2, chromatic=chrom)
    res.holdout = [dict(model='order1', **hold), dict(model='order2', **hold2)]
    res.cents, res.fit, res.poly, res.info, res.geom, res.good = cents, fit, poly, info, geom, good
    track, trange = track_function(cents, good, info, lam_ref)
    res.sr, res.track, res.lam_ref, res.star = sr, track, lam_ref, (ra0, dec0)
    if sens is None:
        return res

    # --- what the sensfunc captured, relative to the star's total counts
    from llamas_pyjamas.Utils.register import _fibre_xy
    ref = load_reference(hdr)
    if ref is None:
        res.summary['capture_note'] = 'standard not in the catalogue: C_ap cannot be formed'
        return res
    if str(sens.meta.get('standard', '')).strip() and ref[0].replace(' ', '').lower() != \
            str(sens.meta.get('standard')).replace(' ', '').lower():
        logger.warning('sensfunc STANDARD=%s but the RSS pointing matches %s', sens.meta.get('standard'), ref[0])
    t_sens = float(sens.meta.get('exptime') or exposure_time(hdr))
    if track is None:
        res.summary['capture_note'] = 'too few good centroid bins for a track'
        return res
    EN, grids, implied = {}, {}, {}
    from adcfit.io import tangent_offsets
    for c in CHANNELS:
        if c not in sens.channels or c not in sib or c not in sr.channels:
            continue
        st = sr.channels[c]
        E, N = tangent_offsets(st.ra, st.dec, ra0, dec0)
        with fits.open(sib[c]) as h:
            nrow = h['WAVE'].data.shape[0]
        if st.n_fibres != nrow:
            logger.warning('standard %s: super-RSS has %d rows, the RSS %d; skipped', c, st.n_fibres, nrow)
            continue
        EN[c] = np.c_[E, N]
        rows = np.flatnonzero(np.isfinite(E) & (np.hypot(E, N) < max(r_tot) + 8.0))
        grids[c] = rss_on_sens_grid(sib[c], sens.channels[c], rows)
        implied[c] = implied_counts(sens.channels[c], t_sens, ref[1], ref[2])
    res.capture = capture_fraction(grids, implied, EN, track, r_tot)
    from adcfit.star import TELLURIC
    for r in res.capture:
        r['extrapolated'] = bool(r['lam'] < trange[0] - 50 or r['lam'] > trange[1] + 50)
        # telluric bands and the unmeasured ends have noisy totals (sky residuals); kept in the
        # table but left out of the verdict
        r['clean'] = bool(not r['extrapolated'] and not any(a - 50 < r['lam'] < b + 50 for a, b in TELLURIC))
    with fits.open(sib.get('green') or next(iter(sib.values()))) as h:
        fm = h['FIBERMAP'].data
        x, y = _fibre_xy(fm['FIBER_ID'], fm['BENCHSIDE'])
    xy_lat = np.c_[x, y].astype(float)
    g0 = next(iter(grids.values()))
    ce, cn = track(lam_ref)[0]
    res.aperture = recover_aperture(grids, implied, xy_lat,
                                    star_dist=np.hypot(EN[g0.channel][g0.rows, 0] - ce, EN[g0.channel][g0.rows, 1] - cn))

    # --- ADC part of f, if the aperture is known: PSF model with the track vs held at lam_ref
    rmid = 'f_R%g' % r_tot[len(r_tot) // 2]
    fw_ok = good[0] & np.isfinite(cents.fwhm[0])
    if res.aperture.get('ok') and fw_ok.sum() >= 2:
        from llamas_pyjamas.Utils.wcsLlamas import ARCSEC_PER_FIBRE
        o = np.argsort(lam[fw_ok])
        fwhm_fn = lambda w: float(np.interp(w, lam[fw_ok][o], cents.fwhm[0][fw_ok][o]))   # noqa: E731
        held = track(lam_ref)[0]
        rows_ap = res.aperture['best']['rows']
        for r in res.capture:
            c = r['channel']
            r['fmod_track'] = float(model_fraction(rows_ap, EN[c], track, fwhm_fn, r['lam'], ARCSEC_PER_FIBRE,
                                                   profile=opts.profile, beta=opts.beta)[0])
            r['fmod_noADC'] = float(model_fraction(rows_ap, EN[c], track, fwhm_fn, r['lam'], ARCSEC_PER_FIBRE,
                                                   held=held, profile=opts.profile, beta=opts.beta)[0])
    # normalised colour error and the 250 A table
    fr = [r for r in res.capture if np.isfinite(r.get(rmid, np.nan))]
    if fr:
        lr = np.array([r['lam'] for r in fr]); fv = np.array([r[rmid] for r in fr])
        f_ref = float(np.interp(lam_ref, lr, fv))
        for r in res.capture:
            r['S_used_over_true'] = 1.0 / r[rmid] if r.get(rmid) else np.nan
            r['colour_err_rel_ref'] = f_ref / r[rmid] - 1.0 if r.get(rmid) else np.nan
        per = {}
        for c in CHANNELS:
            v = np.array([r['colour_err_rel_ref'] for r in fr if r['channel'] == c and r['clean']])
            if v.size:
                per[c] = dict(min=float(v.min()), max=float(v.max()), ptp=float(np.ptp(v)))
        res.summary.update(f_at_lam_ref=f_ref, colour_error=per,
                           r_tot_spread=float(np.nanmax(np.abs(np.diff(
                               np.array([[r[f'f_R{R:g}'] for R in r_tot] for r in fr]), axis=1)))) if len(r_tot) > 1 else 0.0)
        worst = max(max(abs(d['min']), abs(d['max'])) for d in per.values()) if per else 0.0
        res.summary['verdict'] = (
            f'sensfunc colour error vs total flux up to {100 * worst:.1f}% (relative to {lam_ref:.0f} A; '
            'telluric bands and unmeasured ends excluded): '
            + ('REBUILD recommended -- extract the standard with an aperture that follows the star (or a large '
               'one) so S is a total-flux throughput' if worst > 0.02 else 'within 2%: no rebuild needed'))
    res.summary['aperture'] = {k: v for k, v in res.aperture.items() if k not in ('candidates', 'best')}
    if res.aperture.get('best'):
        res.summary['aperture']['best'] = {k: v for k, v in res.aperture['best'].items() if k != 'rows'}
        res.summary['aperture']['best']['rss_rows'] = [int(i) for i in res.aperture['best']['rows']]
    res.EN = EN
    return res
