"""Reference-star detection and per-exposure, per-wavelength-bin centroiding in fibre space.

Lessons from the first analysis (ADC_orig_code/s3_adc.py) built in here:
* the aperture FOLLOWS the star (re-centred per bin, per exposure) instead of sitting at a fixed
  bundle position the dithers move the star away from;
* the PSF width is free per bin (seeing is wider in the blue), or optionally follows lambda^-0.2;
* a background term is fitted, fibres are inverse-variance weighted and outlier fibres rejected;
* every measurement carries quality flags (edge of footprint, fit at a bound, too few fibres, low
  S/N, split image...) so bad bins are excluded rather than silently biasing the fit.

All positions are gnomonic offsets (arcsec, +E, +N) about the star's nominal position, computed from
the fibre RA/Dec of each exposure; the instrument-frame equivalents come from the fibre map.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from adcfit.io import tangent_offsets, tangent_inverse

logger = logging.getLogger(__name__)

FLAGS = dict(EDGE=1, BOUND=2, FEWFIB=4, LOWSN=8, CHI2=16, MOMENT=32, SATCORE=64, HOLE=128,
             MORPH=256, CLIPPED=512, FAILED=1024)
# MORPH (a significant secondary residual peak) is report-only by default: on real data it fires on
# fixed fibre streaks next to the star as often as on genuinely split images; add it to
# --reject-flags to make it fatal.
FATAL_DEFAULT = (FLAGS['EDGE'] | FLAGS['BOUND'] | FLAGS['FEWFIB'] | FLAGS['LOWSN'] | FLAGS['HOLE']
                 | FLAGS['FAILED'])

#: Observed-frame windows excluded from every bin (Angstrom): telluric O2/H2O bands and the
#: brightest night-sky lines (+-6 A). Assumptions; extend with --exclude.
TELLURIC = [(6860, 6960), (7160, 7340), (7590, 7700), (8130, 8350), (8950, 9400)]
SKY_LINES = [5577.34, 5889.95, 5895.92, 6300.30, 6363.78]
SKY_HW = 6.0
BLUE_MIN = 3600.0               #: blue throughput floor (coadd.WHITELIGHT_BLUE_MIN_A)
RED_MAX = 9700.0                #: red cap (fringing / low throughput beyond)
OVERLAP_GAP = 50.0              #: keep fit bins this far from the neighbouring channel's edge


def flag_names(v):
    return '|'.join(k for k, b in FLAGS.items() if int(v) & b) or '-'


def default_exclusions(extra=()):
    return list(TELLURIC) + [(l - SKY_HW, l + SKY_HW) for l in SKY_LINES] + [tuple(e) for e in extra]


# ----------------------------------------------------------------------------- bins
@dataclass
class Bin:
    index: int
    channel: str
    lo: float
    hi: float
    lam_eff: float
    role: str                      #: fit | holdout | overlap | broad


def channel_extent(st, step=7):
    w = st.wave[::step]
    w = w[np.isfinite(w)]
    return float(np.percentile(w, 0.5)), float(np.percentile(w, 99.5))


def clean_ranges(raw: Dict[str, tuple], overrides=None):
    """Per-channel usable ranges: drop the dichroic overlaps (+-OVERLAP_GAP), blue below 3600 A
    and red beyond 9700 A. ``overrides`` {channel: (lo, hi)} replaces the computed range."""
    out = {}
    b, g, r = raw.get('blue'), raw.get('green'), raw.get('red')
    if b:
        out['blue'] = (max(b[0], BLUE_MIN), b[1] if not g else min(b[1], g[0] - OVERLAP_GAP))
    if g:
        out['green'] = (g[0] if not b else max(g[0], b[1] + OVERLAP_GAP),
                        g[1] if not r else min(g[1], r[0] - OVERLAP_GAP))
    if r:
        out['red'] = (r[0] if not g else max(r[0], g[1] + OVERLAP_GAP), min(r[1], RED_MAX))
    for c, v in (overrides or {}).items():
        out[c] = tuple(v)
    return {c: v for c, v in out.items() if v[1] > v[0]}


def _lam_eff(lo, hi, excl):
    grid = np.arange(lo, hi, 0.5)
    ok = np.ones(grid.size, bool)
    for a, b in excl:
        ok &= ~((grid >= a) & (grid < b))
    return (float(grid[ok].mean()) if ok.any() else np.nan), float(ok.mean()) if grid.size else 0.0


def make_bins(sr, nbins, fit_channels, overrides=None, exclusions=None, overlap=True):
    """Equal-width bins over each channel's clean range. Role 'fit' for ``fit_channels``, else
    'holdout'; plus 'overlap' bins in the dichroic overlaps measured in BOTH channels (QA only).
    Bins more than 70% excluded by telluric/sky windows are dropped."""
    excl = exclusions if exclusions is not None else default_exclusions()
    raw = {c: channel_extent(st) for c, st in sr.channels.items()}
    rng = clean_ranges(raw, overrides)
    bins = []
    for c in ('blue', 'green', 'red'):
        if c not in rng:
            continue
        lo, hi = rng[c]
        edges = np.linspace(lo, hi, int(nbins.get(c, 8)) + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            le, frac = _lam_eff(a, b, excl)
            if frac < 0.3:
                continue
            bins.append(Bin(len(bins), c, float(a), float(b), le, 'fit' if c in fit_channels else 'holdout'))
    if overlap:
        for c1, c2 in (('blue', 'green'), ('green', 'red')):
            if c1 in raw and c2 in raw:
                a, b = raw[c2][0] + 40.0, raw[c1][1] - 40.0
                if b - a > 60:
                    le, frac = _lam_eff(a, b, excl)
                    if frac >= 0.3:
                        for c in (c1, c2):
                            bins.append(Bin(len(bins), c, float(a), float(b), le, 'overlap'))
    return bins


def broad_bins(bins):
    """One bin per channel spanning its fit/holdout bins (registration check, references)."""
    out = []
    for c in ('blue', 'green', 'red'):
        bb = [b for b in bins if b.channel == c and b.role != 'overlap']
        if bb:
            out.append(Bin(len(out), c, min(b.lo for b in bb), max(b.hi for b in bb),
                           float(np.mean([b.lam_eff for b in bb])), 'broad'))
    return out


def bin_values(st, rows, lo, hi, excl, nsig=5.0):
    """Robust per-fibre mean over a wavelength bin (native pixels). Pixels deviating from the
    fibre's median by more than ``nsig`` x max(pixel sigma, the fibre's own MAD) are rejected
    (cosmic rays / bad pixels without clipping real spectral structure). Returns (value, var)."""
    w = st.wave[rows]
    inb = (w >= lo) & (w < hi) & ~st.mask[rows] & np.isfinite(st.flux[rows]) & np.isfinite(st.var[rows])
    for a, b in excl:
        inb &= ~((w >= a) & (w < b))
    f = np.where(inb, st.flux[rows].astype(float), np.nan)
    v = np.where(inb, st.var[rows], np.nan)
    with np.errstate(invalid='ignore', all='ignore'):
        med = np.nanmedian(f, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(f - med[:, None]), axis=1)
        thr = nsig * np.maximum(np.sqrt(v), mad[:, None])
        keep = inb & (np.abs(f - med[:, None]) <= thr)
        n = keep.sum(axis=1)
        val = np.where(keep, f, 0.0).sum(axis=1) / np.maximum(n, 1)
        var = np.where(keep, v, 0.0).sum(axis=1) / np.maximum(n, 1) ** 2
    bad = n < 3
    val[bad] = np.nan
    var[bad] = np.nan
    return val, var


# ----------------------------------------------------------------------------- PSF fit
@dataclass
class PsfFit:
    x: float
    y: float
    cov: np.ndarray                #: 2x2 centroid covariance (arcsec^2)
    fwhm: float
    amp: float
    snr: float
    bg: float
    chi2r: float
    nfib: int
    nrej: int
    at_bound: bool
    mx: float = np.nan             #: moment centroid
    my: float = np.nan
    core_deficit: float = 0.0      #: (data - model)/model summed over the core
    ellipticity: float = np.nan
    secondary: float = 0.0         #: brightest significant residual peak / amplitude


def _profile(name, beta):
    if name == 'moffat':
        k = 2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0)

        def prof(r2, w):
            alpha = w / k
            return (1.0 + r2 / alpha ** 2) ** (-beta)
        return prof
    return lambda r2, w: np.exp(-0.5 * r2 / (w / 2.354820045) ** 2)


def fit_psf(x, y, f, v, x0, y0, radius, profile='gaussian', beta=3.5, fwhm_fixed=None, bg='const',
            fwhm_guess=1.2, bound_frac=0.5, min_fib=8, fwhm_bounds=None) -> Optional[PsfFit]:
    """Fit a circular PSF + background to fibre values within ``radius`` of (x0, y0).
    Amplitudes are rescaled to O(1) first (FLAM ~1e-17 otherwise stalls the optimiser)."""
    r0 = np.hypot(x - x0, y - y0)
    sel = (r0 < radius) & np.isfinite(f) & np.isfinite(v) & (v > 0)
    if sel.sum() < min_fib:
        return None
    xs, ys, fs, vs = x[sel], y[sel], f[sel], v[sel]
    outer = r0[sel] > 0.7 * radius
    b0 = float(np.median(fs[outer])) if outer.sum() >= 3 else float(np.percentile(fs, 20))
    scale = float(np.max(fs) - b0)
    if not np.isfinite(scale) or scale <= 0:
        return None
    z, s = fs / scale, np.sqrt(vs) / scale
    prof = _profile(profile, beta)
    free_w = fwhm_fixed is None
    nbg = 3 if bg == 'plane' else 1
    lo_xy = (x0 - bound_frac * radius, y0 - bound_frac * radius)
    hi_xy = (x0 + bound_frac * radius, y0 + bound_frac * radius)

    def unpack(p):
        a, xc, yc = p[:3]
        w = p[3] if free_w else fwhm_fixed
        b = p[4 if free_w else 3:]
        return a, xc, yc, w, b

    def model(p, xx, yy):
        a, xc, yc, w, b = unpack(p)
        m = a * prof((xx - xc) ** 2 + (yy - yc) ** 2, w) + b[0]
        if nbg == 3:
            m = m + b[1] * (xx - x0) + b[2] * (yy - y0)
        return m

    # seed at the flux^2-weighted centroid of the excess (a start far from the star otherwise lets
    # the amplitude collapse to zero), clamped inside the centroid bounds
    wt = np.clip(fs - b0, 0, None) ** 2
    sx, sy = ((np.sum(wt * xs) / wt.sum(), np.sum(wt * ys) / wt.sum()) if wt.sum() > 0 else (x0, y0))
    p0 = [1.0, sx, sy] + ([fwhm_guess] if free_w else []) + [b0 / scale] + [0.0] * (nbg - 1)
    wlo, whi = (max(0.5, fwhm_bounds[0]), min(6.0, fwhm_bounds[1])) if fwhm_bounds else (0.5, 6.0)
    lo = [0.0, lo_xy[0], lo_xy[1]] + ([wlo] if free_w else []) + [-np.inf] * nbg
    hi = [np.inf, hi_xy[0], hi_xy[1]] + ([whi] if free_w else []) + [np.inf] * nbg
    p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    keep = np.ones(xs.size, bool)
    nrej = 0
    for it in range(2):
        try:
            r = least_squares(lambda p: (model(p, xs[keep], ys[keep]) - z[keep]) / s[keep], p0,
                              bounds=(lo, hi), x_scale='jac', max_nfev=400)
        except Exception as exc:                          # noqa: BLE001
            logger.debug('psf fit failed: %s', exc)
            return None
        res = (model(r.x, xs, ys) - z) / s
        bad = keep & (np.abs(res) > 5.0)
        if it == 0 and bad.any() and (keep.sum() - bad.sum()) >= min_fib:
            keep &= ~bad
            nrej = int(bad.sum())
            p0 = r.x
            continue
        break
    npar = r.x.size
    dof = max(int(keep.sum()) - npar, 1)
    chi2r = float(np.sum(r.fun ** 2) / dof)
    try:
        cov = np.linalg.pinv(r.jac.T @ r.jac) * max(1.0, chi2r)
    except np.linalg.LinAlgError:
        return None
    a, xc, yc, w, b = unpack(r.x)
    at_bound = bool(np.isclose(xc, lo_xy[0], atol=1e-3) or np.isclose(xc, hi_xy[0], atol=1e-3)
                    or np.isclose(yc, lo_xy[1], atol=1e-3) or np.isclose(yc, hi_xy[1], atol=1e-3)
                    or (free_w and (w <= wlo * 1.002 or w >= whi * 0.998)))
    sa = np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else np.inf
    out = PsfFit(float(xc), float(yc), cov[1:3, 1:3].copy(), float(w), float(a * scale),
                 float(a / sa), float(b[0] * scale), chi2r, int(keep.sum()), nrej, at_bound)
    # diagnostics: moment centroid, saturation-like core deficit, ellipticity, secondary peak
    from llamas_pyjamas.Utils.centroid import fibre_centroid
    c = fibre_centroid(xs, ys, fs, guess=(xc, yc), radius=max(w, 1.0), power=1.0,
                       background=b[0] * scale)
    if c is not None:
        out.mx, out.my = float(c.x), float(c.y)
    mod = model(r.x, xs, ys)
    rr = np.hypot(xs - xc, ys - yc)
    core = rr < 0.5 * w
    if core.sum() >= 2:
        m_src = mod[core] - b[0]
        out.core_deficit = float(np.sum(z[core] - mod[core]) / max(np.sum(m_src), 1e-12))
    wts = np.clip(z - b[0], 0, None) * (rr < 1.5 * w)
    if wts.sum() > 0:
        dx, dy = xs - xc, ys - yc
        mxx, myy, mxy = (np.sum(wts * dx * dx), np.sum(wts * dy * dy), np.sum(wts * dx * dy))
        out.ellipticity = float(np.hypot(mxx - myy, 2 * mxy) / max(mxx + myy, 1e-12))
    sig = (z - mod) / s
    cand = (sig > 5.0) & (rr > 0.6 * w)
    if cand.any():
        out.secondary = float(np.max((z - mod)[cand]) / max(a, 1e-12))
    return out


def hull_of(E, N):
    ok = np.isfinite(E) & np.isfinite(N)
    if ok.sum() < 3:
        return None
    try:
        return ConvexHull(np.c_[E[ok], N[ok]])
    except Exception:                                    # noqa: BLE001
        return None


def hull_distance(hull, x, y):
    """Signed distance (arcsec) from (x, y) to the footprint edge; positive inside."""
    if hull is None:
        return np.nan
    eq = hull.equations
    return float(-np.max(eq[:, 0] * x + eq[:, 1] * y + eq[:, 2]))


# ----------------------------------------------------------------------------- destriping
ROW_PITCH = 0.75 * 0.8660254        #: arcsec between fibre-map rows (hex lattice)


def row_keys(xy):
    """Fibre-map row index of each fibre from its instrument (x, y) in arcsec; -1 if unknown."""
    y = np.asarray(xy)[:, 1]
    k = np.full(y.size, -1, int)
    ok = np.isfinite(y)
    k[ok] = np.round(y[ok] / ROW_PITCH).astype(int)
    return k


def destripe(v, keys, E=None, N=None, centre=None, excl_r=5.0, min_n=5):
    """Subtract a median per fibre-map row. SKYSUB residuals in LLAMAS are striped along the fibre-
    map rows (on J0111 a per-row median removes 35/23/62% of the red/green/blue background scatter,
    per-benchside or per-column medians < 11%). Fibres within ``excl_r`` of ``centre`` (the star)
    are left out of the medians; rows with fewer than ``min_n`` usable fibres get the global one."""
    out = np.array(v, float)
    use = np.isfinite(out) & (keys >= 0)
    far = np.ones(out.size, bool)
    if centre is not None and E is not None:
        far = np.hypot(E - centre[0], N - centre[1]) > excl_r
    g = use & far
    glob = float(np.median(out[g])) if g.any() else 0.0
    for k in np.unique(keys):
        s = keys == k
        ref = s & g
        out[s] -= float(np.median(out[ref])) if (k >= 0 and ref.sum() >= min_n) else glob
    return out


# ----------------------------------------------------------------------------- star selection
#: per-channel position check: the star must be fitted in every channel's broad band at S/N >= this,
#: within this distance (arcsec) of its red position (a few arcsec covers the ADC/DAR blue-red split)
CHAN_MIN_SNR = 5.0
CHAN_MAX_DPOS = 3.0


class NoStarError(RuntimeError):
    def __init__(self, msg, candidates=None):
        super().__init__(msg)
        self.candidates = candidates or []


@dataclass
class StarChoice:
    ra: float
    dec: float
    source: str                                     #: auto | user
    candidates: List[dict] = field(default_factory=list)
    chosen: Optional[int] = None
    exp_pos: Dict[int, tuple] = field(default_factory=dict)   #: exposure -> (E, N) about (ra, dec)
    exp_missing: List[int] = field(default_factory=list)      #: exposures with no detection of it
    exp_table: List[dict] = field(default_factory=list)       #: per exposure: detected, dist, flux


def _broad_values(sr, e, ch, rng, excl, rows=None):
    st = sr.channels[ch]
    if rows is None:
        rows = np.flatnonzero(st.exposure == e)
    v, var = bin_values(st, rows, rng[0], rng[1], excl)
    return rows, v, var


def find_star(sr, bins, star=None, search_radius=4.0, cluster_radius=2.5, min_snr=30.0,
              max_fwhm=3.0, excl=None, inst=None, chan_min_snr=CHAN_MIN_SNR,
              chan_max_dpos=CHAN_MAX_DPOS) -> StarChoice:
    """Detect compact sources per exposure (green+red broad band, fibre space, row-destriped when
    ``inst`` positions are given), cluster them across exposures on the sky (``cluster_radius``
    must exceed the header-pointing drift between exposures), and pick the brightest compact,
    isolated one seen in at least half the exposures. ``star=(ra, dec)`` overrides the choice (the
    nearest cluster within ``search_radius`` refines it, else the given position is used). Each
    exposure's own detection of the chosen star -- its nearest detection within ``cluster_radius``
    of the chosen position, never a further source -- is returned in ``exp_pos`` as its starting
    point; exposures with none are listed in ``exp_missing`` (the star is not measured there).
    Every candidate is also fitted in each channel's broad band (``chan_ok``: S/N >= ``chan_min_snr``
    and within ``chan_max_dpos`` of the red position in all channels); a candidate that misses the
    full criteria can only be picked if it passes that check. ``reason`` lists what each one failed."""
    from llamas_pyjamas.Utils.register import detect_fibre_sources
    excl = excl if excl is not None else default_exclusions()
    keys = {c: row_keys(xy) for c, xy in (inst or {}).items()}
    bb = {b.channel: (b.lo, b.hi) for b in broad_bins(bins)}
    det_ch = [c for c in ('green', 'red') if c in bb and c in sr.channels] or list(bb)
    gst = sr.channels[det_ch[0]]
    ok = np.isfinite(gst.ra) & (gst.ra > 0)
    ra0, dec0 = float(np.median(gst.ra[ok])), float(np.median(gst.dec[ok]))
    dets = []
    for e in range(sr.n_exposures):
        S, pos = None, None
        for c in det_ch:
            st = sr.channels[c]
            rows, v, _ = _broad_values(sr, e, c, bb[c], excl)
            if rows.size == 0:
                continue
            if c in keys:
                v = destripe(v, keys[c][rows])
            fin = np.isfinite(v)
            if fin.sum() < 20:
                continue
            med = np.median(v[fin])
            mad = 1.4826 * np.median(np.abs(v[fin] - med)) or 1.0
            sn = np.where(fin, (v - med) / mad, np.nan)
            if S is None:
                S = sn
                E, N = tangent_offsets(st.ra[rows], st.dec[rows], ra0, dec0)
                pos = (E, N)
            elif sn.size == S.size:                       # same RSS rows: add the S/N maps
                S = np.where(np.isfinite(S) | np.isfinite(sn), np.nansum(np.c_[S, sn], axis=1), np.nan)
        if S is None:
            continue
        E, N = pos
        srcs = detect_fibre_sources(E / 0.75, N / 0.75, S, nsigma=8.0, min_sep=4.0, max_sources=15)
        for c in srcs:
            dets.append(dict(exp=e, E=c.x * 0.75, N=c.y * 0.75, flux=float(c.flux_sum)))
    # greedy clustering, brightest first
    clusters = []
    for d in sorted(dets, key=lambda q: -q['flux']):
        for cl in clusters:
            if np.hypot(d['E'] - cl['E'], d['N'] - cl['N']) < cluster_radius and d['exp'] not in cl['exps']:
                cl['members'].append(d)
                cl['exps'].add(d['exp'])
                cl['E'] = float(np.median([m['E'] for m in cl['members']]))
                cl['N'] = float(np.median([m['N'] for m in cl['members']]))
                break
        else:
            clusters.append(dict(E=d['E'], N=d['N'], members=[d], exps={d['exp']}))
    cands = []
    for i, cl in enumerate(clusters):
        flux = float(np.median([m['flux'] for m in cl['members']]))
        best = max(cl['members'], key=lambda m: m['flux'])
        ra, dec = tangent_inverse(cl['E'], cl['N'], ra0, dec0)
        row = dict(id=i, ra=float(ra), dec=float(dec), E=cl['E'], N=cl['N'], n_exp=len(cl['exps']),
                   flux=flux, fwhm=np.nan, snr=np.nan, fit_snr=np.nan, ellipticity=np.nan, edge_dist=np.nan,
                   isolated=True, qualifies=False)
        c = 'red' if 'red' in bb else det_ch[0]
        st = sr.channels[c]
        rows, v, var = _broad_values(sr, best['exp'], c, bb[c], excl)
        E, N = tangent_offsets(st.ra[rows], st.dec[rows], ra0, dec0)
        if c in keys:
            v = destripe(v, keys[c][rows], E, N, (best['E'], best['N']))
        pf = fit_psf(E, N, v, var, best['E'], best['N'], 3.5)
        if pf is not None:
            row.update(fwhm=pf.fwhm, fit_snr=pf.snr, ellipticity=pf.ellipticity)
        row['edge_dist'] = hull_distance(hull_of(E, N), best['E'], best['N'])
        ref = (pf.x, pf.y) if pf is not None else (best['E'], best['N'])
        fw0 = pf.fwhm if pf is not None else 1.2
        # aperture S/N (summed flux over the annulus noise), not the fit's amplitude error: the
        # latter is inflated by chi2 for bright stars and tiny for loose fits to noise
        row['snr'] = aperture_snr(E, N, v, ref, fw0)
        bad = []
        for ch in ('blue', 'green', 'red'):
            if ch not in bb or ch not in sr.channels:
                continue
            stc = sr.channels[ch]
            rc, vc, varc = _broad_values(sr, best['exp'], ch, bb[ch], excl)
            Ec, Nc = tangent_offsets(stc.ra[rc], stc.dec[rc], ra0, dec0)
            if ch in keys:
                vc = destripe(vc, keys[ch][rc], Ec, Nc, (best['E'], best['N']))
            pc = fit_psf(Ec, Nc, vc, varc, best['E'], best['N'], 3.5) if rc.size else None
            if pc is None:
                sn = aperture_snr(Ec, Nc, vc, ref, fw0) if rc.size else np.nan
                row.update({f'snr_{ch}': sn, f'fwhm_{ch}': np.nan, f'dpos_{ch}': np.nan,
                            f'ra_{ch}': np.nan, f'dec_{ch}': np.nan})
                bad.append(f'chan:{ch} no fit')
                continue
            sn = aperture_snr(Ec, Nc, vc, (pc.x, pc.y), pc.fwhm)
            rac, decc = tangent_inverse(pc.x, pc.y, ra0, dec0)
            dpos = float(np.hypot(pc.x - ref[0], pc.y - ref[1]))
            row.update({f'snr_{ch}': sn, f'fwhm_{ch}': pc.fwhm, f'dpos_{ch}': dpos,
                        f'ra_{ch}': float(rac), f'dec_{ch}': float(decc)})
            if not (np.isfinite(sn) and sn >= chan_min_snr):
                bad.append(f'chan:{ch} snr {sn:.0f}')
            if not dpos <= chan_max_dpos:
                bad.append(f'chan:{ch} dpos {dpos:.1f}"')
        row['chan_ok'] = not bad
        row['_chan_bad'] = bad
        row['_members'] = cl['members']
        cands.append(row)

    def exp_match(ra, dec):
        """Per exposure, its nearest detection within ``cluster_radius`` of (ra, dec): the starting
        positions (E, N about (ra, dec)), the exposures without one, and a per-exposure table."""
        E0, N0 = tangent_offsets(ra, dec, ra0, dec0)
        pos, missing, table = {}, [], []
        for e in range(sr.n_exposures):
            de = [d for d in dets if d['exp'] == e]
            dist = [float(np.hypot(d['E'] - E0, d['N'] - N0)) for d in de]
            k = int(np.argmin(dist)) if de else -1
            row = dict(exp=sr.exposures[e].exposure_id, detected=False, dist=np.nan, flux=np.nan,
                       nearest_other=min(dist) if dist else np.nan)
            if k >= 0 and dist[k] < cluster_radius:
                mr, md = tangent_inverse(de[k]['E'], de[k]['N'], ra0, dec0)
                pos[e] = tuple(float(x) for x in tangent_offsets(mr, md, ra, dec))
                row.update(detected=True, dist=dist[k], flux=de[k]['flux'], nearest_other=np.nan)
            else:
                missing.append(e)
            table.append(row)
        return pos, missing, table

    def strip(lst):
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in lst]

    for a in cands:
        for b in cands:
            if a is not b and np.hypot(a['E'] - b['E'], a['N'] - b['N']) < 5.0 and b['flux'] > 0.2 * a['flux']:
                a['isolated'] = False
        a['qualifies'] = bool(a['n_exp'] >= max(1, int(np.ceil(sr.n_exposures / 2)))
                              and np.isfinite(a['fwhm']) and 0.6 <= a['fwhm'] <= max_fwhm
                              and np.isfinite(a['snr']) and a['snr'] >= min_snr and a['isolated'])
        why = []
        if a['n_exp'] < max(1, int(np.ceil(sr.n_exposures / 2))):
            why.append('n_exp<half')
        if not (np.isfinite(a['fwhm']) and 0.6 <= a['fwhm'] <= max_fwhm):
            why.append('fwhm')
        if not (np.isfinite(a['snr']) and a['snr'] >= min_snr):
            why.append('snr')
        if not a['isolated']:
            why.append('not isolated')
        a['reason'] = '; '.join(why + a['_chan_bad'])
    cands.sort(key=lambda r: -r['flux'])
    if star is not None:
        E0, N0 = tangent_offsets(star[0], star[1], ra0, dec0)
        near = [c for c in cands if np.hypot(c['E'] - E0, c['N'] - N0) < search_radius]
        if near:
            c = max(near, key=lambda q: q['flux'])
            logger.info('user star matched candidate %d (%.2f" away)', c['id'],
                        np.hypot(c['E'] - E0, c['N'] - N0))
            return StarChoice(float(star[0]), float(star[1]), 'user', strip(cands), c['id'],
                              *exp_match(*star))
        logger.warning('no detected source within %.1f" of the given --star; using it as given',
                       search_radius)
        _, _, table = exp_match(*star)
        return StarChoice(float(star[0]), float(star[1]), 'user', strip(cands), None, exp_table=table)
    good = [c for c in cands if c['qualifies']]
    if not good:
        relaxed = [c for c in cands if np.isfinite(c['fwhm']) and c['fwhm'] <= max_fwhm
                   and np.isfinite(c['snr']) and c['snr'] >= min_snr and c['chan_ok']]
        if relaxed:
            # a preference, not a requirement: a source seen in most exposures beats a brighter one-off
            half = max(1, int(np.ceil(sr.n_exposures / 2)))
            relaxed.sort(key=lambda q: (q['n_exp'] < half, -q['flux']))
            logger.warning('no candidate meets all criteria (seen in >= half the exposures, '
                           'isolated); using the brightest compact one -- check star_candidates.csv')
            good = relaxed
    if not good:
        raise NoStarError('no bright, compact reference star found (need S/N >= %.0f and FWHM <= '
                          '%.1f"); pass --star RA DEC' % (min_snr, max_fwhm), strip(cands))
    c = good[0]
    return StarChoice(c['ra'], c['dec'], 'auto', strip(cands), c['id'], *exp_match(c['ra'], c['dec']))


# ----------------------------------------------------------------------------- measurement
@dataclass
class MeasureOpts:
    profile: str = 'gaussian'
    beta: float = 3.5
    sigma_mode: str = 'auto'                 #: auto | free | kolmogorov
    min_ap_snr: float = 3.0                  #: aperture S/N below this -> LOWSN
    bg: str = 'const'
    fit_radius: Optional[float] = None
    search_radius: float = 4.0
    max_cen_err: float = 0.4                 #: centroid error (arcsec) above this -> LOWSN
    min_fib: int = 12
    max_chi2r: float = 5.0
    moment_tol: float = 0.3
    exclusions: Optional[list] = None
    destripe: bool = True                    #: subtract per-fibre-map-row medians (needs inst)


@dataclass
class Centroids:
    exp_ids: List[str]
    bins: List[Bin]
    ra0: float
    dec0: float
    E: np.ndarray
    N: np.ndarray
    sE: np.ndarray
    sN: np.ndarray
    rho: np.ndarray
    xi: np.ndarray
    yi: np.ndarray
    fwhm: np.ndarray
    snr: np.ndarray
    nfib: np.ndarray
    chi2r: np.ndarray
    flags: np.ndarray
    mE: np.ndarray
    mN: np.ndarray
    ellip: np.ndarray
    edge: np.ndarray
    apsnr: np.ndarray                        #: aperture S/N of the star in the bin (destriped)
    ref: dict = field(default_factory=dict)  #: per-exposure broad red reference (E, N, fwhm, lam, ok)

    @property
    def lam(self):
        return np.array([b.lam_eff for b in self.bins])

    def good(self, fatal=FATAL_DEFAULT):
        return (self.flags & fatal) == 0

    def rows(self):
        out = []
        for e, eid in enumerate(self.exp_ids):
            for j, b in enumerate(self.bins):
                out.append(dict(exp=eid, channel=b.channel, bin=b.index, lam_lo=b.lo, lam_hi=b.hi,
                                lam_eff=b.lam_eff, role=b.role, E=self.E[e, j], N=self.N[e, j],
                                sig_E=self.sE[e, j], sig_N=self.sN[e, j], rho=self.rho[e, j],
                                x_inst=self.xi[e, j], y_inst=self.yi[e, j], fwhm=self.fwhm[e, j],
                                snr=self.snr[e, j], nfib=int(np.nan_to_num(self.nfib[e, j])), chi2r=self.chi2r[e, j],
                                ellipticity=self.ellip[e, j], edge_dist=self.edge[e, j],
                                ap_snr=self.apsnr[e, j],
                                moment_E=self.mE[e, j], moment_N=self.mN[e, j],
                                flags=int(self.flags[e, j]), flag_names=flag_names(self.flags[e, j])))
        return out


def _flags_for(pf, opts, edge_d, hole, apsnr=np.nan):
    f = 0
    if pf is None:
        return FLAGS['FAILED']
    if np.isfinite(edge_d) and edge_d < 1.5 * pf.fwhm:
        f |= FLAGS['EDGE']
    if pf.at_bound:
        f |= FLAGS['BOUND']
    if pf.nfib < opts.min_fib:
        f |= FLAGS['FEWFIB']
    cerr = np.sqrt(max(pf.cov[0, 0], 0) + max(pf.cov[1, 1], 0))
    # signal from the PSF-independent aperture S/N; precision from the centroid error. The fit
    # amplitude S/N is deflated by sqrt(chi2r) wherever the background is structured.
    weak = (np.isfinite(apsnr) and apsnr < opts.min_ap_snr) or pf.amp <= 0
    if weak or not cerr <= opts.max_cen_err:
        f |= FLAGS['LOWSN']
    if pf.chi2r > opts.max_chi2r:
        f |= FLAGS['CHI2']
    if np.isfinite(pf.mx) and np.hypot(pf.mx - pf.x, pf.my - pf.y) > opts.moment_tol:
        f |= FLAGS['MOMENT']
    if pf.core_deficit < -0.10:
        f |= FLAGS['SATCORE']
    if hole:
        f |= FLAGS['HOLE']
    if pf.secondary > 0.35:
        f |= FLAGS['MORPH']
    return f


def aperture_snr(E, N, v, centre, fwhm):
    """S/N of the summed flux within r = max(1", 0.9 FWHM) of ``centre``, with background and
    per-fibre noise from the robust median / MAD of an annulus (3 FWHM .. 3 FWHM + 3"). Independent
    of the PSF fit, so it measures how much signal a bin carries even when the fit fails."""
    r = np.hypot(E - centre[0], N - centre[1])
    fin = np.isfinite(v)
    ap = fin & (r < max(1.0, 0.9 * fwhm))
    ann = fin & (r > 3.0 * fwhm) & (r < 3.0 * fwhm + 3.0)
    if ap.sum() < 3 or ann.sum() < 10:
        return np.nan
    bg = np.median(v[ann])
    sig = 1.4826 * np.median(np.abs(v[ann] - bg))
    return float(np.sum(v[ap] - bg) / (sig * np.sqrt(ap.sum()))) if sig > 0 else np.nan


def bins_for_snr(apsnr_broad, nmax, target, nmin=1):
    """Bins per channel so each bin keeps S/N >= target (S/N scales as sqrt(bin width)):
    clip(floor((S/N_broad / target)^2), nmin, nmax)."""
    if not np.isfinite(apsnr_broad) or target <= 0:
        return nmax
    return int(np.clip(np.floor((apsnr_broad / target) ** 2), min(nmin, nmax), nmax))


def _locate(E, N, v, search_radius, guess=(0.0, 0.0)):
    """Brightest 1"-neighbourhood within ``search_radius`` of ``guess`` -> starting position."""
    fin = np.isfinite(v)
    cand = np.flatnonzero(fin & (np.hypot(E - guess[0], N - guess[1]) < search_radius))
    if cand.size == 0:
        return None
    vv = np.where(fin, v, 0.0)
    score = [vv[np.hypot(E - E[i], N - N[i]) < 1.0].sum() for i in cand]
    i = cand[int(np.argmax(score))]
    return float(E[i]), float(N[i])


def measure_star(sr, star_radec, bins, opts=None, affines=None, shifts=None, inst=None,
                 guesses=None, skip=None) -> Centroids:
    """Centroid the star in every exposure and bin.

    Per exposure: a broad-band reference fit (red, else green; free FWHM) sets the start position,
    the reference FWHM and the aperture radius clip(2.5 FWHM, 2.5", 5") (>= 2.3 FWHM). Per bin the
    fit starts at the reference position and is re-centred up to 3 times so the aperture follows
    the star. ``guesses`` {exposure: (E, N)} (the per-exposure detections) start the search there
    (radius 2") instead of at the nominal position (``search_radius``) -- needed when the header
    pointing drifts between exposures. Exposures in ``skip`` (no detection of the star there) are
    not searched at all: they are flagged FAILED instead of fitting whatever lies nearby. ``inst`` (io.instrument_xy) enables row destriping of every
    bin's fibre values (``opts.destripe``). ``shifts(e, lam) -> (n, 2)`` (a :class:`dar.Correction`)
    moves the fibres first, to re-measure after correction. ``affines`` (io.exposure_affines about
    the same star position) adds instrument-frame centroids."""
    opts = opts or MeasureOpts()
    excl = opts.exclusions if opts.exclusions is not None else default_exclusions()
    ra0, dec0 = star_radec
    nexp, nb = sr.n_exposures, len(bins)
    arr = lambda fill=np.nan: np.full((nexp, nb), fill, float)   # noqa: E731
    E_, N_, sE, sN, rho, xi, yi, fw, snr, nf, chi, mE, mN, el, ed, aps = (arr() for _ in range(16))
    flags = np.zeros((nexp, nb), int)
    ref = dict(E=np.full(nexp, np.nan), N=np.full(nexp, np.nan), fwhm=np.full(nexp, np.nan),
               lam=np.full(nexp, np.nan), channel=[''] * nexp, ok=np.zeros(nexp, bool))
    pos = {c: tangent_offsets(st.ra, st.dec, ra0, dec0) for c, st in sr.channels.items()}
    keys = {c: row_keys(xy) for c, xy in (inst or {}).items()} if opts.destripe else {}
    bb = {b.channel: b for b in broad_bins(bins)}
    lim = 8.0

    for e in range(nexp):
        def coords(c, lam, centre):
            """Fibres within ``lim`` of ``centre`` plus (if destriping) the rest of their rows."""
            st = sr.channels[c]
            rows = np.flatnonzero(st.exposure == e)
            E, N = pos[c][0][rows], pos[c][1][rows]
            if shifts is not None:
                d = shifts(e, [lam])[0]
                E, N = E + d[0], N + d[1]
            fin = np.isfinite(E) & np.isfinite(N)
            near = fin & (np.hypot(E - centre[0], N - centre[1]) < lim)
            k = keys[c][rows] if c in keys else None
            need = (fin & np.isin(k, np.unique(k[near]))) if k is not None else near
            return rows[need], E[need], N[need], (k[need] if k is not None else None), hull_of(E[fin], N[fin])

        def values(c, rows, lo, hi, E, N, k, centre, fwhm):
            v, var = bin_values(sr.channels[c], rows, lo, hi, excl)
            if k is not None:
                v = destripe(v, k, E, N, centre, excl_r=max(5.0, 3.0 * fwhm))
            return v, var

        if skip is not None and e in skip:
            logger.warning('%s: star not detected in this exposure; no star measurement (the ADC '
                           'correction comes from the other exposures)', sr.exposures[e].exposure_id)
            flags[e, :] = FLAGS['FAILED']
            continue
        g0 = (guesses or {}).get(e)
        if g0 is not None and shifts is not None:
            lam_g = bb['red'].lam_eff if 'red' in bb else next(iter(bb.values())).lam_eff
            d = shifts(e, [lam_g])[0]
            g0 = (g0[0] + d[0], g0[1] + d[1])
        centre0 = g0 if g0 is not None else (0.0, 0.0)
        srad = 2.0 if g0 is not None else opts.search_radius

        # broad-band reference
        start = None
        for c in ('red', 'green', 'blue'):
            if c not in bb or c not in sr.channels:
                continue
            rows, E, N, k, _ = coords(c, bb[c].lam_eff, centre0)
            if rows.size < 10:
                continue
            v, var = values(c, rows, bb[c].lo, bb[c].hi, E, N, k, centre0, 1.5)
            g = _locate(E, N, v, srad, centre0)
            if g is None:
                continue
            pf = fit_psf(E, N, v, var, g[0], g[1], 3.0, opts.profile, opts.beta, bg=opts.bg)
            if pf is not None:
                rad = opts.fit_radius or float(np.clip(2.5 * pf.fwhm, 2.5, 5.0))
                rad = max(rad, 2.3 * pf.fwhm)
                pf2 = fit_psf(E, N, v, var, pf.x, pf.y, rad, opts.profile, opts.beta, bg=opts.bg,
                              fwhm_guess=pf.fwhm)
                pf = pf2 or pf
            if pf is not None and not pf.at_bound:
                ref['E'][e], ref['N'][e], ref['fwhm'][e] = pf.x, pf.y, pf.fwhm
                ref['lam'][e], ref['channel'][e], ref['ok'][e] = bb[c].lam_eff, c, True
                start = (pf.x, pf.y)
                break
        if start is None:
            logger.warning('%s: star not found within %.1f" of its expected position in any channel; '
                           'exposure skipped', sr.exposures[e].exposure_id, srad)
            flags[e, :] = FLAGS['FAILED']
            continue
        fw_ref, lam_ref = ref['fwhm'][e], ref['lam'][e]
        radius = opts.fit_radius or float(np.clip(2.5 * fw_ref, 2.5, 5.0))
        radius = max(radius, 2.3 * 1.15 * fw_ref) if opts.fit_radius is None else radius

        for j, b in enumerate(bins):
            if b.channel not in sr.channels:
                flags[e, j] = FLAGS['FAILED']
                continue
            rows, E, N, k, hull = coords(b.channel, b.lam_eff, start)
            if rows.size < 10:
                flags[e, j] = FLAGS['FAILED']
                continue
            v, var = values(b.channel, rows, b.lo, b.hi, E, N, k, start, fw_ref)
            w_exp = fw_ref * (b.lam_eff / lam_ref) ** -0.2           # seeing scaled to this bin

            def follow(fixed, wb):
                """Fit, re-centring the aperture on the star up to 3 times."""
                cx, cy, out = start[0], start[1], None
                for _ in range(4):
                    p = fit_psf(E, N, v, var, cx, cy, radius, opts.profile, opts.beta, fwhm_fixed=fixed,
                                bg=opts.bg, fwhm_guess=w_exp, fwhm_bounds=wb)
                    if p is None:
                        break
                    out = p
                    if np.hypot(p.x - cx, p.y - cy) < 0.05:
                        break
                    cx, cy = p.x, p.y
                return out

            if opts.sigma_mode == 'kolmogorov':
                pf = follow(w_exp, None)
            else:
                pf = follow(None, (0.7 * w_exp, 1.6 * w_exp) if opts.sigma_mode == 'auto' else None)
                if opts.sigma_mode == 'auto' and (pf is None or pf.at_bound):
                    # width ran to its limit (faint bin collapsing onto a fibre, or a smeared one):
                    # a fixed seeing-scaled width still gives an essentially unbiased centroid
                    pfx = follow(w_exp, None)
                    if pfx is not None and not pfx.at_bound:
                        pf = pfx
            ok_c = pf is not None and np.hypot(pf.x - start[0], pf.y - start[1]) < 2.0
            aps[e, j] = aperture_snr(E, N, v, (pf.x, pf.y) if ok_c else start, fw_ref)
            dist = hull_distance(hull, pf.x, pf.y) if pf is not None else np.nan
            hole = False
            if pf is not None:
                nearc = np.hypot(E - pf.x, N - pf.y) < 0.75 * pf.fwhm
                hole = bool(np.any(nearc & ~np.isfinite(v)))
            flags[e, j] = _flags_for(pf, opts, dist, hole, aps[e, j])
            if pf is None:
                continue
            E_[e, j], N_[e, j] = pf.x, pf.y
            sE[e, j], sN[e, j] = np.sqrt(max(pf.cov[0, 0], 0)), np.sqrt(max(pf.cov[1, 1], 0))
            rho[e, j] = pf.cov[0, 1] / max(sE[e, j] * sN[e, j], 1e-30)
            fw[e, j], snr[e, j], nf[e, j], chi[e, j] = pf.fwhm, pf.snr, pf.nfib, pf.chi2r
            mE[e, j], mN[e, j], el[e, j], ed[e, j] = pf.mx, pf.my, pf.ellipticity, dist
            if affines is not None and shifts is None:
                aff = (affines.get(b.channel) or [None] * nexp)[e]
                if aff is not None:
                    xi[e, j], yi[e, j] = aff.to_inst([pf.x, pf.y])[0]
    return Centroids([m.exposure_id for m in sr.exposures], list(bins), ra0, dec0, E_, N_, sE, sN,
                     rho, xi, yi, fw, snr, nf, chi, flags, mE, mN, el, ed, aps, ref)
