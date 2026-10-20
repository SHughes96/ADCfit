"""ADC-corrected copies of a standard star's RSS files, for building a sensitivity function.

The CubeViewer picker sums one fixed set of fibres at every wavelength, so a star whose centroid
moves with wavelength (ADC error + DAR) loses a wavelength-dependent fraction of its light. An RSS
gives each fibre ONE position, so the only way to make a fixed aperture follow the star is to move
the light: at every wavelength the fibre-plane image is resampled so the star sits at its
lambda_ref position,

    new_i(lam) = old( p_i + track(lam) - track(lam_ref) ),

p_i the fibre's sky position and ``track`` the star's measured centroid track. Per channel:

1. every fibre is resampled onto a common wavelength grid (half the native dispersion; neighbouring
   fibres' WAVE grids differ by ~7 pixels, so the spatial step must be done at a common lambda);
2. the grid is split into slabs within which the shift changes by < ``slab_tol``; per slab the
   shifted fibre positions are located in a Delaunay triangulation of the good fibres and the
   values interpolated barycentrically (weights renormalised over good neighbours; variance with
   the squared weights);
3. the result is interpolated back onto each fibre's own WAVE.

The output is a byte-for-byte copy of the input file except for the SKYSUB (or FLUX) and ERROR
data (and FLAM/FLAM_ERR if present) plus HISTORY lines: same name (in another directory), HDUs,
order, dtypes and header keywords, so CubeViewer loads it like any pipeline RSS. Where the corrected
value is undefined (shifted point outside the IFU, dead fibre) the original value is kept; MASK is
not touched and no NaNs are introduced. Full provenance goes to ``<stem>_adcshift.json``.
"""

import json
import logging
import os
from typing import Dict, List

import numpy as np
from astropy.io import fits
from scipy.spatial import Delaunay

from adcfit.io import tangent_offsets
from adcfit.stack import plan_slabs

logger = logging.getLogger(__name__)

#: planes moved as (value, error) pairs; the first present value plane of each pair is used
PLANE_PAIRS = ((('SKYSUB', 'FLUX'), 'ERROR'), (('FLAM',), 'FLAM_ERR'))


def _to_grid(wave, flux, var, dead, wl):
    """Every fibre's (flux, var) linearly interpolated onto ``wl``; bad = outside its coverage,
    non-finite or a dead fibre."""
    n = flux.shape[0]
    X = np.zeros((n, wl.size), np.float64)
    V = np.full((n, wl.size), np.inf)
    for i in range(n):
        if dead[i]:
            continue
        m = np.isfinite(wave[i]) & np.isfinite(flux[i]) & np.isfinite(var[i])
        if m.sum() < 2:
            continue
        o = np.argsort(wave[i][m])
        w = wave[i][m][o]
        X[i] = np.interp(wl, w, flux[i][m][o], left=np.nan, right=np.nan)
        V[i] = np.interp(wl, w, var[i][m][o], left=np.inf, right=np.inf)
    bad = ~np.isfinite(X) | ~np.isfinite(V)
    X[bad] = 0.0
    V[bad] = np.inf
    return X, V, bad


def _weights(E, N, good, shift, targets):
    """Barycentric interpolation of the good-fibre values at (E, N)[targets] + shift.
    Returns (rows, verts (n, 3), bary (n, 3)) for the targets inside the triangulation."""
    vidx = np.flatnonzero(good)
    tri = Delaunay(np.c_[E[vidx], N[vidx]])
    P = np.c_[E[targets], N[targets]] + np.asarray(shift)[None, :]
    simp = tri.find_simplex(P)
    inside = simp >= 0
    T = tri.transform[simp[inside]]
    b = np.einsum('ijk,ik->ij', T[:, :2, :], P[inside] - T[:, 2, :])
    bary = np.c_[b, 1.0 - b.sum(axis=1)]
    return targets[inside], vidx[tri.simplices[simp[inside]]], bary


def correct_channel(wave, planes, E, N, dead, track, lam_ref, slab_tol=0.02, method='cubic'):
    """Move the light of one channel. ``planes`` = [(flux, err), ...] (same rows / WAVE).
    Returns ([(flux_new, err_new), ...], info). Undefined pixels keep the input values."""
    disp = float(np.nanmedian(np.abs(np.diff(wave, axis=1))))
    wl = np.arange(np.nanmin(wave), np.nanmax(wave) + disp / 2, disp / 2)
    shift = track(wl) - track([lam_ref])[0][None, :]                   # (nw, 2) arcsec
    slabs = plan_slabs(shift[None], slab_tol)
    fin = np.isfinite(E) & np.isfinite(N)
    good = fin & ~dead
    targets = np.flatnonzero(fin)
    geo = [(s, *_weights(E, N, good, s.shift[0], targets)) for s in slabs]
    if method == 'cubic':
        from scipy.interpolate import CloughTocher2DInterpolator
        vidx = np.flatnonzero(good)
        tri = Delaunay(np.c_[E[vidx], N[vidx]])
    out, n_kept = [], 0
    for flux, err in planes:
        X, V, bad = _to_grid(wave, flux, np.asarray(err, float) ** 2, dead, wl)
        Y = np.zeros_like(X)
        VY = np.full_like(V, np.inf)
        OK = np.zeros(X.shape, bool)
        for s, rows, verts, bary in geo:
            sl = slice(s.i0, s.i1)
            w = bary[:, :, None] * ~bad[verts, sl]                       # (n, 3, nws)
            ws = w.sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                y = (w * X[verts, sl]).sum(axis=1) / ws
                v = (w ** 2 * np.where(bad[verts, sl], 0.0, V[verts, sl])).sum(axis=1) / ws ** 2
            ok = ws >= 0.5
            if method == 'cubic':
                # C1 piecewise-cubic values (less pixel-locking than linear); bad input pixels are
                # filled with the linear estimate first so they do not propagate
                Xf = X[vidx, sl].copy()
                badv = bad[vidx, sl]
                if badv.any():
                    lin = np.zeros_like(Xf)
                    lin_rows = {r: k for k, r in enumerate(rows)}
                    for k, r in enumerate(vidx):
                        if badv[k].any() and r in lin_rows:
                            lin[k] = np.nan_to_num(y[lin_rows[r]])
                    Xf[badv] = lin[badv]
                P = np.c_[E[rows], N[rows]] + s.shift[0][None, :]
                yc = CloughTocher2DInterpolator(tri, Xf)(P)
                y = np.where(np.isfinite(yc), yc, y)
            Y[rows, sl] = np.where(ok, y, 0.0)
            VY[rows, sl] = np.where(ok, v, np.inf)
            OK[rows, sl] = ok
        fnew = np.array(flux, copy=True)
        enew = np.array(err, copy=True)
        for i in targets:
            if dead[i]:
                continue
            m = np.isfinite(wave[i])
            w_i = wave[i][m]
            okn = np.interp(w_i, wl, OK[i].astype(float)) > 0.999
            vi = np.interp(w_i, wl, np.where(OK[i], VY[i], 0.0))
            fi = np.interp(w_i, wl, Y[i])
            idx = np.flatnonzero(m)[okn]
            fnew[i, idx] = fi[okn]
            enew[i, idx] = np.sqrt(vi[okn])
            n_kept += int((~okn).sum())
        out.append((fnew, enew))
    info = dict(nslab=len(slabs), wave=wl[::20].tolist(), shift_E=shift[::20, 0].tolist(),
                shift_N=shift[::20, 1].tolist(), max_shift=float(np.max(np.hypot(*shift.T))),
                n_dead=int(dead.sum()), n_kept_pixels=n_kept // max(len(planes), 1))
    return out, info


def correct_rss(sib: Dict[str, str], sr, track, lam_ref, star_radec, outdir, slab_tol=0.02,
                overwrite=False, version='', method='cubic') -> dict:
    """Write ADC-corrected copies of the standard's channel RSS files ``sib`` into ``outdir``
    (same file names). ``sr`` = the standard's super-RSS (positions and dead-fibre masks, rows in
    file order). Returns dict(paths={channel: path}, info={channel: ...}, sidecar=path)."""
    outdir = os.path.abspath(outdir)
    for p in sib.values():
        if os.path.dirname(os.path.abspath(p)) == outdir:
            raise ValueError(f'--std-correct-rss must not be the directory of the input RSS ({outdir})')
    targets = {c: os.path.join(outdir, os.path.basename(p)) for c, p in sib.items()}
    clash = [p for p in targets.values() if os.path.exists(p)]
    if clash and not overwrite:
        raise FileExistsError('corrected RSS files exist (use --overwrite):\n  ' + '\n  '.join(clash))
    os.makedirs(outdir, exist_ok=True)
    paths, infos = {}, {}
    for c, src in sib.items():
        if c not in sr.channels:
            logger.warning('standard %s: channel not loaded; not corrected', c)
            continue
        st = sr.channels[c]
        with fits.open(src) as h:
            wave = np.asarray(h['WAVE'].data, float)
            if wave.shape[0] != st.n_fibres:
                raise ValueError(f'{src}: {wave.shape[0]} rows but the super-RSS has {st.n_fibres}')
            pairs = []
            for vals, en in PLANE_PAIRS:
                vn = next((v for v in vals if v in h), None)
                if vn is not None and en in h:
                    pairs.append((vn, en))
            planes = [(np.asarray(h[vn].data), np.asarray(h[en].data)) for vn, en in pairs]
        E, N = tangent_offsets(st.ra, st.dec, *star_radec)
        dead = np.all(st.mask, axis=1)
        new, info = correct_channel(wave, planes, E, N, dead, track, lam_ref, slab_tol, method)
        info['method'] = method
        with fits.open(src) as h:
            for (vn, en), (fnew, enew) in zip(pairs, new):
                h[vn].data = fnew.astype(h[vn].data.dtype, copy=False)
                h[en].data = enew.astype(h[en].data.dtype, copy=False)
            hdr = h[0].header
            hdr['HISTORY'] = f'adcfit {version}: ADC/DAR-corrected copy of {os.path.basename(src)}'
            hdr['HISTORY'] = (f'adcfit: {"/".join(p[0] for p in pairs)} resampled so the star at '
                              f'{star_radec[0]:.6f} {star_radec[1]:+.6f} stays at its {lam_ref:.0f} A position')
            hdr['HISTORY'] = (f'adcfit: max shift {info["max_shift"]:.2f} arcsec, {info["nslab"]} slabs '
                              f'(tol {slab_tol} arcsec); details in *_adcshift.json')
            h.writeto(targets[c], overwrite=True, output_verify='exception')
        paths[c] = targets[c]
        infos[c] = dict(info, source=os.path.abspath(src), planes=[list(p) for p in pairs])
        logger.info('wrote ADC-corrected %s RSS %s (max shift %.2f", %d slabs, %d pixels kept as input)',
                    c, targets[c], info['max_shift'], info['nslab'], info['n_kept_pixels'])
    stem = os.path.basename(next(iter(sib.values()))).split('_RSS_')[0]
    side = os.path.join(outdir, f'{stem}_adcshift.json')
    with open(side, 'w') as fh:
        json.dump(dict(adcfit=version, lam_ref=lam_ref, star=list(star_radec), slab_tol=slab_tol,
                       relation='new(p, lam) = old(p + track(lam) - track(lam_ref))', channels=infos),
                  fh, indent=1)
    return dict(paths=paths, info=infos, sidecar=side)


def compare_rss(orig, new, changed=('SKYSUB', 'FLUX', 'ERROR', 'FLAM', 'FLAM_ERR')) -> List[str]:
    """Structure check of a corrected RSS against its original: same HDU names, types, order,
    shapes, dtypes and header keywords (HISTORY aside); unchanged data everywhere except the
    ``changed`` planes, which must not gain NaNs. Returns the problems (empty = identical)."""
    bad = []
    with fits.open(orig) as a, fits.open(new) as b:
        if [x.name for x in a] != [x.name for x in b]:
            return [f'HDU list {[x.name for x in a]} vs {[x.name for x in b]}']
        for x, y in zip(a, b):
            if type(x) is not type(y):
                bad.append(f'{x.name}: {type(x).__name__} vs {type(y).__name__}')
                continue
            kx = [k for k in x.header if k != 'HISTORY']
            ky = [k for k in y.header if k != 'HISTORY']
            if kx != ky:
                bad.append(f'{x.name}: header keywords differ ({sorted(set(kx) ^ set(ky))})')
            if x.data is None or y.data is None:
                if (x.data is None) != (y.data is None):
                    bad.append(f'{x.name}: data presence differs')
                continue
            if isinstance(x, fits.BinTableHDU):
                if x.data.dtype != y.data.dtype or x.data.tobytes() != y.data.tobytes():
                    bad.append(f'{x.name}: table differs')
                continue
            if x.data.shape != y.data.shape or x.data.dtype != y.data.dtype:
                bad.append(f'{x.name}: {x.data.shape}/{x.data.dtype} vs {y.data.shape}/{y.data.dtype}')
                continue
            if x.name in changed:
                if np.issubdtype(y.data.dtype, np.floating) and np.isnan(y.data).sum() > np.isnan(x.data).sum():
                    bad.append(f'{x.name}: NaNs introduced')
            elif not np.array_equal(x.data, y.data, equal_nan=np.issubdtype(x.data.dtype, np.floating)):
                bad.append(f'{x.name}: data changed')
    return bad


def aperture_fraction(path, E, N, centre, radii, r_tot, centre_tot, width=100.0, plane=None):
    """Aperture captured fraction vs wavelength, the way the picker sums (all fibres within R of
    ``centre`` -- fixed (E, N), or a callable of lambda), relative to the total within ``r_tot`` of
    ``centre_tot(lam)``. Returns rows."""
    with fits.open(path) as h:
        W = np.asarray(h['WAVE'].data, float)
        F = np.asarray(h[plane or next(n for n in ('SKYSUB', 'FLUX', 'FLAM') if n in h)].data, float)
    lo, hi = np.nanpercentile(W, 1), np.nanpercentile(W, 99)
    rows = []
    for a in np.arange(lo, hi - width / 2, width):
        b = a + width
        inb = (W >= a) & (W < b)
        per = np.where(inb, np.nan_to_num(F), 0.0).sum(axis=1) / np.maximum(inb.sum(axis=1), 1)
        lam = 0.5 * (a + b)
        ct = centre_tot(lam)
        tot = per[np.hypot(E - ct[0], N - ct[1]) <= r_tot].sum()
        r = dict(lam=lam)
        ca = centre(lam) if callable(centre) else centre
        for R in radii:
            r[f'f_R{R:g}'] = per[np.hypot(E - ca[0], N - ca[1]) <= R].sum() / tot if tot > 0 else np.nan
        rows.append(r)
    return rows
