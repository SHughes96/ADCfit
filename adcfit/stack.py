"""Cube building with wavelength-dependent fibre positions.

``llamas_pyjamas.Combine.cube.combine_cube`` builds ONE spatial kernel matrix S (n_spaxel x n_fibre)
from 1-D fibre positions and applies it to every wavelength plane, so it cannot apply a shift that
varies with wavelength. Here each channel is resampled onto its wavelength grid ONCE (the package's
``_resample_fibres``), the grid is split into slabs within which every exposure's correction varies by
less than ``slab_tol`` (peak-to-peak), and a separate S is built per slab from the shifted positions.
Everything else (kernel, weighting, variance propagation, coverage / NEXP, WCS, metadata) mirrors
combine_cube exactly; a test pins the zero-shift output to combine_cube.

Coverage and NEXP depend on wavelength once positions move: the 2-D COVERAGE/NEXP maps written are
the per-spaxel MINIMUM over slabs (conservative); the full per-slab NEXP goes to a separate
``*_nexpslab.fits`` file (write_nexpslab) so the cube keeps the pipeline layout.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from astropy.io import fits
from scipy.sparse import coo_matrix

from adcfit.io import shift_radec

logger = logging.getLogger(__name__)


@dataclass
class Slab:
    i0: int
    i1: int
    shift: np.ndarray              #: (nexp, 2) arcsec (E, N) applied in this slab


@dataclass
class StackResult:
    cube: object                   #: llamas_pyjamas CoaddCube
    slabs: List[Slab]
    nexpslab: np.ndarray           #: (nslab, ny, nx)
    meta: dict = field(default_factory=dict)


def kernel_matrix(px, py, ny, nx, r, kern):
    """Sparse S (ny*nx x nfibre): the same pixels and weights as combine_cube's per-fibre loop
    (every integer pixel within ``r`` of the fibre), built without a Python loop. Non-finite
    positions are skipped. Returns (S csr, rows, cols)."""
    n = px.size
    idx = np.flatnonzero(np.isfinite(px) & np.isfinite(py) & (np.abs(px) < 1e9) & (np.abs(py) < 1e9))
    k = int(np.ceil(r))
    offs = np.arange(-k, k + 2)
    fx = np.floor(px[idx]).astype(np.int64)
    fy = np.floor(py[idx]).astype(np.int64)
    gx = fx[:, None, None] + offs[None, None, :]
    gy = fy[:, None, None] + offs[None, :, None]
    d2 = (gx - px[idx, None, None]) ** 2 + (gy - py[idx, None, None]) ** 2
    m = (d2 <= r * r) & (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    fi, iy, ix = np.nonzero(m)
    rows = gy[fi, iy, 0] * nx + gx[fi, 0, ix]
    cols = idx[fi]
    vals = kern(np.sqrt(d2[fi, iy, ix])).astype(np.float32)
    S = coo_matrix((vals, (rows, cols)), shape=(ny * nx, n)).tocsr()
    return S, rows, cols


def plan_slabs(corr, tol):
    """Greedy contiguous slabs: extend while every exposure/axis stays within ``tol`` peak-to-peak;
    each slab applies the mid-range shift (max error tol/2). tol <= 0 -> one slab per plane."""
    nexp, nw, _ = corr.shape
    if tol <= 0:
        return [Slab(i, i + 1, corr[:, i].copy()) for i in range(nw)]
    slabs, i0 = [], 0
    lo, hi = corr[:, 0].copy(), corr[:, 0].copy()
    for i in range(1, nw):
        nlo, nhi = np.minimum(lo, corr[:, i]), np.maximum(hi, corr[:, i])
        if np.any(nhi - nlo > tol):
            slabs.append(Slab(i0, i, 0.5 * (lo + hi)))
            i0, lo, hi = i, corr[:, i].copy(), corr[:, i].copy()
        else:
            lo, hi = nlo, nhi
    slabs.append(Slab(i0, nw, 0.5 * (lo + hi)))
    return slabs


def make_grid(sr, pixscale, kernel, kernel_fwhm, cube_grid=None, center=None):
    """The input cube's spatial grid if given, else the package's shared multi-channel grid
    (exactly what combine_field_cubes uses)."""
    if cube_grid is not None:
        return cube_grid
    from llamas_pyjamas.Combine.cube import field_grid
    return field_grid(sr, list(sr.channels), pixscale=pixscale, kernel=kernel,
                      kernel_fwhm=kernel_fwhm, center=center)


def wave_grid(st, dwave=None, cube_wave=None):
    if cube_wave is not None:
        cw = np.asarray(cube_wave, float)
        return cw, float(np.median(np.diff(cw)))
    from llamas_pyjamas.Combine.cube import _wave_grid
    return _wave_grid(st, dwave, None)


def _pixels(wcs2, ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    px, py = wcs2.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    return np.asarray(px, float), np.asarray(py, float)


def stack_channel(sr, channel, grid, wl, dwave, variants: Dict[str, Optional[Callable]], *,
                  kernel='gaussian', kernel_fwhm=0.9, pixscale=0.5, weighting='ivar', units='sb',
                  min_coverage=1, slab_tol=0.05, drop=None) -> Dict[str, StackResult]:
    """Build one cube per variant for ``channel`` on a fixed spatial ``grid`` and wavelength array.

    ``variants`` maps a name to ``None`` (no shift) or a callable ``fn(e, wl) -> (nw, 2)`` giving the
    shift (arcsec, E, N) to add to exposure e's fibre positions at each wavelength (WAVE frame).
    ``drop`` maps a variant name to exposure indices left out of that variant entirely (no data,
    coverage or NEXP; not listed in its RSSFILn cards)."""
    from llamas_pyjamas.Combine.coadd import _kernel
    from llamas_pyjamas.Combine.cube import _resample_fibres, _wcs3d, CoaddCube
    st = sr.channels[channel]
    wcs2, ny, nx = grid
    npix, nw, nexp = ny * nx, wl.size, sr.n_exposures
    x, V, bad = _resample_fibres(st, wl, units)
    r, kern = _kernel(kernel, kernel_fwhm, pixscale)
    et = np.array([e.exptime for e in sr.exposures], float)[st.exposure][:, None]
    out = {}
    for name, fn in variants.items():
        dropped = sorted(set((drop or {}).get(name, ())))
        keep = ~np.isin(st.exposure, dropped)
        fexp = st.exposure[keep]
        exps = np.unique(fexp)
        if fn is None:
            slabs = [Slab(0, nw, np.zeros((nexp, 2)))]
        else:
            corr = np.zeros((nexp, nw, 2))
            for e in range(nexp):
                if e not in dropped:
                    corr[e] = fn(e, wl)
            slabs = plan_slabs(corr, slab_tol)
        data = np.full((nw, npix), np.nan, np.float32)
        var = np.full((nw, npix), np.nan, np.float32)
        cov_min = np.full(npix, np.iinfo(np.int64).max, np.int64)
        nexp_min = np.full(npix, np.iinfo(np.int32).max, np.int32)
        nexpslab = np.zeros((len(slabs), npix), np.int16)
        for s, sl in enumerate(slabs):
            if np.any(sl.shift != 0):
                ra, dec = shift_radec(st.ra[keep], st.dec[keep], sl.shift[fexp, 0], sl.shift[fexp, 1])
            else:
                ra, dec = st.ra[keep], st.dec[keep]
            px, py = _pixels(wcs2, ra, dec)
            S, rows, cols = kernel_matrix(px, py, ny, nx, r, kern)
            S2 = S.multiply(S)
            j = slice(sl.i0, sl.i1)
            b, Vs, xs = bad[keep, j], V[keep, j], x[keep, j]
            if weighting == 'ivar':
                w = np.where(b, 0.0, 1.0 / Vs)
            elif weighting == 'uniform':
                w = np.where(b, 0.0, 1.0)
            elif weighting == 'exptime':
                w = np.where(b, 0.0, et[keep])
            else:
                raise ValueError(f'unknown weighting {weighting!r}')
            Vsafe = np.where(b, 1.0, Vs)
            wx = w * xs
            w2V = np.where(b, 0.0, w * w * Vsafe)
            with np.errstate(invalid='ignore', divide='ignore'):
                den = S.dot(w)
                d = S.dot(wx) / den
                vv = S2.dot(w2V) / den ** 2
            coverage = np.bincount(rows, minlength=npix).astype(np.int64)
            below = coverage < min_coverage
            d[below, :] = np.nan
            vv[below, :] = np.nan
            ne = np.zeros(npix, np.int32)
            ecol = fexp[cols]
            for e in exps:
                touched = np.zeros(npix, bool)
                touched[rows[ecol == e]] = True
                ne += touched
            ne[below] = 0
            data[j] = d.T
            var[j] = vv.T
            cov_min = np.minimum(cov_min, coverage)
            nexp_min = np.minimum(nexp_min, ne)
            nexpslab[s] = ne
        meta = dict(FIELD=sr.field, CHANNEL=channel, WGHT=weighting, UNITS=units,
                    KERNFWHM=kernel_fwhm, PIXSCALE=pixscale, DWAVE=dwave,
                    NEXPTOT=sr.n_exposures - len(dropped),
                    exposure_paths=[x.path for e, x in enumerate(sr.exposures) if e not in dropped])
        unit = (sr.bunit + '/arcsec2') if units == 'sb' else sr.bunit
        cube = CoaddCube(data=data.reshape(nw, ny, nx), var=var.reshape(nw, ny, nx), wave=wl,
                         coverage=cov_min.reshape(ny, nx), nexp=nexp_min.reshape(ny, nx),
                         wcs=_wcs3d(wcs2, wl[0], dwave), bunit=unit, meta=meta)
        out[name] = StackResult(cube, slabs, nexpslab.reshape(len(slabs), ny, nx))
        logger.info('%s %s: %d slab(s)%s', channel, name, len(slabs),
                    f', exposures {[e + 1 for e in dropped]} left out' if dropped else '')
    return out


def fibre_image(ra, dec, value, var, grid, kernel='gaussian', kernel_fwhm=0.9, pixscale=0.5):
    """Inverse-variance 2-D coadd of per-fibre scalars on a FIXED grid (per-exposure QA images that
    share pixels, unlike combine_image which sizes a grid per call)."""
    from llamas_pyjamas.Combine.coadd import _kernel
    wcs2, ny, nx = grid
    r, kern = _kernel(kernel, kernel_fwhm, pixscale)
    ok = np.isfinite(value) & np.isfinite(var) & (var > 0)
    px, py = _pixels(wcs2, np.asarray(ra)[ok], np.asarray(dec)[ok])
    S, _, _ = kernel_matrix(px, py, ny, nx, r, kern)
    w = 1.0 / np.asarray(var)[ok]
    with np.errstate(invalid='ignore', divide='ignore'):
        img = S.dot(w * np.asarray(value)[ok]) / S.dot(w)
    return img.reshape(ny, nx)


#: the pipeline cube layout (CoaddCube.write); adcfit never changes these HDUs
PIPELINE_HDUS = ('PRIMARY', 'VAR', 'COVERAGE', 'NEXP', 'WAVELENGTH')
_WCS_KEYS = ('NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'WCSAXES', 'CTYPE1', 'CTYPE2', 'CTYPE3', 'CUNIT1',
             'CUNIT2', 'CUNIT3', 'CRPIX1', 'CRPIX2', 'CRPIX3', 'CRVAL1', 'CRVAL2', 'CRVAL3', 'CDELT1',
             'CDELT2', 'CDELT3', 'BUNIT', 'BITPIX')


def write_cube(res: StackResult, path, exp_ids, cards=None, model_rows=None, covfrac=None):
    """CoaddCube.write, untouched (HDUs 0-4 exactly as the pipeline writes them), followed by two
    binary tables with scalar columns only: ADCINFO (provenance cards in its header, one row per
    exposure with the authoritative model) and ADCSLABS (one row per slab and exposure). The
    per-slab NEXP cube is NOT put in the cube (see write_nexpslab). ``covfrac`` reproduces the
    COVFRAC card CubeViewer writes and reads back as the white-light display floor."""
    if covfrac is not None:
        res.cube.meta['COVFRAC'] = float(covfrac)
    res.cube.write(path)
    with fits.open(path, mode='append') as h:
        h.append(_adcinfo_hdu(exp_ids, cards, model_rows))
        h.append(_slabs_hdu(res, exp_ids))
        h.flush()
    return path


def _adcinfo_hdu(exp_ids, cards, model_rows):
    rows = model_rows or [dict(exp=e) for e in exp_ids]
    k = 3
    get = lambda r, a, i: (list(r.get(f'coef_{a}', [])) + [np.nan] * k)[i]   # noqa: E731
    cols = [fits.Column('EXP_INDEX', 'J', array=np.arange(1, len(rows) + 1)),
            fits.Column('EXP', '40A', array=[str(r['exp']) for r in rows]),
            fits.Column('STATUS', '12A', array=[str(r.get('status', '')) for r in rows]),
            fits.Column('ORDER', 'J', array=[int(r.get('order', 0)) for r in rows]),
            fits.Column('LAMREF', 'D', unit='Angstrom', array=[r.get('lam_ref', np.nan) for r in rows]),
            fits.Column('VELCORR', 'D', array=[r.get('velcorr', np.nan) for r in rows]),
            fits.Column('PRES', 'D', unit='mbar', array=[r.get('P', np.nan) for r in rows]),
            fits.Column('TEMP', 'D', unit='deg_C', array=[r.get('T', np.nan) for r in rows]),
            fits.Column('HUMID', 'D', unit='percent', array=[r.get('RH', np.nan) for r in rows])]
    for a in 'EN':
        for i in range(k):
            cols.append(fits.Column(f'C{i}_{a}', 'D', unit='arcsec',
                                    array=[get(r, a, i) for r in rows]))
    th = fits.BinTableHDU.from_columns(cols, name='ADCINFO')
    for key, v in (cards or {}).items():
        th.header[key] = v
    th.header['COMMENT'] = ('offset_a(lam) = C0_a + C1_a*dn + C2_a*dn^2 (arcsec, a = E,N), dn = 1e6*'
                            '[n(lam/VELCORR) - n(LAMREF/VELCORR)], n: Filippenko82 air at PRES/TEMP/HUMID')
    th.header['COMMENT'] = 'HDUs 0-4 are the unmodified LLAMAS cube layout; ADCSLABS lists the applied shifts'
    return th


def _slabs_hdu(res, exp_ids):
    wl = res.cube.wave
    rows = dict(SLAB=[], I0=[], I1=[], WAVE_LO=[], WAVE_HI=[], EXP_INDEX=[], SHIFT_E=[], SHIFT_N=[],
                NEXP_MIN=[], NEXP_MAX=[])
    for k, s in enumerate(res.slabs):
        ne = res.nexpslab[k]
        for e in range(len(exp_ids)):
            for key, v in (('SLAB', k), ('I0', s.i0), ('I1', s.i1), ('WAVE_LO', wl[s.i0]),
                           ('WAVE_HI', wl[s.i1 - 1]), ('EXP_INDEX', e + 1), ('SHIFT_E', s.shift[e, 0]),
                           ('SHIFT_N', s.shift[e, 1]), ('NEXP_MIN', int(ne[ne > 0].min()) if (ne > 0).any() else 0),
                           ('NEXP_MAX', int(ne.max()))):
                rows[key].append(v)
    fmt = dict(SLAB='J', I0='J', I1='J', EXP_INDEX='J', NEXP_MIN='J', NEXP_MAX='J')
    unit = dict(WAVE_LO='Angstrom', WAVE_HI='Angstrom', SHIFT_E='arcsec', SHIFT_N='arcsec')
    th = fits.BinTableHDU.from_columns([fits.Column(k, fmt.get(k, 'D'), unit=unit.get(k), array=v)
                                        for k, v in rows.items()], name='ADCSLABS')
    th.header['COMMENT'] = ('planes I0..I1-1 were built with fibre positions shifted by (SHIFT_E, SHIFT_N) '
                            'for exposure EXP_INDEX (ADCINFO row)')
    return th


def write_nexpslab(res: StackResult, path):
    """Per-slab, per-spaxel exposure count (nslab, ny, nx) as a separate QA file."""
    hdr = res.cube.wcs.celestial.to_header()
    hdr['COMMENT'] = 'NEXP per wavelength slab; slab k spans the ADCSLABS rows with SLAB = k'
    fits.HDUList([fits.PrimaryHDU(res.nexpslab.astype(np.int16), header=hdr)]).writeto(path, overwrite=True)
    return path


def coverage_summary(res: StackResult):
    """How the per-slab exposure count changes with wavelength (from ``res.nexpslab``): spaxels at
    full depth in every slab vs in at least one, spaxels that lose an exposure somewhere, and the
    slab with the fewest full-depth spaxels."""
    ne = res.nexpslab.reshape(len(res.slabs), -1).astype(int)
    nmax = int(ne.max()) if ne.size else 0
    covered = ne.max(axis=0) > 0
    full = ne == nmax
    nfull = full.sum(axis=1)
    k = int(np.argmin(nfull))
    wl = res.cube.wave
    s = res.slabs[k]
    return dict(nslab=len(res.slabs), nexp_max=nmax, covered=int(covered.sum()),
                full_all=int(full.all(axis=0).sum()), full_any=int(full.any(axis=0).sum()),
                varying=int((covered & (ne.max(axis=0) != ne.min(axis=0))).sum()),
                worst_slab=k, worst_lo=float(wl[s.i0]), worst_hi=float(wl[s.i1 - 1]), worst_full=int(nfull[k]))


def check_structure(path, reference=None):
    """HDUs 0-4 must match the pipeline layout (names, types, shapes, dtypes, WCS cards; the reference
    cube's when given) and every later HDU must be a binary table with scalar columns only.
    Returns a list of problems (empty = OK)."""
    bad = []
    with fits.open(path) as h:
        names = [x.name for x in h]
        if tuple(names[:5]) != PIPELINE_HDUS:
            bad.append(f'first HDUs {names[:5]} != {list(PIPELINE_HDUS)}')
        for x in h[5:]:
            if not isinstance(x, fits.BinTableHDU):
                bad.append(f'{x.name}: extra HDU is {type(x).__name__}, not a binary table')
            elif any(c.format.repeat != 1 and 'A' not in str(c.format) for c in x.columns):
                bad.append(f'{x.name}: has vector columns')
        if reference and os.path.exists(reference):
            with fits.open(reference) as r:
                for i, n in enumerate(PIPELINE_HDUS):
                    if i >= len(r) or i >= len(h):
                        bad.append(f'HDU {i} missing')
                        continue
                    a, b = h[i], r[i]
                    if type(a) is not type(b) or a.name != b.name:
                        bad.append(f'HDU {i}: {type(a).__name__} {a.name} vs {type(b).__name__} {b.name}')
                        continue
                    if a.is_image and (a.shape != b.shape or a.header.get('BITPIX') != b.header.get('BITPIX')):
                        bad.append(f'{n}: shape/BITPIX {a.shape}/{a.header.get("BITPIX")} vs '
                                   f'{b.shape}/{b.header.get("BITPIX")}')
                    for k in _WCS_KEYS:
                        va, vb = a.header.get(k), b.header.get(k)
                        if (va is None) != (vb is None) or (va is not None and not _same(va, vb)):
                            bad.append(f'{n}: {k} = {va!r} vs reference {vb!r}')
                extra = sorted(set(h[0].header) - set(r[0].header) - {''})
                extra = [k for k in extra if not k.startswith(('RSSFIL', 'COMMENT', 'HISTORY'))]
                if extra:
                    bad.append(f'PRIMARY: cards not in the reference: {extra}')
    return bad


def _same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return np.isclose(float(a), float(b), rtol=1e-9, atol=0)
    return str(a).strip() == str(b).strip()


def compare_to_delivered(cube, path):
    """Rebuild fidelity: median and 99th-percentile |rebuilt - delivered| / sigma, if shapes match."""
    try:
        with fits.open(path) as h:
            d = h[0].data.astype(float)
            v = h['VAR'].data.astype(float)
    except Exception as exc:                             # noqa: BLE001
        return dict(ok=False, note=str(exc))
    if d.shape != cube.data.shape:
        return dict(ok=False, note=f'shape {d.shape} vs rebuilt {cube.data.shape}')
    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.abs(cube.data - d) / np.sqrt(v)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return dict(ok=False, note='no overlapping finite voxels')
    return dict(ok=True, median=float(np.median(z)), p99=float(np.percentile(z, 99)),
                frac_gt1=float(np.mean(z > 1)))
