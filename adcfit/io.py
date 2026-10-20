"""Inputs, per-exposure header metadata and coordinate frames.

Everything that touches FITS files lives here, so the measurement / fitting / stacking modules can
be driven by synthetic data in the tests.

Header timing (verified on LLAMAS 2026 frames): ``MJD-OBS``, ``UTC`` and ``LST`` are taken at
SHUTTER OPEN; every ``TEL *`` card (``TEL HA``, ``TEL PARANGL``, ``TEL AIRMASS``, ``TEL LST``,
``TEL UTC``) is a snapshot at SHUTTER CLOSE. ``AIRMASS`` is start, ``AIRMASSE`` is end.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

CHANNELS = ('blue', 'green', 'red')
ARCSEC_PER_RAD = 206264.80624709636

# Weather defaults when a card is missing (Las Campanas typical); every use is recorded.
DEFAULT_P_MBAR = 770.0
DEFAULT_T_C = 10.0
DEFAULT_RH = 20.0


# ----------------------------------------------------------------------------- input discovery
@dataclass
class Inputs:
    rss_paths: List[str]
    cube_path: Optional[str] = None
    cube_hdr: Optional[fits.Header] = None


def _is_cube(path):
    try:
        h = fits.getheader(path, 0)
    except Exception:                                    # noqa: BLE001
        return False
    return int(h.get('NAXIS', 0)) == 3 and 'NRSSFILE' in h


def resolve_inputs(inputs, directory=None, obj=None, rss_dir=None) -> Inputs:
    """Turn the CLI inputs into a list of RSS paths (one per exposure, any channel).

    * one combined cube (NAXIS=3 with ``RSSFILn`` provenance) -> its RSS files. Stale absolute paths
      (cube moved between machines) are retried as ``rss_dir/<basename>`` and
      ``<cube_dir>/../extractions/<basename>``;
    * RSS files of any channel;
    * ``directory`` + ``obj`` -> ``discover_field`` (green RSS whose OBJECT starts with ``obj``).
    """
    inputs = list(inputs or [])
    if directory:
        from llamas_pyjamas.Combine.combineField import discover_field
        paths = discover_field(directory, obj or '')
        if not paths:
            raise FileNotFoundError(f'no *_RSS_green.fits with OBJECT starting {obj!r} in {directory}')
        return Inputs(paths)
    if len(inputs) == 1 and _is_cube(inputs[0]):
        cube = os.path.abspath(inputs[0])
        hdr = fits.getheader(cube, 0)
        out, missing = [], []
        for i in range(1, int(hdr['NRSSFILE']) + 1):
            p = str(hdr[f'RSSFIL{i}'])
            cands = [p]
            if rss_dir:
                cands.append(os.path.join(rss_dir, os.path.basename(p)))
            cands.append(os.path.join(os.path.dirname(os.path.dirname(cube)), 'extractions',
                                      os.path.basename(p)))
            hit = next((c for c in cands if os.path.exists(c)), None)
            (out if hit else missing).append(hit or p)
        if missing:
            raise FileNotFoundError('RSS files named in the cube header were not found (pass '
                                    '--rss-dir):\n  ' + '\n  '.join(missing))
        return Inputs(out, cube, hdr)
    if not inputs:
        raise ValueError('no inputs: give a cube, RSS files, or --dir/--object')
    for p in inputs:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    return Inputs([os.path.abspath(p) for p in inputs])


def cube_settings(cube_path, cube_hdr):
    """Combine settings and grids of a delivered cube, so the rebuild matches it.

    Returns dict(pixscale, kernel_fwhm, weighting, units, plane, grid=(wcs2, ny, nx),
    waves={channel: wavelength array}) -- ``waves`` includes sibling ``*_cube_{channel}.fits``.
    The kernel NAME is not stored in pipeline cubes; 'gaussian' is assumed (logged)."""
    h = cube_hdr
    w2 = WCS(h).celestial
    out = dict(pixscale=float(h.get('PIXSCALE', 0.5)), kernel_fwhm=float(h.get('KERNFWHM', 0.9)),
               weighting=str(h.get('WGHT', 'ivar')).strip(), units=str(h.get('UNITS', 'sb')).strip(),
               plane='flam' if 'erg' in str(h.get('BUNIT', '')) else 'skysub',
               grid=(w2, int(h['NAXIS2']), int(h['NAXIS1'])), waves={})
    m = re.search(r'_cube_(blue|green|red)', os.path.basename(cube_path))
    for c in CHANNELS:
        p = cube_path if m is None else cube_path.replace(m.group(0), f'_cube_{c}')
        if os.path.exists(p):
            try:
                with fits.open(p) as hd:
                    ch = str(hd[0].header.get('CHANNEL', c)).strip() or c
                    out['waves'][ch] = np.asarray(hd['WAVELENGTH'].data['WAVELENGTH'], float)
            except Exception as exc:                     # noqa: BLE001
                logger.warning('could not read the WAVELENGTH table of %s (%s)', p, exc)
    logger.info('input cube settings: pixscale %.3f", kernel FWHM %.2f" (kernel assumed gaussian), '
                'weighting %s, units %s; wavelength grids for %s', out['pixscale'],
                out['kernel_fwhm'], out['weighting'], out['units'], sorted(out['waves']))
    return out


def load_field(paths, plane='auto', scales=None, reject_bad_fibres=True):
    from llamas_pyjamas.Combine.superRSS import build_super_rss
    return build_super_rss(paths, plane=plane, channels=list(CHANNELS), scales=scales,
                           reject_bad_fibres=reject_bad_fibres)


# ----------------------------------------------------------------------------- header metadata
def _hours(v):
    """'21:02:50', '20h12m23s', '-3:47:37' or a number -> float hours (nan if absent)."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    s = str(v).strip().lower().replace('h', ':').replace('m', ':').replace('s', '')
    if not s:
        return np.nan
    sign = -1.0 if s.startswith('-') else 1.0
    try:
        parts = [float(p) for p in s.lstrip('+-').split(':') if p != '']
    except ValueError:
        return np.nan
    return sign * sum(p / 60.0 ** i for i, p in enumerate(parts))


def _num(hdr, *keys):
    from llamas_pyjamas.Combine.superRSS import _hdr_float
    return _hdr_float(*[hdr.get(k) for k in keys])


@dataclass
class ExposureInfo:
    """Everything the DAR geometry and the QA tables need about one exposure."""
    index: int
    exp_id: str
    path: str
    mjd: float
    exptime: float
    ra: float                      #: pointing, deg
    dec: float
    lst_start: float               #: hours (LST card, shutter open)
    tel_lst_end: float             #: hours (TEL LST, shutter close)
    ha_end: float                  #: hours (TEL HA, shutter close)
    parang_end: float              #: deg  (TEL PARANGL, shutter close)
    airmass_start: float
    airmass_end: float
    tel_rot: float
    pressure: float                #: mbar
    temperature: float             #: deg C
    humidity: float                #: percent
    seeing: float
    velcorr: float = 1.0
    adc_state: str = ''
    defaulted: List[str] = field(default_factory=list)


def read_exposure_info(sr) -> List[ExposureInfo]:
    """One :class:`ExposureInfo` per ``sr.exposures`` entry (same order = exposure index)."""
    from llamas_pyjamas.Utils.utils import exposure_time
    from llamas_pyjamas.Utils.wcsLlamas import pointing_from_header
    out = []
    for i, m in enumerate(sr.exposures):
        h = fits.getheader(m.path, 0)
        ra, dec, rot = pointing_from_header(h)
        d = []
        P, T, RH = _num(h, 'WTH-PRES'), _num(h, 'WTH-TEMP'), _num(h, 'WTH-HUMD')
        if not np.isfinite(P):
            P = DEFAULT_P_MBAR; d.append('WTH-PRES')
        if not np.isfinite(T):
            T = DEFAULT_T_C; d.append('WTH-TEMP')
        if not np.isfinite(RH):
            RH = DEFAULT_RH; d.append('WTH-HUMD')
        vc = _num(h, 'VELCORR')
        if not np.isfinite(vc):
            vc = 1.0
        et = exposure_time(h, default=None)
        out.append(ExposureInfo(
            index=i, exp_id=m.exposure_id, path=m.path, mjd=_num(h, 'MJD-OBS'),
            exptime=float(et) if et else float(m.exptime),
            ra=np.nan if ra is None else float(ra), dec=np.nan if dec is None else float(dec),
            lst_start=_hours(h.get('LST')), tel_lst_end=_hours(h.get('TEL LST')),
            ha_end=_hours(h.get('TEL HA')), parang_end=_num(h, 'TEL PARANGL'),
            airmass_start=_num(h, 'AIRMASS', 'TEL AIRMASS'), airmass_end=_num(h, 'AIRMASSE'),
            tel_rot=float(rot or 0.0), pressure=P, temperature=T, humidity=RH,
            seeing=_num(h, 'SEEING-1', 'SEEING-2'), velcorr=vc,
            adc_state=str(h.get('FRONTEND STATE', '')).strip(), defaulted=d))
        if d:
            logger.warning('%s: missing %s; using defaults (%.0f mbar, %.0f C, %.0f%%)',
                           m.exposure_id, ', '.join(d), P, T, RH)
    states = {e.adc_state for e in out}
    if len(states) > 1:
        logger.warning('exposures have different FRONTEND STATE values: %s', sorted(states))
    return out


# ----------------------------------------------------------------------------- coordinate frames
def tangent_offsets(ra, dec, ra0, dec0):
    """Gnomonic offsets (E, N) in arcsec of (ra, dec) about (ra0, dec0); all in deg."""
    ra, dec = np.deg2rad(np.asarray(ra, float)), np.deg2rad(np.asarray(dec, float))
    a0, d0 = np.deg2rad(ra0), np.deg2rad(dec0)
    da = ra - a0
    cosc = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(da)
    xi = np.cos(dec) * np.sin(da) / cosc
    eta = (np.cos(d0) * np.sin(dec) - np.sin(d0) * np.cos(dec) * np.cos(da)) / cosc
    return xi * ARCSEC_PER_RAD, eta * ARCSEC_PER_RAD


def tangent_inverse(E, N, ra0, dec0):
    """Inverse of :func:`tangent_offsets` -> (ra, dec) deg."""
    xi = np.asarray(E, float) / ARCSEC_PER_RAD
    eta = np.asarray(N, float) / ARCSEC_PER_RAD
    a0, d0 = np.deg2rad(ra0), np.deg2rad(dec0)
    den = np.cos(d0) - eta * np.sin(d0)
    ra = a0 + np.arctan2(xi, den)
    dec = np.arctan2(np.sin(d0) + eta * np.cos(d0), np.hypot(xi, den))
    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def shift_radec(ra, dec, dE, dN):
    """Move positions by small sky offsets (arcsec, +E, +N)."""
    dec = np.asarray(dec, float)
    return (np.asarray(ra, float) + np.asarray(dE, float) / 3600.0 / np.cos(np.deg2rad(dec)),
            dec + np.asarray(dN, float) / 3600.0)


@dataclass
class Affine:
    """EN = A @ (x, y) + t: instrument fibre-map frame (arcsec) -> sky tangent offsets (arcsec)."""
    A: np.ndarray
    t: np.ndarray
    rms: float

    @property
    def parity(self):                     # +1 = ordinary rotation, -1 = mirrored
        return int(np.sign(np.linalg.det(self.A)))

    @property
    def pa_x(self):                       # sky PA (deg E of N) of the instrument +x axis
        return float(np.rad2deg(np.arctan2(self.A[0, 0], self.A[1, 0])) % 360.0)

    @property
    def scale(self):
        return float(np.sqrt(abs(np.linalg.det(self.A))))

    def to_inst(self, EN):
        return np.linalg.solve(self.A, (np.atleast_2d(EN) - self.t).T).T

    def vec_to_inst(self, v):
        """Transform a displacement (no translation) from sky to instrument frame."""
        return np.linalg.solve(self.A, np.atleast_2d(v).T).T


def fit_affine(xy, EN) -> Optional[Affine]:
    ok = np.all(np.isfinite(xy), axis=1) & np.all(np.isfinite(EN), axis=1)
    if ok.sum() < 10:
        return None
    X = np.c_[xy[ok], np.ones(ok.sum())]
    sol, *_ = np.linalg.lstsq(X, EN[ok], rcond=None)
    A, t = sol[:2].T, sol[2]
    res = EN[ok] - (xy[ok] @ A.T + t)
    return Affine(A, t, float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))))


def instrument_xy(sr) -> Dict[str, np.ndarray]:
    """Per channel, an (N, 2) array of instrument-frame fibre positions (arcsec, fibre-map axes)
    aligned with ``sr.channels[c]`` rows. From FIBERWCS X/Y_FIBERMAP when present, else the
    fibre-map LUT via FIBERMAP (FIBER_ID, BENCHSIDE)."""
    from llamas_pyjamas.CubeViewer.cubeViewRSS import channel_siblings
    from llamas_pyjamas.Utils.wcsLlamas import ARCSEC_PER_FIBRE
    out = {}
    for c, st in sr.channels.items():
        xy = np.full((st.n_fibres, 2), np.nan)
        for e, m in enumerate(sr.exposures):
            rows = np.flatnonzero(st.exposure == e)
            if rows.size == 0:
                continue
            path = channel_siblings(m.path).get(c)
            if path is None:
                continue
            with fits.open(path) as hd:
                if 'FIBERWCS' in hd and 'X_FIBERMAP' in hd['FIBERWCS'].columns.names:
                    t = hd['FIBERWCS'].data
                    x, y = np.asarray(t['X_FIBERMAP'], float), np.asarray(t['Y_FIBERMAP'], float)
                else:
                    from llamas_pyjamas.Utils.register import _fibre_xy
                    t = hd['FIBERMAP'].data
                    x, y = _fibre_xy(t['FIBER_ID'], t['BENCHSIDE'])
            if x.size != rows.size:
                logger.warning('%s %s: fibre table has %d rows but the super-RSS block has %d; '
                               'instrument frame unavailable for this exposure', m.exposure_id, c,
                               x.size, rows.size)
                continue
            bad = (x < 0) & (y < 0)
            xy[rows, 0] = np.where(bad, np.nan, x) * ARCSEC_PER_FIBRE
            xy[rows, 1] = np.where(bad, np.nan, y) * ARCSEC_PER_FIBRE
        out[c] = xy
    return out


def exposure_affines(sr, inst, ra0, dec0) -> Dict[str, List[Optional[Affine]]]:
    """Per channel, per exposure: the instrument -> sky (about ra0, dec0) affine implied by the
    fibre RA/Dec the pipeline assigned (i.e. the WCS in use)."""
    out = {}
    for c, st in sr.channels.items():
        E, N = tangent_offsets(st.ra, st.dec, ra0, dec0)
        lst = []
        for e in range(sr.n_exposures):
            rows = st.exposure == e
            if c not in inst or not rows.any():
                lst.append(None)
                continue
            lst.append(fit_affine(inst[c][rows], np.c_[E[rows], N[rows]]))
        out[c] = lst
    return out


def check_channel_positions(sr, tol=0.05):
    """Max position difference (arcsec) between channels for the same exposure + fibre row. A stale
    per-channel FIBERWCS would fake a chromatic offset, so this is checked before anything else."""
    res = {}
    chans = [c for c in CHANNELS if c in sr.channels]
    for e, m in enumerate(sr.exposures):
        worst = 0.0
        blocks = {c: np.flatnonzero(sr.channels[c].exposure == e) for c in chans}
        ref = next((c for c in chans if blocks[c].size), None)
        for c in chans:
            if c == ref or blocks[c].size != blocks[ref].size:
                continue
            a, b = sr.channels[ref], sr.channels[c]
            ok = np.isfinite(a.ra[blocks[ref]]) & np.isfinite(b.ra[blocks[c]]) & (a.ra[blocks[ref]] > 0)
            if not ok.any():
                continue
            E, N = tangent_offsets(b.ra[blocks[c]][ok], b.dec[blocks[c]][ok],
                                   a.ra[blocks[ref]][ok], a.dec[blocks[ref]][ok])
            worst = max(worst, float(np.nanmax(np.hypot(E, N))))
        res[m.exposure_id] = worst
        if worst > tol:
            logger.warning('%s: channels disagree on fibre positions by up to %.3f" -- a chromatic '
                           'offset of this size would be an artefact of the fibre WCS', m.exposure_id, worst)
    return res


def llamas_pyjamas_version():
    import subprocess
    import llamas_pyjamas
    root = os.path.dirname(os.path.dirname(llamas_pyjamas.__file__))
    try:
        h = subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                           text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return h + ('-dirty' if dirty else '') if h else 'unknown'
    except Exception:                                    # noqa: BLE001
        return 'unknown'


__all__ = ['Inputs', 'resolve_inputs', 'cube_settings', 'load_field', 'ExposureInfo',
           'read_exposure_info', 'tangent_offsets', 'tangent_inverse', 'shift_radec', 'Affine',
           'fit_affine', 'instrument_xy', 'exposure_affines', 'check_channel_positions',
           'llamas_pyjamas_version']
