"""Refraction physics, predicted (no-ADC) dispersion, and the per-exposure refractivity-basis fit.

Model (per exposure, per sky axis a in {E, N}):

    offset_a(lam) = c0_a + c1_a * dn(lam) [+ c2_a * dn(lam)^2],    dn = 1e6 * [n(lam) - n(lam_ref)]

with n the refractive index of air at that exposure's pressure / temperature / humidity, evaluated
at the observed-frame VACUUM wavelength (RSS WAVE is vacuum x VELCORR, so lam_obs = WAVE/VELCORR).
Atmospheric dispersion is exactly proportional to dn, and a mis-set ADC adds a dispersion of almost
the same shape, so this basis extrapolates sanely into the blue -- unlike a polynomial in lambda.
c1 is in arcsec per 1e-6 of refractivity; an uncorrected atmosphere gives c1 = 0.206265 tan(z)
pointing to the zenith (position angle q, the parallactic angle).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from adcfit.io import ARCSEC_PER_RAD

logger = logging.getLogger(__name__)

LAT_LCO = -29.0146              #: Las Campanas (Magellan) latitude, deg
LON_LCO = -70.6926              #: deg (east positive)
SIDEREAL_RATE = 1.0027379093    #: sidereal hours per solar hour
ERR_FLOOR = 0.01                #: arcsec, added in quadrature to every centroid error


# ----------------------------------------------------------------------------- refractivity
def refractivity(lam_vac_A, P_mbar=None, T_C=None, RH=None):
    """Air refractivity n-1 at vacuum wavelength ``lam_vac_A`` (Angstrom).

    Standard air (15 C, 760 mmHg, dry) from the package formula (Edlen 1953 / Filippenko 1982), then
    scaled to (P, T) and corrected for water vapour as in Filippenko (1982, PASP 94, 715). With
    ``P_mbar=None`` the standard-air value is returned."""
    from llamas_pyjamas.Combine.spectrum import _refractivity
    lam = np.asarray(lam_vac_A, float)
    n = _refractivity(lam)
    if P_mbar is None:
        return n
    T = 15.0 if T_C is None else float(T_C)
    Pmm = float(P_mbar) * 0.750062
    n = n * Pmm * (1.0 + (1.049 - 0.0157 * T) * 1e-6 * Pmm) / (720.883 * (1.0 + 0.003661 * T))
    if RH:
        e_sat = 6.1094 * np.exp(17.625 * T / (T + 243.04))            # hPa (Magnus)
        f = float(RH) / 100.0 * e_sat * 0.750062                       # mmHg
        s2 = 1.0 / (lam / 1e4) ** 2
        n = n - (0.0624 - 0.000680 * s2) / (1.0 + 0.003661 * T) * f * 1e-6
    return n


def dn_e6(lam_wave, info, lam_ref_wave):
    """1e6 x [n(lam) - n(lam_ref)] for one exposure; inputs in the RSS WAVE frame."""
    vc = getattr(info, 'velcorr', 1.0) or 1.0
    P, T, RH = info.pressure, info.temperature, info.humidity
    return 1e6 * (refractivity(np.asarray(lam_wave, float) / vc, P, T, RH)
                  - refractivity(float(lam_ref_wave) / vc, P, T, RH))


def basis(lam_wave, info, lam_ref_wave, order):
    d = np.atleast_1d(dn_e6(lam_wave, info, lam_ref_wave))
    return np.stack([np.ones_like(d), d, d ** 2][:order + 1], axis=1)


# ----------------------------------------------------------------------------- geometry
def zenith_parallactic(H_hours, dec_deg, lat_deg=LAT_LCO):
    """Zenith distance (rad) and parallactic angle q (deg, E of N, position angle of the zenith as
    seen from the object) for hour angle ``H_hours``."""
    H = np.deg2rad(np.asarray(H_hours, float) * 15.0)
    d, phi = np.deg2rad(dec_deg), np.deg2rad(lat_deg)
    cosz = np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(H)
    q = np.arctan2(np.sin(H), np.tan(phi) * np.cos(d) - np.sin(d) * np.cos(H))
    return np.arccos(np.clip(cosz, -1, 1)), np.rad2deg(q)


def _wrap_h(h):
    return (np.asarray(h, float) + 12.0) % 24.0 - 12.0


def _wrap_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


@dataclass
class Geometry:
    ok: bool
    source: str = ''                       #: which header card fixed the hour angle
    H_start: float = np.nan                #: hours
    H_end: float = np.nan
    vec: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))   #: <tan z (sin q, cos q)>
    tanz_mean: float = np.nan
    airmass_mean: float = np.nan
    q_mean: float = np.nan                 #: direction of the mean vector, deg
    q_start: float = np.nan
    q_end: float = np.nan
    airmass_start: float = np.nan          #: computed (sec z)
    airmass_end: float = np.nan
    checks: dict = field(default_factory=dict)

    @property
    def p(self):
        """Predicted no-ADC c1 vector (E, N): arcsec per 1e-6 refractivity."""
        return self.vec * ARCSEC_PER_RAD * 1e-6


def exposure_geometry(info, nsub=11, lat=LAT_LCO) -> Geometry:
    """Time-averaged dispersion direction/amplitude over the exposure.

    The hour angle at shutter open comes from ``LST`` - RA (both shutter-open); fallbacks are
    ``TEL HA`` - duration, ``TEL LST`` - RA - duration, then astropy sidereal time from MJD-OBS.
    The displacement vector tan(z)(sin q, cos q) is averaged over ``nsub`` sub-exposures, which is
    the correct mean for a centroid integrated over the exposure. Computed start airmass, end
    parallactic angle and end hour angle are cross-checked against the header cards."""
    if not (np.isfinite(info.dec) and np.isfinite(info.ra)):
        return Geometry(False, 'no pointing')
    dur = (info.exptime if np.isfinite(info.exptime) else 0.0) / 3600.0 * SIDEREAL_RATE
    if np.isfinite(info.lst_start):
        H0, src = info.lst_start - info.ra / 15.0, 'LST(start)-RA'
    elif np.isfinite(info.ha_end):
        H0, src = info.ha_end - dur, 'TEL HA(end)-exptime'
    elif np.isfinite(info.tel_lst_end):
        H0, src = info.tel_lst_end - info.ra / 15.0 - dur, 'TEL LST(end)-RA-exptime'
    elif np.isfinite(info.mjd):
        try:
            from astropy.time import Time
            from astropy.utils import iers
            iers.conf.auto_download = False
            lst = Time(info.mjd, format='mjd').sidereal_time('mean', longitude=LON_LCO).hour
            H0, src = lst - info.ra / 15.0, 'MJD-OBS sidereal time'
        except Exception as exc:                          # noqa: BLE001
            logger.warning('%s: no usable time for the DAR prediction (%s)', info.exp_id, exc)
            return Geometry(False, 'no time')
    else:
        return Geometry(False, 'no time')
    H0 = float(_wrap_h(H0))
    t = (np.arange(nsub) + 0.5) / nsub
    z, q = zenith_parallactic(H0 + dur * t, info.dec, lat)
    vec = np.array([np.mean(np.tan(z) * np.sin(np.deg2rad(q))),
                    np.mean(np.tan(z) * np.cos(np.deg2rad(q)))])
    z0, q0 = zenith_parallactic(H0, info.dec, lat)
    z1, q1 = zenith_parallactic(H0 + dur, info.dec, lat)
    g = Geometry(True, src, H0, float(_wrap_h(H0 + dur)), vec, float(np.mean(np.tan(z))),
                 float(np.mean(1.0 / np.cos(z))), float(np.rad2deg(np.arctan2(vec[0], vec[1]))),
                 float(q0), float(q1), float(1 / np.cos(z0)), float(1 / np.cos(z1)))
    chk = {}
    if np.isfinite(info.airmass_start):
        chk['airmass_start'] = g.airmass_start - info.airmass_start
    if np.isfinite(info.airmass_end):
        chk['airmass_end'] = g.airmass_end - info.airmass_end
    if np.isfinite(info.parang_end):
        chk['parang_end_deg'] = float(_wrap_deg(g.q_end - info.parang_end))
    if np.isfinite(info.ha_end):
        chk['ha_end_min'] = float(_wrap_h(g.H_end - info.ha_end) * 60.0)
    g.checks = chk
    bad = [k for k, v in chk.items() if (('deg' in k and abs(v) > 0.5) or ('airmass' in k and abs(v) > 0.01)
                                         or ('min' in k and abs(v) > 1.0))]
    if bad:
        logger.warning('%s: computed geometry disagrees with the header (%s); the predicted DAR uses '
                       'the computed values', info.exp_id,
                       ', '.join(f'{k} {chk[k]:+.3f}' for k in bad))
    return g


def predicted_dar(info, geom, lam_wave, lam_ref_wave):
    """Uncorrected-atmosphere centroid offset (n, 2) arcsec (E, N) relative to lam_ref."""
    d = np.atleast_1d(dn_e6(lam_wave, info, lam_ref_wave))
    return d[:, None] * geom.p[None, :]


# ----------------------------------------------------------------------------- fitting
@dataclass
class AdcFit:
    exp_id: str
    index: int
    order: int
    lam_ref: float
    info: object
    coef: np.ndarray                       #: (2, k)   rows E, N
    cov: np.ndarray                        #: (2, k, k)
    chi2r: np.ndarray                      #: (2,)
    nfit: int = 0
    used: Optional[np.ndarray] = None      #: (nbin,) bins in the final fit
    clipped: Optional[np.ndarray] = None   #: (nbin,) bins rejected by sigma clipping
    bic: float = np.nan
    status: str = 'ok'                     #: ok | fallback | model | regonly | failed
    note: str = ''
    chi2: float = np.nan                   #: total chi2 (both axes) of the final fit
    beta: float = np.nan                   #: shared curvature (c2 = beta c1), if fitted that way
    c1_source: str = 'own'                 #: own | neighbour <exp> | pooled <model> | none
    chi2r_model: float = np.nan            #: reduced chi2 of the c1 estimator (not 'own' fits)
    gate_failed: bool = False              #: c1 estimator failed the --model-max-chi2r gate

    def design(self, lam):
        return basis(lam, self.info, self.lam_ref, self.order)

    def offset(self, lam):
        return self.design(lam) @ self.coef.T

    def offset_err(self, lam):
        B = self.design(lam)
        return np.sqrt(np.stack([np.einsum('ij,jk,ik->i', B, self.cov[a], B) for a in (0, 1)], axis=1))

    @property
    def c1(self):
        return self.coef[:, 1] if self.coef.shape[1] > 1 else np.zeros(2)


def _wls(B, y, s):
    W = 1.0 / s ** 2
    M = B.T @ (B * W[:, None])
    c = np.linalg.solve(M, B.T @ (W * y))
    r = y - B @ c
    return c, np.linalg.inv(M), r, float(np.sum(W * r ** 2))


def fit_exposure(lam, off, err, fitmask, info, lam_ref, order=1, err_floor=ERR_FLOOR, clip=3.0,
                 rounds=2, exp_id='', index=0, beta=None) -> AdcFit:
    """Weighted least squares per axis on the fit-role bins; up to ``rounds`` of sigma clipping
    (a bin is clipped if EITHER axis deviates by > ``clip`` scaled sigma), never leaving fewer than
    k+2 bins. Covariances are scaled by max(1, chi2_r).

    With ``beta`` the curvature is fixed: offset = c0 + c1 (dn + beta dn^2), returned as an order-2
    fit with c2 = beta c1 (``order`` is ignored; the uncertainty of beta is not in ``cov``)."""
    k = order + 1 if beta is None else 2
    ko = k if beta is None else 3
    nb = len(lam)
    sel = (np.asarray(fitmask, bool) & np.all(np.isfinite(off), axis=1)
           & np.all(np.isfinite(err), axis=1))
    clipped = np.zeros(nb, bool)
    empty = AdcFit(exp_id, index, ko - 1, lam_ref, info, np.full((2, ko), np.nan),
                   np.full((2, ko, ko), np.nan), np.full(2, np.nan), 0, sel & False, clipped,
                   status='failed')
    if sel.sum() < k + 2:
        empty.note = f'only {int(sel.sum())} usable fit bins (need {k + 2})'
        return empty
    if beta is None:
        B_all = basis(lam, info, lam_ref, order)
    else:
        B2 = basis(lam, info, lam_ref, 2)
        B_all = np.c_[B2[:, 0], B2[:, 1] + beta * B2[:, 2]]
    s_all = np.sqrt(np.asarray(err, float) ** 2 + err_floor ** 2)
    for _ in range(rounds + 1):
        use = sel & ~clipped
        coef, cov, chi2r, chi2, z = np.zeros((2, k)), np.zeros((2, k, k)), np.zeros(2), 0.0, np.zeros(nb)
        for a in (0, 1):
            c, C, r, x2 = _wls(B_all[use], off[use, a], s_all[use, a])
            dof = use.sum() - k
            chi2r[a] = x2 / dof if dof > 0 else np.nan
            scale = max(1.0, chi2r[a]) if np.isfinite(chi2r[a]) else 1.0
            coef[a], cov[a], chi2 = c, C * scale, chi2 + x2
            za = np.zeros(nb)
            za[use] = np.abs(r) / (s_all[use, a] * np.sqrt(scale))
            z = np.maximum(z, za)
        cand = np.flatnonzero(use & (z > clip))
        if cand.size == 0 or rounds == 0:
            break
        room = use.sum() - (k + 2)
        if room <= 0:
            break
        clipped[cand[np.argsort(z[cand])[::-1][:room]]] = True
    use = sel & ~clipped
    n = int(use.sum())
    if beta is not None:
        J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, beta]])
        coef = coef @ J.T
        cov = np.stack([J @ cov[a] @ J.T for a in (0, 1)])
    return AdcFit(exp_id, index, ko - 1, lam_ref, info, coef, cov, chi2r, n, use, clipped,
                  bic=float(chi2 + 2 * k * np.log(2 * n)), chi2=float(chi2),
                  beta=np.nan if beta is None else float(beta))


def fit_shared_beta(lam, offs, errs, masks, infos, lam_ref, exp_ids, err_floor=ERR_FLOOR, clip=3.0,
                    rounds=2):
    """Per-exposure c0, c1 with ONE curvature shared by all exposures and both axes:
    offset_e(lam) = c0_e + c1_e (dn + beta dn^2), i.e. c2_e = beta c1_e. A misaligned ADC leaves a
    residual of a fixed spectral shape, so beta is constrained by every exposure's bins at the cost of
    one parameter overall. Clipping is decided at beta = 0, beta then minimises the summed chi2 on
    those bins, and the final fits are re-clipped at that beta. Returns (fits, beta, sigma_beta,
    edge) -- ``edge`` is True if beta ran into the search bound. The per-exposure covariances do not
    include the beta uncertainty."""
    n = len(offs)
    kw = dict(err_floor=err_floor, clip=clip)
    f0 = [fit_exposure(lam, offs[e], errs[e], masks[e], infos[e], lam_ref, rounds=rounds, beta=0.0,
                       exp_id=exp_ids[e], index=e, **kw) for e in range(n)]
    ok = [e for e in range(n) if f0[e].status == 'ok']
    if not ok:
        return f0, np.nan, np.nan, False
    dmax = max(float(np.max(np.abs(dn_e6(lam[f0[e].used], infos[e], lam_ref)))) for e in ok)
    bound = 2.0 / max(dmax, 1e-3)

    def total(b):
        return sum(fit_exposure(lam, offs[e], errs[e], f0[e].used, infos[e], lam_ref, rounds=0, beta=b,
                                **kw).chi2 for e in ok)

    grid = np.linspace(-bound, bound, 41)
    chi = np.array([total(b) for b in grid])
    i = int(np.argmin(chi))
    step = grid[1] - grid[0]
    from scipy.optimize import minimize_scalar
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    beta = float(minimize_scalar(total, bounds=(lo, hi), method='bounded',
                                 options=dict(xatol=step * 1e-4)).x) if hi > lo else float(grid[i])
    edge = i in (0, grid.size - 1)
    h = step * 0.05
    c0, cm, cp = total(beta), total(beta - h), total(beta + h)
    d2 = (cp - 2 * c0 + cm) / h ** 2
    ndat = sum(2 * f0[e].nfit for e in ok)
    dof = ndat - 2 * 2 * len(ok) - 1
    scale = max(1.0, c0 / dof) if dof > 0 else 1.0
    sig = float(np.sqrt(2.0 * scale / d2)) if d2 > 0 else np.nan
    fits = [fit_exposure(lam, offs[e], errs[e], masks[e], infos[e], lam_ref, rounds=rounds, beta=beta,
                         exp_id=exp_ids[e], index=e, **kw) for e in range(n)]
    for f in fits:
        if f.status == 'ok':
            f.note = f'shared curvature beta = {beta:+.4f} +- {sig:.4f}'
    return fits, beta, sig, edge


def shared_vs_order1(fits1, fits_s):
    """BIC(shared) - BIC(order 1) summed over exposures fitted in both: the shared fit adds one
    parameter. Negative favours the shared curvature."""
    pairs = [(a, b) for a, b in zip(fits1, fits_s) if a.status == 'ok' and b.status == 'ok']
    if not pairs:
        return np.nan
    N = sum(2 * a.nfit for a, _ in pairs)
    return float(sum(b.chi2 - a.chi2 for a, b in pairs) + np.log(N))


def holdout_stats(fit, lam, off, err, mask, err_floor=ERR_FLOOR):
    """Residuals of bins NOT used in the fit (blue): bias, rms and chi2/n per axis, with the model's
    own extrapolation uncertainty included in chi2."""
    m = np.asarray(mask, bool) & np.all(np.isfinite(off), axis=1)
    if fit.status == 'failed' or not m.any():
        return dict(n=0, bias_E=np.nan, bias_N=np.nan, rms_E=np.nan, rms_N=np.nan,
                    chi2n_E=np.nan, chi2n_N=np.nan)
    r = off[m] - fit.offset(lam[m])
    s2 = err[m] ** 2 + err_floor ** 2 + fit.offset_err(lam[m]) ** 2
    c = np.mean(r ** 2 / s2, axis=0)
    return dict(n=int(m.sum()), bias_E=float(np.mean(r[:, 0])), bias_N=float(np.mean(r[:, 1])),
                rms_E=float(np.sqrt(np.mean(r[:, 0] ** 2))), rms_N=float(np.sqrt(np.mean(r[:, 1] ** 2))),
                chi2n_E=float(c[0]), chi2n_N=float(c[1]))


def relative_holdout(fits, lam, E, N, sE, sN, good, err_floor=ERR_FLOOR, min_n=3):
    """Blue test on RELATIVE positions at one wavelength ``lam``: each exposure's measured position
    minus the median over exposures, against the model treated the same way. A common offset (blue
    vs green/red registration) cancels; it is returned separately as ``common`` (median of measured
    minus model). Exposures used: good measurement and a fit with a chromatic term (not
    'failed'/'model'/'regonly'). Returns (rows, common) with rows=[] if fewer than ``min_n``."""
    meas = np.c_[E, N]
    S = [e for e, f in enumerate(fits) if good[e] and f.status in ('ok', 'fallback')
         and np.all(np.isfinite(meas[e]))]
    if len(S) < min_n:
        return [], None
    mod = np.array([fits[e].offset([lam])[0] for e in S])
    merr = np.array([fits[e].offset_err([lam])[0] for e in S])
    mm, md = np.median(meas[S], axis=0), np.median(mod, axis=0)
    rows = []
    for i, e in enumerate(S):
        rm, rd = meas[e] - mm, mod[i] - md
        r = rm - rd
        s2 = np.array([sE[e], sN[e]]) ** 2 + err_floor ** 2 + merr[i] ** 2
        rows.append(dict(exp=fits[e].exp_id, status=fits[e].status, lam=float(lam),
                         meas_E=rm[0], meas_N=rm[1], model_E=rd[0], model_N=rd[1],
                         resid_E=r[0], resid_N=r[1], chi2_E=float(r[0] ** 2 / s2[0]),
                         chi2_N=float(r[1] ** 2 / s2[1])))
    common = np.median(meas[S] - mod, axis=0)
    return rows, common


@dataclass
class PolyFit:
    """Plain polynomial in lambda (comparison only): x = (lam - lam_ref)/1000."""
    lam_ref: float
    coef: List[np.ndarray]

    def offset(self, lam):
        x = (np.asarray(lam, float) - self.lam_ref) / 1000.0
        return np.stack([np.polyval(c, x) for c in self.coef], axis=1)


def fit_poly_lambda(lam, off, err, mask, deg, lam_ref, err_floor=ERR_FLOOR) -> Optional[PolyFit]:
    m = np.asarray(mask, bool) & np.all(np.isfinite(off), axis=1)
    if m.sum() < deg + 2:
        return None
    x = (lam[m] - lam_ref) / 1000.0
    s = np.sqrt(err[m] ** 2 + err_floor ** 2)
    return PolyFit(lam_ref, [np.polyfit(x, off[m, a], deg, w=1.0 / s[:, a]) for a in (0, 1)])


def select_order(fits1, fits2, threshold=-10.0):
    """Global choice between order 1 and 2: order 2 only if the summed BIC improves by > 10."""
    d = [f2.bic - f1.bic for f1, f2 in zip(fits1, fits2)
         if f1.status == 'ok' and f2.status == 'ok' and np.isfinite(f1.bic) and np.isfinite(f2.bic)]
    total = float(np.sum(d)) if d else np.nan
    return (2 if (np.isfinite(total) and total < threshold) else 1), total, d


# ----------------------------------------------------------------------------- summaries
def _ang(v):
    return float(np.rad2deg(np.arctan2(v[0], v[1])))


def chromatic_summary(fits, geoms, lam_blue=4000.0, lam_red=8000.0):
    """Per exposure: measured vs predicted dispersion (amplitude ratio = ADC residual fraction;
    direction residual), and the model's blue-minus-red offset vs the no-ADC prediction."""
    rows = []
    for f, g in zip(fits, geoms):
        r = dict(exp=f.exp_id, status=f.status)
        if f.status == 'failed':
            rows.append(r)
            continue
        c1 = f.c1
        d_meas = f.offset([lam_blue])[0] - f.offset([lam_red])[0]
        r.update(c1_E=c1[0], c1_N=c1[1], c1_amp=float(np.hypot(*c1)), c1_pa=_ang(c1),
                 meas_b_minus_r_E=d_meas[0], meas_b_minus_r_N=d_meas[1],
                 meas_b_minus_r=float(np.hypot(*d_meas)))
        if g.ok:
            p = g.p
            d_pred = (predicted_dar(f.info, g, [lam_blue], f.lam_ref)[0]
                      - predicted_dar(f.info, g, [lam_red], f.lam_ref)[0])
            r.update(airmass=g.airmass_mean, q=g.q_mean, p_E=p[0], p_N=p[1], p_amp=float(np.hypot(*p)),
                     amp_ratio=float(np.hypot(*c1) / np.hypot(*p)),
                     along_frac=float(np.dot(c1, p) / np.dot(p, p)),
                     dir_resid_deg=float(_wrap_deg(_ang(c1) - _ang(p))),
                     pred_b_minus_r_E=d_pred[0], pred_b_minus_r_N=d_pred[1],
                     pred_b_minus_r=float(np.hypot(*d_pred)))
        rows.append(r)
    return rows


@dataclass
class DispersionModel:
    """Pooled description of measured c1 vectors across exposures (complex E + iN):
    v = g*p (+k) etc. Used as the fallback for exposures whose own fit failed, and to test the
    sky orientation (a mirror-parity error makes v follow conj(p) instead of p)."""
    name: str
    params: dict
    chi2: float
    npar: int
    n: int
    bic: float

    def predict(self, geom):
        p = complex(*geom.p)
        g, k = self.params.get('g', 0j), self.params.get('k', 0j)
        v = g * (p.conjugate() if self.name == 'mirror' else p) + k
        return np.array([v.real, v.imag])


def dispersion_models(fits, geoms):
    """Fit candidate models to the measured c1 vectors of the 'ok' exposures. Returns a list of
    DispersionModel sorted by BIC (empty if < 2 usable exposures)."""
    v, p, w = [], [], []
    for f, g in zip(fits, geoms):
        if f.status != 'ok' or not g.ok or not np.all(np.isfinite(f.c1)):
            continue
        v.append(complex(*f.c1))
        p.append(complex(*g.p))
        s2 = 0.5 * (f.cov[0, 1, 1] + f.cov[1, 1, 1])
        w.append(1.0 / max(s2, 1e-8))
    n = len(v)
    if n < 2:
        return []
    v, p, w = np.array(v), np.array(p), np.array(w)
    out = []

    def add(name, model, params, npar):
        chi2 = float(np.sum(w * np.abs(v - model) ** 2))
        out.append(DispersionModel(name, params, chi2, npar, n, chi2 + npar * np.log(2 * n)))

    add('perfect_adc', np.zeros(n), {}, 0)
    add('no_adc', p, {'g': 1 + 0j}, 0)
    for name, x in (('gain', p), ('mirror', p.conjugate())):
        g = np.sum(w * x.conjugate() * v) / np.sum(w * np.abs(x) ** 2)
        add(name, g * x, {'g': g}, 2)
    k = np.sum(w * v) / np.sum(w)
    add('const', np.full(n, k), {'k': k}, 2)
    if n >= 3:
        X = np.c_[p, np.ones(n)] * np.sqrt(w)[:, None]
        sol, *_ = np.linalg.lstsq(X, v * np.sqrt(w), rcond=None)
        add('gain+const', sol[0] * p + sol[1], {'g': sol[0], 'k': sol[1]}, 4)
    return sorted(out, key=lambda m: m.bic)


def orientation_verdict(models):
    """Plain-language reading of :func:`dispersion_models`."""
    if not models:
        return 'not enough exposures with good fits to test the dispersion direction'
    by = {m.name: m for m in models}
    best = models[0]
    lines = [f'best model: {best.name} (BIC {best.bic:.1f})']
    if 'gain' in by and 'mirror' in by:
        dg, dm = by['gain'], by['mirror']
        lines.append(f"gain   v = g p:       |g| = {abs(dg.params['g']):.2f}, arg g = "
                     f"{np.rad2deg(np.angle(dg.params['g'])):+.1f} deg, chi2 = {dg.chi2:.1f}")
        lines.append(f"mirror v = g conj(p): |g| = {abs(dm.params['g']):.2f}, arg g = "
                     f"{np.rad2deg(np.angle(dm.params['g'])):+.1f} deg, chi2 = {dm.chi2:.1f}")
        if dm.bic < dg.bic - 10:
            lines.append('WARNING: the dispersion follows the MIRRORED parallactic angle -- the '
                         'fibre WCS parity (or the ADC sense) is suspect.')
        elif dg.bic < dm.bic - 10:
            lines.append('dispersion tracks the parallactic angle with the WCS parity in use; '
                         'arg g mixes any WCS rotation error with ADC misalignment (degenerate).')
        else:
            lines.append('parity not decided: the parallactic angle does not vary enough across '
                         'these exposures.')
    if 'const' in by and best.name == 'const':
        lines.append('a CONSTANT dispersion vector fits best: the residual is fixed on the sky/'
                     'instrument (e.g. ADC stuck or misaligned), not following the atmosphere.')
    if best.name == 'perfect_adc':
        lines.append('no significant residual dispersion: the ADC appears to be working.')
    return '\n'.join(lines)


# ----------------------------------------------------------------------------- corrections
@dataclass
class C1Estimate:
    """Chromatic term for an exposure without its own fit."""
    c1: np.ndarray                         #: (2,) E, N
    var: np.ndarray                        #: (2,) per-axis variance
    chi2r: float                           #: reduced chi2 of the estimator on the measured exposures
    source: str
    ntest: int = 0                         #: measured exposures the estimator was tested on


def _ok_c1(fits, geoms):
    return [i for i, (f, g) in enumerate(zip(fits, geoms))
            if f.status == 'ok' and g.ok and np.all(np.isfinite(f.c1))]


def _c1_var(f):
    return np.array([f.cov[0, 1, 1], f.cov[1, 1, 1]])


def _nearest(e, cands, infos):
    """Nearest exposure in time (MJD-OBS); exposure order when a time is missing."""
    t = getattr(infos[e], 'mjd', np.nan)
    def dist(j):
        tj = getattr(infos[j], 'mjd', np.nan)
        return abs(tj - t) if (np.isfinite(t) and np.isfinite(tj)) else 1e6 + abs(j - e)
    return min(cands, key=lambda j: (dist(j), j))


def _transfer(j, e, fits, geoms):
    """c1 of exposure j carried to exposure e's geometry: v_e = v_j p_e / p_j (complex E + iN), the
    gain model with a gain fitted locally to j."""
    r = complex(*geoms[e].p) / complex(*geoms[j].p)
    v = complex(*fits[j].c1) * r
    return np.array([v.real, v.imag]), np.full(2, np.mean(_c1_var(fits[j])) * abs(r) ** 2)


def neighbour_c1(e, fits, geoms, infos) -> Optional[C1Estimate]:
    """c1 from the nearest-in-time measured exposure, scaled by the ratio of the no-ADC dispersion
    predictions. Its quality is tested by predicting every measured exposure from ITS nearest
    measured neighbour: the rms of those prediction errors per axis is the variance (if larger than
    the propagated one) and their chi2 per point is the estimator's reduced chi2."""
    ok = [j for j in _ok_c1(fits, geoms) if j != e]
    if not ok or not geoms[e].ok:
        return None
    j = _nearest(e, ok, infos)
    c1, var = _transfer(j, e, fits, geoms)
    d, chi = [], []
    for i in ok:
        others = [k for k in ok if k != i]
        if not others:
            continue
        pred, pv = _transfer(_nearest(i, others, infos), i, fits, geoms)
        r = fits[i].c1 - pred
        d.append(r)
        chi.append(r ** 2 / np.maximum(pv + _c1_var(fits[i]), 1e-12))
    chi2r = float(np.mean(chi)) if chi else np.nan
    if d:
        var = np.maximum(var, np.mean(np.array(d) ** 2, axis=0))
    return C1Estimate(c1, var, chi2r, f'neighbour {fits[j].exp_id}', len(d))


def pooled_c1(e, fits, geoms, model) -> Optional[C1Estimate]:
    """c1 from a pooled :class:`DispersionModel`; variance = scatter of the measured c1 about it."""
    if model is None or not geoms[e].ok:
        return None
    ok = _ok_c1(fits, geoms)
    r = np.array([fits[i].c1 - model.predict(geoms[i]) for i in ok])
    var = np.sum(r ** 2, axis=0) / max(model.n - model.npar / 2, 1)
    chi2r = model.chi2 / (2 * model.n - model.npar) if 2 * model.n > model.npar else np.nan
    return C1Estimate(model.predict(geoms[e]), var, float(chi2r), f'pooled {model.name}', model.n)


def fallback_fit(index, exp_id, info, lam_ref, est: Optional[C1Estimate], ref_off, ref_lam, order=1,
                 beta=None, ref_var=ERR_FLOOR ** 2):
    """Coefficients for an exposure whose own fit failed: chromatic term from ``est``, zero point
    from the broad-band reference position (status 'fallback'). Without a reference position (the
    star is not in the exposure) the status is 'model': chromatic term only, zero point unknown (0),
    and :class:`Correction` applies it as an ADC-only shift with the registration untouched. With
    ``beta`` (order 2, shared curvature) c2 = beta c1. The covariance carries the estimator's
    variance through the shape dn + beta dn^2 (and the reference position's into c0)."""
    k = order + 1
    coef, cov = np.zeros((2, k)), np.zeros((2, k, k))
    have_ref = ref_off is not None and np.all(np.isfinite(ref_off))
    if est is None:
        if not have_ref:
            return AdcFit(exp_id, index, order, lam_ref, info, np.full((2, k), np.nan), cov,
                          np.full(2, np.nan), status='failed', note='no reference position and no c1 estimate',
                          c1_source='none')
        coef[:, 0] = ref_off
        cov[:, 0, 0] = ref_var
        return AdcFit(exp_id, index, order, lam_ref, info, coef, cov, np.full(2, np.nan), status='regonly',
                      note='no estimate of the chromatic term; registration only', c1_source='none')
    b = float(beta) if (beta is not None and order == 2) else 0.0
    J = np.array([0.0, 1.0, b])[:k]
    coef[:, 1:] = np.asarray(est.c1)[:, None] * J[None, 1:]
    status, note = 'model', f'star not in exposure: chromatic term from {est.source}, no registration'
    if have_ref:
        d = float(dn_e6([ref_lam], info, lam_ref)[0])
        J[0] = -(d + b * d * d) if k > 2 else -d
        coef[:, 0] = np.asarray(ref_off) + J[0] * np.asarray(est.c1)
        status, note = 'fallback', f'chromatic term from {est.source}, zero point from the reference position'
    for a in (0, 1):
        cov[a] = est.var[a] * np.outer(J, J)
        if have_ref:
            cov[a, 0, 0] += ref_var
    return AdcFit(exp_id, index, order, lam_ref, info, coef, cov, np.full(2, np.nan), status=status,
                  note=note, beta=b if k > 2 else np.nan, c1_source=est.source, chi2r_model=est.chi2r)


@dataclass
class Correction:
    """Per-exposure shift (arcsec, E, N) to ADD to fibre positions at wavelength lam (WAVE frame):
    anchor='star' -> A - model_e(lam) (registration + ADC); 'none' -> model_e(lam_anchor) -
    model_e(lam) (ADC only, registration untouched). 'model' fits (star not in the exposure, zero
    point unknown) are always applied ADC-only."""
    fits: List[AdcFit]
    anchor: np.ndarray
    lam_anchor: float
    mode: str = 'star'

    def __call__(self, e, lam):
        f = self.fits[e]
        lam = np.atleast_1d(np.asarray(lam, float))
        if f.status == 'failed':
            return np.zeros((lam.size, 2))
        m = f.offset(lam)
        if self.mode == 'none' or f.status == 'model':
            return f.offset([self.lam_anchor]) - m
        return self.anchor[None, :] - m


def build_correction(fits, lam_anchor, mode='star'):
    good = [f for f in fits if f.status == 'ok'] or [f for f in fits if f.status not in ('failed', 'model')]
    if not good:
        return None
    A = np.median(np.array([f.offset([lam_anchor])[0] for f in good]), axis=0)
    return Correction(fits, A, float(lam_anchor), mode)


def shift_grid_table(fits, corr, lam_grid):
    rows = []
    for f in fits:
        if f.status == 'failed':
            continue
        m, s = f.offset(lam_grid), f.offset_err(lam_grid)
        c = corr(f.index, lam_grid) if corr is not None else np.full_like(m, np.nan)
        for i, lam in enumerate(lam_grid):
            rows.append(dict(exp=f.exp_id, status=f.status, lam=float(lam), model_E=m[i, 0],
                             model_N=m[i, 1], err_E=s[i, 0], err_N=s[i, 1], corr_E=c[i, 0],
                             corr_N=c[i, 1]))
    return rows
