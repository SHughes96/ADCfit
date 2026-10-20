"""Refraction physics, geometry and the refractivity-basis fit."""

import numpy as np

from llamas_pyjamas.Combine.spectrum import _refractivity

from adcfit.dar import (refractivity, zenith_parallactic, exposure_geometry, fit_exposure, basis,
                        dispersion_models, build_correction, select_order, fallback_fit, dn_e6,
                        C1Estimate, neighbour_c1, pooled_c1, fit_shared_beta, shared_vs_order1,
                        relative_holdout)
from adcfit.io import _hours
from tests.synth import make_info, run_tests


def test_refractivity_standard_air_matches_package():
    lam = np.array([3600.0, 5000.0, 9000.0])
    assert np.allclose(refractivity(lam, 1013.25, 15.0, 0.0), _refractivity(lam), rtol=1e-3)
    # lower pressure / higher temperature -> less refraction; humidity slightly reduces it
    assert np.all(refractivity(lam, 765.0, 10.0, 0.0) < _refractivity(lam))
    assert np.all(refractivity(lam, 765.0, 20.0, 0.0) < refractivity(lam, 765.0, 0.0, 0.0))
    assert np.all(refractivity(lam, 765.0, 10.0, 80.0) < refractivity(lam, 765.0, 10.0, 0.0))


def test_parallactic_angle_matches_tcs_header():
    # LLAMAS Feige110 header: TEL HA 2h36m33s, Dec -5.1667 -> TEL PARANGL 127.51
    z, q = zenith_parallactic(_hours('2h36m33s'), -5.16667)
    assert abs(q - 127.51) < 0.15, q
    assert abs(1 / np.cos(z) - 1.391) < 0.01, 1 / np.cos(z)
    # TEL HA written as 0..24 h (20h12m) is a negative hour angle -> negative (east) q
    _, q2 = zenith_parallactic(_hours('20h12m23s') - 24.0, -56.626)
    assert q2 < 0


def test_exposure_average_lies_between_start_and_end():
    info = make_info(0, H0=-1.5, exptime=3600.0)
    g = exposure_geometry(info)
    assert g.ok and g.q_start < g.q_mean < g.q_end
    assert min(g.airmass_start, g.airmass_end) <= g.airmass_mean + 1e-9


def test_basis_fit_recovers_exact_coefficients():
    info = make_info(0, H0=0.5)
    lam = np.linspace(4900, 9600, 14)
    B = basis(lam, info, 8000.0, 1)
    truth = np.array([[0.3, 0.12], [-0.7, -0.05]])
    off = B @ truth.T
    f = fit_exposure(lam, off, np.full_like(off, 0.01), np.ones(lam.size, bool), info, 8000.0)
    assert f.status == 'ok'
    assert np.allclose(f.coef, truth, atol=1e-8)
    # extrapolation into the blue follows the refractivity shape exactly
    assert np.allclose(f.offset([3700.0]), basis([3700.0], info, 8000.0, 1) @ truth.T, atol=1e-8)


def test_sigma_clipping_rejects_one_bad_bin():
    info = make_info(0, H0=0.5)
    lam = np.linspace(4900, 9600, 14)
    off = basis(lam, info, 8000.0, 1) @ np.array([[0.0, 0.1], [0.0, 0.05]]).T
    rng = np.random.default_rng(3)
    off = off + rng.normal(0, 0.01, off.shape)
    off[5, 1] += 0.5
    f = fit_exposure(lam, off, np.full_like(off, 0.01), np.ones(lam.size, bool), info, 8000.0)
    assert f.clipped[5] and f.clipped.sum() == 1
    assert abs(f.c1[1] - 0.05) < 0.005


def test_too_few_bins_fails_cleanly():
    info = make_info(0, H0=0.5)
    lam = np.array([5000.0, 7000.0])
    f = fit_exposure(lam, np.zeros((2, 2)), np.full((2, 2), 0.01), np.ones(2, bool), info, 8000.0)
    assert f.status == 'failed'


def test_order_selection_and_models():
    infos = [make_info(e, H0=h) for e, h in enumerate([-2.0, -0.5, 1.0, 2.5])]
    geoms = [exposure_geometry(i) for i in infos]
    lam = np.linspace(4900, 9600, 16)
    f1, f2 = [], []
    for i, g in zip(infos, geoms):
        c1 = 0.5 * g.p
        off = basis(lam, i, 8000.0, 1) @ np.array([[0.0, c1[0]], [0.0, c1[1]]]).T
        off += np.random.default_rng(i.index).normal(0, 0.01, off.shape)
        m = np.ones(lam.size, bool)
        f1.append(fit_exposure(lam, off, np.full_like(off, 0.01), m, i, 8000.0, order=1, index=i.index))
        f2.append(fit_exposure(lam, off, np.full_like(off, 0.01), m, i, 8000.0, order=2, index=i.index))
    order, _, _ = select_order(f1, f2)
    assert order == 1
    models = dispersion_models(f1, geoms)
    gain = [m for m in models if m.name == 'gain'][0]
    assert abs(gain.params['g'] - 0.5) < 0.05
    assert models[0].name in ('gain', 'gain+const')
    corr = build_correction(f1, 8000.0)
    # after correction every exposure's model maps to the common anchor at all wavelengths
    for f in f1:
        moved = f.offset(lam) + corr(f.index, lam)
        assert np.allclose(moved, corr.anchor[None, :], atol=1e-12)



def test_model_status_is_adc_only():
    """No star in the exposure: the estimated chromatic term is applied with the registration untouched
    (zero shift at the anchor wavelength) in both anchor modes, and it never sets the anchor."""
    est = C1Estimate(np.array([-0.4, 0.05]), np.array([0.01, 0.004]), 2.0, 'test')
    info = make_info(0, 1.0)
    lam_ref = 8300.0
    f = fallback_fit(0, info.exp_id, info, lam_ref, est, None, np.nan)
    assert f.status == 'model', f.status
    lam = np.array([4000.0, 6000.0, lam_ref, 9000.0])
    d = dn_e6(lam, info, lam_ref) - dn_e6([lam_ref], info, lam_ref)[0]
    want = -np.outer(d, [-0.4, 0.05])
    other = fallback_fit(1, 'x', info, lam_ref, est, np.array([3.0, -2.0]), lam_ref)
    assert other.status == 'fallback'
    for mode in ('star', 'none'):
        corr = build_correction([f, other], lam_ref, mode=mode)
        assert np.allclose(corr(0, lam), want, atol=1e-9), (mode, corr(0, lam), want)
        assert np.allclose(corr.anchor, other.offset([lam_ref])[0])
    # the estimator's variance reaches the shift uncertainty (no more err = 0)
    assert np.allclose(f.offset_err(lam), np.sqrt(np.outer(np.abs(d) ** 2, [0.01, 0.004])), atol=1e-12)
    assert np.all(f.offset_err([4000.0]) > 0)
    # shared curvature: c2 = beta c1
    f2 = fallback_fit(0, info.exp_id, info, lam_ref, est, None, np.nan, order=2, beta=0.03)
    assert np.allclose(f2.coef[:, 2], 0.03 * est.c1)
    assert fallback_fit(0, info.exp_id, info, lam_ref, None, None, np.nan).status == 'failed'
    assert fallback_fit(0, info.exp_id, info, lam_ref, None, np.array([1.0, 0.0]), lam_ref).status == 'regonly'


def _gain_fits(H, g, noise=0.0, seed=0):
    """Order-1 'ok' fits whose c1 follow v = g p (complex), exposures one hour of hour angle apart."""
    infos = [make_info(e, H0=h) for e, h in enumerate(H)]
    geoms = [exposure_geometry(i) for i in infos]
    lam = np.linspace(4900, 9600, 14)
    rng = np.random.default_rng(seed)
    fits = []
    for i, gm in zip(infos, geoms):
        v = g * complex(*gm.p) + noise * complex(*rng.normal(0, 1, 2))
        off = basis(lam, i, 8000.0, 1) @ np.array([[0.0, v.real], [0.0, v.imag]]).T
        off += rng.normal(0, 0.005, off.shape)
        fits.append(fit_exposure(lam, off, np.full_like(off, 0.005), np.ones(lam.size, bool), i, 8000.0,
                                 index=i.index, exp_id=i.exp_id))
    return infos, geoms, fits


def test_neighbour_estimator_recovers_c1_with_uncertainty():
    H = [-2.0, -1.0, 0.0, 1.0, 2.0]
    infos, geoms, fits = _gain_fits(H, 0.6 - 0.3j)
    truth = fits[2].c1.copy()
    fits[2].status = 'failed'
    est = neighbour_c1(2, fits, geoms, infos)
    assert est is not None and est.source.startswith('neighbour')
    assert est.source.split()[-1] in (fits[1].exp_id, fits[3].exp_id)
    assert np.all(np.abs(est.c1 - truth) < 0.01), (est.c1, truth)
    assert np.all(est.var > 0)
    assert est.chi2r < 10


def test_pooled_variance_is_scatter_and_gate_flags_bad_set():
    H = [-2.0, -1.0, 0.0, 1.0, 2.0, 2.5]
    infos, geoms, fits = _gain_fits(H, 0.6 - 0.3j, noise=0.05, seed=4)
    models = dispersion_models(fits, geoms)
    gain = [m for m in models if m.name == 'gain'][0]
    est = pooled_c1(0, fits, geoms, gain)
    r = np.array([f.c1 - gain.predict(g) for f, g in zip(fits, geoms)])
    assert np.allclose(est.var, np.sum(r ** 2, axis=0) / (gain.n - 1))
    assert abs(est.chi2r - gain.chi2 / (2 * gain.n - 2)) < 1e-9
    # scatter of 0.05 "/1e-6 against c1 errors of ~0.002 -> far above a gate of 10
    assert est.chi2r > 10
    assert neighbour_c1(0, fits, geoms, infos).chi2r > 10
    # a consistent set passes
    infos2, geoms2, fits2 = _gain_fits(H, 0.6 - 0.3j, seed=5)
    g2 = [m for m in dispersion_models(fits2, geoms2) if m.name == 'gain'][0]
    assert pooled_c1(0, fits2, geoms2, g2).chi2r < 10


def test_shared_beta_recovers_curvature():
    infos = [make_info(e, H0=h) for e, h in enumerate([-2.0, -0.5, 1.0, 2.5])]
    lam = np.linspace(4900, 9600, 14)
    beta = 0.06
    rng = np.random.default_rng(7)
    offs, errs, masks, c1s = [], [], [], []
    for i in infos:
        c1 = rng.normal(0, 0.3, 2)
        c0 = rng.normal(0, 0.5, 2)
        B = basis(lam, i, 8000.0, 2)
        off = c0[None, :] + np.outer(B[:, 1] + beta * B[:, 2], c1) + rng.normal(0, 0.003, (lam.size, 2))
        offs.append(off); errs.append(np.full_like(off, 0.003)); masks.append(np.ones(lam.size, bool))
        c1s.append(c1)
    fits, b, sb, edge = fit_shared_beta(lam, offs, errs, masks, infos, 8000.0, [i.exp_id for i in infos])
    assert not edge and abs(b - beta) < max(3 * sb, 0.01), (b, sb)
    for f, c1 in zip(fits, c1s):
        assert f.status == 'ok' and f.order == 2
        assert np.allclose(f.c1, c1, atol=0.01)
        assert np.allclose(f.coef[:, 2], b * f.c1)
    f1 = [fit_exposure(lam, o, e, m, i, 8000.0) for o, e, m, i in zip(offs, errs, masks, infos)]
    assert shared_vs_order1(f1, fits) < -10


def test_relative_holdout_cancels_common_offset():
    infos = [make_info(e, H0=h) for e, h in enumerate([-2.0, -0.5, 1.0, 2.5])]
    lam = np.linspace(4900, 9600, 14)
    fits = []
    for e, i in enumerate(infos):
        c1 = np.array([0.1 * e, -0.05 * e])
        off = basis(lam, i, 8000.0, 1) @ np.array([[0.0, c1[0]], [0.0, c1[1]]]).T
        fits.append(fit_exposure(lam, off, np.full_like(off, 0.005), np.ones(lam.size, bool), i, 8000.0,
                                 index=e, exp_id=i.exp_id))
    lb = 4000.0
    meas = np.array([f.offset([lb])[0] for f in fits]) + np.array([0.7, -0.4])   # common blue offset
    n = len(fits)
    rows, common = relative_holdout(fits, lb, meas[:, 0], meas[:, 1], np.full(n, 0.02), np.full(n, 0.02),
                                    np.ones(n, bool))
    assert len(rows) == n
    assert np.allclose([r['resid_E'] for r in rows], 0, atol=1e-9)
    assert np.allclose([r['resid_N'] for r in rows], 0, atol=1e-9)
    assert np.allclose(common, [0.7, -0.4], atol=1e-9)
    assert relative_holdout(fits, lb, meas[:, 0], meas[:, 1], np.full(n, 0.02), np.full(n, 0.02),
                            np.array([True, True, False, False]))[0] == []


if __name__ == '__main__':
    run_tests(globals())
