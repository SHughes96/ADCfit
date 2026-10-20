"""Star detection and per-bin centroiding on a synthetic field with the real fibre geometry."""

import numpy as np

from adcfit.star import (fit_psf, make_bins, measure_star, find_star, NoStarError, FLAGS, hull_of,
                         hull_distance)
from tests.synth import make_field, lut_xy, run_tests


def _hex_star(x0, y0, fwhm=1.2, amp=1e-17, noise=2e-20, seed=0, sx=0.0, sy=0.0):
    xy = lut_xy()
    xy = xy - xy.mean(axis=0)
    rng = np.random.default_rng(seed)
    sig = fwhm / 2.354820045
    f = amp * np.exp(-0.5 * ((xy[:, 0] - x0) ** 2 + (xy[:, 1] - y0) ** 2) / sig ** 2)
    f = f + 3e-19 + rng.normal(0, noise, f.size)
    return xy[:, 0], xy[:, 1], f, np.full(f.size, noise ** 2)


def test_fit_recovers_off_centre_star_at_flam_scale():
    # the original s3 failure: star 2" from the aperture start position, FLAM-scale amplitudes
    for (x0, y0) in [(0.3, -0.2), (2.0, 0.0), (1.2, -1.6)]:
        x, y, f, v = _hex_star(x0, y0)
        pf = fit_psf(x, y, f, v, 0.0, 0.0, 5.0)          # centroid bound = radius/2 = 2.5"
        assert pf is not None and not pf.at_bound
        assert np.hypot(pf.x - x0, pf.y - y0) < 0.02, (x0, y0, pf.x, pf.y)
        assert abs(pf.fwhm - 1.2) < 0.05


def test_edge_distance_and_flag():
    x, y, f, v = _hex_star(0, 0)
    h = hull_of(x, y)
    assert hull_distance(h, 0.0, 0.0) > 10
    assert hull_distance(h, 100.0, 0.0) < 0


def test_measure_star_tracks_known_dispersion():
    sr, infos, inst, truth = make_field(nexp=3)
    bins = make_bins(sr, {'blue': 6, 'green': 6, 'red': 6}, ('green', 'red'))
    cents = measure_star(sr, truth['star_radec'], bins)
    good = cents.good()
    assert good.mean() > 0.9, [b.channel for b in bins]
    # measured offsets relative to the nominal star position = reg + dispersion track
    for e in range(sr.n_exposures):
        t = truth['offset'](e, cents.lam) - truth['s0'][None, :]
        d = np.c_[cents.E[e], cents.N[e]] - t
        ok = good[e]
        assert np.nanmax(np.abs(d[ok])) < 0.06, np.round(d[ok], 3)


def test_find_star_and_no_star():
    sr, infos, inst, truth = make_field(nexp=2)
    bins = make_bins(sr, {'blue': 4, 'green': 4, 'red': 4}, ('green', 'red'))
    ch = find_star(sr, bins)
    from adcfit.io import tangent_offsets
    E, N = tangent_offsets(ch.ra, ch.dec, *truth['star_radec'])
    assert np.hypot(E, N) < 1.0
    sr0, *_ = make_field(nexp=2, star=False)
    try:
        find_star(sr0, bins)
    except NoStarError:
        pass
    else:
        raise AssertionError('expected NoStarError on an empty field')


def test_find_star_channel_check():
    """Every candidate carries its per-channel fit; the true star is found in all three channels
    at a consistent position."""
    sr, infos, inst, truth = make_field(nexp=2)
    bins = make_bins(sr, {'blue': 4, 'green': 4, 'red': 4}, ('green', 'red'))
    ch = find_star(sr, bins)
    c = next(q for q in ch.candidates if q['id'] == ch.chosen)
    for k in ('blue', 'green', 'red'):
        assert np.isfinite(c[f'snr_{k}']) and c[f'snr_{k}'] >= 5, (k, c[f'snr_{k}'])
        assert c[f'dpos_{k}'] < 3.0, (k, c[f'dpos_{k}'])
    assert c['chan_ok'] and c['reason'] == '', c['reason']
    # selection S/N is the aperture S/N; noise candidates stay far below the star
    assert c['snr'] >= 30
    assert all(not np.isfinite(q['snr']) or q['snr'] < 0.2 * c['snr'] for q in ch.candidates if q['id'] != ch.chosen)
    # an impossible per-channel requirement leaves only the full criteria; the star still qualifies
    ch2 = find_star(sr, bins, chan_min_snr=1e9)
    assert ch2.chosen == ch.chosen


def test_no_star_writes_candidate_pdf():
    """With no star in the field the CLI exits 2 but still writes the candidate figures and PDF."""
    import os
    import tempfile
    from adcfit.cli import main
    from tests.synth import write_rss
    with tempfile.TemporaryDirectory() as tmp:
        sr, infos, inst, truth = make_field(nexp=2, star=False)
        paths = write_rss(sr, infos, os.path.join(tmp, 'extractions'))
        qa_dir = os.path.join(tmp, 'qa')
        rc = main(paths + ['--outdir', qa_dir, '--cube-outdir', os.path.join(tmp, 'cubes'),
                           '--bins', '4', '4', '4'])
        assert rc == 2
        assert os.path.getsize(os.path.join(qa_dir, 'adc_qa.pdf')) > 0
        pngs = os.listdir(os.path.join(qa_dir, 'qa'))
        assert any('star_selection' in f for f in pngs) and any('star_channels' in f for f in pngs), pngs
        assert os.path.exists(os.path.join(qa_dir, 'star_candidates.txt'))


def test_star_missing_from_an_exposure_is_not_replaced():
    """The star is absent from exposure 0, which instead holds another source 5" away: that
    exposure must be reported missing and never measured on the other source."""
    from adcfit.io import tangent_offsets
    sr, infos, inst, truth = make_field(nexp=3, star_exps={1, 2}, dither=1.0,
                                        decoy=dict(exp=0, dE=5.0, dN=0.0, flux=3e-16))
    bins = make_bins(sr, {'blue': 4, 'green': 4, 'red': 4}, ('green', 'red'))
    ch = find_star(sr, bins, star=truth['star_radec'])
    E, N = tangent_offsets(ch.ra, ch.dec, *truth['star_radec'])
    assert np.hypot(E, N) < 1.0
    assert ch.exp_missing == [0], ch.exp_missing
    assert 0 not in ch.exp_pos and set(ch.exp_pos) == {1, 2}
    assert [r['detected'] for r in ch.exp_table] == [False, True, True]
    # automatic choice: the one-exposure source is not preferred over the star seen in the others
    auto = find_star(sr, bins)
    E, N = tangent_offsets(auto.ra, auto.dec, *truth['star_radec'])
    assert np.hypot(E, N) < 1.0 and auto.exp_missing == [0]
    cents = measure_star(sr, truth['star_radec'], bins, guesses=ch.exp_pos, skip=set(ch.exp_missing))
    assert np.all(cents.flags[0] == FLAGS['FAILED']) and not cents.ref['ok'][0]
    good = cents.good()
    for e in (1, 2):
        d = np.c_[cents.E[e], cents.N[e]] - (truth['offset'](e, cents.lam) - truth['s0'][None, :])
        assert np.nanmax(np.abs(d[good[e]])) < 0.1


def test_flags_bitmask_roundtrip():
    from adcfit.star import flag_names
    assert flag_names(FLAGS['EDGE'] | FLAGS['LOWSN']) == 'EDGE|LOWSN'
    assert flag_names(0) == '-'


if __name__ == '__main__':
    run_tests(globals())
