"""ADC-corrected standard RSS: same file structure as the input, loads through the CubeViewer RSS
reader and sensfunc builder, flattens the star's track, and makes a fixed aperture's captured
fraction flat in wavelength while conserving the star's total light."""

import hashlib
import os
import tempfile

import numpy as np
from astropy.io import fits

from adcfit import rsscorr, standard, star as astar
from adcfit.io import load_field, tangent_offsets
from tests.synth import make_field, write_rss, run_tests


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def _with_skysub(path):
    """Synthetic RSS carry FLAM only; add SKYSUB/ERROR like a pipeline (pre-fluxcal) RSS."""
    with fits.open(path) as h:
        hdus = [h[0].copy(), fits.ImageHDU(h['FLAM'].data.copy(), name='SKYSUB'),
                fits.ImageHDU(h['FLAM_ERR'].data.copy(), name='ERROR')] + [x.copy() for x in h[1:]]
        fits.HDUList(hdus).writeto(path, overwrite=True)


def _band_totals(path, E, N, centre_fn, r, lams, width=100.0):
    with fits.open(path) as h:
        W, F = h['WAVE'].data, h['SKYSUB'].data
    out = []
    for lam in lams:
        inb = (W >= lam - width / 2) & (W < lam + width / 2)
        per = np.where(inb, F, 0.0).sum(axis=1) / np.maximum(inb.sum(axis=1), 1)
        c = centre_fn(lam)
        out.append(per[np.hypot(E - c[0], N - c[1]) <= r].sum())
    return np.array(out)


def test_corrected_standard_rss():
    with tempfile.TemporaryDirectory() as tmp:
        sr0, infos, inst, truth = make_field(nexp=1, nw=600, gain=2.5 + 0.0j, s0=(0.0, 0.0), noise=2e-20,
                                             flux0=3e-15, dither=0.0, reg_sigma=0.0)
        for st in sr0.channels.values():
            st.flux -= np.float32(1e-18)
        green = write_rss(sr0, infos, os.path.join(tmp, 'ex'), obj='SYNSTD')[0]
        sib = standard.std_siblings(green)
        for p in sib.values():
            _with_skysub(p)
        md5 = {c: _md5(p) for c, p in sib.items()}

        opts = astar.MeasureOpts(exclusions=astar.default_exclusions(), destripe=False)
        res = standard.run_standard(green, None, [], opts, excl=astar.default_exclusions())
        outdir = os.path.join(tmp, 'std_adc')
        out = rsscorr.correct_rss(sib, res.sr, res.track, res.lam_ref, res.star, outdir)

        # same names, same structure, originals untouched, refuses to clobber / write in place
        for c, p in sib.items():
            q = out['paths'][c]
            assert os.path.basename(q) == os.path.basename(p) and os.path.dirname(q) == outdir
            assert rsscorr.compare_rss(p, q) == [], rsscorr.compare_rss(p, q)
            assert _md5(p) == md5[c]
        assert set(standard.std_siblings(out['paths']['green'])) == {'blue', 'green', 'red'}
        assert os.path.exists(out['sidecar'])
        for bad_dir, exc in ((outdir, FileExistsError), (os.path.dirname(green), ValueError)):
            try:
                rsscorr.correct_rss(sib, res.sr, res.track, res.lam_ref, res.star, bad_dir)
                raise AssertionError('should have refused')
            except exc:
                pass

        # loads through CubeViewer's RSS reader; a picker-style aperture builds a sensfunc
        from llamas_pyjamas.CubeViewer.cubeViewRSS import RSSScene
        from llamas_pyjamas.CubeViewer.cubeViewScene import combine
        from llamas_pyjamas.Flux.sensFunc import build_sensfunc
        scene = RSSScene(out['paths'])
        with fits.open(out['paths']['red']) as h:
            peak = int(np.argmax(np.nanmedian(h['SKYSUB'].data, axis=1)))
            key = (str(h['FIBERMAP'].data['BENCHSIDE'][peak]).strip(), int(h['FIBERMAP'].data['FIBER_ID'][peak]))
        c0 = scene.positions[scene.keys.index(key)]
        keys = [k for k, xy in zip(scene.keys, scene.positions) if np.hypot(*(xy - c0)) <= 4.0]
        spectra = {s.channel: s.good() for s in combine(scene.spectra_of(keys), mode='sum')}
        assert set(spectra) == {'blue', 'green', 'red'}
        sf = build_sensfunc(spectra, 1200.0, np.array([2000.0, 12000.0]), np.array([1e-16, 1e-16]))
        assert set(sf.channels) == {'blue', 'green', 'red'}

        # the star's track is flat in the corrected RSS
        sr2 = load_field([out['paths']['green']], plane='auto')
        ref = tuple(res.track([res.lam_ref])[0])
        c2 = astar.measure_star(sr2, res.star, res.cents.bins, opts, guesses={0: ref})
        js = [j for j, b in enumerate(res.cents.bins) if b.role != 'overlap']
        g1 = [j for j in js if res.cents.good()[0, j]]
        g2 = [j for j in js if c2.good()[0, j]]
        assert max(np.ptp(res.cents.E[0, g1]), np.ptp(res.cents.N[0, g1])) > 1.0
        # residual = interpolation bias on the 0.75" lattice (~+-0.03"; the measurement itself is exact here)
        assert max(np.ptp(c2.E[0, g2]), np.ptp(c2.N[0, g2])) < 0.08, (np.ptp(c2.E[0, g2]), np.ptp(c2.N[0, g2]))

        # fixed 2" aperture: in the corrected RSS it captures what an aperture FOLLOWING the star
        # captures in the input (only the PSF width changes with lambda); fixed on the input it does not
        st = res.sr.channels['blue']
        E, N = tangent_offsets(st.ra, st.dec, *res.star)
        follow = lambda l: res.track([l])[0]                       # noqa: E731
        fx = {}
        for name, files, cen, ctot in (('fixed_before', sib, ref, follow), ('follow_before', sib, follow, follow),
                                       ('fixed_after', out['paths'], ref, lambda l: ref)):
            fx[name] = np.concatenate([[r['f_R2'] for r in rsscorr.aperture_fraction(files[c], E, N, cen, (2.0,), 6.0, ctot)]
                                       for c in ('blue', 'green', 'red')])
        assert np.max(np.abs(fx['fixed_before'] - fx['follow_before'])) > 0.15
        dev = np.abs(fx['fixed_after'] / fx['follow_before'] - 1)
        assert np.nanmax(dev) < 0.03, np.nanmax(dev)
        for c, lams in (('blue', np.arange(3700, 4800, 200)), ('red', np.arange(7000, 9800, 400))):
            t0 = _band_totals(sib[c], E, N, lambda l: res.track([l])[0], 6.0, lams)
            t1 = _band_totals(out['paths'][c], E, N, lambda l: ref, 6.0, lams)
            assert np.all(np.abs(t1 / t0 - 1) < 0.01), (c, t1 / t0)


if __name__ == '__main__':
    run_tests(globals())
