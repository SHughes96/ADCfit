"""Standard-star / sensfunc diagnostic on a synthetic single exposure: a sensfunc built from a known
fibre aperture must be traced back to that aperture, and the captured fraction it implies must match
the star's actual fraction in that aperture."""

import os
import tempfile

import numpy as np

from llamas_pyjamas.Flux.sensFunc import SensChannel, SensFunc
from llamas_pyjamas.Utils.register import _fibre_xy
from astropy.io import fits

from adcfit import standard, star as astar
from tests.synth import make_field, write_rss, run_tests

T_EXP, F_REF = 1200.0, 1e-16


def _sensfunc(paths, rows_ap, ref_row, out):
    """S from summing ``rows_ap`` interpolated onto ``ref_row``'s grid, as the picker does."""
    chans = {}
    for c, p in paths.items():
        with fits.open(p) as h:
            W, F = h['WAVE'].data.astype(float), h['FLAM'].data.astype(float)
        w = W[ref_row]
        C = sum(np.nan_to_num(np.interp(w, W[r], F[r], left=np.nan, right=np.nan)) for r in rows_ap)
        raw = F_REF * T_EXP / C
        chans[c] = SensChannel(c, w, raw, raw, np.isfinite(raw) & (raw > 0))
    SensFunc(chans, dict(standard='SYNSTD', exptime=T_EXP, airmass=1.2)).save(out)
    return out


def test_standard_aperture_and_capture():
    with tempfile.TemporaryDirectory() as tmp:
        sr, infos, inst, truth = make_field(nexp=1, nw=600, gain=1.2 + 0.0j, s0=(0.0, 0.0), noise=2e-20,
                                            flux0=3e-15, dither=0.0, reg_sigma=0.0)
        for st in sr.channels.values():
            st.flux -= np.float32(1e-18)                 # synth's constant floor (a sky-subtracted RSS has none)
        green = write_rss(sr, infos, os.path.join(tmp, 'ex'), obj='SYNSTD')[0]
        sib = standard.std_siblings(green)
        with fits.open(sib['red']) as h:
            fm, F = h['FIBERMAP'].data, h['FLAM'].data
            xy = np.column_stack(_fibre_xy(fm['FIBER_ID'], fm['BENCHSIDE'])).astype(float)
        ref_row = int(np.nanargmax(np.nanmedian(F, axis=1)))           # brightest red fibre
        rows_ap = np.flatnonzero(np.hypot(*(xy - xy[ref_row]).T) <= 2.0 + 1e-3)
        sens = _sensfunc(sib, rows_ap, ref_row, os.path.join(tmp, 'std_sensfunc.fits'))

        standard.load_reference = lambda hdr: ('SYNSTD', np.array([2000.0, 12000.0]), np.array([F_REF, F_REF]))
        opts = astar.MeasureOpts(exclusions=astar.default_exclusions(), destripe=False)
        res = standard.run_standard(green, sens, [], opts, excl=astar.default_exclusions(), r_tot=(5.0, 6.0))
        assert res.fit.status == 'ok'
        # the picker aperture is recovered exactly
        ap = res.aperture
        assert ap['ok'] and ap['ref_row'] == ref_row, ap.get('note')
        assert sorted(ap['best']['rows']) == sorted(rows_ap), (ap['best']['nfib'], rows_ap.size)
        # captured fraction: data (C_ap / total within 6") vs a direct sum over the whole IFU
        for r in res.capture:
            c, lo, hi = r['channel'], r['lam'] - 50, r['lam'] + 50
            with fits.open(sib[c]) as h:
                W, F = h['WAVE'].data, h['FLAM'].data
            s = (W[0] >= lo) & (W[0] < hi)
            f_true = F[rows_ap][:, s].sum() / F[:, s].sum()
            assert abs(r['f_R6'] - f_true) < 0.005, (c, r['lam'], r['f_R6'], f_true)
            assert abs(r['fmod_track'] - f_true) < 0.03, (c, r['lam'], r['fmod_track'], f_true)
        # the star's dispersion moves it out of the aperture to the blue: f falls blueward
        f = {c: np.mean([r['f_R6'] for r in res.capture if r['channel'] == c]) for c in ('blue', 'red')}
        assert f['blue'] < f['red'] - 0.05, f
        assert 'REBUILD' in res.summary['verdict']


if __name__ == '__main__':
    run_tests(globals())
