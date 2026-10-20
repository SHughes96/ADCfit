"""Full CLI run on mock RSS files with a known per-exposure dispersion: the fit must recover it,
predict the held-out blue, and the corrected cubes must have a flat star centroid vs wavelength."""

import csv
import json
import os
import tempfile

import numpy as np
from astropy.io import fits

from adcfit.cli import main
from tests.synth import make_field, write_rss, run_tests


def _read_csv(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def _run(tmp, **kw):
    sr, infos, inst, truth = make_field(nexp=3, nw=150, **kw)
    paths = write_rss(sr, infos, os.path.join(tmp, 'extractions'))
    qa_dir, cube_dir = os.path.join(tmp, 'qa'), os.path.join(tmp, 'cubes')
    rc = main(paths + ['--outdir', qa_dir, '--cube-outdir', cube_dir, '--no-pdf',
                       '--bins', '6', '6', '6'])
    return rc, truth, qa_dir, cube_dir, paths


def test_end_to_end_recovers_and_corrects():
    with tempfile.TemporaryDirectory() as tmp:
        rc, truth, qa_dir, cube_dir, paths = _run(tmp)
        assert rc == 0
        coefs = json.load(open(os.path.join(qa_dir, 'adc_coeffs.json')))
        for e, r in enumerate(coefs):
            assert r['status'] == 'ok', r
            c1 = np.array([r['c1_E'], r['c1_N']])
            # chromatic term recovered (arcsec per 1e-6; dn spans ~10 over the band)
            assert np.all(np.abs(c1 - truth['c1'][e]) < 0.006), (c1, truth['c1'][e])
        # held-out blue predicted by the green+red fit
        hold = [h for h in _read_csv(os.path.join(qa_dir, 'blue_holdout.csv')) if h['model'] == 'order1']
        for h in hold:
            assert float(h['rms_E']) < 0.05 and float(h['rms_N']) < 0.05, h
        # outputs in the requested places
        for v in ('adc', 'noadc'):
            for c in ('blue', 'green', 'red'):
                p = os.path.join(cube_dir, f'SYN_{v}_cube_{c}.fits')
                assert os.path.exists(p), p
        adc_blue = os.path.join(cube_dir, 'SYN_adc_cube_blue.fits')
        with fits.open(adc_blue) as h:
            # pipeline layout untouched; provenance only in appended scalar tables
            assert [x.name for x in h] == ['PRIMARY', 'VAR', 'COVERAGE', 'NEXP', 'WAVELENGTH', 'ADCINFO', 'ADCSLABS']
            assert not any(k.startswith('ADC') for k in h[0].header)
            assert h['ADCINFO'].header['ADCCORR'] is True
            assert h['ADCINFO'].header['ADCNSLAB'] > 1
            assert len(h['ADCINFO'].data) == 3 and np.all(np.isfinite(h['ADCINFO'].data['C1_E']))
        assert os.path.exists(adc_blue.replace('.fits', '_nexpslab.fits'))
        # same layout and WCS as a plain combine_cube().write of the same field
        from llamas_pyjamas.Combine.cube import combine_cube
        from adcfit.io import load_field
        from adcfit.stack import check_structure
        ref = os.path.join(tmp, 'ref_cube_blue.fits')
        combine_cube(load_field(paths), 'blue').write(ref)
        assert check_structure(adc_blue) == []
        assert check_structure(adc_blue, ref) == []
        for f in ('adc_shift_grid.txt', 'adc_shift_grid.csv', 'centroids.csv', 'inventory.csv',
                  'validation.csv', 'adc_summary.json'):
            assert os.path.exists(os.path.join(qa_dir, f)), f
        assert os.path.exists(os.path.join(cube_dir, 'slabs.csv'))
        # stacked-cube star centroid: flat after correction, drifting before
        val = _read_csv(os.path.join(qa_dir, 'validation.txt').replace('.txt', '.csv'))
        def ptp(v, k):
            x = np.array([float(r[k]) for r in val if r['variant'] == v and r[k] not in ('', 'nan')])
            return np.ptp(x)
        before = max(ptp('noadc', 'E'), ptp('noadc', 'N'))
        after = max(ptp('adc', 'E'), ptp('adc', 'N'))
        assert after < 0.1 and before > 2 * after, (before, after)
        # refuses to overwrite without --overwrite
        rc2 = main([p for p in []] + [os.path.join(tmp, 'extractions', f) for f in
                                      sorted(os.listdir(os.path.join(tmp, 'extractions'))) if f.endswith('_green.fits')]
                   + ['--outdir', qa_dir, '--cube-outdir', cube_dir, '--no-pdf'])
        assert rc2 == 1



def test_exposure_without_star_gets_model_correction():
    with tempfile.TemporaryDirectory() as tmp:
        sr, infos, inst, truth = make_field(nexp=4, nw=150, star_exps={1, 2, 3},
                                            decoy=dict(exp=0, dE=5.0, dN=0.0, flux=3e-16))
        paths = write_rss(sr, infos, os.path.join(tmp, 'extractions'))
        qa_dir, cube_dir = os.path.join(tmp, 'qa'), os.path.join(tmp, 'cubes')
        rc = main(paths + ['--outdir', qa_dir, '--cube-outdir', cube_dir, '--no-pdf',
                           '--bins', '6', '6', '6', '--star', *map(str, truth['star_radec']),
                           '--order', 'auto', '--model-exposures', 'both', '--drop', 'blue', 'green'])
        assert rc == 0
        coefs = json.load(open(os.path.join(qa_dir, 'adc_coeffs.json')))
        assert coefs[0]['status'] == 'model', coefs[0]
        assert coefs[0]['c1_source'].startswith('neighbour'), coefs[0]
        assert np.isfinite(coefs[0]['chi2r_model'])
        assert all(r['status'] == 'ok' for r in coefs[1:]), [r['status'] for r in coefs]
        rows = _read_csv(os.path.join(qa_dir, 'star_per_exposure.csv'))
        assert [r['detected'] for r in rows] == ['False', 'True', 'True', 'True'], rows
        # the estimated exposure carries an uncertainty
        grid = [r for r in _read_csv(os.path.join(qa_dir, 'adc_shift_grid.csv')) if r['exp'] == coefs[0]['exp']]
        assert max(float(r['err_E']) for r in grid) > 0 and max(float(r['err_N']) for r in grid) > 0
        # two versions in each --drop channel: everything, and without the estimated exposure
        from adcfit.stack import check_structure
        for c in ('blue', 'green'):
            full = os.path.join(cube_dir, f'SYN_adc_cube_{c}.fits')
            dropped = os.path.join(cube_dir, f'SYN_adcdrop_cube_{c}.fits')
            assert check_structure(dropped, full) == []
            with fits.open(full) as h1, fits.open(dropped) as h2:
                assert h1[0].header['NEXPTOT'] == 4 and h2[0].header['NEXPTOT'] == 3
                assert h2['NEXP'].data.max() == h1['NEXP'].data.max() - 1
                st = [x.strip() for x in h2['ADCINFO'].data['STATUS']]
                assert st == ['dropped', 'ok', 'ok', 'ok'], st
        assert not os.path.exists(os.path.join(cube_dir, 'SYN_adcdrop_cube_red.fits'))
        for f in ('blue_relative.txt', 'blue_relative.csv', 'coverage_slabs.txt', 'shift_extremes.txt'):
            assert os.path.exists(os.path.join(qa_dir, f)), f
        rel = [r for r in _read_csv(os.path.join(qa_dir, 'blue_relative.csv')) if r['source'] == 'broad']
        assert len(rel) == 3 and all(abs(float(r['resid_E'])) < 0.1 and abs(float(r['resid_N'])) < 0.1 for r in rel), rel
        summ = json.load(open(os.path.join(qa_dir, 'adc_summary.json')))
        assert summ['curvature'] in ('1', 'shared')
        assert {'adcdrop_blue', 'adcdrop_green'} <= set(summ['cubes']) and 'adcdrop_red' not in summ['cubes']
        assert summ['model_exposures']['drop_channels'] == ['blue', 'green']


if __name__ == '__main__':
    run_tests(globals())
