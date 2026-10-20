"""The slab stacker must reproduce combine_cube exactly at zero shift, and shift in the right sense."""

import numpy as np

from llamas_pyjamas.Combine.cube import combine_cube, field_grid
from llamas_pyjamas.Combine.coadd import _kernel

from adcfit.stack import kernel_matrix, plan_slabs, stack_channel, wave_grid
from tests.synth import make_field, run_tests


def _loop_S(px, py, ny, nx, r, kern):
    """combine_cube's per-fibre loop, verbatim."""
    from scipy.sparse import coo_matrix
    rows, cols, vals = [], [], []
    for i in range(px.size):
        xi, yi = px[i], py[i]
        x0, x1 = max(0, int(np.floor(xi - r))), min(nx - 1, int(np.ceil(xi + r)))
        y0, y1 = max(0, int(np.floor(yi - r))), min(ny - 1, int(np.ceil(yi + r)))
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)
        d2 = (xs[None, :] - xi) ** 2 + (ys[:, None] - yi) ** 2
        inside = d2 <= r * r
        if not inside.any():
            continue
        yy, xx = np.nonzero(inside)
        rows.append((y0 + yy) * nx + (x0 + xx)); cols.append(np.full(yy.size, i))
        vals.append(kern(np.sqrt(d2[inside])).astype(np.float32))
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ny * nx, px.size)).tocsr()


def test_kernel_matrix_equals_loop():
    rng = np.random.default_rng(0)
    px, py = rng.uniform(-3, 40, 500), rng.uniform(-3, 30, 500)
    for name, fwhm in (('gaussian', 0.9), ('tophat', 1.4), ('gaussian', 2.5)):
        r, kern = _kernel(name, fwhm, 0.5)
        A, _, _ = kernel_matrix(px, py, 28, 38, r, kern)
        B = _loop_S(px, py, 28, 38, r, kern)
        assert (A != B).nnz == 0


def test_plan_slabs_respects_tolerance():
    nw = 400
    corr = np.zeros((3, nw, 2))
    corr[:, :, 0] = np.linspace(0.6, 0.0, nw)[None] * np.array([1.0, 0.5, -0.8])[:, None]
    slabs = plan_slabs(corr, 0.05)
    assert slabs[0].i0 == 0 and slabs[-1].i1 == nw
    for s in slabs:
        seg = corr[:, s.i0:s.i1]
        assert np.all(np.abs(seg - s.shift[:, None, :]) <= 0.025 + 1e-12)
    assert len(plan_slabs(np.zeros((3, nw, 2)), 0.05)) == 1
    assert len(plan_slabs(corr, 0.0)) == nw


def test_zero_shift_matches_combine_cube():
    sr, *_ = make_field(nexp=2, nw=60)
    grid = field_grid(sr, list(sr.channels), pixscale=0.5, kernel='gaussian', kernel_fwhm=0.9)
    for c in ('blue', 'red'):
        ref = combine_cube(sr, c, grid=grid)
        wl, dw = wave_grid(sr.channels[c])
        zero = lambda e, w: np.zeros((w.size, 2))           # noqa: E731
        out = stack_channel(sr, c, grid, wl, dw, {'none': None, 'zero_multi': zero}, slab_tol=-1)
        for name in ('none', 'zero_multi'):
            cube = out[name].cube
            assert np.allclose(cube.wave, ref.wave)
            fin = np.isfinite(ref.data)
            assert np.array_equal(fin, np.isfinite(cube.data))
            assert np.allclose(cube.data[fin], ref.data[fin], rtol=1e-5, atol=0)
            assert np.allclose(cube.var[fin], ref.var[fin], rtol=1e-5, atol=0)
            assert np.array_equal(cube.coverage, ref.coverage)
            assert np.array_equal(cube.nexp, ref.nexp)


def test_shift_sign_moves_star_east():
    sr, infos, inst, truth = make_field(nexp=2, nw=40)
    grid = field_grid(sr, list(sr.channels), pixscale=0.5, kernel='gaussian', kernel_fwhm=0.9)
    wl, dw = wave_grid(sr.channels['red'])
    east = lambda e, w: np.tile([1.0, 0.0], (w.size, 1))    # noqa: E731
    out = stack_channel(sr, 'red', grid, wl, dw, {'none': None, 'east': east})
    from llamas_pyjamas.Combine.spectrum import fit_gaussian_image
    w2 = grid[0]
    px0, py0 = w2.world_to_pixel_values(*truth['star_radec'])
    pos = {}
    for k in ('none', 'east'):
        img = np.nanmedian(out[k].cube.data, axis=0)
        g = fit_gaussian_image(np.nan_to_num(img), float(px0), float(py0), 6)
        pos[k] = np.array(w2.pixel_to_world_values(g.x_mean.value, g.y_mean.value))
    from adcfit.io import tangent_offsets
    dE, dN = tangent_offsets(pos['east'][0], pos['east'][1], pos['none'][0], pos['none'][1])
    assert abs(dE - 1.0) < 0.05 and abs(dN) < 0.05, (dE, dN)


def _without(sr, d):
    """The same field with exposure ``d`` removed (fibre rows dropped, exposure indices renumbered)."""
    import dataclasses
    from llamas_pyjamas.Combine.superRSS import SuperRSS
    remap = np.array([e - (e > d) for e in range(sr.n_exposures)])
    chans = {}
    for c, st in sr.channels.items():
        k = st.exposure != d
        chans[c] = dataclasses.replace(st, **{f.name: getattr(st, f.name)[k] for f in dataclasses.fields(st)
                                              if f.name != 'channel'})
        chans[c].exposure = remap[chans[c].exposure]
    return SuperRSS(sr.field, sr.plane, sr.bunit, [x for e, x in enumerate(sr.exposures) if e != d], chans)


def test_drop_equals_stack_without_that_exposure():
    from adcfit.stack import coverage_summary
    sr, *_ = make_field(nexp=3, nw=50)
    grid = field_grid(sr, list(sr.channels), pixscale=0.5, kernel='gaussian', kernel_fwhm=0.9)
    wl, dw = wave_grid(sr.channels['green'])
    shift = lambda e, w: np.outer(np.linspace(0, 0.6, w.size), [1.0, -0.5]) * (e + 1)   # noqa: E731
    out = stack_channel(sr, 'green', grid, wl, dw, {'all': shift, 'drop1': shift}, drop={'drop1': [1]})
    sr2 = _without(sr, 1)
    shift2 = lambda e, w: shift([0, 2][e], w)                                            # noqa: E731
    ref = stack_channel(sr2, 'green', grid, wl, dw, {'x': shift2})['x']
    got = out['drop1']
    fin = np.isfinite(ref.cube.data)
    assert np.array_equal(fin, np.isfinite(got.cube.data))
    assert np.allclose(got.cube.data[fin], ref.cube.data[fin], rtol=1e-5)
    assert np.array_equal(got.cube.nexp, ref.cube.nexp) and np.array_equal(got.cube.coverage, ref.cube.coverage)
    assert got.cube.nexp.max() == out['all'].cube.nexp.max() - 1
    assert got.cube.meta['NEXPTOT'] == 2 and len(got.cube.meta['exposure_paths']) == 2
    cs = coverage_summary(out['all'])
    assert cs['nexp_max'] == 3 and cs['nslab'] == len(out['all'].slabs) > 1
    assert cs['full_all'] <= cs['full_any'] <= cs['covered'] and cs['varying'] > 0


if __name__ == '__main__':
    run_tests(globals())
