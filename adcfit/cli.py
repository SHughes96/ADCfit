"""Command line: measure, fit and correct ADC / DAR errors in a LLAMAS dither stack.

    python -m adcfit <cube.fits | RSS files...> [--outdir DIR] [--cube-outdir DIR] [...]
    python -m adcfit --dir <extractions> --object <OBJECT> --measure-only

Stages (QA under --outdir; cubes under --cube-outdir):
  1 inventory      exposure geometry & weather, header cross-checks
  2 registration   broad-band star position / FWHM per exposure and channel (uncorrected)
  3 star           reference-star selection
  4 centroids      star centroid vs wavelength per exposure (sky + instrument frames)
  5 fit            refractivity-basis fit on green+red; blue held out; tables of shift vs lambda
  6 stack          corrected (adc) + uncorrected (noadc) cubes on identical grids
  7 validation     fibre-space residuals after correction; cube centroid/FWHM/concentration vs lambda
"""

import argparse
import logging
import os
import re
import sys
from typing import List, Optional

import numpy as np

logger = logging.getLogger('adcfit')


def _parser():
    p = argparse.ArgumentParser(prog='adcfit', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('input')
    g.add_argument('inputs', nargs='*', help='one combined cube (uses its RSSFILn) or RSS files')
    g.add_argument('--dir', help='directory of RSS files (with --object)')
    g.add_argument('--object', default='', help='OBJECT prefix for --dir')
    g.add_argument('--rss-dir', help='where to look for the RSS files named in a cube header')
    g.add_argument('--plane', choices=['auto', 'flam', 'skysub'], default='auto')
    g.add_argument('--keep-bad-fibres', action='store_true', help='do not mask negative outlier fibres')
    g = p.add_argument_group('reference star')
    g.add_argument('--star', nargs=2, metavar=('RA', 'DEC'), help='deg or sexagesimal')
    g.add_argument('--star-pix', nargs=2, type=float, metavar=('X', 'Y'),
                   help='1-based (DS9) pixel on the input cube')
    g.add_argument('--search-radius', type=float, default=4.0,
                   help='arcsec around the nominal/given position (default 4)')
    g.add_argument('--cluster-radius', type=float, default=2.5,
                   help='max distance (arcsec) at which a detection in an exposure counts as the same '
                        'star (default 2.5); exposures with no detection that close are not measured')
    g.add_argument('--min-snr', type=float, default=30.0,
                   help='star selection S/N: aperture S/N of the broad red band in the exposure where '
                        'the source is brightest (default 30)')
    g.add_argument('--max-fwhm', type=float, default=3.0)
    g.add_argument('--chan-min-snr', type=float, default=5.0,
                   help='per-channel check: S/N a candidate needs in each broad band (default 5)')
    g.add_argument('--chan-max-dpos', type=float, default=3.0,
                   help='per-channel check: max distance (arcsec) of each channel\'s position from the '
                        'red one (default 3)')
    g = p.add_argument_group('measurement')
    g.add_argument('--bins', nargs=3, type=int, default=[8, 8, 8], metavar=('B', 'G', 'R'),
                   help='bins per channel (the maximum when --target-snr > 0)')
    g.add_argument('--target-snr', type=float, default=10.0,
                   help='per-bin aperture S/N to aim for: fewer bins for a faint star (0 = fixed --bins)')
    g.add_argument('--range', nargs=3, action='append', default=[], metavar=('CH', 'LO', 'HI'),
                   help='override a channel\'s clean wavelength range (repeatable)')
    g.add_argument('--exclude', nargs=2, type=float, action='append', default=[], metavar=('LO', 'HI'),
                   help='extra wavelength window to exclude from all bins (repeatable)')
    g.add_argument('--psf', choices=['gaussian', 'moffat'], default='gaussian')
    g.add_argument('--moffat-beta', type=float, default=3.5)
    g.add_argument('--sigma', choices=['auto', 'free', 'kolmogorov'], default='auto',
                   help='PSF width per bin: auto = free within 0.7-1.6x the seeing-scaled width, '
                        'falling back to that fixed width; free; kolmogorov = fixed lambda^-0.2')
    g.add_argument('--bg', choices=['const', 'plane'], default='const')
    g.add_argument('--no-destripe', action='store_true',
                   help='do not subtract per-fibre-map-row medians before fitting')
    g.add_argument('--fit-radius', type=float, help='aperture radius (arcsec); default from FWHM')
    g.add_argument('--min-ap-snr', type=float, default=3.0, help='aperture S/N floor (LOWSN)')
    g.add_argument('--max-cen-err', type=float, default=0.4, help='centroid error ceiling, arcsec (LOWSN)')
    g.add_argument('--reject-flags', default='EDGE,BOUND,FEWFIB,LOWSN,HOLE,FAILED',
                   help='flags that exclude a bin from the fit')
    g = p.add_argument_group('fit')
    g.add_argument('--fit-channels', default='green,red')
    g.add_argument('--order', choices=['1', '2', 'shared', 'auto'], default='1',
                   help="1: c0 + c1 dn; 2: + an independent c2 dn^2 per exposure; shared: c2 = beta c1 with "
                        "one beta for all exposures; auto: 1 or shared, by BIC")
    g.add_argument('--model-c1', choices=['neighbour', 'pooled'], default='neighbour',
                   help='chromatic term for exposures without their own fit: the nearest-in-time measured '
                        'exposure scaled by the ratio of no-ADC predictions, or the best pooled model')
    g.add_argument('--model-max-chi2r', type=float, default=10.0,
                   help='reduced chi2 of that estimator on the measured exposures above which the exposure '
                        'is flagged and extra --drop channel cubes without it are written')
    g.add_argument('--model-exposures', choices=['apply', 'exclude', 'exclude-blue', 'both'], default='apply',
                   help='exposures without their own fit (status model/fallback): apply the estimate '
                        '(gate-flagged ones also get <field>_adcdrop_cube_<ch>.fits without them for each --drop '
                        'channel); exclude them from the ADC cubes of the --drop channels; apply AND write the '
                        'adcdrop cubes without all of them (both). exclude-blue = exclude --drop blue (deprecated)')
    g.add_argument('--drop', nargs='+', choices=['blue', 'green', 'red'], default=['blue'], metavar='CH',
                   help='channels from which exposures without their own fit may be dropped (see '
                        '--model-exposures; default blue; channels not stacked are ignored)')
    g.add_argument('--poly-deg', type=int, default=2, help='comparison polynomial in lambda')
    g.add_argument('--clip', type=float, default=3.0)
    g.add_argument('--err-floor', type=float, default=0.01, help='arcsec added in quadrature')
    g.add_argument('--lam-ref', type=float, help='reference/anchor wavelength (default: mean red bin)')
    g.add_argument('--anchor', choices=['star', 'none'], default='star',
                   help="star: align all exposures to the star's median position (registration + "
                        "ADC); none: remove only the chromatic part")
    g.add_argument('--shift-step', type=float, default=250.0, help='A step of the printed shift table')
    g = p.add_argument_group('stacking')
    g.add_argument('--channels', default='blue,green,red')
    g.add_argument('--slab-tol', type=float, default=0.05, help='arcsec peak-to-peak per slab')
    g.add_argument('--max-extrap-shift', type=float, default=3.0,
                   help='warn when the chromatic shift of an exposure without its own fit (or of any '
                        'exposure in a held-out channel) exceeds this many arcsec')
    g.add_argument('--pixscale', type=float)
    g.add_argument('--kernel', choices=['gaussian', 'tophat'])
    g.add_argument('--fwhm', type=float, help='resampling kernel FWHM (arcsec)')
    g.add_argument('--weight', choices=['ivar', 'uniform', 'exptime'])
    g.add_argument('--units', choices=['sb', 'flux'])
    g.add_argument('--dwave', type=float)
    g.add_argument('--min-coverage', type=int, default=1)
    g.add_argument('--also-regonly', action='store_true', help='also write registration-only cubes')
    g = p.add_argument_group('standard star / sensitivity function (diagnostic only; nothing is changed)')
    g.add_argument('--std-rss', help='RSS file (any channel) of the standard exposure the sensfunc was built from')
    g.add_argument('--sensfunc', help='the sensitivity function (*_sensfunc.fits) applied to the science FLAM')
    g.add_argument('--std-star', nargs=2, metavar=('RA', 'DEC'), help='standard position if auto-detection fails')
    g.add_argument('--std-target-snr', type=float, default=30.0, help='per-bin S/N for the standard (default 30)')
    g.add_argument('--std-rtot', nargs='+', type=float, default=[4.0, 5.0, 6.0],
                   help='radii (arcsec) of the total-flux apertures that follow the star (default 4 5 6)')
    g.add_argument('--std-correct-rss', metavar='OUTDIR',
                   help='write ADC-corrected copies of the standard RSS files (same names) here, for building '
                        'a new sensitivity function in CubeViewer; must not be the input directory')
    g.add_argument('--std-slab-tol', type=float, default=0.02,
                   help='arcsec: max change of the shift within one wavelength slab when correcting the RSS')
    g = p.add_argument_group('output')
    g.add_argument('--outdir', help='QA plots/tables/JSON (default <reduced>/combined/adcfit_<field>)')
    g.add_argument('--cube-outdir', help='where cubes are written (default: --outdir)')
    g.add_argument('--overwrite', action='store_true', help='replace existing output cubes')
    g.add_argument('--measure-only', action='store_true', help='stop after the fit (stage 5)')
    g.add_argument('--no-cubes', action='store_true', help='skip cube building (stage 6)')
    g.add_argument('--no-pdf', action='store_true')
    g.add_argument('-v', '--verbose', action='store_true')
    return p


def _coord(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if ':' in str(ra) or 'h' in str(ra):
        c = SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg))
    else:
        c = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
    return float(c.ra.deg), float(c.dec.deg)


def _safe(s):
    return re.sub(r'[^A-Za-z0-9_.+-]', '_', s or 'field')


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    for noisy in ('matplotlib', 'PIL'):          # never pre-create 'astropy': it installs its own Logger class
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return run(args)
    except Exception as exc:                              # noqa: BLE001
        from adcfit.star import NoStarError
        if isinstance(exc, NoStarError):
            logger.error('%s', exc)
            return 2
        if isinstance(exc, (FileNotFoundError, ValueError, FileExistsError)):
            logger.error('%s', exc)
            return 1
        raise


def _broadband_images(sr, bins, grid, pixscale, inst, args, excl):
    """Broad-band images on the common grid: the red channel's per-exposure images (for the
    registration figure) and each channel's all-exposure combined image."""
    from adcfit import star as astar, stack as ast
    imgs, chan_imgs = [], {}
    rc = 'red' if 'red' in sr.channels else list(sr.channels)[0]
    for b in astar.broad_bins(bins):
        if b.channel not in sr.channels:
            continue
        st = sr.channels[b.channel]
        allv = dict(ra=[], dec=[], v=[], var=[])
        for e in range(sr.n_exposures):
            rows = np.flatnonzero(st.exposure == e)
            v, var = astar.bin_values(st, rows, b.lo, b.hi, excl)
            if inst and not args.no_destripe and b.channel in inst:
                v = astar.destripe(v, astar.row_keys(inst[b.channel][rows]))
            if b.channel == rc:
                imgs.append(ast.fibre_image(st.ra[rows], st.dec[rows], v, var, grid, pixscale=pixscale))
            allv['ra'].append(st.ra[rows]); allv['dec'].append(st.dec[rows]); allv['v'].append(v); allv['var'].append(var)
        chan_imgs[b.channel] = ast.fibre_image(*(np.concatenate(allv[k]) for k in ('ra', 'dec', 'v', 'var')),
                                               grid, pixscale=pixscale)
    return imgs, chan_imgs


def run(args) -> int:
    from llamas_pyjamas.Combine.superRSS import combined_dir
    from adcfit import __version__
    from adcfit import io as aio, dar, star as astar, stack as ast, qa

    if args.std_correct_rss and not args.std_rss:
        raise ValueError('--std-correct-rss needs --std-rss')
    if args.std_rss and not (args.inputs or args.dir):
        return run_standard_only(args)

    # ------------------------------------------------------------------ inputs
    inp = aio.resolve_inputs(args.inputs, args.dir, args.object, args.rss_dir)
    cs = aio.cube_settings(inp.cube_path, inp.cube_hdr) if inp.cube_path else None
    plane = args.plane if (args.plane != 'auto' or cs is None) else cs['plane']
    sr = aio.load_field(inp.rss_paths, plane=plane, reject_bad_fibres=not args.keep_bad_fibres)
    field = _safe(sr.field)
    outdir = os.path.abspath(args.outdir or os.path.join(combined_dir(inp.rss_paths), f'adcfit_{field}'))
    cube_outdir = os.path.abspath(args.cube_outdir or outdir)
    os.makedirs(outdir, exist_ok=True)
    stack_chans = [c for c in args.channels.split(',') if c in sr.channels]
    variants = ['noadc', 'adc'] + (['regonly'] if args.also_regonly else [])
    cube_paths = {(v, c): os.path.join(cube_outdir, f'{field}_{v}_cube_{c}.fits')
                  for v in variants for c in stack_chans}
    drop_chans = set(args.drop) | ({'blue'} if args.model_exposures == 'exclude-blue' else set())
    if args.model_exposures in ('apply', 'both'):
        # written only if some exposure is dropped from them; reserved now for the clash check
        for c in stack_chans:
            if c in drop_chans:
                cube_paths[('adcdrop', c)] = os.path.join(cube_outdir, f'{field}_adcdrop_cube_{c}.fits')
    if not (args.measure_only or args.no_cubes):
        os.makedirs(cube_outdir, exist_ok=True)
        clash = [p for p in cube_paths.values() if os.path.exists(p)]
        if clash and not args.overwrite:
            raise FileExistsError('output cubes exist (use --overwrite or another --cube-outdir):\n  '
                                  + '\n  '.join(clash))
        if inp.cube_path and os.path.abspath(inp.cube_path) in cube_paths.values():
            raise FileExistsError('refusing to overwrite the input cube')
    logger.info('QA output: %s', outdir)
    if not (args.measure_only or args.no_cubes):
        logger.info('cube output: %s', cube_outdir)
    rep = qa.QAReport(outdir, pdf=not args.no_pdf)
    summary = dict(field=sr.field, inputs=inp.rss_paths, input_cube=inp.cube_path, plane=sr.plane,
                   outdir=outdir, cube_outdir=cube_outdir, args=vars(args),
                   llamas_pyjamas=aio.llamas_pyjamas_version())
    fatal = 0
    for name in args.reject_flags.split(','):
        if name.strip():
            fatal |= astar.FLAGS[name.strip().upper()]
    fit_channels = tuple(c.strip() for c in args.fit_channels.split(','))

    # ------------------------------------------------------------------ 1 inventory
    infos = aio.read_exposure_info(sr)
    geoms = [dar.exposure_geometry(i) for i in infos]
    chpos = aio.check_channel_positions(sr)
    inv = []
    for i, g in zip(infos, geoms):
        inv.append(dict(exp=i.exp_id, mjd=i.mjd, exptime=i.exptime, X_start_hdr=i.airmass_start,
                        X_end_hdr=i.airmass_end, X_mean=g.airmass_mean, q_end_hdr=i.parang_end,
                        q_start=g.q_start, q_mean=g.q_mean, q_end=g.q_end,
                        dq_end=g.checks.get('parang_end_deg', np.nan), HA_start=g.H_start, HA_end=g.H_end,
                        P=i.pressure, T=i.temperature, RH=i.humidity, seeing=i.seeing, tel_rot=i.tel_rot,
                        adc=i.adc_state, velcorr=i.velcorr, chan_pos_diff=chpos.get(i.exp_id, np.nan),
                        defaults=','.join(i.defaulted) or '-', geom_src=g.source))
    qa.write_csv(os.path.join(outdir, 'inventory.csv'), inv)
    qa.emit_table(os.path.join(outdir, 'inventory.txt'), inv,
                  ['exp', 'exptime', 'X_start_hdr', 'X_mean', 'X_end_hdr', 'q_end_hdr', 'q_mean', 'dq_end',
                   'HA_start', 'P', 'T', 'RH', 'seeing', 'tel_rot', 'adc', 'chan_pos_diff', 'defaults'],
                  dict(exptime='%.0f', X_start_hdr='%.3f', X_mean='%.3f', X_end_hdr='%.3f', q_end_hdr='%+.1f',
                       q_mean='%+.1f', dq_end='%+.2f', HA_start='%+.3f', P='%.1f', T='%.1f', RH='%.0f',
                       seeing='%.2f', tel_rot='%.2f', chan_pos_diff='%.3f'),
                  title=f'Stage 1: {sr.field}: {sr.n_exposures} exposures, plane {sr.plane}')
    rep.save(qa.fig_inventory(infos, geoms), 'inventory')
    summary['exposures'] = inv

    try:
        inst = aio.instrument_xy(sr)
    except Exception as exc:                              # noqa: BLE001
        logger.warning('instrument-frame positions unavailable (%s)', exc)
        inst = {}

    # ------------------------------------------------------------------ bins
    excl = astar.default_exclusions(args.exclude)
    overrides = {r[0]: (float(r[1]), float(r[2])) for r in args.range}
    bins = astar.make_bins(sr, dict(zip(('blue', 'green', 'red'), args.bins)), fit_channels,
                           overrides, excl)
    qa.emit_table(os.path.join(outdir, 'bins.txt'),
                  [dict(bin=b.index, channel=b.channel, lo=b.lo, hi=b.hi, lam_eff=b.lam_eff, role=b.role)
                   for b in bins], ['bin', 'channel', 'lo', 'hi', 'lam_eff', 'role'],
                  dict(lo='%.0f', hi='%.0f', lam_eff='%.0f'), title='wavelength bins')
    ranges = {b.channel: (b.lo, b.hi) for b in astar.broad_bins(bins)}
    val_bins = list(bins)            # full binning for validating the stacked cubes (8x deeper)

    # ------------------------------------------------------------------ 3 star (needed by 2)
    grid = ast.make_grid(sr, args.pixscale or (cs or {}).get('pixscale', 0.5),
                         args.kernel or 'gaussian', args.fwhm or (cs or {}).get('kernel_fwhm', 0.9),
                         cube_grid=(cs or {}).get('grid'))
    pixscale = args.pixscale or (cs or {}).get('pixscale', 0.5)
    cand_cols = ['id', 'ra', 'dec', 'n_exp', 'flux', 'fwhm', 'snr', 'fit_snr', 'ellipticity', 'edge_dist',
                   'isolated', 'snr_blue', 'snr_green', 'snr_red', 'dpos_blue', 'dpos_green', 'chan_ok',
                   'qualifies', 'reason']
    cand_fmt = dict(ra='%.6f', dec='%.6f', flux='%.3g', fwhm='%.2f', snr='%.0f', ellipticity='%.2f',
                    fit_snr='%.0f', edge_dist='%.1f', snr_blue='%.0f', snr_green='%.0f', snr_red='%.0f', dpos_blue='%.2f',
                    dpos_green='%.2f')
    star_in = None
    if args.star:
        star_in = _coord(*args.star)
    elif args.star_pix:
        if cs is None:
            raise ValueError('--star-pix needs a cube input')
        star_in = tuple(float(v) for v in cs['grid'][0].pixel_to_world_values(args.star_pix[0] - 1,
                                                                               args.star_pix[1] - 1))
    try:
        choice = astar.find_star(sr, bins, star=star_in, search_radius=args.search_radius,
                                 cluster_radius=args.cluster_radius, inst=None if args.no_destripe else inst,
                                 min_snr=args.min_snr, max_fwhm=args.max_fwhm, excl=excl,
                                 chan_min_snr=args.chan_min_snr, chan_max_dpos=args.chan_max_dpos)
    except astar.NoStarError as exc:
        qa.write_csv(os.path.join(outdir, 'star_candidates.csv'), exc.candidates)
        qa.emit_table(os.path.join(outdir, 'star_candidates.txt'), exc.candidates, cand_cols, cand_fmt,
                      title='Stage 3: candidates; NO STAR SELECTED (reason = criteria each one failed)')
        _, chan_imgs = _broadband_images(sr, bins, grid, pixscale, inst, args, excl)
        rep.save(qa.fig_star(chan_imgs.get('red', next(iter(chan_imgs.values()))), grid, exc.candidates,
                             None, None), 'star_selection')
        rep.save(qa.fig_star_channels(chan_imgs, grid, exc.candidates, None), 'star_channels')
        rep.close()
        raise
    qa.write_csv(os.path.join(outdir, 'star_candidates.csv'), choice.candidates)
    qa.emit_table(os.path.join(outdir, 'star_candidates.txt'), choice.candidates, cand_cols, cand_fmt,
                  title=f'Stage 3: candidates; chosen id={choice.chosen} ({choice.source}) at '
                        f'RA {choice.ra:.6f} Dec {choice.dec:.6f}')
    star_radec = (choice.ra, choice.dec)
    summary['star'] = dict(ra=choice.ra, dec=choice.dec, source=choice.source, chosen=choice.chosen)
    affines = aio.exposure_affines(sr, inst, *star_radec) if inst else None
    if affines:
        summary['wcs_affine'] = [dict(exp=sr.exposures[e].exposure_id, channel=c, pa_x=a.pa_x,
                                      parity=a.parity, scale=a.scale, rms=a.rms)
                                 for c, lst in affines.items() for e, a in enumerate(lst) if a is not None]
    opts = _measure_opts(args, excl)
    mkw = dict(inst=inst or None, guesses=choice.exp_pos, skip=set(choice.exp_missing))
    if choice.exp_table:
        qa.write_csv(os.path.join(outdir, 'star_per_exposure.csv'), choice.exp_table)
        qa.emit_table(os.path.join(outdir, 'star_per_exposure.txt'), choice.exp_table,
                      ['exp', 'detected', 'dist', 'flux', 'nearest_other'],
                      dict(dist='%.2f', flux='%.3g', nearest_other='%.2f'),
                      title=f'Stage 3: detection of the star in each exposure (within {args.cluster_radius:.1f}"; '
                            'dist / nearest_other in arcsec)')
    for e in choice.exp_missing:
        logger.warning('%s: the star is not detected within %.1f" in this exposure; it gets the ADC '
                       'correction predicted from the other exposures, no re-registration',
                       sr.exposures[e].exposure_id, args.cluster_radius)

    # ------------------------------------------------------------------ 2 registration
    reg = astar.measure_star(sr, star_radec, astar.broad_bins(bins), opts, affines, **mkw)
    reg_rows = []
    for j, b in enumerate(reg.bins):
        ok = (reg.flags[:, j] & fatal) == 0
        mE, mN = (np.nanmedian(reg.E[ok, j]), np.nanmedian(reg.N[ok, j])) if ok.any() else (np.nan, np.nan)
        for e, eid in enumerate(reg.exp_ids):
            reg_rows.append(dict(exp=eid, channel=b.channel, dE=reg.E[e, j] - mE, dN=reg.N[e, j] - mN,
                                 dR=np.hypot(reg.E[e, j] - mE, reg.N[e, j] - mN), fwhm=reg.fwhm[e, j],
                                 snr=reg.snr[e, j], flags=astar.flag_names(reg.flags[e, j])))
    qa.write_csv(os.path.join(outdir, 'registration_uncorrected.csv'), reg_rows)
    if args.target_snr > 0:
        nmax = dict(zip(('blue', 'green', 'red'), args.bins))
        nb, brow = {}, []
        for j, b in enumerate(reg.bins):
            s_b = float(np.nanmedian(reg.apsnr[:, j])) if np.isfinite(reg.apsnr[:, j]).any() else np.nan
            nb[b.channel] = astar.bins_for_snr(s_b, nmax.get(b.channel, 8), args.target_snr,
                                               nmin=2 if b.channel in fit_channels else 1)
            brow.append(dict(channel=b.channel, broad_ap_snr=s_b, bins=nb[b.channel], max=nmax.get(b.channel, 8)))
        print(qa.text_table(brow, ['channel', 'broad_ap_snr', 'bins', 'max'], dict(broad_ap_snr='%.1f'),
                            title=f'bins per channel for a per-bin S/N of ~{args.target_snr:.0f} '
                                  '(median single-exposure broad-band aperture S/N)'))
        bins = astar.make_bins(sr, nb, fit_channels, overrides, excl)
        summary['bins_per_channel'] = brow
    scat = []
    for b in reg.bins:
        d = np.array([r['dR'] for r in reg_rows if r['channel'] == b.channel and np.isfinite(r['dR'])])
        scat.append(dict(channel=b.channel, n=d.size, rms=float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan,
                         max=float(d.max()) if d.size else np.nan))
    qa.emit_table(os.path.join(outdir, 'registration_uncorrected.txt'), reg_rows,
                  ['exp', 'channel', 'dE', 'dN', 'dR', 'fwhm', 'snr', 'flags'],
                  dict(dE='%+.3f', dN='%+.3f', dR='%.3f', fwhm='%.2f', snr='%.0f'),
                  title='Stage 2: star position per exposure relative to the median over exposures '
                        '(uncorrected; arcsec)')
    print(qa.text_table(scat, ['channel', 'n', 'rms', 'max'], dict(rms='%.3f', max='%.3f'),
                        title='registration scatter (arcsec)'))
    summary['registration_scatter'] = scat
    imgs, chan_imgs = _broadband_images(sr, bins, grid, pixscale, inst, args, excl)
    rc = 'red' if 'red' in chan_imgs else next(iter(chan_imgs))
    rep.save(qa.fig_registration(reg, imgs, chan_imgs[rc], grid, pixscale, star_radec), 'registration')
    rep.save(qa.fig_star(chan_imgs[rc], grid, choice.candidates, choice.chosen, star_radec), 'star_selection')
    rep.save(qa.fig_star_channels(chan_imgs, grid, choice.candidates, choice.chosen), 'star_channels')
    del imgs

    # ------------------------------------------------------------------ 4 centroids
    cents = astar.measure_star(sr, star_radec, bins, opts, affines, **mkw)
    qa.write_csv(os.path.join(outdir, 'centroids.csv'), cents.rows())
    rep.save(qa.fig_centroids(cents, fatal, ranges, 'sky'), 'centroids_sky')
    if affines:
        rep.save(qa.fig_centroids(cents, fatal, ranges, 'inst'), 'centroids_instrument')
    rep.save(qa.fig_fwhm(cents, fatal, ranges), 'fwhm_snr')
    good = cents.good(fatal)
    flag_rows = [dict(channel=c, bins=int(sum(b.channel == c for b in bins)) * sr.n_exposures,
                      good=int(sum(good[e, j] for e in range(sr.n_exposures)
                                   for j, b in enumerate(bins) if b.channel == c)))
                 for c in ('blue', 'green', 'red') if c in sr.channels]
    print(qa.text_table(flag_rows, ['channel', 'bins', 'good'], title='Stage 4: usable bins'))
    ov = []
    obins = [b for b in bins if b.role == 'overlap']
    for b1 in obins:
        for b2 in obins:
            if b1.channel < b2.channel and abs(b1.lo - b2.lo) < 1e-6:
                j1, j2 = b1.index, b2.index
                for e, eid in enumerate(cents.exp_ids):
                    ov.append(dict(exp=eid, lam=b1.lam_eff, pair=f'{b1.channel}-{b2.channel}',
                                   dE=cents.E[e, j1] - cents.E[e, j2], dN=cents.N[e, j1] - cents.N[e, j2],
                                   sig=np.hypot(np.hypot(cents.sE[e, j1], cents.sE[e, j2]),
                                                np.hypot(cents.sN[e, j1], cents.sN[e, j2]))))
    if ov:
        qa.emit_table(os.path.join(outdir, 'overlap_consistency.txt'), ov,
                      ['exp', 'pair', 'lam', 'dE', 'dN', 'sig'], dict(lam='%.0f', dE='%+.3f', dN='%+.3f', sig='%.3f'),
                      title='same wavelength measured in two channels (should be ~0: tests channel-'
                            'dependent fibre positions)')
    morph = []
    for e, eid in enumerate(cents.exp_ids):
        r = dict(exp=eid)
        for c in ('blue', 'green', 'red'):
            js = [j for j, b in enumerate(bins) if b.channel == c and b.role != 'overlap']
            if js:
                r[f'ell_{c}'] = float(np.nanmedian(cents.ellip[e, js]))
                r[f'morph_{c}'] = int(sum(cents.flags[e, j] & astar.FLAGS['MORPH'] > 0 for j in js))
        morph.append(r)
    qa.emit_table(os.path.join(outdir, 'morphology.txt'), morph, list(morph[0].keys()),
                  {k: '%.2f' for k in morph[0] if k.startswith('ell')},
                  title='star morphology per channel (a split blue image => fibre-ID / trace problem)')

    # ------------------------------------------------------------------ 5 fit
    lam = cents.lam
    roles = np.array([b.role for b in bins])
    fit_role, hold_role = roles == 'fit', roles == 'holdout'
    red_fit = fit_role & np.array([b.channel == 'red' for b in bins])
    lam_ref = float(args.lam_ref or (np.mean(lam[red_fit]) if red_fit.any() else np.mean(lam[fit_role])))
    fits1, fits2, polys = [], [], []
    offs = [np.c_[cents.E[e], cents.N[e]] for e in range(sr.n_exposures)]
    errs = [np.c_[cents.sE[e], cents.sN[e]] for e in range(sr.n_exposures)]
    masks = [fit_role & good[e] for e in range(sr.n_exposures)]
    for e in range(sr.n_exposures):
        off, err, m = offs[e], errs[e], masks[e]
        kw = dict(err_floor=args.err_floor, clip=args.clip, exp_id=cents.exp_ids[e], index=e)
        fits1.append(dar.fit_exposure(lam, off, err, m, infos[e], lam_ref, order=1, **kw))
        fits2.append(dar.fit_exposure(lam, off, err, m, infos[e], lam_ref, order=2, **kw))
        polys.append(dar.fit_poly_lambda(lam, off, err, m, args.poly_deg, lam_ref, args.err_floor))
    fitsS, beta, sbeta, bedge = dar.fit_shared_beta(lam, offs, errs, masks, infos, lam_ref, cents.exp_ids,
                                                    args.err_floor, args.clip)
    _, dbic, _ = dar.select_order(fits1, fits2)
    dbic_s = dar.shared_vs_order1(fits1, fitsS)
    curve = args.order if args.order != 'auto' else ('shared' if np.isfinite(dbic_s) and dbic_s < -10 else '1')
    order = 1 if curve == '1' else 2
    logger.info('curvature: %s (summed BIC - BIC(order 1): independent order 2 %.1f, shared beta %.1f; '
                'beta = %+.4f +- %.4f)', curve, dbic, dbic_s, beta, sbeta)
    if bedge:
        logger.warning('shared beta hit its search bound (%+.4f): the curvature is not well described by '
                       'dn^2', beta)
    if curve == '2':
        logger.warning('order 2: the dn^2 term is poorly constrained by green+red alone and can '
                       'extrapolate badly into the blue -- check blue_relative.txt (or use --order shared)')
    fits = list({'1': fits1, '2': fits2, 'shared': fitsS}[curve])
    models = dar.dispersion_models(fits, geoms)
    fb_model = next((m for m in models if m.name in ('gain', 'gain+const', 'const', 'mirror')), None)
    for e, f in enumerate(fits):
        if f.status == 'failed':
            ref_off = (np.array([cents.ref['E'][e], cents.ref['N'][e]]) if cents.ref['ok'][e] else None)
            est = (dar.neighbour_c1(e, fits, geoms, infos) if args.model_c1 == 'neighbour'
                   else dar.pooled_c1(e, fits, geoms, fb_model))
            if est is None and args.model_c1 == 'neighbour':
                est = dar.pooled_c1(e, fits, geoms, fb_model)
            fits[e] = dar.fallback_fit(e, f.exp_id, infos[e], lam_ref, est, ref_off, cents.ref['lam'][e], order,
                                       beta=beta if curve == 'shared' else None, ref_var=args.err_floor ** 2)
            logger.warning('%s: own fit failed (%s) -> %s: %s', f.exp_id, f.note, fits[e].status, fits[e].note)
            if est is not None and np.isfinite(est.chi2r) and est.chi2r > args.model_max_chi2r:
                fits[e].gate_failed = True
                logger.warning('%s: the c1 estimator (%s) has reduced chi2 %.1f > %.1f on the measured '
                               'exposures -- the exposure is flagged; %s cube(s) without it will be written '
                               'unless --model-exposures exclude', f.exp_id, est.source,
                               est.chi2r, args.model_max_chi2r, '/'.join(sorted(drop_chans)))
            elif est is not None and not np.isfinite(est.chi2r):
                logger.warning('%s: the c1 estimator (%s) could not be tested (too few measured exposures)',
                               f.exp_id, est.source)
    hold = []
    for e in range(sr.n_exposures):
        off = np.c_[cents.E[e], cents.N[e]]
        err = np.c_[cents.sE[e], cents.sN[e]]
        hm = hold_role & good[e]
        for lab, ff in (('order1', fits1[e]), ('order2', fits2[e]), ('shared', fitsS[e])):
            hold.append(dict(exp=cents.exp_ids[e], model=lab, **dar.holdout_stats(ff, lam, off, err, hm, args.err_floor)))
        if polys[e] is not None:
            r = off[hm] - polys[e].offset(lam[hm])
            hold.append(dict(exp=cents.exp_ids[e], model=f'poly{args.poly_deg}', n=int(hm.sum()),
                             bias_E=float(np.mean(r[:, 0])) if r.size else np.nan,
                             bias_N=float(np.mean(r[:, 1])) if r.size else np.nan,
                             rms_E=float(np.sqrt(np.mean(r[:, 0] ** 2))) if r.size else np.nan,
                             rms_N=float(np.sqrt(np.mean(r[:, 1] ** 2))) if r.size else np.nan))
    qa.write_csv(os.path.join(outdir, 'blue_holdout.csv'), hold)
    qa.emit_table(os.path.join(outdir, 'blue_holdout.txt'), hold,
                  ['exp', 'model', 'n', 'bias_E', 'bias_N', 'rms_E', 'rms_N', 'chi2n_E', 'chi2n_N'],
                  {k: '%+.3f' if 'bias' in k else '%.3f' for k in
                   ('bias_E', 'bias_N', 'rms_E', 'rms_N', 'chi2n_E', 'chi2n_N')},
                  title='Stage 5: held-out (blue) residuals of each model -- the extrapolation test')
    h1 = [h for h in hold if h['model'] == {'1': 'order1', '2': 'order2', 'shared': 'shared'}[curve] and h.get('n', 0)]
    hold_by_exp = {h['exp']: h for h in h1}
    rel_rows, rel_common = _blue_relative(fits, reg, cents, fatal, fit_channels, args.err_floor)
    rel_sum = _relative_summary(rel_rows, rel_common)
    if rel_rows:
        qa.write_csv(os.path.join(outdir, 'blue_relative.csv'), rel_rows)
        qa.emit_table(os.path.join(outdir, 'blue_relative.txt'), rel_rows,
                      ['exp', 'status', 'source', 'lam', 'meas_E', 'model_E', 'resid_E', 'meas_N', 'model_N',
                       'resid_N', 'chi2_E', 'chi2_N'],
                      {**{k: '%+.3f' for k in ('meas_E', 'model_E', 'resid_E', 'meas_N', 'model_N', 'resid_N')},
                       'lam': '%.0f', 'chi2_E': '%.2f', 'chi2_N': '%.2f'},
                      title='Stage 5: held-out channel test on RELATIVE positions (each minus the median over '
                            'exposures; a common offset cancels) -- decides whether the blue stacks sharply')
        with open(os.path.join(outdir, 'blue_relative.txt'), 'a') as fh:
            fh.write('\n' + rel_sum['text'] + '\n')
        print(rel_sum['text'])
        broad = [r for r in rel_rows if r['source'] == 'broad']
        bad = [r for r in broad if max(r['chi2_E'], r['chi2_N']) > 4]
        if broad and len(bad) > len(broad) / 2:
            logger.warning('relative blue test chi2 > 4 in %d/%d exposures: the model does not predict how the '
                           'blue moves between exposures (the blue will not stack sharply)', len(bad), len(broad))
    else:
        logger.warning('relative blue test: fewer than 3 exposures with a good held-out measurement')

    coef_rows = []
    for f, g in zip(fits, geoms):
        r = dict(exp=f.exp_id, status=f.status, order=f.order, nfit=f.nfit, lam_ref=f.lam_ref,
                 chi2r_E=f.chi2r[0], chi2r_N=f.chi2r[1], bic=f.bic, c1_source=f.c1_source,
                 chi2r_model=f.chi2r_model, gate='FAIL' if f.gate_failed else '', note=f.note)
        for k in range(f.coef.shape[1]):
            for a, nm in enumerate('EN'):
                r[f'c{k}_{nm}'] = f.coef[a, k]
                r[f'sig_c{k}_{nm}'] = float(np.sqrt(f.cov[a, k, k])) if np.isfinite(f.cov[a, k, k]) else np.nan
        coef_rows.append(r)
    qa.write_csv(os.path.join(outdir, 'adc_coeffs.csv'), coef_rows)
    qa.write_json(os.path.join(outdir, 'adc_coeffs.json'), coef_rows)
    ccols = ['exp', 'status', 'nfit'] + [f'{p}c{k}_{a}' for k in range(order + 1) for a in 'EN' for p in ('', 'sig_')] \
        + ['chi2r_E', 'chi2r_N', 'c1_source', 'chi2r_model', 'gate']
    qa.emit_table(os.path.join(outdir, 'adc_coeffs.txt'), coef_rows, ccols,
                  {**{c: '%+.4f' for c in ccols if c[0] == 'c' and c[1].isdigit()},
                   **{c: '%.4f' for c in ccols if c.startswith('sig')}, 'chi2r_E': '%.2f', 'chi2r_N': '%.2f',
                   'chi2r_model': '%.1f'},
                  title=f'Stage 5: coefficients, offset = c0 + c1*dn [+ c2*dn^2], dn = 1e6*[n(lam)-n({lam_ref:.0f})] '
                        '(arcsec; c0 is the star position at lam_ref relative to the nominal star position)')
    bin_rows = []
    for f in fits:
        e = f.index
        mod = f.offset(lam) if f.status != 'failed' else np.full((lam.size, 2), np.nan)
        for j, b in enumerate(bins):
            bin_rows.append(dict(exp=f.exp_id, channel=b.channel, role=b.role, lam=b.lam_eff,
                                 used=bool(f.used[j]) if f.used is not None else False,
                                 clipped=bool(f.clipped[j]) if f.clipped is not None else False,
                                 flags=astar.flag_names(cents.flags[e, j]),
                                 meas_E=cents.E[e, j], meas_N=cents.N[e, j], sig_E=cents.sE[e, j],
                                 sig_N=cents.sN[e, j], model_E=mod[j, 0], model_N=mod[j, 1],
                                 resid_E=cents.E[e, j] - mod[j, 0], resid_N=cents.N[e, j] - mod[j, 1]))
    qa.write_csv(os.path.join(outdir, 'adc_shift_bins.csv'), bin_rows)

    chrom = dar.chromatic_summary(fits, geoms)
    qa.write_csv(os.path.join(outdir, 'chromatic_summary.csv'), chrom)
    qa.emit_table(os.path.join(outdir, 'chromatic_summary.txt'), chrom,
                  ['exp', 'status', 'airmass', 'q', 'c1_E', 'c1_N', 'p_E', 'p_N', 'amp_ratio', 'along_frac',
                   'dir_resid_deg', 'meas_b_minus_r', 'pred_b_minus_r'],
                  dict(airmass='%.3f', q='%+.1f', c1_E='%+.4f', c1_N='%+.4f', p_E='%+.4f', p_N='%+.4f',
                       amp_ratio='%.2f', along_frac='%+.2f', dir_resid_deg='%+.0f', meas_b_minus_r='%.3f',
                       pred_b_minus_r='%.3f'),
                  title='Stage 5: measured dispersion (c1) vs the no-ADC atmosphere (p); amp_ratio = residual '
                        'fraction; b_minus_r = |offset(4000 A) - offset(8000 A)| in arcsec')
    verdict = dar.orientation_verdict(models)
    mrows = [dict(model=m.name, chi2=m.chi2, npar=m.npar, bic=m.bic,
                  params=', '.join(f'{k}={complex(v).real:+.3f}{complex(v).imag:+.3f}i' for k, v in m.params.items()))
             for m in models]
    mtxt = qa.text_table(mrows, ['model', 'chi2', 'npar', 'bic', 'params'], dict(chi2='%.1f', bic='%.1f'),
                         title='pooled dispersion models (complex c1 = E + iN; p = no-ADC prediction)') if mrows else ''
    print('\n' + mtxt + '\n' + verdict)
    btxt = (f'curvature: {curve}; shared beta = {beta:+.4f} +- {sbeta:.4f} (c2 = beta c1); summed '
            f'BIC - BIC(order 1): shared {dbic_s:.1f}, independent order 2 {dbic:.1f}')
    print(btxt)
    with open(os.path.join(outdir, 'dispersion_models.txt'), 'w') as fh:
        fh.write(mtxt + '\n\n' + verdict + '\n\n' + btxt + '\n')

    corr = dar.build_correction(fits, lam_ref, mode=args.anchor)
    lam_grid = np.arange(3500.0, 10000.0 + 1, args.shift_step)
    grid_rows = dar.shift_grid_table(fits, corr, lam_grid)
    qa.write_csv(os.path.join(outdir, 'adc_shift_grid.csv'), grid_rows)
    _print_shift_grid(outdir, fits, corr, lam_grid, ranges)

    lam_plot = np.linspace(3500, 9800, 300)
    for e, f in enumerate(fits):
        rep.save(qa.fig_fit_exposure(e, cents, f, polys[e], infos[e], geoms[e], fatal, ranges,
                                     hold_by_exp.get(f.exp_id), lam_plot), f'fit_{qa.short_id(f.exp_id)}')
    rep.save(qa.fig_fit_summary(fits, geoms, infos, chrom, lam_plot, ranges), 'fit_summary')
    if rel_rows:
        rep.save(qa.fig_blue_relative(rel_rows, rel_common, [f.exp_id for f in fits]), 'blue_relative')
    rep.text('Dispersion direction / orientation test', mtxt + '\n\n' + verdict, 'orientation_test')
    summary.update(lam_ref=lam_ref, order=order, curvature=curve, beta=beta, sigma_beta=sbeta,
                   delta_bic_shared=dbic_s, blue_relative=rel_sum['stats'], delta_bic_order2=dbic,
                   fits=coef_rows, chromatic=chrom, dispersion_models=mrows, verdict=verdict,
                   anchor=dict(mode=args.anchor, E=None if corr is None else corr.anchor[0],
                               N=None if corr is None else corr.anchor[1], lam=lam_ref))
    if args.std_rss or args.sensfunc:
        summary['standard'] = _standard_stage(args, inp, opts, excl, fatal, fit_channels, rep, outdir, chrom)

    if corr is None:
        logger.error('no exposure has a usable fit or reference position: no correction can be made')
        _finish(rep, outdir, summary)
        return 3

    # exposures without their own fit: which ADC cubes leave them out
    est_exps = {f.index for f in fits if f.status in ('model', 'fallback')}
    gated = {f.index for f in fits if f.gate_failed}
    mode = args.model_exposures
    adc_drop = {c: set(est_exps) if (mode in ('exclude', 'exclude-blue') and c in drop_chans) else set()
                for c in ('blue', 'green', 'red')}
    extra = set(est_exps) if mode == 'both' else (gated if mode == 'apply' else set())
    if len(extra) >= sr.n_exposures:
        logger.warning('every exposure would be dropped: no adcdrop cubes')
        extra = set()
    for c, d in adc_drop.items():
        if d and len(d) >= sr.n_exposures:
            logger.warning('--model-exposures %s would leave no exposure in the %s ADC cube: keeping them', mode, c)
            adc_drop[c] = set()
    summary['model_exposures'] = dict(mode=mode, drop_channels=sorted(drop_chans), estimated=[fits[e].exp_id for e in sorted(est_exps)],
                                      gate_failed=[fits[e].exp_id for e in sorted(gated)],
                                      adc_dropped={c: [fits[e].exp_id for e in sorted(d)] for c, d in adc_drop.items()},
                                      adcdrop_dropped=[fits[e].exp_id for e in sorted(extra)])
    for c, d in adc_drop.items():
        if d and c in stack_chans:
            logger.info('%s ADC cube: leaving out %s (--model-exposures %s)', c,
                        ', '.join(fits[e].exp_id for e in sorted(d)), mode)
    if extra and not (args.measure_only or args.no_cubes):
        for c in stack_chans:
            if c in drop_chans:
                logger.info('also writing %s without %s', os.path.basename(cube_paths[('adcdrop', c)]),
                            ', '.join(fits[e].exp_id for e in sorted(extra)))

    # ------------------------------------------------------------------ 7a fibre-space validation
    after = astar.measure_star(sr, star_radec, bins, opts, shifts=corr, **mkw)
    good_a = after.good(fatal)
    res_rows = []
    for e, eid in enumerate(after.exp_ids):
        for j, b in enumerate(bins):
            res_rows.append(dict(exp=eid, channel=b.channel, role=b.role, lam=b.lam_eff,
                                 dE=after.E[e, j] - corr.anchor[0], dN=after.N[e, j] - corr.anchor[1],
                                 good=bool(good_a[e, j])))
    qa.write_csv(os.path.join(outdir, 'residuals_after.csv'), res_rows)
    rsum = []
    for c in ('blue', 'green', 'red'):
        for when, cc, gg in (('before', cents, good), ('after', after, good_a)):
            js = [j for j, b in enumerate(bins) if b.channel == c and b.role != 'overlap']
            if not js:
                continue
            dE = cc.E[:, js] - (corr.anchor[0] if when == 'after' else np.nanmedian(cc.E[:, js][gg[:, js]]))
            dN = cc.N[:, js] - (corr.anchor[1] if when == 'after' else np.nanmedian(cc.N[:, js][gg[:, js]]))
            m = gg[:, js].copy()
            if when == 'after':
                m[sorted(adc_drop[c]), :] = False
            rsum.append(dict(channel=c, when=when, n=int(m.sum()),
                             rms_E=float(np.sqrt(np.nanmean(dE[m] ** 2))) if m.any() else np.nan,
                             rms_N=float(np.sqrt(np.nanmean(dN[m] ** 2))) if m.any() else np.nan,
                             max_R=float(np.nanmax(np.hypot(dE[m], dN[m]))) if m.any() else np.nan))
    qa.emit_table(os.path.join(outdir, 'residuals_after.txt'), rsum,
                  ['channel', 'when', 'n', 'rms_E', 'rms_N', 'max_R'], dict(rms_E='%.3f', rms_N='%.3f', max_R='%.3f'),
                  title='Stage 7: fibre-space star position scatter across exposures and bins, before '
                        '(about the median) and after correction (about the anchor); blue = held-out test')
    rep.save(qa.fig_resid_after(after, corr.anchor, fatal, ranges), 'residuals_after')
    summary['residuals'] = rsum

    if args.measure_only or args.no_cubes:
        _finish(rep, outdir, summary)
        return 0

    # ------------------------------------------------------------------ 6 stack + 7b cube validation
    def regonly(e, w):
        f = fits[e]
        if f.status in ('failed', 'model'):
            return np.zeros((np.size(w), 2))
        return np.tile(corr.anchor - f.offset([lam_ref])[0], (np.size(w), 1))
    var_fns = {'noadc': None, 'adc': corr}
    if args.also_regonly:
        var_fns['regonly'] = regonly
    kernel = args.kernel or 'gaussian'
    kfwhm = args.fwhm or (cs or {}).get('kernel_fwhm', 0.9)
    weighting = args.weight or (cs or {}).get('weighting', 'ivar')
    units = args.units or (cs or {}).get('units', 'sb')
    model_rows = [dict(exp=f.exp_id, status=f.status, order=f.order, lam_ref=f.lam_ref,
                       velcorr=f.info.velcorr, P=f.info.pressure, T=f.info.temperature, RH=f.info.humidity,
                       coef_E=list(np.nan_to_num(f.coef[0], nan=0.0)), coef_N=list(np.nan_to_num(f.coef[1], nan=0.0)))
                  for f in fits]
    metrics = {v: [] for v in var_fns}
    stamps, slab_plot, slab_rows, fidelity, structure, written, cov_rows = [], {}, [], {}, {}, {}, []
    exp_ids = [m.exposure_id for m in sr.exposures]
    _shift_guard(outdir, fits, corr, sr, stack_chans, fit_channels, lam_ref, args, cs, ast)
    for c in stack_chans:
        wl, dw = ast.wave_grid(sr.channels[c], args.dwave, (cs or {}).get('waves', {}).get(c))
        logger.info('stacking %s: %d planes, %.0f-%.0f A', c, wl.size, wl[0], wl[-1])
        vf, drop = dict(var_fns), {'adc': adc_drop[c]}
        if c in drop_chans and extra:
            vf['adcdrop'], drop['adcdrop'] = corr, extra
        res = ast.stack_channel(sr, c, grid, wl, dw, vf, kernel=kernel, kernel_fwhm=kfwhm,
                                pixscale=pixscale, weighting=weighting, units=units,
                                min_coverage=args.min_coverage, slab_tol=args.slab_tol, drop=drop)
        cb = [(b.lo, b.hi, b.lam_eff, b.channel) for b in val_bins if b.channel == c and b.role != 'overlap']
        for v, r in res.items():
            metrics.setdefault(v, [])
            metrics[v] += qa.cube_star_metrics(r.cube, *star_radec, cb)
            cov_rows.append(dict(variant=v, channel=c, **ast.coverage_summary(r)))
        half = int(round(5.0 / pixscale))
        for (lo, hi, le, _) in (cb[0], cb[-1]) if cb else ():
            sel = (wl >= lo) & (wl < hi)
            with np.errstate(all='ignore'):
                b_img = np.nanmedian(res['noadc'].cube.data[sel], axis=0)
                a_img = np.nanmedian(res['adc'].cube.data[sel], axis=0)
            ar, ad = aio.tangent_inverse(corr.anchor[0], corr.anchor[1], *star_radec)
            stamps.append((f'{c} {le:.0f} A', qa._stamp(b_img, grid[0], ar, ad, half),
                           qa._stamp(a_img, grid[0], ar, ad, half)))
        corr_arr = np.stack([corr(e, wl) for e in range(sr.n_exposures)])
        slab_plot[c] = (wl, corr_arr, res['adc'].slabs)
        for s in res['adc'].slabs:
            for e, eid in enumerate(exp_ids):
                slab_rows.append(dict(channel=c, i0=s.i0, i1=s.i1, wave_lo=wl[s.i0], wave_hi=wl[s.i1 - 1],
                                      exp=eid, shift_E=s.shift[e, 0], shift_N=s.shift[e, 1]))
        if inp.cube_path:
            m = re.search(r'_cube_(blue|green|red)', os.path.basename(inp.cube_path))
            delivered = inp.cube_path.replace(m.group(0), f'_cube_{c}') if m else inp.cube_path
            if os.path.exists(delivered):
                fidelity[c] = ast.compare_to_delivered(res['noadc'].cube, delivered)
        for v, r in res.items():
            dset = drop.get(v, set())
            cards = dict(ADCCORR=(v in ('adc', 'adcdrop'), 'ADC/DAR wavelength-dependent shifts applied'),
                         ADCREG=(v in ('adc', 'adcdrop', 'regonly') and args.anchor == 'star',
                                 'exposures registered on the star'),
                         ADCBASIS=('refractivity', 'offset = sum c_i dn^i (see ADCMODEL)'),
                         ADCREFR=('Filippenko82', 'air refractivity with P/T/RH per exposure'),
                         ADCORDER=(order, 'order in dn'), ADCCURV=(curve, 'curvature: 1, 2 (per exposure), shared'),
                         ADCBETA=(float(beta) if curve == 'shared' else 0.0, 'shared curvature c2 = beta*c1'),
                         ADCLREF=(lam_ref, '[A] reference wavelength'),
                         ADCFITCH=(','.join(fit_channels), 'channels fitted (others held out)'),
                         ADCANCH=(args.anchor, 'anchor mode'),
                         ADCANCE=(float(corr.anchor[0]), '[arcsec] anchor E of reference star'),
                         ADCANCN=(float(corr.anchor[1]), '[arcsec] anchor N of reference star'),
                         ADCSTRA=(float(star_radec[0]), '[deg] reference star RA'),
                         ADCSTDE=(float(star_radec[1]), '[deg] reference star Dec'),
                         ADCSLTOL=(args.slab_tol, '[arcsec] slab tolerance (peak-to-peak)'),
                         ADCNSLAB=(len(r.slabs), 'wavelength slabs'),
                         ADCVERS=(__version__, 'adcfit version'))
            for i, f in enumerate(fits, 1):
                st_i = 'dropped' if (i - 1) in dset else f.status
                cards[f'ADCE{i}'] = (f'{f.exp_id[-24:]} {st_i}', 'per-exposure ADC fit status')
            mrows = [dict(r_, status='dropped') if i in dset else r_ for i, r_ in enumerate(model_rows)]
            ref = _reference_cube(inp, c, sr.field)
            covfrac = None
            if ref:
                covfrac = fits_getval(ref, 'COVFRAC')
            ast.write_cube(r, cube_paths[(v, c)], exp_ids, cards, mrows, covfrac=covfrac)
            written[f'{v}_{c}'] = cube_paths[(v, c)]
            ast.write_nexpslab(r, cube_paths[(v, c)].replace('.fits', '_nexpslab.fits'))
            problems = ast.check_structure(cube_paths[(v, c)], ref)
            structure[f'{v}_{c}'] = dict(reference=ref, ok=not problems, problems=problems)
            for msg in problems:
                logger.warning('%s: structure differs from the pipeline cube: %s',
                               os.path.basename(cube_paths[(v, c)]), msg)
            logger.info('wrote %s', cube_paths[(v, c)])
        del res
    qa.write_csv(os.path.join(cube_outdir, 'slabs.csv'), slab_rows)
    ccols = ['variant', 'channel', 'nslab', 'nexp_max', 'covered', 'full_all', 'full_any', 'varying',
             'worst_slab', 'worst_lo', 'worst_hi', 'worst_full']
    ctitle = ('Stage 6: per-slab exposure coverage (from *_nexpslab.fits): spaxels covered at all, at full depth '
              '(NEXP = nexp_max) in EVERY slab / in at least one, spaxels whose NEXP changes with wavelength, and '
              'the slab with the fewest full-depth spaxels')
    qa.emit_table(os.path.join(outdir, 'coverage_slabs.txt'), cov_rows, ccols,
                  dict(worst_lo='%.0f', worst_hi='%.0f'), title=ctitle)
    rep.text('Coverage per wavelength slab', qa.text_table(cov_rows, ccols, dict(worst_lo='%.0f', worst_hi='%.0f'),
                                                           title=ctitle), 'coverage_slabs')
    summary['coverage_slabs'] = cov_rows
    rep.save(qa.fig_slabs(slab_plot, exp_ids), 'slabs')
    rep.save(qa.fig_validation(metrics, None, corr.anchor, ranges), 'validation_cubes')
    if stamps:
        rep.save(qa.fig_stamps(stamps, pixscale), 'stamps')
    vrows = []
    for v, rows in metrics.items():
        for r in rows:
            vrows.append(dict(variant=v, **r, dE=r['E'] - corr.anchor[0], dN=r['N'] - corr.anchor[1]))
    qa.write_csv(os.path.join(outdir, 'validation.csv'), vrows)
    vsum = []
    for v in metrics:
        for c in stack_chans:
            rr = [r for r in vrows if r['variant'] == v and r['channel'] == c]
            if not rr:
                continue
            E = np.array([r['E'] for r in rr]); N = np.array([r['N'] for r in rr])
            vsum.append(dict(variant=v, channel=c, ptp_E=float(np.nanmax(E) - np.nanmin(E)),
                             ptp_N=float(np.nanmax(N) - np.nanmin(N)),
                             fwhm=float(np.nanmedian([r['fwhm'] for r in rr])),
                             conc=float(np.nanmedian([r['conc'] for r in rr]))))
    allc = {}
    for v in metrics:
        rr = [r for r in vrows if r['variant'] == v]
        E = np.array([r['E'] for r in rr]); N = np.array([r['N'] for r in rr])
        allc[v] = dict(variant=v, channel='all', ptp_E=float(np.nanmax(E) - np.nanmin(E)),
                       ptp_N=float(np.nanmax(N) - np.nanmin(N)),
                       fwhm=float(np.nanmedian([r['fwhm'] for r in rr])),
                       conc=float(np.nanmedian([r['conc'] for r in rr])))
    vsum += list(allc.values())
    qa.emit_table(os.path.join(outdir, 'validation.txt'), vsum,
                  ['variant', 'channel', 'ptp_E', 'ptp_N', 'fwhm', 'conc'],
                  dict(ptp_E='%.3f', ptp_N='%.3f', fwhm='%.2f', conc='%.3f'),
                  title='Stage 7: stacked-cube star centroid peak-to-peak drift over wavelength (arcsec), '
                        'median FWHM and r<1"/r<3" concentration')
    if fidelity:
        frows = [dict(channel=c, **d) for c, d in fidelity.items()]
        qa.emit_table(os.path.join(outdir, 'rebuild_fidelity.txt'), frows,
                      ['channel', 'ok', 'median', 'p99', 'frac_gt1', 'note'],
                      dict(median='%.3f', p99='%.2f', frac_gt1='%.4f'),
                      title='uncorrected rebuild vs the delivered cube: |diff|/sigma (should be ~0 if the '
                            'combine settings match)')
    srows = [dict(cube=k, ok=d['ok'], reference=os.path.basename(d['reference'] or '-'),
                  problems='; '.join(d['problems']) or '-') for k, d in structure.items()]
    qa.emit_table(os.path.join(outdir, 'cube_structure.txt'), srows, ['cube', 'ok', 'reference', 'problems'],
                  title='output cube layout vs the pipeline cube (HDUs 0-4 identical; only scalar tables after)')
    summary.update(validation=vsum, rebuild_fidelity=fidelity, cube_structure=structure,
                   cubes=written)
    _finish(rep, outdir, summary)
    return 0


def _measure_opts(args, excl):
    from adcfit.star import MeasureOpts
    return MeasureOpts(profile=args.psf, beta=args.moffat_beta, sigma_mode=args.sigma, bg=args.bg,
                       fit_radius=args.fit_radius, search_radius=args.search_radius,
                       min_ap_snr=args.min_ap_snr, max_cen_err=args.max_cen_err, exclusions=excl,
                       destripe=not args.no_destripe)


def _fatal(args):
    from adcfit.star import FLAGS
    fatal = 0
    for name in args.reject_flags.split(','):
        if name.strip():
            fatal |= FLAGS[name.strip().upper()]
    return fatal


def run_standard_only(args) -> int:
    """--std-rss without science inputs: the standard's checks, track, sensfunc diagnostic and
    (with --std-correct-rss) the ADC-corrected RSS copies."""
    from types import SimpleNamespace
    from llamas_pyjamas.Combine.superRSS import combined_dir
    from adcfit import qa, star as astar
    stem = os.path.basename(args.std_rss).split('_RSS_')[0]
    outdir = os.path.abspath(args.outdir or os.path.join(combined_dir([args.std_rss]), f'adcfit_std_{stem}'))
    os.makedirs(outdir, exist_ok=True)
    logger.info('QA output: %s', outdir)
    rep = qa.QAReport(outdir, pdf=not args.no_pdf)
    excl = astar.default_exclusions(args.exclude)
    summary = dict(mode='standard-only', outdir=outdir, args=vars(args))
    sm = _standard_stage(args, SimpleNamespace(rss_paths=[]), _measure_opts(args, excl), excl, _fatal(args),
                         tuple(c.strip() for c in args.fit_channels.split(',')), rep, outdir, [])
    summary['standard'] = sm
    _finish(rep, outdir, summary)
    return 1 if sm.get('error') else 0


def _correct_standard_rss(args, res, opts, fatal, rep, outdir):
    """Write the ADC-corrected standard RSS copies and check them: same structure as the input,
    star track flat, fixed-aperture captured fraction flat."""
    from adcfit import __version__, io as aio, qa, rsscorr, star as astar
    if res.track is None:
        raise ValueError('standard: no usable centroid track, so the RSS cannot be corrected')
    sib = res.summary['standard_rss']
    out = rsscorr.correct_rss(sib, res.sr, res.track, res.lam_ref, res.star, args.std_correct_rss,
                              slab_tol=args.std_slab_tol, overwrite=args.overwrite, version=__version__)
    struct = {c: rsscorr.compare_rss(sib[c], p) for c, p in out['paths'].items()}
    srows = [dict(channel=c, file=out['paths'][c], identical_structure=not v, problems='; '.join(v) or '-',
                  max_shift=out['info'][c]['max_shift'], slabs=out['info'][c]['nslab'],
                  pixels_kept=out['info'][c]['n_kept_pixels']) for c, v in struct.items()]
    for c, v in struct.items():
        for msg in v:
            logger.error('corrected %s RSS differs from the input structure: %s', c, msg)
    # re-measure the star on the corrected files
    green = out['paths'].get('green') or next(iter(out['paths'].values()))
    sr2 = aio.load_field([green], plane='auto')
    try:
        inst2 = aio.instrument_xy(sr2)
    except Exception:                                     # noqa: BLE001
        inst2 = {}
    ref = tuple(res.track([res.lam_ref])[0])
    c2 = astar.measure_star(sr2, res.star, res.cents.bins, opts, inst=inst2 or None, guesses={0: ref})
    trows = []
    for c in ('blue', 'green', 'red'):
        js = [j for j, b in enumerate(res.cents.bins) if b.channel == c and b.role != 'overlap']
        for when, cc in (('before', res.cents), ('after', c2)):
            g = [j for j in js if cc.good(fatal)[0, j]]
            if g:
                trows.append(dict(channel=c, when=when, n=len(g), ptp_E=float(np.ptp(cc.E[0, g])),
                                  ptp_N=float(np.ptp(cc.N[0, g])), mean_E=float(np.mean(cc.E[0, g])),
                                  mean_N=float(np.mean(cc.N[0, g]))))
    allb = {w: [j for j, b in enumerate(res.cents.bins) if b.role != 'overlap' and cc.good(fatal)[0, j]]
            for w, cc in (('before', res.cents), ('after', c2))}
    for w, cc in (('before', res.cents), ('after', c2)):
        g = allb[w]
        if g:
            trows.append(dict(channel='all', when=w, n=len(g), ptp_E=float(np.ptp(cc.E[0, g])),
                              ptp_N=float(np.ptp(cc.N[0, g])), mean_E=float(np.mean(cc.E[0, g])),
                              mean_N=float(np.mean(cc.N[0, g]))))
    # fixed-aperture captured fraction, before vs after
    radii = (1.5, 2.0, 3.0)
    cap = []
    for c, p in out['paths'].items():
        st = res.sr.channels[c]
        E, N = aio.tangent_offsets(st.ra, st.dec, *res.star)
        for when, path, ctot in (('before', sib[c], lambda l: res.track([l])[0]), ('after', p, lambda l: ref)):
            for r in rsscorr.aperture_fraction(path, E, N, ref, radii, 6.0, ctot):
                cap.append(dict(channel=c, when=when, **r))
    crow = []
    for c in out['paths']:
        for when in ('before', 'after'):
            rr = [r for r in cap if r['channel'] == c and r['when'] == when]
            d = dict(channel=c, when=when)
            for R in radii:
                v = np.array([r[f'f_R{R:g}'] for r in rr])
                d[f'R{R:g}_min'], d[f'R{R:g}_max'] = float(np.nanmin(v)), float(np.nanmax(v))
            crow.append(d)
    qa.emit_table(os.path.join(outdir, 'std_rss_correction.txt'), srows,
                  ['channel', 'identical_structure', 'max_shift', 'slabs', 'pixels_kept', 'problems', 'file'],
                  dict(max_shift='%.2f'),
                  title='ADC-corrected standard RSS (same names/HDUs/dtypes as the input; only SKYSUB/ERROR '
                        '[+FLAM/FLAM_ERR] data changed)')
    qa.emit_table(os.path.join(outdir, 'std_rss_track.txt'), trows,
                  ['channel', 'when', 'n', 'ptp_E', 'ptp_N', 'mean_E', 'mean_N'],
                  {k: '%.3f' for k in ('ptp_E', 'ptp_N', 'mean_E', 'mean_N')},
                  title='standard star centroid over wavelength (arcsec), input vs corrected RSS')
    qa.write_csv(os.path.join(outdir, 'std_rss_capture.csv'), cap)
    ccols = ['channel', 'when'] + [f'R{R:g}_{m}' for R in radii for m in ('min', 'max')]
    qa.emit_table(os.path.join(outdir, 'std_rss_capture.txt'), crow, ccols, {k: '%.3f' for k in ccols[2:]},
                  title='fraction of the star in a FIXED picker aperture (R arcsec about its lambda_ref '
                        'position) vs the total within 6"; range over wavelength per channel')
    rep.save(qa.fig_std_rss_corrected(res.cents, c2, fatal, cap, radii), 'std_rss_corrected')
    logger.info('ADC-corrected standard RSS written to %s. Build the new sensitivity function in CubeViewer: '
                'open %s, pick an aperture on the star (a radius of >= 3x FWHM), Build Sensitivity Function, '
                'save; then re-run apply_fluxcal on the science RSS with it.', os.path.abspath(args.std_correct_rss),
                os.path.basename(green))
    return dict(out, structure=struct, track=trows, capture=crow)


def _standard_stage(args, inp, opts, excl, fatal, fit_channels, rep, outdir, sci_chrom):
    """Standard star + sensfunc diagnostic: provenance checks, the standard's chromatic track
    (green+red fit, blue held out), and the fraction of the star the sensfunc's aperture captured."""
    from adcfit import standard as astd, qa
    from adcfit.star import NoStarError, broad_bins
    r_tot = tuple(args.std_rtot)
    try:
        res = astd.run_standard(args.std_rss, args.sensfunc, inp.rss_paths, opts, fit_channels=fit_channels,
                                target_snr=args.std_target_snr, excl=excl, fatal=fatal, r_tot=r_tot,
                                star=_coord(*args.std_star) if args.std_star else None,
                                err_floor=args.err_floor, clip=args.clip)
    except NoStarError as exc:
        logger.error('standard: %s', exc)
        return dict(error=str(exc))
    qa.emit_table(os.path.join(outdir, 'standard_checks.txt'), res.checks, ['check', 'value', 'expected', 'ok', 'note'],
                  title='standard / sensfunc provenance (the science FLAM must have been made with this sensfunc)')
    bad = [c for c in res.checks if not c['ok']]
    for c in bad:
        logger.warning('standard check failed: %s = %s (expected %s) %s', c['check'], c['value'], c['expected'], c['note'])
    sm = res.summary
    if res.cents is not None:
        qa.write_csv(os.path.join(outdir, 'standard_centroids.csv'), res.cents.rows())
        ranges = {b.channel: (b.lo, b.hi) for b in broad_bins(res.cents.bins)}
        lam_plot = np.linspace(3500, 9800, 300)
        rep.save(qa.fig_fit_exposure(0, res.cents, res.fit, res.poly, res.info, res.geom, fatal, ranges,
                                     res.holdout[0], lam_plot), 'std_fit')
        ch = sm['chromatic'][0]
        rows = [dict(source='standard ' + str(sm.get('object')), **{k: ch.get(k) for k in
                ('airmass', 'q', 'c1_E', 'c1_N', 'p_E', 'p_N', 'amp_ratio', 'dir_resid_deg', 'meas_b_minus_r')})]
        rows += [dict(source='science ' + qa.short_id(c['exp']), **{k: c.get(k) for k in
                 ('airmass', 'q', 'c1_E', 'c1_N', 'p_E', 'p_N', 'amp_ratio', 'dir_resid_deg', 'meas_b_minus_r')})
                 for c in sci_chrom]
        txt = qa.text_table(rows, ['source', 'airmass', 'q', 'c1_E', 'c1_N', 'p_E', 'p_N', 'amp_ratio', 'dir_resid_deg',
                                   'meas_b_minus_r'],
                            dict(airmass='%.3f', q='%+.1f', c1_E='%+.4f', c1_N='%+.4f', p_E='%+.4f', p_N='%+.4f',
                                 amp_ratio='%.2f', dir_resid_deg='%+.0f', meas_b_minus_r='%.3f'),
                            title='standard vs science: measured dispersion c1 vs no-ADC prediction p')
        htxt = qa.text_table(res.holdout, ['model', 'n', 'bias_E', 'bias_N', 'rms_E', 'rms_N', 'chi2n_E', 'chi2n_N'],
                             {k: '%+.3f' if 'bias' in k else '%.3f' for k in
                              ('bias_E', 'bias_N', 'rms_E', 'rms_N', 'chi2n_E', 'chi2n_N')},
                             title='standard: blue held-out residuals of the green+red fit (arcsec)')
        with open(os.path.join(outdir, 'standard_fit.txt'), 'w') as fh:
            fh.write(txt + '\n\n' + htxt + '\n')
        print('\n' + txt + '\n\n' + htxt)
        qa.write_json(os.path.join(outdir, 'standard_fit.json'), dict(fit=sm.get('fit'), holdout=res.holdout,
                                                                       chromatic=sm.get('chromatic')))
    if res.capture:
        cols = ['channel', 'lam', 'C_ap'] + [f'f_R{R:g}' for R in r_tot] + \
            (['fmod_track', 'fmod_noADC'] if 'fmod_track' in res.capture[0] else []) + \
            ['S_used_over_true', 'colour_err_rel_ref', 'track_E', 'track_N', 'clean']
        qa.write_csv(os.path.join(outdir, 'sensfunc_diagnostic.csv'), res.capture)
        qa.emit_table(os.path.join(outdir, 'sensfunc_diagnostic.txt'), res.capture, cols,
                      {**{c: '%.3f' for c in cols[3:]}, 'lam': '%.0f', 'C_ap': '%.4g'},
                      title='sensfunc: f = counts in its aperture / total counts of the star (total = fibres within '
                            'R" of the measured track); S_used/S_true = 1/f\n' + str(sm.get('verdict', '')))
        rep.save(qa.fig_sensfunc_diag(res.capture, r_tot, sm['lam_ref'], sm.get('verdict')), 'sensfunc_diag')
    ap = res.aperture
    if ap:
        b = ap.get('best') or {}
        lines = [f"reference fibre (its wavelength grid is the sensfunc's): RSS row {ap.get('ref_row', '-')}",
                 f"result: {ap.get('note', '')}"]
        for cnd in ap.get('candidates', [])[:5]:
            lines.append(f"  R={cnd['R']:.1f} fib  centre offset ({cnd['dx']:+.1f}, {cnd['dy']:+.1f}) fib  nfib={cnd['nfib']:3d}  "
                         f"rms log ratio={cnd['rms']:.3f}  sum/C_ap blue/green/red = "
                         + ' / '.join(f"{cnd.get('ratio_' + c, np.nan):.3f}" for c in ('blue', 'green', 'red')))
        with open(os.path.join(outdir, 'sensfunc_aperture.txt'), 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print('\nsensfunc aperture search\n' + '\n'.join(lines))
        if res.cents is not None and res.EN:
            rep.save(qa.fig_std_aperture(res.EN, b.get('rows'), res.cents, res.good, ap.get('note', '')), 'std_aperture')
    if sm.get('verdict'):
        logger.warning('standard: %s', sm['verdict'])
    if args.std_correct_rss:
        sm['rss_correction'] = _correct_standard_rss(args, res, opts, fatal, rep, outdir)
    return sm


def fits_getval(path, key):
    from astropy.io import fits
    try:
        return fits.getheader(path, 0).get(key)
    except OSError:
        return None


def _reference_cube(inp, channel, field):
    """The delivered pipeline cube for this channel: the input cube's sibling, else
    <combined>/<field>_cube_<channel>.fits when it exists."""
    if inp.cube_path:
        m = re.search(r'_cube_(blue|green|red)', os.path.basename(inp.cube_path))
        p = inp.cube_path.replace(m.group(0), f'_cube_{channel}') if m else inp.cube_path
        return p if os.path.exists(p) else None
    from llamas_pyjamas.Combine.superRSS import combined_dir
    try:
        d = combined_dir(inp.rss_paths)
    except Exception:                                      # noqa: BLE001
        return None
    for name in (f'{field}_cube_{channel}.fits', f'{_safe(field)}_cube_{channel}.fits'):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def _shift_guard(outdir, fits, corr, sr, stack_chans, fit_channels, lam_ref, args, cs, ast):
    """Largest CHROMATIC shift (applied correction minus its value at lam_ref) per exposure and
    channel over the stacked wavelength grid. Warns when an exposure without its own fit exceeds
    --max-extrap-shift or 2x the largest shift of the measured exposures in that channel, or when
    any exposure exceeds --max-extrap-shift in a held-out channel. Nothing is clipped."""
    from adcfit.qa import emit_table
    rows = []
    for c in stack_chans:
        wl, _ = ast.wave_grid(sr.channels[c], args.dwave, (cs or {}).get('waves', {}).get(c))
        amp = {}
        for f in fits:
            if f.status == 'failed':
                continue
            ch = corr(f.index, wl) - corr(f.index, [lam_ref])[0][None, :]
            k = int(np.argmax(np.hypot(ch[:, 0], ch[:, 1])))
            amp[f.index] = (float(np.hypot(*ch[k])), float(wl[k]), float(np.hypot(*corr(f.index, wl)[k])))
        ref = max([amp[f.index][0] for f in fits if f.status == 'ok' and f.index in amp] or [np.nan])
        for f in fits:
            if f.index not in amp:
                continue
            a, lam_at, tot = amp[f.index]
            why = []
            if f.status != 'ok' and a > args.max_extrap_shift:
                why.append(f'> {args.max_extrap_shift:g}"')
            if f.status != 'ok' and np.isfinite(ref) and a > 2 * ref:
                why.append(f'> 2x measured max {ref:.2f}"')
            if f.status == 'ok' and c not in fit_channels and a > args.max_extrap_shift:
                why.append(f'> {args.max_extrap_shift:g}" (held-out channel)')
            rows.append(dict(channel=c, exp=f.exp_id, status=f.status, max_chromatic=a, at_lam=lam_at,
                             total_there=tot, ok_max=ref, warning='; '.join(why)))
            if why:
                logger.warning('%s %s (%s): chromatic shift reaches %.2f" at %.0f A (%s) -- consider '
                               '--model-exposures both (or exclude) --drop %s', f.exp_id, c, f.status, a, lam_at,
                               ', '.join(why), c)
    emit_table(os.path.join(outdir, 'shift_extremes.txt'), rows,
               ['channel', 'exp', 'status', 'max_chromatic', 'at_lam', 'total_there', 'ok_max', 'warning'],
               dict(max_chromatic='%.2f', at_lam='%.0f', total_there='%.2f', ok_max='%.2f'),
               title='largest chromatic shift (correction minus its value at lambda_ref, arcsec) per exposure and '
                     'channel; ok_max = largest among exposures with their own fit')
    return rows


def _blue_relative(fits, reg, cents, fatal, fit_channels, err_floor):
    """Relative held-out test (dar.relative_holdout) on the broad-band registration positions of
    every held-out channel, then on each held-out bin with >= 3 good exposures. Returns (rows,
    common) with ``common`` the broad-band common offset per held-out channel."""
    from adcfit.dar import relative_holdout
    rows, common = [], {}
    for src, cc in (('broad', reg), ('bin', cents)):
        g = cc.good(fatal)
        for j, b in enumerate(cc.bins):
            if b.channel in fit_channels or b.role == 'overlap':
                continue
            rr, cm = relative_holdout(fits, b.lam_eff, cc.E[:, j], cc.N[:, j], cc.sE[:, j], cc.sN[:, j],
                                      g[:, j], err_floor)
            for r in rr:
                r.update(source=src, channel=b.channel)
            rows += rr
            if src == 'broad' and cm is not None:
                common[b.channel] = dict(lam=b.lam_eff, E=float(cm[0]), N=float(cm[1]))
    return rows, common


def _relative_summary(rows, common):
    """One line per (channel, source) of the relative test plus the common offsets."""
    stats, lines = [], []
    for key in sorted({(r['channel'], r['source']) for r in rows}):
        rr = [r for r in rows if (r['channel'], r['source']) == key]
        rE, rN = np.array([r['resid_E'] for r in rr]), np.array([r['resid_N'] for r in rr])
        st = dict(channel=key[0], source=key[1], n=len(rr), nexp=len({r['exp'] for r in rr}),
                  rms_E=float(np.sqrt(np.mean(rE ** 2))), rms_N=float(np.sqrt(np.mean(rN ** 2))),
                  chi2n_E=float(np.mean([r['chi2_E'] for r in rr])),
                  chi2n_N=float(np.mean([r['chi2_N'] for r in rr])), max_R=float(np.max(np.hypot(rE, rN))))
        stats.append(st)
        lines.append(f"relative {key[0]} ({key[1]}): {st['nexp']} exposures, {st['n']} points, rms E "
                     f"{st['rms_E']:.3f}\" N {st['rms_N']:.3f}\", chi2/n {st['chi2n_E']:.2f}, {st['chi2n_N']:.2f}, "
                     f"max {st['max_R']:.3f}\"")
    for c, cm in common.items():
        lines.append(f"common {c} offset (measured - model, median over exposures) at {cm['lam']:.0f} A: "
                     f"E {cm['E']:+.3f}\" N {cm['N']:+.3f}\" (matters only for aligning {c} with the fitted channels)")
    return dict(stats=stats, common=common, text='\n'.join(lines))


def _print_shift_grid(outdir, fits, corr, lam_grid, ranges):
    """Wide tables: the fitted shift (model) and the applied correction vs wavelength, one column
    pair (E, N) per exposure. Printed and written to adc_shift_grid.txt."""
    from adcfit.qa import text_table
    ok = [f for f in fits if f.status != 'failed']
    blocks = []
    for title, fn in (('fitted star offset vs wavelength, relative to lambda_ref (arcsec, E/N)',
                       lambda f, l: f.offset(l) - f.offset([f.lam_ref])[0][None, :]),
                      ('model uncertainty (1 sigma, arcsec, E/N)', lambda f, l: f.offset_err(l)),
                      ('APPLIED correction added to the fibre positions (arcsec, E/N)',
                       lambda f, l: corr(f.index, l))):
        rows = []
        vals = {f.exp_id: fn(f, lam_grid) for f in ok}
        for i, lam in enumerate(lam_grid):
            ch = next((c for c, (lo, hi) in ranges.items() if lo <= lam <= hi), '')
            r = dict(lam=f'{lam:.0f}', ch=ch)
            for k, f in enumerate(ok):
                r[f'E{k + 1}'] = vals[f.exp_id][i, 0]
                r[f'N{k + 1}'] = vals[f.exp_id][i, 1]
            rows.append(r)
        cols = ['lam', 'ch'] + [f'{a}{k + 1}' for k in range(len(ok)) for a in 'EN']
        f = '%.3f' if title.startswith('model uncertainty') else '%+.3f'
        blocks.append(text_table(rows, cols, {c: f for c in cols[2:]}, title=title))
    key = '\n'.join(f'  {k + 1}: {f.exp_id} ({f.status})' for k, f in enumerate(ok))
    txt = ('Stage 5: ADC shifts vs wavelength (fit on green+red; blue extrapolated)\nexposures:\n' + key
           + '\n\n' + '\n\n'.join(blocks))
    with open(os.path.join(outdir, 'adc_shift_grid.txt'), 'w') as fh:
        fh.write(txt + '\n')
    print('\n' + txt)


def _finish(rep, outdir, summary):
    from adcfit import qa
    summary['qa_figures'] = rep.files
    rep.close()
    qa.write_json(os.path.join(outdir, 'adc_summary.json'), summary)
    logger.info('QA report: %s', os.path.join(outdir, 'adc_qa.pdf'))
    logger.info('summary: %s', os.path.join(outdir, 'adc_summary.json'))
    if summary.get('cubes'):
        for k, p in summary['cubes'].items():
            logger.info('cube %-14s %s', k, p)


if __name__ == '__main__':
    sys.exit(main())
