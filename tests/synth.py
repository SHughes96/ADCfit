"""Synthetic LLAMAS field: real fibre-map geometry, dithers, registration errors, and a star whose
centroid follows a known per-exposure  s0 + reg_e + c1_e * dn(lambda)  track."""

import os

import numpy as np
from astropy.table import Table

from llamas_pyjamas.Combine.superRSS import ChannelStack, ExposureMeta, SuperRSS

from adcfit.dar import dn_e6, exposure_geometry
from adcfit.io import ExposureInfo, tangent_inverse

RA0, DEC0 = 30.0, -30.0
CH_RANGES = {'blue': (3200.0, 4880.0), 'green': (4555.0, 7117.0), 'red': (6644.0, 10052.0)}


def lut_table():
    from llamas_pyjamas.config import LUT_DIR
    return Table.read(os.path.join(LUT_DIR, 'LLAMAS_FiberMap_rev04.dat'), format='ascii.fixed_width')


def lut_xy():
    t = lut_table()
    return np.c_[np.asarray(t['xpos'], float), np.asarray(t['ypos'], float)] * 0.75


def _sexa(h):
    h = float(h) % 24.0
    hh = int(h); mm = int((h - hh) * 60); ss = (h - hh - mm / 60) * 3600
    return f'{hh:02d}:{mm:02d}:{ss:04.1f}'


def write_rss(sr, infos, directory, obj='SYN'):
    """Write the synthetic field as LLAMAS-like RSS files (FLAM/FLAM_ERR/MASK/WAVE/FIBERMAP/
    FIBERWCS + the header cards adcfit reads). Returns the green paths."""
    from astropy.io import fits
    t = lut_table()
    xy = lut_xy() / 0.75
    os.makedirs(directory, exist_ok=True)
    out = []
    for e, info in enumerate(infos):
        stem = f'LLAMAS_2026-01-01_00-{e:02d}-00.0'
        for c, st in sr.channels.items():
            rows = st.exposure == e
            h = fits.Header()
            h['OBJECT'] = obj
            h['MJD-OBS'] = info.mjd
            h['SEXPTIME'] = info.exptime
            h['RA'] = info.ra
            h['DEC'] = info.dec
            h['LST'] = _sexa(info.lst_start)
            h['HIERARCH WTH-PRES'] = info.pressure
            h['HIERARCH WTH-TEMP'] = info.temperature
            h['HIERARCH WTH-HUMD'] = info.humidity
            h['VELCORR'] = 1.0
            h['HIERARCH FRONTEND STATE'] = 'ADC'
            h['HIERARCH TEL ROT'] = 0.0
            h['FIBAREA'] = 0.4418
            h['CHANNEL'] = c
            cols = [fits.Column('FIBER_ID', 'J', array=np.asarray(t['fiber'])),
                    fits.Column('BENCHSIDE', '10A', array=np.asarray(t['bench']).astype(str))]
            fm = fits.BinTableHDU.from_columns(cols + [fits.Column('FIBER_TYPE', '10A', array=['SCI'] * len(t)),
                                                       fits.Column('RA', 'D', array=st.ra[rows]),
                                                       fits.Column('DEC', 'D', array=st.dec[rows])], name='FIBERMAP')
            fw = fits.BinTableHDU.from_columns(cols + [fits.Column('X_FIBERMAP', 'D', array=xy[:, 0]),
                                                       fits.Column('Y_FIBERMAP', 'D', array=xy[:, 1]),
                                                       fits.Column('RA', 'D', array=st.ra[rows]),
                                                       fits.Column('DEC', 'D', array=st.dec[rows])], name='FIBERWCS')
            hdus = fits.HDUList([fits.PrimaryHDU(header=h),
                                 fits.ImageHDU(st.flux[rows], name='FLAM'),
                                 fits.ImageHDU(np.sqrt(st.var[rows]).astype(np.float32), name='FLAM_ERR'),
                                 fits.ImageHDU(st.mask[rows].astype(np.int16), name='MASK'),
                                 fits.ImageHDU(st.wave[rows], name='WAVE'), fm, fw])
            p = os.path.join(directory, f'{stem}_RSS_{c}.fits')
            hdus.writeto(p, overwrite=True)
            if c == 'green':
                out.append(p)
    return out


def make_info(e, H0, exptime=1200.0, P=765.0, T=10.0, RH=30.0):
    ra = RA0
    return ExposureInfo(index=e, exp_id=f'LLAMAS_SYN_{e:02d}', path=f'syn{e}', mjd=61000.0 + e * 0.02,
                        exptime=exptime, ra=ra, dec=DEC0, lst_start=(ra / 15.0 + H0) % 24.0,
                        tel_lst_end=np.nan, ha_end=np.nan, parang_end=np.nan, airmass_start=np.nan,
                        airmass_end=np.nan, tel_rot=0.0, pressure=P, temperature=T, humidity=RH,
                        seeing=1.0, velcorr=1.0)


def make_field(nexp=4, nw=150, seed=1, gain=0.5 + 0.2j, const=0.0j, c2=0.0, star=True,
               s0=(2.0, -3.0), fwhm=1.1, flux0=3e-16, noise=2e-19, lam_ref=8000.0, pa=1.2,
               reg_sigma=0.3, dither=3.0, bright_blue=1.0, star_exps=None, decoy=None):
    """``star_exps``: exposures that contain the star (default all); ``decoy``: dict(exp, dE, dN,
    flux) adds another point source (on the sky, arcsec about the star) to that exposure only.
    Returns (sr, infos, inst, truth). truth['offset'](e, lam) -> (n, 2) true star track in the
    RECORDED fibre coordinates (arcsec about RA0, DEC0); truth['c1'][e] the true c1 vector."""
    rng = np.random.default_rng(seed)
    xy = lut_xy()
    nf = xy.shape[0]
    p_ = np.deg2rad(pa)
    A = np.array([[np.sin(p_), -np.cos(p_)], [np.cos(p_), np.sin(p_)]])
    ctr = xy.mean(axis=0)
    Hs = np.linspace(-2.0, 2.5, nexp) if nexp > 1 else np.array([1.0])
    infos = [make_info(e, H) for e, H in enumerate(Hs)]
    geoms = [exposure_geometry(i) for i in infos]
    dith = rng.uniform(-dither, dither, (nexp, 2))
    reg = rng.normal(0, reg_sigma, (nexp, 2))
    c1 = np.array([[(gain * complex(*g.p) + const).real, (gain * complex(*g.p) + const).imag]
                   for g in geoms])
    sig0 = fwhm / 2.354820045
    chans, metas = {}, []
    for e in range(nexp):
        metas.append(ExposureMeta(infos[e].exp_id, infos[e].path, infos[e].exptime, 1.2, infos[e].mjd))
    for c, (lo, hi) in CH_RANGES.items():
        wl = np.linspace(lo, hi, nw)
        acc = dict(ra=[], dec=[], wave=[], flux=[], var=[], mask=[], solid=[], exp=[])
        for e in range(nexp):
            true_EN = (xy - ctr) @ A.T + dith[e]                       # where each fibre looks
            rec_EN = true_EN + reg[e]                                  # what the RSS records
            d = dn_e6(wl, infos[e], lam_ref)
            cx = s0[0] + c1[e, 0] * d + c2 * d ** 2
            cy = s0[1] + c1[e, 1] * d + c2 * d ** 2
            sig = sig0 * (wl / 7000.0) ** -0.2
            r2 = (true_EN[:, 0, None] - cx[None]) ** 2 + (true_EN[:, 1, None] - cy[None]) ** 2
            spec = flux0 * (wl / 7000.0) ** (1.0 if bright_blue == 1.0 else bright_blue)
            psf = np.exp(-0.5 * r2 / sig[None] ** 2) / (2 * np.pi * sig[None] ** 2) * 0.4418
            has = star and (star_exps is None or e in star_exps)
            f = (spec[None] * psf if has else 0.0) + 1e-18 + rng.normal(0, noise, (nf, nw))
            if decoy is not None and decoy['exp'] == e:
                r2d = ((true_EN[:, 0, None] - s0[0] - decoy['dE']) ** 2
                       + (true_EN[:, 1, None] - s0[1] - decoy['dN']) ** 2)
                f = f + decoy['flux'] * (wl / 7000.0)[None] * np.exp(-0.5 * r2d / sig[None] ** 2) \
                    / (2 * np.pi * sig[None] ** 2) * 0.4418
            ra, dec = tangent_inverse(rec_EN[:, 0], rec_EN[:, 1], RA0, DEC0)
            acc['ra'].append(ra); acc['dec'].append(dec)
            acc['wave'].append(np.tile(wl, (nf, 1))); acc['flux'].append(f.astype(np.float32))
            acc['var'].append(np.full((nf, nw), noise ** 2)); acc['mask'].append(np.zeros((nf, nw), bool))
            acc['solid'].append(np.full(nf, 0.4418)); acc['exp'].append(np.full(nf, e, np.int32))
        chans[c] = ChannelStack(c, *(np.concatenate(acc[k]) for k in
                                     ('ra', 'dec', 'wave', 'flux', 'var', 'mask', 'solid', 'exp')))
    sr = SuperRSS('SYN', 'flam', 'erg/s/cm2/Angstrom', metas, chans)
    inst = {c: np.tile(xy, (nexp, 1)) for c in CH_RANGES}

    def offset(e, lam):
        d = dn_e6(np.atleast_1d(lam), infos[e], lam_ref)
        return np.c_[s0[0] + reg[e, 0] + c1[e, 0] * d + c2 * d ** 2,
                     s0[1] + reg[e, 1] + c1[e, 1] * d + c2 * d ** 2]

    truth = dict(offset=offset, c1=c1, reg=reg, s0=np.array(s0), lam_ref=lam_ref, geoms=geoms,
                 star_radec=tangent_inverse(s0[0], s0[1], RA0, DEC0))
    return sr, infos, inst, truth


def run_tests(glb):
    import sys
    fns = [(k, v) for k, v in sorted(glb.items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
