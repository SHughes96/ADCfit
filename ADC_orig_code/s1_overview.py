import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_body, SkyCoord
import astropy.units as u
from astropy.visualization import ZScaleInterval
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
t = sys.argv[1]
loc = EarthLocation(lat=-29.015 * u.deg, lon=-70.692 * u.deg, height=2380 * u.m)
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
for i, c in enumerate(['blue', 'green', 'red']):
    with fits.open(cube_path(t, c)) as h:
        f = h[0].data; nexp = h['NEXP'].data; wl = h['WAVELENGTH'].data.field(0)
        nfull = int(h[0].header['NEXPTOT'])
    sel = (wl > 3700) & (wl < 4700) if c == 'blue' else np.ones(len(wl), bool)
    img = np.nanmedian(f[sel], axis=0); img[nexp == 0] = np.nan
    np.save(f'{scr_dir(t)}/wl_{c}.npy', img)
    lo, hi = ZScaleInterval().get_limits(img[np.isfinite(img)])
    p = ax[0, i].imshow(img, origin='lower', vmin=lo, vmax=hi, cmap='gray')
    ax[0, i].set_title(f'{c} cube median' + (' 3700–4700 Å' if c == 'blue' else ''), fontsize=10)
    plt.colorbar(p, ax=ax[0, i], fraction=0.046)
    print(f'{c}: shape {f.shape}, NaN frac {np.isnan(f).mean():.3f}')
p = ax[1, 0].imshow(nexp, origin='lower', cmap='viridis', vmin=0, vmax=nfull)
ax[1, 0].set_title(f'NEXP (of {nfull} exposures)'); plt.colorbar(p, ax=ax[1, 0], fraction=0.046)
np.save(f'{scr_dir(t)}/nexp.npy', nexp)
rows = []
for k, p in enumerate(rss_paths(t, 'blue')):
    with fits.open(p) as h:
        hd = h[0].header; sky = h['SKY'].data; wave = h['WAVE'].data; ss = h['SKYSUB'].data
        live = np.nanmedian(np.abs(h['COUNTS'].data), axis=1) > 0
        bs = np.array(h['FIBERMAP'].data['BENCHSIDE']).astype(str)
    band = (wave > 3800) & (wave < 4500)
    t0 = Time(hd['MJD-OBS'], format='mjd')          # DATE-OBS is the LOCAL date after UT midnight
    tm = t0 + float(hd['SEXPTIME']) / 2 * u.s
    aa = AltAz(obstime=tm, location=loc)
    sun = get_sun(tm).transform_to(aa).alt.deg
    moon = get_body('moon', tm, loc)
    malt = moon.transform_to(aa).alt.deg
    sunc = get_sun(tm)
    elong = sunc.separation(moon).deg
    illum = (1 - np.cos(np.radians(elong))) / 2
    tgt = SkyCoord(hd['RA'] * u.deg, hd['DEC'] * u.deg)
    msep = moon.separation(tgt).deg
    skyv = float(np.nanmedian(np.where(band[live], sky[live], np.nan)))
    resid = float(np.nanmedian(np.where(band[live], ss[live], np.nan)))
    lost = sorted(set(bs[~live]))
    rows.append(dict(exp=exp_id(p), mid=tm.isot[11:19], sun=sun, moon_alt=malt, illum=illum, moon_sep=msep,
                     airmass=hd.get('AIRMASS'), sky=skyv, resid=resid, dead=lost, nlive=int(live.sum())))
    print('%s mid %s UT  sun %5.1f  moon alt %5.1f illum %.2f sep %5.1f  X %.3f  blue SKY %.1f  median SKYSUB %+.1f  dead benchsides %s (live fibres %d)' % (
        rows[-1]['exp'], rows[-1]['mid'], sun, malt, illum, msep, hd.get('AIRMASS'), skyv, resid, lost, live.sum()))
json.dump(rows, open(f'{scr_dir(t)}/conditions.json', 'w'), indent=1)
x = np.arange(len(rows))
ax[1, 1].bar(x, [r['sky'] for r in rows], color=['C3' if r['moon_alt'] > 0 else 'C0' for r in rows])
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels([r['exp'] for r in rows], rotation=60, fontsize=7)
ax[1, 1].set_ylabel('median blue SKY counts 3800–4500 Å'); ax[1, 1].set_title('blue sky per exposure (red = Moon up)', fontsize=10)
ax[1, 2].plot(x, [r['moon_alt'] for r in rows], 'o-', label='Moon altitude [deg]')
ax[1, 2].plot(x, [r['sun'] for r in rows], 's-', label='Sun altitude [deg]')
ax[1, 2].axhline(0, color='k', lw=0.5); ax[1, 2].axhline(-18, color='gray', lw=0.5, ls=':')
ax[1, 2].set_xticks(x); ax[1, 2].set_xticklabels([r['exp'] for r in rows], rotation=60, fontsize=7)
ax[1, 2].legend(fontsize=8); ax[1, 2].set_title('observing conditions (Moon illum %.0f%%)' % (100 * rows[0]['illum']), fontsize=10)
fig.suptitle(t, fontsize=14)
plt.tight_layout(); plt.savefig(f'{out_dir(t)}/01_overview_and_conditions.png', dpi=100)
