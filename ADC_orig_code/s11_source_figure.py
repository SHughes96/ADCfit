"""Publication-style figure for one confident Lyα source.
Usage: s11_source_figure.py TARGET X Y LAM LABEL   (cube pixel coords of the blue cube in use)"""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
import astropy.units as u
from scipy.ndimage import median_filter, gaussian_filter, map_coordinates
from scipy.optimize import curve_fit
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'font.family': 'DejaVu Sans'})
t, x0, y0, lam0, label = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
sd = scr_dir(t); use = np.load(f'{sd}/use_mask.npy')
with fits.open(cube_path(t, 'blue')) as h:
    f = h[0].data.astype(float); wl = h['WAVELENGTH'].data.field(0); cw = WCS(h[0].header).celestial
    nexp = h['NEXP'].data
ny, nx = f.shape[1:]; yy, xx = np.mgrid[:ny, :nx]; pix = 0.5; dw = np.median(np.diff(wl))
cont = median_filter(np.nan_to_num(f), size=(101, 1, 1), mode='nearest')
res = np.where(np.isfinite(f), f - cont, np.nan)
ra0, dec0 = [float(q) for q in cw.pixel_to_world_values(x0, y0)]
sc = SkyCoord(ra0 * u.deg, dec0 * u.deg)
# --- 1D aperture spectrum + empirical error from random apertures (same size)
R = 3.0
ap = np.hypot(xx - x0, yy - y0) <= R
spec = np.nansum(res[:, ap], axis=1) * pix ** 2
spec[np.isfinite(res[:, ap]).sum(axis=1) < 0.5 * ap.sum()] = np.nan      # no data -> blank, not zero
rng = np.random.default_rng(5)
pool = np.argwhere(use & (np.hypot(xx - x0, yy - y0) > 10))
rs = np.array([np.nansum(res[:, np.hypot(xx - px, yy - py) <= R], axis=1) * pix ** 2 for py, px in pool[rng.integers(0, len(pool), 120)]])
err = 1.4826 * np.nanmedian(np.abs(rs - np.nanmedian(rs, axis=0)), axis=0)
err = median_filter(err, 9)
# Gaussian line fit
g = lambda x, A, mu, s: A * np.exp(-(x - mu) ** 2 / (2 * s ** 2))
sel = np.abs(wl - lam0) < 30
p, cv = curve_fit(g, wl[sel], np.nan_to_num(spec[sel]), p0=[np.nanmax(spec[sel]), lam0, 2.5], sigma=err[sel], absolute_sigma=True)
mu, sig = p[1], abs(p[2]); fwhm = 2.355 * sig; efwhm = 2.355 * np.sqrt(cv[2, 2])
z = mu / 1215.67 - 1; ez = np.sqrt(cv[1, 1]) / 1215.67
win = np.abs(wl - mu) < max(1.5 * fwhm / 2.355 * 2, 4)
flux = np.nansum(spec[win]) * dw
rint = np.nansum(rs[:, win], axis=1) * dw                    # same window integrated at random apertures
eflux = 1.4826 * np.median(np.abs(rint - np.median(rint)))   # includes spectral+spatial correlated noise
res_fwhm = mu / 1738.0; v_int = 3e5 * np.sqrt(max(fwhm ** 2 - res_fwhm ** 2, 0)) / mu
kpc_per_as = 1 / Planck18.arcsec_per_kpc_proper(z).value
# --- narrowband S/N map (line window), empirical per-pixel noise from same window elsewhere
k = (np.abs(wl - mu) < fwhm)
nb = np.nansum(res[k], axis=0) * dw
nbs = gaussian_filter(np.nan_to_num(nb), 1.0)
noise = 1.4826 * np.nanmedian(np.abs(nbs[use] - np.nanmedian(nbs[use])))
nbsn = nbs / noise; nbsn[nexp == 0] = np.nan
# continuum (green+red white light) on the blue grid
cimg = None
for ch in ['red', 'green']:
    with fits.open(cube_path(t, ch)) as h:
        w2 = WCS(h[0].header).celestial; d2 = np.nanmedian(h[0].data, axis=0)
    r2, d2_ = cw.pixel_to_world_values(xx, yy); px, py = w2.world_to_pixel_values(r2, d2_)
    rr = map_coordinates(np.nan_to_num(d2), [py, px], order=1, cval=np.nan)
    cimg = rr if cimg is None else cimg + rr
cimg = gaussian_filter(cimg, 0.7)
# stamp geometry: offsets in arcsec, E to the left
half = 10.0
ext = [(-(0 - x0) * pix) - (-half), 0, 0, 0]
def stamp(img):
    xs = slice(int(x0 - half / pix), int(x0 + half / pix) + 1); ys = slice(int(y0 - half / pix), int(y0 + half / pix) + 1)
    return img[ys, xs]
extent = [ (int(x0 - half / pix) - 0.5 - x0) * pix, (int(x0 + half / pix) + 0.5 - x0) * pix,
           (int(y0 - half / pix) - 0.5 - y0) * pix, (int(y0 + half / pix) + 0.5 - y0) * pix]
# blue cube +x pixel = West (CDELT1<0) -> plot with x = -ΔRA so East is left: imshow shows +x right = West. Good (E left).
fig = plt.figure(figsize=(14, 11.5))
gs = GridSpec(3, 3, figure=fig, height_ratios=[1.15, 0.9, 0.9], hspace=0.45, wspace=0.45)
def decorate(ax, title):
    ax.set_xlabel('Δα (arcsec; East ←  → West)'); ax.set_ylabel('Δδ (arcsec)'); ax.set_title(title, fontsize=10)
    ax.add_patch(Circle((-half + 1.6, -half + 1.6), 1.3 / 2, fill=False, color='k', lw=1.2))
    ax.text(-half + 1.6, -half + 2.6, 'PSF', ha='center', fontsize=7)
    ax.plot([half - 6.5, half - 1.5], [-half + 1.2] * 2, 'k-', lw=2)
    ax.text(half - 4, -half + 1.7, f'5" = {5 * kpc_per_as:.0f} kpc', ha='center', fontsize=7)
    cx_, cy_ = half - 1.2, half - 5.0                                  # compass origin (upper right)
    ax.annotate('', xy=(cx_, cy_ + 3.0), xytext=(cx_, cy_), arrowprops=dict(arrowstyle='->', lw=1.1, color='k'))
    ax.annotate('', xy=(cx_ - 3.0, cy_), xytext=(cx_, cy_), arrowprops=dict(arrowstyle='->', lw=1.1, color='k'))
    ax.text(cx_, cy_ + 3.3, 'N', ha='center', va='bottom', fontsize=8, weight='bold')
    ax.text(cx_ - 3.3, cy_, 'E', ha='right', va='center', fontsize=8, weight='bold')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
# note: N arrow points down from the corner? fix: arrow from corner going +y is North
extra = json.load(open(f'{sd}/source_extra_{label}.json')) if os.path.exists(f'{sd}/source_extra_{label}.json') else {}
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(stamp(nbsn), origin='lower', extent=extent, cmap='viridis', vmin=-2, vmax=max(6, np.nanmax(stamp(nbsn))))
ax1.contour(stamp(nbsn), levels=[3, 4, 5, 6], colors='w', linewidths=0.7, extent=extent, origin='lower')
cb = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.02); cb.set_label('narrowband S/N')
decorate(ax1, f'(a) Lyα narrowband {mu - fwhm:.0f}–{mu + fwhm:.0f} Å')
ax2 = fig.add_subplot(gs[0, 1])
cs = stamp(cimg); lo, hi = np.nanpercentile(cs, [5, 99.5])
ax2.imshow(cs, origin='lower', extent=extent, cmap='Greys', vmin=lo, vmax=hi)
ax2.contour(stamp(nbsn), levels=[3, 4, 5, 6], colors='C1', linewidths=0.9, extent=extent, origin='lower')
decorate(ax2, '(b) continuum 4700–10000 Å + Lyα contours (3–6σ)')
note = extra.get('note_b', 'No continuum counterpart detected')
ax2.text(0.02, 0.97, note, transform=ax2.transAxes, fontsize=7.5, va='top', ha='left', bbox=dict(fc='w', ec='0.6', alpha=0.85))
ax3 = fig.add_subplot(gs[0, 2]); ax3.axis('off')
rows = [('Position (ICRS)', sc.to_string('hmsdms', precision=2, sep=':')),
        ('Line centre', f'{mu:.1f} ± {np.sqrt(cv[1, 1]):.1f} Å'),
        ('Redshift (Lyα)', f'z = {z:.4f} ± {max(ez, 0.0005):.4f}'),
        ('Line flux', f'({flux*1e17:.1f} ± {eflux*1e17:.1f})×10⁻¹⁷ erg s⁻¹ cm⁻²  (r = 1.5")'),
        ('S/N, conservative', f'{flux / eflux:.1f}  (random apertures in cube)'),
        ('FWHM', f'{fwhm:.1f} ± {efwhm:.1f} Å  (≈{v_int:.0f} km/s, deconvolved)'),
        ('Instrument', 'LLAMAS / Magellan, R ≈ 1740'),
        ('Field', t), ('Exposure', '8 × 1350 s')]

pe0 = json.load(open(f'{sd}/perexp_{label}.json')) if os.path.exists(f'{sd}/perexp_{label}.json') else None
if pe0:
    v0, e0 = np.array(pe0['val']), np.array(pe0['err'])
    rows.insert(5, ('S/N, per exposure', f'{np.nansum(v0 / e0**2) / np.sqrt(np.nansum(1 / e0**2)):.1f}  (8 exposures combined)'))
rows += [(k_, v_) for k_, v_ in extra.get('rows', [])]
for i, (k_, v_) in enumerate(rows):
    ax3.text(-0.05, 1 - i * 0.072, k_, fontsize=8.5, weight='bold', transform=ax3.transAxes, va='top')
    ax3.text(0.42, 1 - i * 0.072, v_, fontsize=8.5, transform=ax3.transAxes, va='top', wrap=True)
ax3.set_title(f'{label}: properties', fontsize=10, loc='left')
# (c) full blue spectrum
ax4 = fig.add_subplot(gs[1, :2])
good = (wl > 3500) & (wl < 4750)
ax4.fill_between(wl[good], -err[good] * 1e17, err[good] * 1e17, color='0.85', lw=0, label='±1σ (empirical)')
ax4.plot(wl[good], spec[good] * 1e17, 'k', lw=0.6, label='aperture spectrum (r = 1.5", continuum-subtracted)')
ax4.axvline(mu, color='C3', lw=0.8, ls='--'); ax4.text(mu + 6, np.nanmax(spec[good]) * 1e17 * 0.92, f'Lyα z={z:.3f}', color='C3', fontsize=8)
ax4.set_xlim(3500, wl[np.where(np.isfinite(spec))[0].max()] + 10); ax4.set_ylabel('F$_λ$ [10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]'); ax4.legend(fontsize=7, loc='upper left')
ax4.set_title('(c) blue-channel spectrum', fontsize=10)
sec = ax4.secondary_xaxis('top', functions=(lambda l: l / 1215.67 - 1, lambda zz: (zz + 1) * 1215.67)); sec.set_xlabel('Lyα redshift', fontsize=8)
# (d) zoom with fit + sky
ax5 = fig.add_subplot(gs[1, 2])
zsel = np.abs(wl - mu) < 45
ax5.fill_between(wl[zsel], -err[zsel] * 1e17, err[zsel] * 1e17, color='0.85', lw=0)
ax5.step(wl[zsel], spec[zsel] * 1e17, 'k', where='mid', lw=0.8)
xf = np.linspace(mu - 45, mu + 45, 400); ax5.plot(xf, g(xf, *p) * 1e17, 'C3', lw=1.2, label=f'Gaussian: FWHM {fwhm:.1f} Å')
sk = json.load(open(f'{sd}/sky_{label}.json')) if os.path.exists(f'{sd}/sky_{label}.json') else None
if sk:
    sw, sv = np.array(sk['w']), np.array(sk['v']); ssel = np.abs(sw - mu) < 45
    yl = ax5.get_ylim()
    ax5.plot(sw[ssel], yl[0] + (sv[ssel] - sv[ssel].min()) / np.ptp(sv[ssel]) * 0.25 * (yl[1] - yl[0]), color='C0', lw=0.7, label='sky (scaled)')
ax5.axhline(0, color='0.5', lw=0.5); ax5.legend(fontsize=7); ax5.set_ylabel('F$_λ$ [10$^{-17}$ cgs Å$^{-1}$]'); ax5.set_xlabel('observed λ [Å]')
ax5.set_title('(d) line zoom', fontsize=10)
# (e) position-wavelength cut (E-W pseudo-slit, ±1" in Dec)
ax6 = fig.add_subplot(gs[2, 0])
rows_ = slice(int(y0 - 2), int(y0 + 3)); cols = slice(int(x0 - half / pix), int(x0 + half / pix) + 1)
pv = np.nanmean(res[zsel][:, rows_, cols], axis=1)
pvs = gaussian_filter(np.nan_to_num(pv), (1.0, 0.7))
m_ = np.nanpercentile(np.abs(pvs), 99)
ax6.imshow(pvs.T, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-m_, vmax=m_,
           extent=[wl[zsel][0], wl[zsel][-1], (cols.start - 0.5 - x0) * pix, (cols.stop - 0.5 - x0) * pix])
ax6.set_xlabel('observed λ [Å]'); ax6.set_ylabel('Δ (arcsec, West →)'); ax6.set_title('(e) E–W pseudo-slit (2.5" wide)', fontsize=10)
# (f) per-exposure consistency (filled by s11 helper json if present)
ax7 = fig.add_subplot(gs[2, 1:])
pe = json.load(open(f'{sd}/perexp_{label}.json')) if os.path.exists(f'{sd}/perexp_{label}.json') else None
if pe:
    xs_ = np.arange(len(pe['exp'])); v_ = np.array(pe['val']); e_ = np.array(pe['err'])
    v_, e_ = v_ * 1e17, e_ * 1e17
    ax7.errorbar(xs_, v_, yerr=e_, fmt='o', color='k', ms=4, capsize=2)
    wm = np.nansum(v_ / e_ ** 2) / np.nansum(1 / e_ ** 2)
    ax7.axhline(wm, color='C3', ls='--', lw=0.8, label=f'weighted mean ({wm / np.sqrt(1 / np.nansum(1 / e_ ** 2)):.1f}σ combined)')
    ax7.axhline(0, color='0.5', lw=0.5); ax7.set_xticks(xs_); ax7.set_xticklabels(pe['exp'], rotation=35, fontsize=7)
    ax7.set_ylabel('line flux [10$^{-17}$ erg s$^{-1}$ cm$^{-2}$]'); ax7.legend(fontsize=7)
    ax7.set_title('(f) line flux in each exposure (fibres < 1.2"; detector-defect fibres excluded; empirical errors)', fontsize=10)
fig.suptitle(f'{label} — LLAMAS Lyα emitter candidate at z = {z:.3f} in the field of {t}', fontsize=13, y=0.985)
for ext_ in ['png', 'pdf']:
    fig.savefig(f'{out_dir(t)}/LYA_{label}.{ext_}', dpi=300 if ext_ == 'png' else None, bbox_inches='tight')
json.dump(dict(label=label, target=t, ra=ra0, dec=dec0, mu=mu, z=z, ez=ez, flux=flux, eflux=eflux, fwhm=fwhm, efwhm=efwhm, v=v_int,
               kpc_per_arcsec=kpc_per_as), open(f'{sd}/fig_{label}.json', 'w'), indent=1)
print(f'{label}: z={z:.4f}±{ez:.4f} flux={flux:.2e}±{eflux:.1e} ({flux/eflux:.1f}σ) FWHM={fwhm:.1f}±{efwhm:.1f} Å ({v_int:.0f} km/s); {kpc_per_as:.2f} kpc/"')
