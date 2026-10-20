"""Assemble the colleague PDF: summary, C2 figure, limits/non-detections, data-quality appendix."""
import sys, os, json, textwrap, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
import common
from common import ROOT, OUT, SCR
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
PDF = os.path.join(OUT, 'LLAMAS_Lya_summary_2026-10-16.pdf')
c2 = json.load(open(f'{SCR}/SPT-CLJ0158-5500/fig_C2.json'))
pe = json.load(open(f'{SCR}/SPT-CLJ0158-5500/perexp_C2.json'))
v, e = np.array(pe['val']), np.array(pe['err']); snr8 = np.nansum(v / e**2) / np.sqrt(np.nansum(1 / e**2))
def c50(path):
    d = json.load(open(path)); return np.c_[d['lam'], d['c50']]
lim58 = c50(f'{SCR}/SPT-CLJ0158-5500/limits_c50.json')
lim11 = c50(f'{SCR}/SPT-CLJ0111-5637/staraligned/limits_c50.json')
def L(arr, l): return arr[np.argmin(abs(arr[:, 0] - l)), 1]
W, H = 8.27, 11.69                                   # A4 portrait

def text_page(pdf, title, blocks):
    fig = plt.figure(figsize=(W, H)); fig.text(0.07, 0.955, title, fontsize=15, weight='bold', va='top')
    y = 0.915
    for head, body in blocks:
        if head:
            fig.text(0.07, y, head, fontsize=11, weight='bold', va='top'); y -= 0.024
        for para in body:
            lines = textwrap.wrap(para, 105)
            for ln in lines:
                fig.text(0.085 if para.startswith('•') else 0.07, y, ln, fontsize=9, va='top'); y -= 0.0165
            y -= 0.006
        y -= 0.008
    fig.text(0.07, 0.025, 'LLAMAS commissioning data, Magellan · reduced with llamas-pyjamas · analysis 2026-10-17', fontsize=7, color='0.4')
    pdf.savefig(fig); plt.close(fig)

with PdfPages(PDF) as pdf:
    # ---- page 1: summary
    text_page(pdf, 'LLAMAS Lyα search in two SPT cluster fields (2026-10-16)', [
        ('Summary', [
            f'We searched the blue channel (≈3500–4750 Å; Lyα at 1.9 ≲ z ≲ 2.9) of two LLAMAS IFU fields, each 8 × 1350 s, for '
            f'Lyα emission. One confident emission-line source is found in SPT-CLJ0158-5500 (C2): a single line at '
            f'{c2["mu"]:.1f} Å, which we interpret as Lyα at z = {c2["z"]:.3f}. No confident emitter is found in SPT-CLJ0111-5637.',
            f'C2: RA, Dec = 01:58:56.12, −55:00:28.3; F(Lyα) = ({c2["flux"]*1e17:.1f} ± {c2["eflux"]*1e17:.1f})×10⁻¹⁷ erg s⁻¹ cm⁻² '
            f'(r = 1.5" aperture); S/N = {c2["flux"]/c2["eflux"]:.1f} (conservative, cube) to {snr8:.1f} (8 exposures combined); '
            f'FWHM {c2["fwhm"]:.1f} Å (≈{c2["v"]:.0f} km/s deconvolved); spatially extended (~3–4", ≈25–35 kpc); no continuum counterpart.']),
        ('Observations', [
            '• SPT-CLJ0158-5500: 8 × 1350 s, 03:56–07:06 UT; dark (Moon below horizon), airmass 1.11–1.17. Dithers of 0–3.5" '
            'were executed and the frames combine on the telescope pointing to ~0.3–0.5" (verified with C2 itself, which moves across '
            'the fibre bundle in step with the dithers while staying fixed on the sky). Absolute astrometry: telescope pointing (±~1").',
            '• SPT-CLJ0111-5637: 8 × 1350 s, 00:07–03:24 UT; Moon up (27% illuminated, 41°→6° altitude, 80° away), airmass 1.59→1.16; '
            'blue sky 2–3× brighter than for 0158. The telescope header pointing drifted by up to ~7" relative to the true field, so a '
            'header-based combine is misaligned (star scatter 1.2–1.3" rms, FWHM 2.0" vs 1.4" single-frame). We rebuilt the cubes with '
            'every exposure and channel aligned on the field star (Gaia position): residual scatter 0.18–0.35" (red/green), no PSF '
            'broadening. The Lyα search uses these star-aligned cubes.']),
        ('Detection limits (50% completeness, point-like line)', [
            f'• SPT-CLJ0158-5500: {L(lim58, 3650)*1e17:.1f}, {L(lim58, 4000)*1e17:.1f} and {L(lim58, 4300)*1e17:.1f} ×10⁻¹⁷ erg s⁻¹ cm⁻² '
            'at Lyα z = 2.0, 2.3 and 2.5.',
            f'• SPT-CLJ0111-5637: {L(lim11, 3650)*1e17:.1f}, {L(lim11, 4000)*1e17:.1f} and {L(lim11, 4300)*1e17:.1f} ×10⁻¹⁷ erg s⁻¹ cm⁻² '
            'at the same redshifts. Measured by injecting 30 fake lines per flux level; 80% completeness is ≈2.5× higher. Lyα below z ≈ 1.66 is unobservable (atmospheric cut-off); in practice the search is useful above z ≈ 1.9.']),
        ('Method in brief', [
            'Continuum-subtracted (running median) blue cubes were searched with a 3-D matched filter (1.2" PSF × ~4 Å), with noise '
            'measured empirically (the pipeline variance underestimates it by ~2×). The number of NEGATIVE peaks provides the '
            'false-positive benchmark. Every candidate was then checked in the individual fibre spectra of all 8 exposures, against '
            'field-wide systematic wavelengths, against detector-fixed defects (same fibre and wavelength in the other field), and for '
            'companion lines at alternative redshifts.']),
        ('Important caveats', [
            '• The atmospheric dispersion corrector (ADC) was not correcting on this night: blue light lands ≈1.5" West of red '
            '(0.6–1.9" per exposure; measured on the 0111 star). For 0111 this is removed by the per-channel star alignment; for 0158 '
            '(no star) a red/IR counterpart of C2 would lie ~1.4" East of the Lyα position.',
            '• A detector defect on camera 3A (fibres ~1421–1425) produces a spurious feature at the same wavelength as C2; C2 is '
            'unchanged when all such fibres are removed (13.7σ, positive in all 8 exposures).',
            '• Two weaker lines near C2 at the same redshift are NOT confirmed (see page 3).']),
    ])
    # ---- page 2: C2 figure
    img = plt.imread(os.path.join(OUT, 'SPT-CLJ0158-5500', 'LYA_C2.png'))
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.03, 0.30, 0.94, 0.66]); ax.imshow(img, interpolation='none'); ax.axis('off')
    cap = ('Figure 1. Lyα emitter candidate C2 in the field of SPT-CLJ0158-5500. (a) Continuum-subtracted narrowband image '
           f'over the line (±1 FWHM), shown as S/N with 3–6σ contours; the circle marks the PSF. (b) Combined green+red continuum with '
           'the Lyα contours: no counterpart is detected (note the ADC offset described in the text). (c) Continuum-subtracted 1.5"-radius '
           'aperture spectrum of the blue channel with the empirical 1σ noise (grey). (d) Zoom on the line with a Gaussian fit and the '
           'scaled sky spectrum. (e) Position–wavelength cut along E–W. (f) Line flux measured independently in each of the 8 '
           'exposures, using only fibres within 1.2" and excluding fibres affected by a detector defect at this wavelength: the line is '
           'present in every exposure. The [O II] 3727 alternative (z = 0.125) is disfavoured by the absence of Hα (< 0.13× the line flux), '
           'Hβ (< 0.17×) and continuum; C IV, He II and C III] are undetected (< 0.11–0.24×), as expected for a star-forming Lyα emitter.')
    fig.text(0.07, 0.28, '\n'.join(textwrap.wrap(cap, 118)), fontsize=8.5, va='top')
    pdf.savefig(fig); plt.close(fig)
    # ---- page 3: limits + non-detection + tentative
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.12, 0.60, 0.80, 0.30])
    ax.semilogy(lim58[:, 0], lim58[:, 1] * 1e17, 'C0', label='SPT-CLJ0158-5500 (dark)')
    ax.semilogy(lim11[:, 0], lim11[:, 1] * 1e17, 'C1', label='SPT-CLJ0111-5637 (Moon up; star-aligned)')
    ax.plot(c2['mu'], c2['flux'] * 1e17, '*', color='C3', ms=14, label='C2 (measured flux)')
    ax.set_xlabel('observed wavelength [Å]'); ax.set_ylabel('50% completeness line flux [10$^{-17}$ erg s$^{-1}$ cm$^{-2}$]')
    sec = ax.secondary_xaxis('top', functions=(lambda l: l / 1215.67 - 1, lambda z: (z + 1) * 1215.67)); sec.set_xlabel('Lyα redshift')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([3, 4, 6, 10, 15, 20, 30])); ax.yaxis.set_major_formatter(FixedFormatter(['3', '4', '6', '10', '15', '20', '30']))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3, which='both')
    ax.set_title('Point-source line detection limits: 50% completeness from injected fake lines', fontsize=10, pad=28)
    blocks = [
        ('SPT-CLJ0111-5637: no confident Lyα', [
            f'The star-aligned blue cube (1019 arcsec² searched) has 0 positive and 2 negative regions above 5σ, i.e. nothing beyond '
            'the noise. The independent fibre-level search (lines repeated in ≥3 of 8 exposures in ≥2 different fibres) finds no '
            'candidate outside two field-wide systematic wavelength bands (4535–4550 and 4640–4720 Å, the blue band edge).']),
        ('SPT-CLJ0158-5500: tentative, unconfirmed lines near C2', [
            '• C1  01:58:57.16 −55:00:19.8, 4178.2 Å (z = 2.437), 12" (≈100 kpc) from C2, Δv ≈ −1100 km/s: 7σ combined but carried '
            'by two exposures (inconsistent across the 8 frames).',
            '• C4  01:58:57.63 −55:00:24.3, 4185.0 Å (z = 2.443), 14" (≈110 kpc) from C2, Δv ≈ −600 km/s: 6.5σ, absent in 2 of 8 exposures.',
            'If real, C1 and C4 would suggest a small structure at z ≈ 2.44 around C2; deeper data are needed.']),
        ('Suggested follow-up', [
            '• Confirm C2 (deep imaging for a counterpart ~1.4" E of the Lyα position; or a second LLAMAS visit in dark time with the ADC working).',
            '• Re-observe 0111 in dark time; ensure dithers are recorded/guided so the combine can rely on the pointing.']),
    ]
    y = 0.50
    for head, body in blocks:
        fig.text(0.07, y, head, fontsize=11, weight='bold', va='top'); y -= 0.024
        for para in body:
            for ln in textwrap.wrap(para, 105):
                fig.text(0.085 if para.startswith('•') else 0.07, y, ln, fontsize=9, va='top'); y -= 0.0165
            y -= 0.006
        y -= 0.01
    pdf.savefig(fig); plt.close(fig)
    # ---- page 4: appendix data quality (J0111 combine + ADC)
    fig = plt.figure(figsize=(W, H))
    fig.text(0.07, 0.965, 'Appendix: data quality for SPT-CLJ0111-5637', fontsize=13, weight='bold', va='top')
    anchor = json.load(open(f'{SCR}/SPT-CLJ0111-5637/staralign_shifts.json'))['anchor']
    for j, ch in enumerate(['red', 'green', 'blue']):
        band = {'red': (7000, 9500), 'green': (5000, 6800), 'blue': (3800, 4600)}[ch]
        for i, (lab, path) in enumerate([('header-pointing combine', os.path.join(ROOT, 'combined', f'SPT-CLJ0111-5637_cube_{ch}.fits')),
                                         ('star-aligned (this work)', os.path.join(OUT, 'SPT-CLJ0111-5637', 'star_aligned', f'SPT-CLJ0111-5637_staraligned_cube_{ch}.fits'))]):
            with fits.open(path) as h:
                wl = h['WAVELENGTH'].data.field(0); wcs = WCS(h[0].header).celestial
                im = np.nanmedian(h[0].data[(wl > band[0]) & (wl < band[1])], axis=0)
            x0, y0 = [int(round(float(q))) for q in wcs.world_to_pixel_values(*anchor)]
            st = im[y0 - 9:y0 + 10, x0 - 9:x0 + 10]; st = st - np.nanmedian(st)
            a = fig.add_axes([0.08 + j * 0.30, 0.66 - i * 0.22, 0.24, 0.20])
            a.imshow(st, origin='lower', cmap='magma', extent=(-4.75, 4.75, -4.75, 4.75), vmin=0, vmax=np.nanmax(st))
            a.plot(0, 0, '+', color='c', ms=12)
            a.set_title(f'{ch}: {lab}', fontsize=8); a.set_xticks([-4, 0, 4]); a.set_yticks([-4, 0, 4])
    fig.text(0.07, 0.425, '\n'.join(textwrap.wrap(
        'Field star (cyan + = Gaia position), 9.5" stamps. Top: cubes combined on the telescope pointing (misaligned by up to ~2.6"; '
        'FWHM 2.0"). Bottom: each exposure and channel aligned on the star (red/green scatter 0.18–0.35", FWHM 1.38–1.48", equal to '
        'single frames). In blue the star sits ≈1" West without per-channel alignment because of the ADC (see below); a residual '
        'N–S streak below the star is the fixed fibre-column pattern and is present in every version.', 120)), fontsize=8, va='top')
    adc = plt.imread(os.path.join(OUT, 'SPT-CLJ0111-5637', '05_ADC_dispersion_star.png'))
    a = fig.add_axes([0.03, 0.10, 0.94, 0.25]); a.imshow(adc, interpolation='none'); a.axis('off')
    fig.text(0.07, 0.09, '\n'.join(textwrap.wrap(
        'ADC test: star centroid vs wavelength in each exposure (relative to red), compared with the uncorrected atmospheric '
        'dispersion (red line). The blue light is displaced ≈1–2" West in every frame, in a roughly fixed direction that does not follow '
        'the parallactic angle and in later frames exceeds the uncorrected prediction: the ADC was misaligned.', 120)), fontsize=8, va='top')
    pdf.savefig(fig); plt.close(fig)
    d = pdf.infodict(); d['Title'] = 'LLAMAS Lyα search, SPT-CLJ0111-5637 and SPT-CLJ0158-5500'; d['Subject'] = 'Lyα emitter candidate at z=2.450'
print('wrote', PDF)
