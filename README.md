# adcfit: ADC / differential-refraction correction for LLAMAS dither stacks

`adcfit` measures how a reference star's position changes with wavelength in every exposure. It fits that track on the **green and red channels only**, applies the fitted shift(λ) to every wavelength slice of **all three channels**, and re-stacks the cubes. Along the way it also checks how well the exposures register.

```
conda activate llamas_data_reduction        # needs llamas_pyjamas importable
cd /Users/slhughes/Work/independent/ADC_fitting
python -m adcfit <combined cube | RSS files ...> [options]
python -m adcfit --dir <reduced>/extractions --object SPT-CLJ0111 --outdir qa_J0111 --cube-outdir cubes_J0111
```

## What it does

| stage | what | key outputs (in `--outdir`) |
|---|---|---|
| 1 inventory | Exposure geometry and weather. Hour angle, airmass and parallactic angle are computed over the exposure and cross-checked against the header cards. | `inventory.txt/.csv`, `qa/01_inventory.png` |
| 3 star | Detects compact sources per exposure (destriped), clusters them across exposures, and picks the brightest compact, isolated star. `--star` overrides the choice. | `star_candidates.txt/.csv`, `qa/03_star_selection.png` |
| 2 registration | Broad-band star position and FWHM per exposure and channel, with fibre positions as delivered. | `registration_uncorrected.txt/.csv`, `qa/02_registration.png` |
| 4 centroids | Star centroid vs wavelength per exposure, in the sky and instrument frames, with errors and quality flags. | `centroids.csv`, `morphology.txt`, `overlap_consistency.txt`, `qa/04-06_*` |
| 5 fit | Per-exposure fit on green+red (below); blue held out. | **`adc_shift_grid.txt/.csv`** (shift vs λ every 250 Å, printed), `adc_coeffs.*`, `adc_shift_bins.csv`, `blue_holdout.txt`, `chromatic_summary.txt`, `dispersion_models.txt`, **`qa/NN_fit_<exp>.png`** (one per exposure), `qa/*_fit_summary.png` |
| 7a residuals | Re-measures the star with the correction applied (fibre space). Blue is the held-out test. | `residuals_after.txt/.csv`, `qa/*_residuals_after.png` |
| 5s standard | With `--std-rss`/`--sensfunc`: the standard's track and the sensfunc's captured fraction (below). | `standard_*.txt`, `sensfunc_diagnostic.*`, `sensfunc_aperture.txt` |
| 6 stack | Corrected (`adc`) and uncorrected (`noadc`) cubes on identical spatial and wavelength grids, plus `regonly` with `--also-regonly`. | cubes, `*_nexpslab.fits` and `slabs.csv` in **`--cube-outdir`**, `cube_structure.txt`, `qa/*_slabs.png` |
| 7b validation | Star centroid, FWHM and concentration vs λ in the stacked cubes, before and after. | `validation.txt/.csv`, `rebuild_fidelity.txt` (cube input only), `qa/*_validation_cubes.png`, `qa/*_stamps.png` |

All figures are also collected in `adc_qa.pdf`. Every number, setting and output path goes into `adc_summary.json`.

## The model

For each exposure and each sky axis a ∈ {E, N}:

    offset_a(λ) = c0_a + c1_a · dn(λ) [+ c2_a · dn(λ)²],     dn = 1e6 · [n(λ) − n(λ_ref)]

- **n(λ)** is the refractive index of air at that exposure's `WTH-PRES`, `WTH-TEMP` and `WTH-HUMD` (Filippenko 1982). It is evaluated at the observed-frame vacuum wavelength, WAVE / VELCORR.
- **Why this basis.** Atmospheric dispersion is exactly proportional to dn, and a mis-set ADC adds a dispersion of almost the same shape. So a fit to green and red extrapolates sensibly into the blue. A polynomial in λ does not, so it is only drawn for comparison.
- **c0** is the star's position at λ_ref, i.e. the exposure's registration.
- **c1** is the dispersion, in arcsec per 1e-6 of refractivity. With no ADC it would equal 0.206265·tan z, pointing to the zenith (parallactic angle q).
- **The correction added to each exposure's fibre positions** is `anchor − offset_e(λ)`. The anchor is the median over exposures of `offset_e(λ_ref)`. This removes both the registration error and the ADC error. `--anchor none` removes only the chromatic part.
- **Blue is never fitted.** Its measured centroids are compared with the extrapolated model in `blue_holdout.txt`. That comparison is the test of whether the green+red model describes the blue.
- **Exposures whose own fit fails** get their chromatic term from the best pooled model across exposures (`dispersion_models.txt`) and their zero point from the broad-band red position. Their status shows as `fallback`.

### Reading the dispersion diagnostics

- `chromatic_summary.txt` compares the measured c1 with the no-ADC prediction p for each exposure:
  - `amp_ratio` = |c1|/|p| is the fraction of the atmospheric dispersion left over. It is ~0 if the ADC works, 1 with no ADC, and above 1 if the ADC adds dispersion.
  - `dir_resid_deg` is the angle between the measured and predicted dispersion directions.
- `dispersion_models.txt` fits all exposures' c1 together:
  - `gain`: v = g·p
  - `mirror`: v = g·conj(p). If this fits clearly better, suspect the sky parity of the fibre WCS.
  - `const`: v = k, a residual fixed on the sky or instrument, such as a stuck ADC
  - `gain+const`, plus the `perfect_adc` and `no_adc` references
  - arg(g) mixes any WCS rotation error with ADC misalignment; the two cannot be separated with a single rotator angle.
- `overlap_consistency.txt`: the same wavelength measured in two channels should agree. A difference points to channel-specific fibre positions (trace or fibre-ID problems), not to the atmosphere.

## Measurement details

- The **aperture follows the star**. Each exposure starts from that exposure's own detection, and each bin's fit re-centres up to 3 times. The original analysis instead used a fixed bundle aperture while dithers moved the star 1–6".
- **PSF width** (`--sigma auto`, the default): the width is free within 0.7–1.6× the seeing-scaled width, FWHM_ref·(λ/λ_ref)^−0.2. If the free fit runs to a limit, the tool falls back to that fixed width; a fixed width has a negligible effect on the centroid.
- **Row destriping.** SKYSUB residuals are striped along the fibre-map rows. On J0111 a per-row median removes 35%, 23% and 62% of the background scatter in red, green and blue. Per-row medians are subtracted, with fibres near the star excluded from the median. `--no-destripe` turns this off.
- **Bins.** The number per channel is set so each bin has a single-exposure aperture S/N of about `--target-snr` (default 10), capped at `--bins` (default 8 8 8). Fit channels always get at least 2 bins.
  - Telluric bands, bright sky lines and the dichroic overlaps are excluded.
  - The overlaps are measured separately in both channels as a consistency check.
- **Quality flags** (in `centroids.csv`). The fatal ones (`--reject-flags`) are EDGE, BOUND, FEWFIB, LOWSN, HOLE and FAILED. Report-only: CHI2, MOMENT, SATCORE and MORPH. MORPH also fires on fixed fibre streaks next to the star.
- **The predicted dispersion** averages tan z·(sin q, cos q) over the exposure. The hour angle comes from `LST` (taken at shutter open), and the computed values are checked against `TEL PARANGL`/`TEL HA`, which are taken at shutter close. The original analysis combined the start-of-exposure `AIRMASS` with the end-of-exposure `TEL PARANGL`, which biased the prediction by about 0.05–0.08" towards North.

## Stacking

`combine_cube` uses one wavelength-independent kernel matrix, so it cannot move fibres as a function of wavelength. `adcfit.stack` reproduces it exactly, with the same resampling, kernel, weights, variance and coverage; a test checks that it matches `combine_cube` at zero shift.

- **Slabs.** The wavelength grid is split into slabs within which every exposure's correction varies by less than `--slab-tol` (default 0.05" peak-to-peak), and each slab is built with its own shifted positions.
- **Grids.** When a cube is given as input, its spatial grid, wavelength grids and combine settings are reused, so `noadc` should reproduce it (see `rebuild_fidelity.txt`).
- **Output cubes have the pipeline layout, unchanged.**
  - HDUs 0–4 (PRIMARY, VAR, COVERAGE, NEXP, WAVELENGTH) are exactly what `CoaddCube.write` produces. The primary header has no extra cards; `COVFRAC` is copied from the delivered pipeline cube when one exists, so CubeViewer keeps the same display floor.
  - The ADC information is appended after them as two binary tables with scalar columns only:
    - `ADCINFO`: one row per exposure with the authoritative model (C0/C1/C2 for E and N, LAMREF, VELCORR, P/T/RH, status). Its header holds the provenance cards (`ADCCORR`, `ADCORDER`, `ADCLREF`, `ADCANCH`, `ADCSTRA/DE`, `ADCSLTOL`, `ADCNSLAB`, ...).
    - `ADCSLABS`: one row per wavelength slab and exposure, with the plane range and the shift applied.
  - The per-slab NEXP cube goes to a separate `<cube>_nexpslab.fits`, not into the cube.
  - Every cube is checked against the delivered pipeline cube (or a plain `combine_cube` write) after writing; see `cube_structure.txt`.
- **2-D maps.** `COVERAGE`/`NEXP` hold the per-spaxel minimum over slabs.
- **Safety.** Existing cubes are never overwritten without `--overwrite`, and the input cube never is.

Note: CubeViewer's optimal extraction rebuilds the super-RSS from `RSSFILn`, so it does not see this correction.

## Standard star and sensitivity function (`--std-rss`, `--sensfunc`)

This is a diagnostic only: nothing in the RSS files, the sensfunc or the science flux is changed.

```
python -m adcfit --dir <reduced>/extractions --object SPT-CLJ0111 \
    --std-rss <reduced>/extractions/LLAMAS_<std>_RSS_green.fits \
    --sensfunc <reduced>/extractions/LLAMAS_<std>_RSS_green_sensfunc.fits --outdir ... --cube-outdir ...
```

**Should the standard be ADC-corrected before the sensitivity function is built and applied?**
- S(λ) must be a total-flux throughput. The CubeViewer picker builds it from the SKYSUB counts summed over one fixed fibre set at every wavelength, with no aperture correction.
- If the standard's centroid moves with wavelength (ADC error + DAR at its airmass and parallactic angle) or its PSF widens to the blue, that fixed set captures a wavelength-dependent fraction f(λ). S then carries a 1/f(λ) colour error into every FLAM spectrum calibrated with it.
- So yes: the standard's chromatic offset has to be removed from S. For a single standard exposure that means extracting it with an aperture that follows the star at each wavelength (or one large enough to hold all of it), not restacking it.
- On the science side the order does not matter. S multiplies each fibre's spectrum and the ADC correction only moves fibre positions, so applying S before or after the correction gives the same result.

**What adcfit reports**
- `standard_checks.txt`: the sensfunc's STANDARD/EXPTIME/AIRMASS against the standard RSS, and every science FLAM plane's `SENSFILE`, `STDNAME`, `STDAIR` and `EXTCORR`.
- `standard_fit.txt/.json`, `qa/*_std_fit.png`:
  - the standard's track, fitted on green+red with blue held out — a high-S/N test of the blue extrapolation the science star cannot provide;
  - its c1 next to the science exposures' c1 and the no-ADC prediction.
- `sensfunc_diagnostic.txt/.csv`, `qa/*_sensfunc_diag.png`:
  - f(λ) = C_ap/C_total, where C_ap = F_ref·EXPTIME/RAW are the counts the sensfunc implies and C_total the star's counts within `--std-rtot` arcsec of its measured track. This works whatever aperture was used.
  - `S_used_over_true` = 1/f, and the colour error relative to λ_ref. The verdict recommends rebuilding S when that error exceeds 2%.
  - Spread between the `--std-rtot` radii shows whether the total is converged. `extrapolated` marks bins beyond the measured track.
- `sensfunc_aperture.txt`, `qa/*_std_aperture.png`:
  - The picker aperture is searched for: circles of R fibre spacings about the fibre whose wavelength grid the sensfunc uses (found exactly).
  - If one reproduces the sensfunc to < 2%, the PSF model also splits f into its ADC part (star on its track vs held at λ_ref).
  - A hand-grown aperture, or an RSS re-extracted after the sensfunc was built, is reported as not reproduced.

### Writing ADC-corrected standard RSS files (`--std-correct-rss OUTDIR`)

This writes corrected copies of the standard's blue/green/red RSS files. You then load them in the llamas-pyjamas CubeViewer and build a new sensitivity function with the picker as usual.

```
python -m adcfit --std-rss <reduced>/extractions/LLAMAS_<std>_RSS_green.fits \
    --std-correct-rss <reduced>/std_adc --outdir <reduced>/std_adc_qa
```

Science inputs are optional. Without them only the standard is processed; with them it runs alongside the science stages.

**What is moved.** An RSS gives each fibre one position, so a fixed picker aperture can only follow the star if the light itself is moved:
- At every wavelength the fibre-plane image is resampled so the star sits where it is at λ_ref (red): `new(p, λ) = old(p + track(λ) − track(λ_ref))`.
- `track` is the standard's own measured centroid track, in all three channels. Blue is measured, not extrapolated.
- Each fibre is resampled to a common wavelength grid at half the native dispersion. Neighbouring fibres' WAVE grids differ by about 7 pixels, so this resampling is needed.
- The spatial resampling is piecewise-cubic (Clough–Tocher) on the fibre triangulation, done in wavelength slabs within which the shift changes by less than `--std-slab-tol` (0.02").
- Results are interpolated back onto each fibre's own WAVE.
- The variance is propagated with the linear weights.

**The files have exactly the pipeline structure.**
- Same file names, written to OUTDIR. Writing into the input directory is refused, and existing files are not overwritten without `--overwrite`.
- Same HDUs, order, dtypes and header keywords.
- Only the SKYSUB (or FLUX) and ERROR data change, plus FLAM/FLAM_ERR if present. COUNTS, SKY, SKYRESID, FWHM, WAVE, MASK and the tables are byte-identical; COUNTS − SKY therefore no longer equals SKYSUB.
- Where a corrected value is undefined (shifted point outside the IFU, dead fibre), the input value is kept, so no NaNs or mask bits are introduced.
- Provenance is in HISTORY lines and `<stem>_adcshift.json`.
- Every write is checked against its input (`std_rss_correction.txt`).

**Checks** (in `--outdir`):
- `std_rss_track.txt`: the star's centroid range over wavelength, input vs corrected.
- `std_rss_capture.txt/.csv`: the fraction of the star inside fixed apertures (R = 1.5, 2, 3") before and after.
- `qa/*_std_rss_corrected.png`: plots of both.

On a synthetic star with 1.4" of dispersion:
- the corrected track is flat to ±0.03", the remainder being interpolation bias on the 0.75" fibre lattice;
- a fixed 2" aperture captures the same light as an aperture following the star, to 0.7% (17% error before);
- the star's total light is conserved to better than 1%.

**Limits.**
- The seeing PSF still widens to the blue (∝ λ^−0.2), and correcting the position does not change that. Use a generous picker aperture (radius ≳ 3× FWHM, e.g. 3–4") so the remaining loss is small.
- The interpolation smooths the PSF slightly (FWHM +2% in the synthetic test).

**Next steps.**
1. Open the corrected green file from OUTDIR in CubeViewer.
2. Pick the aperture on the star, then Build Sensitivity Function, then Save. The default name lands in OUTDIR.
3. Re-run `apply_fluxcal` on the science RSS with the new sensfunc.
4. Re-run adcfit with `--sensfunc` pointing at it. The diagnostic should then show a colour error of a few % at most.

## Options worth knowing

`--star RA DEC` / `--star-pix X Y` · `--std-rss` / `--sensfunc` / `--std-star` / `--std-rtot` / `--std-correct-rss` / `--std-slab-tol` · `--measure-only` (stop after the fit) · `--no-cubes` · `--order {1,2,auto}` · `--fit-channels green,red` · `--anchor {star,none}` · `--target-snr` · `--psf moffat` · `--bg plane` · `--channels` · `--pixscale/--kernel/--fwhm/--weight/--units/--dwave` (default to the input cube's values) · `--cube-outdir` · `--overwrite`. Run `python -m adcfit -h` for all options.

Exit codes: 0 ok · 1 bad input or existing outputs · 2 no usable reference star (see `star_candidates.csv`; use `--star`) · 3 no exposure could be fitted.

## Tests

```
pytest tests -q            # ~15 s
```

The tests use a synthetic field: the real fibre-map LUT, dithers, registration errors, and a star with a known per-exposure dispersion. They check that:

- the refraction physics and parallactic angle are right (checked against a TCS header);
- an off-centre star is centroided correctly;
- the slab stacker matches `combine_cube` exactly at zero shift;
- a +E shift moves the star +E;
- end to end, the injected dispersion is recovered, the held-out blue is predicted to < 0.05", and the corrected cubes are flat in wavelength;
- the output cubes have exactly the pipeline layout for HDUs 0–4, with only scalar tables after them;
- a sensfunc built from a known fibre aperture is traced back to that aperture, and its captured fraction matches the truth to 0.5%;
- the ADC-corrected standard RSS has exactly the input structure, loads through CubeViewer's RSS reader and sensfunc builder, has a flat star track, and conserves flux.

## Reused llamas_pyjamas internals

The tool uses these private functions from `llamas_pyjamas` (pinned by the tests):

- `Combine.cube`: `_resample_fibres`, `_wave_grid`, `_wcs3d`
- `Combine.coadd`: `_kernel`
- `Combine.spectrum`: `_refractivity`
- `Combine.superRSS`: `_hdr_float`
- `Utils.register`: `_fibre_xy`
- `CubeViewer.cubeViewRSS`: `channel_siblings` (and `RSSScene` in the tests)

It never calls the `register_*` functions, because those write into the RSS files.
