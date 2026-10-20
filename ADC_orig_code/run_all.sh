#!/bin/bash
# Reproduce the Lyα search + dither-combine analysis of SPT-CLJ0111-5637 and SPT-CLJ0158-5500.
# Commands are listed in the order they were run on 2026-10-17. See ../ANALYSIS_LOG.md for what each
# step does and what it found.
#
# Environment: conda env `llamas_data_reduction_clean` (Python 3.12.8, numpy 2.2.1, scipy 1.15.1,
# astropy 7.0.0, matplotlib 3.10.8, pypdf 6.13.3) with llamas-pyjamas importable (repo commit
# bfce0d7, branch sensfunc-standard-match, with uncommitted local changes at the time).
# Network: Gaia DR3 TAP (ESA) is queried by s10_staralign.py and adhoc/a06.
# Intermediates go to $LYA_WORK (default: ../work). Figures go to ../<TARGET>/.
#
# Usage:  bash run_all.sh [A|B|C|D|E|F|all]     (default: all of A, C, D; B is historical, see below)
set -euo pipefail
cd "$(dirname "$0")"
PY="conda run -n llamas_data_reduction_clean python"
J0111=SPT-CLJ0111-5637; J0158=SPT-CLJ0158-5500
WORK=${LYA_WORK:-$(cd .. && pwd)/work}
STEP=${1:-all}

# ---------------------------------------------------------------- A. SPT-CLJ0158-5500 (fully reproducible)
if [[ $STEP == A || $STEP == all ]]; then
  $PY adhoc/a01_inventory.py                       # which RSS built which cube; per-exposure provenance
  $PY adhoc/a02_dateobs_vs_mjd.py                  # DATE-OBS is the local date before local midnight
  $PY s1_overview.py $J0158                        # 01_overview_and_conditions.png (Sun/Moon via MJD-OBS)
  $PY s1b_pointing.py $J0158                       # fibre-map centres vs header pointing
  $PY s2_dithers.py $J0158 --nsig 4                # 02_dither_combine.png (weak test: only an edge source)
  $PY s2d_xcorr.py $J0158                          # leave-one-out cross-correlation (found to track fixed pattern; not used)
  $PY s2e_instrument_frame.py $J0158               # brightest peak on the bundle (turned out to be an edge artefact)
  $PY s2f_sources_on_bundle.py $J0158              # 04_bundle_sources_spectra.png (instrument-fixed artefacts)
  $PY s4_cube_search.py $J0158                     # 06_lya_cube_search.png + injections (~10 min)
  $PY s5_rss_search.py $J0158                      # 07_rss_line_hits.png (fibre level, >=3 exposures)
  $PY s6_rebuild.py $J0158                         # independent cube rebuild vs delivered cube
  $PY s7_vet.py $J0158                             # 08_candidate_vetting.png (per-exposure empirical S/N)
  $PY s8_c2.py                                     # C2: dithers real? detector-artefact test (uses J0111 RSS too)
  $PY s8b_c2_noartefact.py                         # C1/C2/C4 with defect fibres removed
  $PY s8c_c2_identity.py                           # C2 width, EW, size, companion-line limits
  $PY s9_dither_fig_0158.py                        # 03_dither_check_with_C2.png
  $PY adhoc/a05_c2_neighbours.py                   # coordinates/separations of C1, C2, C4
fi

# ---------------------------------------------------------------- B. SPT-CLJ0111-5637, FIRST ATTEMPT (historical)
# These ran against the first-attempt cube (now combined/firstattempt_*) AND the first-attempt fibre
# maps (relative-source registration). Those fibre maps were overwritten when 'Auto-register block'
# was run (all frames -> rough-header), so these steps will NOT reproduce the first-attempt numbers
# on the current files. To re-run them faithfully you would need the RSS files as they were before
# 2026-10-17 08:53, and to point common.cube_path at combined/firstattempt_*.
if [[ $STEP == B ]]; then
  $PY s1_overview.py $J0111
  $PY s1b_pointing.py $J0111
  $PY s2_dithers.py $J0111
  $PY s2_dithers.py $J0111 --shift 00-31-08.4 -22.71 23.67 --tag _exp1_at_header --nsig 6
  $PY s2b_frame_quality.py $J0111
  $PY s2c_exp1_fibremap.py
  $PY s2d_xcorr.py $J0111
  $PY s2d_xcorr.py $J0111 --shift 00-31-08.4 -22.71 23.67 --tag _exp1_at_header
  $PY s2e_instrument_frame.py $J0111
  $PY s2f_sources_on_bundle.py $J0111
  $PY s4_cube_search.py $J0111
  $PY s5_rss_search.py $J0111                      # used a fixed exp-1 shift (-23.60, +23.91) at the time
  $PY s6_rebuild.py $J0111
  $PY s7_vet.py $J0111
  $PY adhoc/a03_exp1_star_stored_position.py
  $PY adhoc/a06_which_frames_wrong_gaia.py
fi

# ---------------------------------------------------------------- C. ADC test (reproducible: uses relative fibre positions only)
if [[ $STEP == C || $STEP == all ]]; then
  $PY s3_adc.py $J0111 13.8 -10.2 star             # 05_ADC_dispersion_star.png; writes work/$J0111/adc_star.json
  $PY adhoc/a04_adc_models_and_fibremaps.py        # ADC model fits; identical fibre maps across channels
fi

# ---------------------------------------------------------------- D. SPT-CLJ0111-5637, SECOND ATTEMPT (current files)
if [[ $STEP == D || $STEP == all ]]; then
  $PY adhoc/a07_current_wcs_state.py               # all frames rough-header after the failed auto-register
  $PY s2_dithers.py $J0111 --tag _header_combine --nsig 6           # header combine is misaligned
  $PY s2_dithers.py $J0111 --tag _header_combine_withblue --blue --nsig 6
  $PY s10_staralign.py                             # writes ../$J0111/star_aligned/*_staraligned_cube_*.fits (Gaia query)
  $PY s2_dithers.py $J0111 --nsig 6 --tag _star_aligned --blue --shifts "$WORK/$J0111/staralign_shifts.json"
  $PY s10b_blue_choice.py                          # blue alignment recipes compared (needs adc_star.json from C)
  $PY s10c_blue_stamps.py                          # blue star stamps per recipe (kept recipe a = per-exposure blue fit)
  $PY adhoc/a08_star_concentration.py
  LYA_VARIANT=staraligned $PY s1_overview.py $J0111
  LYA_VARIANT=staraligned $PY s4_cube_search.py $J0111   # Lyα search on star-aligned cube (~10 min)
  LYA_VARIANT=staraligned $PY s5_rss_search.py $J0111
  # ---- colleague figure + PDF
  $PY s11a_prep.py $J0158 49 45 4193.4 C2          # sky spectrum + per-exposure fluxes (defect fibres excluded)
  $PY s11_source_figure.py $J0158 49 45 4193.4 C2  # ../$J0158/LYA_C2.{png,pdf}; text rows from work/$J0158/source_extra_C2.json
  $PY s13_completeness.py                          # 50%-completeness limits from injection recovery
  $PY s12_pdf.py                                   # ../LLAMAS_Lya_summary_2026-10-16.pdf
  bash adhoc/a09_render_pdf_pages.sh               # render pages for checking
fi

# ---------------------------------------------------------------- E. User's CubeViewer narrowband check (0158)
if [[ $STEP == E || $STEP == all ]]; then
  $PY s14_narrowband_check.py                      # marks C2 on ../../narrowband.fits; writes contsub narrowband FITS + C2.reg
  $PY adhoc/a10_second_blob.py                     # the blob at (18,58): line-window vs line-free windows
  $PY s11a_prep.py $J0158 18 58 4194.0 blob_x18y58 # per-exposure test of that blob (not real: 1.5σ)
fi

# ---------------------------------------------------------------- F. ADC tables and write-up (J0111; JSON only)
if [[ $STEP == F || $STEP == all ]]; then
  $PY s15_adc_tables.py                            # ../$J0111/ADC/: tables, figures, ADC_tables_printable.pdf
fi
