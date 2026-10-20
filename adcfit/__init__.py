"""adcfit: measure and correct ADC / differential-refraction errors in LLAMAS dither stacks.

Per exposure, the chromatic centroid track of a reference star is measured in fibre space, fitted
in a refractivity basis on green+red only (blue is held out as validation), and the fitted shift
is applied to every wavelength slab of every channel before re-stacking. Run ``python -m adcfit -h``.
"""

__version__ = '0.1.0'
