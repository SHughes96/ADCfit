"""Leave-one-out image cross-correlation: each exposure's red image (destriped) vs the mean of the
others, all resampled onto one grid. Measures residual mis-registration using ALL sky structure,
not just one pinned source. Optional --shift as in s2_dithers."""
import sys, json, argparse, dataclasses, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import map_coordinates, gaussian_filter
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.coadd import combine_image
from llamas_pyjamas.Combine.spectrum import _subselect
ap = argparse.ArgumentParser(); ap.add_argument('target'); ap.add_argument('--shift', nargs=3); ap.add_argument('--tag', default='')
a = ap.parse_args(); t = a.target
with fits.open(cube_path(t, 'red')) as h:
    center = (h[0].header['CRVAL1'], h[0].header['CRVAL2'])
sr = build_super_rss(rss_paths(t, 'red'), channels=['red'])
ids = [e.exposure_id[18:] for e in sr.exposures]
if a.shift:
    k = [i for i, e in enumerate(ids) if a.shift[0] in e][0]
    st = sr.channels['red']; sel = st.exposure == k
    st.ra[sel] += float(a.shift[1]) / 3600 / np.cos(np.radians(st.dec[sel].mean())); st.dec[sel] += float(a.shift[2]) / 3600
comb = combine_image(sr, 7000, 9500, channels=['red'], center=center)
ny, nx = comb.data.shape
yy, xx = np.mgrid[:ny, :nx]
wra, wdec = comb.wcs.pixel_to_world_values(xx, yy)
stack = []
for i in range(len(ids)):
    one = dataclasses.replace(sr, channels={'red': _subselect(sr.channels['red'], sr.channels['red'].exposure == i)})
    im = combine_image(one, 7000, 9500, channels=['red'], center=center)
    px, py = im.wcs.world_to_pixel_values(wra, wdec)
    d = map_coordinates(np.nan_to_num(im.data, nan=np.nan), [py, px], order=1, cval=np.nan)
    stack.append(d)
stack = np.array(stack)
def clean(d):
    d = d - np.nanmedian(d, axis=0, keepdims=True)              # vertical stripes (constant along Dec)
    d = d - gaussian_filter(np.nan_to_num(d), 6)                 # large-scale background
    return d
rows = []
for i, e in enumerate(ids):
    ref = np.nanmean(np.delete(stack, i, axis=0), axis=0)
    A, B = clean(stack[i]), clean(ref)
    m = np.isfinite(A) & np.isfinite(B)
    A = np.where(m, A, 0); B = np.where(m, B, 0)
    clipA = np.nanpercentile(abs(A[m]), 99.5); clipB = np.nanpercentile(abs(B[m]), 99.5)
    A = np.clip(A, -clipA, clipA); B = np.clip(B, -clipB, clipB)
    F = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(A) * np.conj(np.fft.rfft2(B)), s=A.shape))
    cy, cx = ny // 2, nx // 2
    win = F[cy - 12:cy + 13, cx - 12:cx + 13]
    py_, px_ = np.unravel_index(np.argmax(win), win.shape)
    def parab(v):
        return 0.5 * (v[0] - v[2]) / (v[0] - 2 * v[1] + v[2]) if v[0] - 2 * v[1] + v[2] != 0 else 0
    sy = py_ - 12 + (parab(win[py_ - 1:py_ + 2, px_]) if 0 < py_ < 24 else 0)
    sx = px_ - 12 + (parab(win[py_, px_ - 1:px_ + 2]) if 0 < px_ < 24 else 0)
    contrast = (win.max() - np.median(win)) / (np.std(win) + 1e-30)
    # +x pixel = -RA (CDELT1 < 0); pixel = 0.5"
    rows.append(dict(exp=e, dra=float(-sx * 0.5), ddec=float(sy * 0.5), contrast=float(contrast)))
    print('  %s  shift vs others: dRA %+.2f" dDec %+.2f"   (peak contrast %.1f)' % (e, -sx * 0.5, sy * 0.5, contrast))
json.dump(rows, open(f'{scr_dir(t)}/xcorr{a.tag}.json', 'w'), indent=1)
