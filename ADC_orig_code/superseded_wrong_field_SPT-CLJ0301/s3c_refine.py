"""Second pass: per-spaxel noise normalisation, continuum-source mask, positive vs negative
region counts (the negative side is the false-positive benchmark)."""
import sys, json, numpy as np
from astropy.io import fits
from scipy.ndimage import label, find_objects, binary_dilation, binary_erosion
S, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'real')
snr = np.load(f'{S}/snr_{tag}.npy').astype(float)
with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_blue.fits') as h:
    wl = h['WAVELENGTH'].data.field(0); nexp = h['NEXP'].data
snr[~np.isfinite(snr)] = 0
core = binary_erosion(nexp == 3, iterations=3)
zgood = (wl > 3300) & (wl < 4780)                       # drop sensfunc edges
snr[~zgood] = 0
# per-spaxel robust std along wavelength
q = np.where(snr != 0, snr, np.nan)
sp = 1.4826 * np.nanmedian(np.abs(q - np.nanmedian(q, axis=0)), axis=0)
sp[~np.isfinite(sp) | (sp == 0)] = 1
snr2 = snr / sp[None]
# continuum-source mask from the white light (green: best S/N continuum), dilated by 4 px
wlimg = np.load(f'{S}/wl_green.npy')
wv = wlimg[core]; med = np.nanmedian(wv); s = 1.4826 * np.nanmedian(np.abs(wv - med))
src = np.nan_to_num((wlimg - med) / s) > 5
src = binary_dilation(src, iterations=4)
np.save(f'{S}/srcmask.npy', src)
use = core & ~src
snr2[:, ~use] = 0
np.save(f'{S}/snr2_{tag}.npy', snr2.astype(np.float32))
print('per-spaxel noise factor percentiles', np.percentile(sp[core], [5, 50, 95]))
print('snr2 std over used voxels', np.std(snr2[zgood][:, use]))
print('masked continuum-source spaxels', int(src.sum()))
def regions(cube, thr, minvox):
    lab, n = label(cube > thr)
    out = []
    for i, sl in enumerate(find_objects(lab), 1):
        m = lab[sl] == i
        if m.sum() < minvox: continue
        pk = np.unravel_index(np.argmax(np.where(m, cube[sl], -1e9)), m.shape)
        z0, y0, x0 = [a.start + b for a, b in zip(sl, pk)]
        out.append(dict(lam=float(wl[z0]), y=int(y0), x=int(x0), snr=float(cube[z0, y0, x0]),
                        nvox=int(m.sum()), area=int(m.any(axis=0).sum()), dz=int(sl[0].stop - sl[0].start)))
    return sorted(out, key=lambda c: -c['snr'])
for thr in [4.5, 5, 6, 7]:
    pos = regions(snr2, thr, 10); neg = regions(-snr2, thr, 10)
    print(f'thr {thr}: positive regions {len(pos):4d}   negative regions {len(neg):4d}')
pos = regions(snr2, 5, 10); neg = regions(-snr2, 5, 10)
print('\nTop positive (S/N>5, >=10 voxels):')
for c in pos[:30]:
    print('  lam %7.1f  z_Lya %.3f  (x,y)=(%2d,%2d)  S/N %5.1f  nvox %4d  area %3d  dlam_px %3d' %
          (c['lam'], c['lam'] / 1215.67 - 1, c['x'], c['y'], c['snr'], c['nvox'], c['area'], c['dz']))
print('\nTop negative:')
for c in neg[:10]:
    print('  lam %7.1f  (x,y)=(%2d,%2d)  S/N %5.1f  nvox %4d' % (c['lam'], c['x'], c['y'], c['snr'], c['nvox']))
json.dump(dict(pos=pos, neg=neg), open(f'{S}/regions_{tag}.json', 'w'), indent=1)
