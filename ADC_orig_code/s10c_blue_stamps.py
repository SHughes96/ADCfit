import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import *
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.coadd import combine_image
t = 'SPT-CLJ0111-5637'
B = json.load(open(f'{scr_dir(t)}/blue_choice.json')); ra0, dec0 = json.load(open(f'{scr_dir(t)}/staralign_shifts.json'))['anchor']
base = build_super_rss(rss_paths(t, 'blue'), channels=['blue'])
ids = [e.exposure_id[18:] for e in base.exposures]
st = base.channels['blue']; ro, do = st.ra.copy(), st.dec.copy()
recs = {'header (as combined)': {e: (0, 0) for e in ids}, **B['recipes']}
fig, ax = plt.subplots(2, len(recs), figsize=(3.3 * len(recs), 6.6))
for j, (name, rec) in enumerate(recs.items()):
    st.ra[:] = ro; st.dec[:] = do
    for k, e in enumerate(ids):
        sel = st.exposure == k
        st.ra[sel] += rec[e][0] / 3600 / np.cos(np.radians(dec0)); st.dec[sel] += rec[e][1] / 3600
    for row, (lo, hi) in enumerate([(3800, 4200), (4200, 4600)]):
        img = combine_image(base, lo, hi, channels=['blue'], center=(ra0, dec0))
        x0, y0 = [int(round(float(q))) for q in img.wcs.world_to_pixel_values(ra0, dec0)]
        stamp = img.data[y0 - 10:y0 + 11, x0 - 10:x0 + 11]
        stamp = stamp - np.nanmedian(stamp)
        ax[row, j].imshow(stamp, origin='lower', cmap='magma', extent=(-5.25, 5.25, -5.25, 5.25), vmin=0, vmax=np.nanmax(stamp))
        ax[row, j].plot(0, 0, 'c+', ms=14)
        ax[row, j].set_title(f'{name}\n{lo}-{hi} Å', fontsize=8)
fig.suptitle('SPT-CLJ0111-5637 blue: field star (cyan + = Gaia) under each alignment recipe; axes in arcsec (E left)', fontsize=10)
plt.tight_layout(); plt.savefig(f'{scr_dir(t)}/blue_recipes.png', dpi=90)
