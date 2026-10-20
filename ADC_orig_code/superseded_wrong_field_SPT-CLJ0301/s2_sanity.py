import sys, numpy as np
from astropy.io import fits
S = sys.argv[1]
for c in ['blue', 'green', 'red']:
    with fits.open(f'{S}/data/SPT-CLJ0301-5142_cube_{c}.fits') as h:
        f = h[0].data; v = h['VAR'].data; nexp = h['NEXP'].data; cov = h['COVERAGE'].data
        wl = h['WAVELENGTH'].data.field(0)
        print(c, 'shape', f.shape, 'wl', wl[0], wl[-1], 'dwl', np.median(np.diff(wl)))
        print('   NaN frac flux %.3f var %.3f' % (np.isnan(f).mean(), np.isnan(v).mean()))
        print('   NEXP values', np.unique(nexp, return_counts=True))
        print('   COVERAGE values', np.unique(cov, return_counts=True)[0][:10])
        good = nexp == 3
        med = np.nanmedian(f[:, good], axis=1)
        sig = np.sqrt(np.nanmedian(v[:, good], axis=1))
        for lo, hi in [(wl[0], wl[0] + 200), (3600, 3700), (4000, 4100), (wl[-1] - 200, wl[-1])]:
            s = (wl >= lo) & (wl <= hi)
            print('   %6.0f-%6.0f  median SB %.2e  median sigma %.2e' % (lo, hi, np.nanmedian(med[s]), np.nanmedian(sig[s])))
with fits.open(f'{S}/data/LLAMAS_2026-10-15_08-33-54.7_RSS_blue.fits') as h:
    w = h['WAVE'].data
    print('blue RSS WAVE range', np.nanmin(w[w > 0]), np.nanmax(w))
    print('FLAM BUNIT', h['FLAM'].header.get('BUNIT'), 'SENSFILE', h['FLAM'].header.get('SENSFILE'), 'EXTCORR', h['FLAM'].header.get('EXTCORR'))
    fm = h['FIBERMAP'].data
    print('FIBERMAP cols', fm.columns.names)
