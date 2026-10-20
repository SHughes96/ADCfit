"""Honest limits: 50% completeness from the injection recovery fractions (band average), with the
wavelength shape taken from the measured noise curve. Writes limits_c50.json per field."""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import SCR
for t, sub in [('SPT-CLJ0158-5500', ''), ('SPT-CLJ0111-5637', 'staraligned'), ('SPT-CLJ0111-5637', '')]:
    d = json.load(open(f'{SCR}/{t}/{sub}/search.json'.replace('//', '/')))
    F = np.array(sorted(float(k) for k in d['inj'])); frac = np.array([np.mean([r['hit'] for r in d['inj'][str(f) if str(f) in d['inj'] else repr(f)]]) for f in F])
    F50 = 10 ** np.interp(0.5, frac, np.log10(F))
    F80 = 10 ** np.interp(0.8, frac, np.log10(F)) if frac.max() >= 0.8 else np.nan
    lim = np.array(d['lim']); band = (lim[:, 0] > 3550) & (lim[:, 0] < 4750)
    shape = lim[:, 1] / np.nanmedian(lim[band, 1])
    c50 = F50 * shape
    print(f'{t} {sub or "delivered"}: recovery {dict(zip([f"{f:.1e}" for f in F], np.round(frac, 2)))} -> 50% completeness {F50:.1e} (band average), 80%: {F80:.1e}')
    for l in [3650, 4000, 4300]:
        print('    λ %d (z=%.2f): 50%%-completeness line flux %.1e   (old optimistic "5σ" %.1e)' % (l, l / 1215.67 - 1, c50[np.argmin(abs(lim[:, 0] - l))], lim[np.argmin(abs(lim[:, 0] - l)), 1]))
    json.dump(dict(F50=F50, F80=F80, lam=lim[:, 0].tolist(), c50=c50.tolist(), frac=frac.tolist(), F=F.tolist()), open(f'{SCR}/{t}/{sub}/limits_c50.json'.replace('//', '/'), 'w'))
