import sys, json, numpy as np
S, tag = sys.argv[1], sys.argv[2]
fakes = json.load(open(f'{S}/cands_{tag}.json'))['fakes']
pos = json.load(open(f'{S}/regions_{tag}.json'))['pos']
src = np.load(f'{S}/srcmask.npy')
n_ok = n_tot = 0
for fk in fakes:
    if src[int(round(fk['y'])), int(round(fk['x']))]:
        continue                                       # fell on a masked continuum source
    n_tot += 1
    hit = [c for c in pos if abs(c['lam'] - fk['lam']) < 4 and np.hypot(c['x'] - fk['x'], c['y'] - fk['y']) < 2.5]
    n_ok += bool(hit)
    print('   %7.1f A  (%2.0f,%2.0f)  %s' % (fk['lam'], fk['x'], fk['y'], 'S/N %.1f' % hit[0]['snr'] if hit else '-'))
print(f'{tag}: recovered {n_ok}/{n_tot}')
