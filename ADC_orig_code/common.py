"""Shared paths/helpers for the two-target Lyα + dither analysis."""
import glob, os
import numpy as np
from astropy.io import fits

ROOT = '/Users/slh/Documents/Projects/michael_reduction'
OUT = os.path.join(ROOT, 'lya_analysis')
# Working directory for intermediates (S/N cubes, JSON results). Override with LYA_WORK.
SCR = os.environ.get('LYA_WORK', os.path.join(OUT, 'work'))
TARGETS = ['SPT-CLJ0111-5637', 'SPT-CLJ0158-5500']


VARIANT = os.environ.get('LYA_VARIANT', '')       # 'staraligned' -> use the star-aligned J0111 cubes


def cube_path(target, channel='blue'):
    if VARIANT == 'staraligned' and target == 'SPT-CLJ0111-5637':
        return os.path.join(OUT, target, 'star_aligned', f'{target}_staraligned_cube_{channel}.fits')
    return os.path.join(ROOT, 'combined', f'{target}_cube_{channel}.fits')


def exposure_shifts(target):
    """Per-exposure (dE, dN) arcsec to apply to the CURRENT blue fibre maps so they match the cube."""
    if VARIANT == 'staraligned' and target == 'SPT-CLJ0111-5637':
        import json
        sh = json.load(open(os.path.join(SCR, target, 'staralign_shifts.json')))['shifts']
        return {e: tuple(v['blue']) for e, v in sh.items()}
    return {}


def rss_paths(target, channel='blue'):
    """The RSS files a delivered cube was built from (RSSFILn -> same exposure, this channel)."""
    h = fits.getheader(cube_path(target, 'blue'))
    out = []
    for i in range(1, int(h['NRSSFILE']) + 1):
        stem = os.path.basename(h[f'RSSFIL{i}']).split('_RSS_')[0]
        out.append(os.path.join(ROOT, 'extractions', f'{stem}_RSS_{channel}.fits'))
    return out


def exp_id(path):
    return os.path.basename(path)[18:28]


def out_dir(target):
    if VARIANT == 'staraligned' and target == 'SPT-CLJ0111-5637':
        return os.path.join(OUT, target, 'star_aligned')
    return os.path.join(OUT, target)


def scr_dir(target):
    d = os.path.join(SCR, target, VARIANT) if (VARIANT and target == 'SPT-CLJ0111-5637') else os.path.join(SCR, target)
    os.makedirs(d, exist_ok=True)
    return d
