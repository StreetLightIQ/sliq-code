"""Schematic light map for five example lamps (top view and centre-line profile). Uses no data.
    python -m sliq.figures.lightmap_schematic
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sliq.common import paths

OUT = os.path.join(paths.RUNS, 'figures', 'lightmap_schematic')
H = 9.0                    # mounting height (m)
LIT, DARK = 0.2, 0.05      # thresholds on E / E_ref
LAMPS = [dict(s=14, x=-4.5, I=1, n=8, label='LED\n$n=8$'),
         dict(s=40, x=4.5, I=1, n=8, label='lamp $i$\n(LED)'),
         dict(s=67, x=-4.5, I=1, n=2, label='sodium\n$n=2$'),
         dict(s=94, x=4.5, I=0, n=8, label='Off\n(undetected)'),
         dict(s=120, x=4.5, I=0.5, n=8, label='Dim\n($I=0.5$)')]
FOCUS = 1                  # lamp whose loss-if-off is hatched
COLOURS = dict(dark='#1d2440', mid='#8a7f6a', lit='#f7d774')


def set_fonts():
    fonts = glob.glob('/usr/share/fonts/opentype/linux-libertine/LinLibertine_R*.otf')
    for p in fonts:
        fm.fontManager.addfont(p)
    plt.rcParams.update({'font.size': 8, 'hatch.linewidth': 0.8})
    if fonts:
        plt.rcParams.update({'font.family': 'Linux Libertine O', 'mathtext.fontset': 'custom',
                             'mathtext.rm': 'Linux Libertine O', 'mathtext.it': 'Linux Libertine O:italic'})


def light_map(step=0.1):
    s = np.arange(0, 136 + step / 2, step)
    x = np.arange(-7, 7 + step / 2, step)
    S, X = np.meshgrid(s, x, indexing='ij')
    per = []
    for l in LAMPS:
        d2 = (S - l['s']) ** 2 + (X - l['x']) ** 2 + H * H
        per.append(l['I'] * (H / np.sqrt(d2)) ** (l['n'] + 1) / d2)
    per = np.array(per) * H ** 2          # E_ref = 1 / H^2
    return s, x, S, X, per


def dark_runs(s, dark):
    runs, start = [], None
    for i, v in enumerate(np.r_[dark, False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((s[start], s[i - 1]))
            start = None
    return runs


def main():
    set_fonts()
    s, x, S, X, per = light_map()
    E = per.sum(0)
    lit, dark = E >= LIT, E < DARK
    zone = np.where(lit, 2, np.where(dark, 0, 1))
    on = [k for k, l in enumerate(LAMPS) if l['I'] > 0]
    owner = np.array(on)[np.argmax(per[on], 0)]
    loss = lit & (E - per[FOCUS] < LIT)
    centre = np.argmin(abs(x))
    runs = dark_runs(s, dark[:, centre])
    longest = max(runs, key=lambda r: r[1] - r[0])

    fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 2.75), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    a.imshow(zone.T, origin='lower', extent=[s[0], s[-1], x[0], x[-1]], aspect='auto', interpolation='nearest',
             cmap=ListedColormap([COLOURS['dark'], COLOURS['mid'], COLOURS['lit']]), vmin=0, vmax=2)
    a.contourf(S, X, loss.astype(float), levels=[0.5, 1.5], colors='none', hatches=['////'])
    a.contour(S, X, loss.astype(float), levels=[0.5], colors='#b22222', linewidths=1.0)
    a.contour(S, X, owner.astype(float), levels=np.array(on[:-1]) + 0.5, colors='#7fd3ff', linewidths=1.4, linestyles='--')
    a.axhline(0, color='white', lw=0.5, alpha=0.6)
    for r in runs:
        a.plot(r, [0, 0], color='#ff3b3b', lw=3.2, solid_capstyle='butt')
    for l in LAMPS:
        if l['I'] > 0:
            a.plot(l['s'], l['x'], 'o', ms=6, mfc='white', mec='black', mew=0.9)
        else:
            a.plot(l['s'], l['x'], 'X', ms=7, mfc='#ff3b3b', mec='black', mew=0.6)
        a.text(l['s'] + 2.0, l['x'], l['label'], va='center', ha='left', fontsize=7, color='white', linespacing=0.95,
               bbox=dict(fc=COLOURS['dark'], ec='none', alpha=0.8, pad=0.8))
    a.set_ylim(-7, 7)
    a.set_yticks([-7, 0, 7])
    a.set_ylabel('across (m)')
    a.text(-0.5, 7.6, '(a) Light map seen from above', fontsize=8.5, va='bottom')

    ec = E[:, centre]
    for r in runs:
        b.axvspan(r[0], r[1], color='#ff3b3b', alpha=0.22, lw=0)
    b.fill_between(s, 0, ec, where=lit[:, centre], color=COLOURS['lit'], lw=0)
    b.plot(s, ec, color='black', lw=1.1)
    b.axhline(LIT, color='#c08a00', lw=0.9, ls='--')
    b.axhline(DARK, color='#3050a0', lw=0.9, ls='--')
    b.text(136.8, LIT, '$\\lambda=0.2$ (lit)', va='center', fontsize=7, color='#c08a00')
    b.text(136.8, DARK, '$\\mu=0.05$ (dark)', va='center', fontsize=7, color='#3050a0')
    b.annotate('', xy=(longest[0], 0.5), xytext=(longest[1], 0.5), arrowprops=dict(arrowstyle='<->', lw=0.8))
    b.text(sum(longest) / 2, 0.53, 'longest dark stretch %.0f m' % (longest[1] - longest[0]), ha='center', va='bottom', fontsize=7)
    b.set_ylim(0, 0.65)
    b.set_yticks([0, 0.2, 0.4, 0.6])
    b.set_ylabel('$E/E_{\\mathrm{ref}}$ on centre line')
    b.set_xlim(0, 136)
    b.set_xlabel('along the road (m)')
    b.text(-0.5, 0.68, '(b) Modelled illuminance along the road centre line', fontsize=8.5, va='bottom')

    handles = [mp.Patch(fc=COLOURS['lit'], ec='k', lw=0.4, label='lit ($E\\geq\\lambda E_{\\mathrm{ref}}$)'),
               mp.Patch(fc=COLOURS['mid'], ec='k', lw=0.4, label='in between'),
               mp.Patch(fc=COLOURS['dark'], ec='k', lw=0.4, label='dark ($E<\\mu E_{\\mathrm{ref}}$)'),
               plt.Line2D([], [], color='#4fb3e8', lw=1.4, ls='--', label='served-area boundary'),
               mp.Patch(fc='white', ec='#b22222', hatch='////', label='loss-if-off of lamp $i$'),
               plt.Line2D([], [], color='#ff3b3b', lw=3, label='dark stretch on centre line')]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.015),
               handlelength=1.8, columnspacing=1.0)
    plt.subplots_adjust(left=0.07, right=0.905, top=0.93, bottom=0.22, hspace=0.32)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + '.pdf')
    fig.savefig(OUT + '.png', dpi=220)
    print('saved', OUT + '.pdf')


if __name__ == '__main__':
    main()
