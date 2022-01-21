"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.

Usage:
    publication_kippenhahn.py <root_dir>

"""
import re
from collections import OrderedDict
import h5py
from mpi4py import MPI
from docopt import docopt
args = docopt(__doc__)
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.optimize as sop
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import logging
logger = logging.getLogger(__name__)

import dedalus.public as de
from plotpal.file_reader import SingleTypeReader, match_basis
plt.style.use('./apj.mplstyle')

import palettable.colorbrewer.qualitative as bqual

root_dir    = '{}/{}'.format(args['<root_dir>'], 'top_cz')
logger.info("reading data from {}".format(root_dir))


with h5py.File('{:s}/data_top_cz.h5'.format(root_dir), 'r') as f:
    times = f['times'][()]
    write_nums = f['write_nums'][()]
    L_d002s = f['L_d002s'][()]
    L_d05s = f['L_d05s'][()]
    L_d080s = f['L_d080s'][()]
    N2_switch = f['N2_switch'][()]
    cz_bound = f['cz_bound'][()]
    cz_bound_schwarz = f['cz_bound_schwarz'][()]
    oz_bound = f['oz_bound'][()]
    fconv2 = f['fconv2'][()]
    N2max = f['N2max'][()]
    S_measured = f['S_measured'][()]
    yL_switch = f['yL_switch'][()]
    yS_switch = f['yS_switch'][()]

Lz = 3
#kippenhahn fig
colors = bqual.Dark2_4.mpl_colors
GREEN  = colors[0]
ORANGE = colors[1]
PURPLE = colors[2]
PINK   = colors[3]
kfig = plt.figure(figsize=(3.25, 2))
ax = kfig.add_subplot(1,1,1)
print(yS_switch, yL_switch)
plt.fill_between(times, yS_switch, Lz*np.ones_like(times), facecolor=PURPLE, alpha=0.4)
plt.fill_between(times, yL_switch, yS_switch, facecolor=GREEN, alpha=0.4)
plt.fill_between(times, np.zeros_like(times), yL_switch, facecolor=ORANGE, alpha=0.4)
plt.fill_between(times, yL_switch, oz_bound, facecolor=PINK, alpha=0.3)
plt.plot(times, yS_switch, c=GREEN, lw=2, zorder=1)
plt.plot(times, yL_switch, c=ORANGE, lw=2)
plt.fill_between(times, yL_switch, oz_bound, hatch='//', edgecolor='k', facecolor="none", zorder=2)
#    plt.plot(times, N2_switch, c='k')
#plt.plot(times, L_d080s, c='k', ls='--')
end_entrainment = times[yS_switch > yL_switch*1.01][-1]/times.max()
x_max = times.max()
if end_entrainment < 0.5:
    x_max = end_entrainment*times.max()*2
    plt.xlim(0, x_max)
    end_entrainment = 0.5
len_overshoot   = (1-end_entrainment)
plt.ylim(0, Lz)
plt.xlabel('time')
plt.ylabel('z')


arrow_kwargs = { 'edgecolor' : 'k', 'facecolor' : 'k', 'width' : 0.001, 'head_width' : 0.007, 'head_length': 0.003}
arrow_room = end_entrainment/20
plt.arrow(x=end_entrainment/2, y=2.9/3, dx=end_entrainment/2-arrow_room, dy=0, transform=ax.transAxes, **arrow_kwargs)
plt.arrow(x=end_entrainment/2, y=2.9/3, dx=-end_entrainment/2+arrow_room, dy=0, transform=ax.transAxes, **arrow_kwargs)
plt.arrow(x=end_entrainment + len_overshoot/2, y=2.9/3, dx=len_overshoot/2-arrow_room, dy=0, transform=ax.transAxes, **arrow_kwargs)
plt.arrow(x=end_entrainment + len_overshoot/2, y=2.9/3, dx=-len_overshoot/2+arrow_room, dy=0, transform=ax.transAxes, **arrow_kwargs)
plt.text(x=end_entrainment/2, y=2.65/3, s='entrainment', ha='center', transform=ax.transAxes)
plt.text(x=end_entrainment + len_overshoot/2, y=2.65/3, s='overshoot', ha='center', transform=ax.transAxes)




x_line = end_entrainment*np.ones(100)
y_line = np.linspace(2.83, 2.97, 100)/3
plt.plot(x_line, y_line, c='k', lw=0.5, transform=ax.transAxes)



plt.text(x=0.02+100/times.max(), y=-0.03 +yL_switch[times > 100][0]/Lz, s=r'$\mathcal{Y}_{\rm{L}} = 0$', ha='left', transform=ax.transAxes, fontsize = 9, rotation=23, color=ORANGE)
plt.text(x=0.01, y=-0.08 +yS_switch[times > 50][0]/Lz, s=r'$\mathcal{Y}_{\rm{S}} = 0$', ha='left', transform=ax.transAxes, fontsize = 9, color=GREEN)
plt.savefig('{:s}/{:s}.png'.format(root_dir, 'kippenhahn'), dpi=400, bbox_inches='tight')
plt.savefig('{:s}/{:s}.pdf'.format(root_dir, 'kippenhahn'), dpi=400, bbox_inches='tight')

