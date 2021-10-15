"""
Script for plotting a movie of the evolution of a 2D dedalus simulation.  
This script plots time evolution of the fields specified in 'fig_type'

Usage:
    plot_slices.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --static_cbar                       If flagged, don't evolve the cbar with time
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 4]
    --row_inch=<in>                     Number of inches / row [default: 2]

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T - horiz_avg(T), w
                                        [default: 1]
"""
import numpy as np
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter
import logging
logger = logging.getLogger(__name__)


start_fig = int(args['--start_fig'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)
start_file  = int(args['--start_file'])

root_dir    = args['<root_dir>']
if root_dir is None:
    logger.error('No dedalus output dir specified, exiting')
    import sys
    sys.exit()
fig_name   = args['--fig_name']

plotter = SlicePlotter(root_dir, file_dir='slices', fig_name=fig_name, start_file=start_file, n_files=n_files)
convergence_plotter = SlicePlotter(root_dir, file_dir='slices', fig_name=fig_name, start_file=start_file, n_files=n_files)

plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']), 'padding' : 100 }
if int(args['--fig_type']) == 1:
    plotter.setup_grid(1, 3, **plotter_kwargs)
    fnames = [  (("T",),         {'remove_x_mean' : True, 'label' : 'T - horiz_avg(T)'}),
            (('mu',),        {'cmap': 'Greens', 'vmin' : 0, 'vmax' : 1}),
                (('w',),         {'cmap': 'PuOr_r'})]

for tup in fnames:
    plotter.add_colormesh(*tup[0], **tup[1])

plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
