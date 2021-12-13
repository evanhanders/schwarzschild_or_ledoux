"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.

Usage:
    find_top_cz.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: top_cz]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]
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

resolution_regex_2d = re.compile('(.*)x(.*)')
resolution_regex_3d = re.compile('(.*)x(.*)x(.*)')

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)
start_file  = int(args['--start_file'])
start_fig = int(args['--start_fig']) - 1

root_dir    = args['<root_dir>']
if root_dir is None:
    logger.error('No dedalus output dir specified, exiting')
    import sys
    sys.exit()
fig_name   = args['--fig_name']
logger.info("reading data from {}".format(root_dir))

#therm_mach2 = float(root_dir.split("Ma2t")[-1].split("_")[0])
for string in root_dir.split('_'):
    val = string.split('/')[0]
    if 'Pe' in val:
        Pe_in = float(val.split('Pe')[-1])
    elif 'invR' in val:
        inv_R_in = float(val.split('invR')[-1])
    elif resolution_regex_3d.match(val):
        res_strs = val.split('x')
        nx, ny, nz = [int(s) for s in res_strs]
    elif resolution_regex_2d.match(val):
        res_strs = val.split('x')
        nx, nz = [int(s) for s in res_strs]
Lz = 3
z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=1)
domain = de.Domain([z_basis,], grid_dtype=np.float64, mesh=None, comm=MPI.COMM_SELF)
dense_scales=20
z_dense = domain.grid(0, scales=dense_scales)

rolled_reader = SingleTypeReader(root_dir, 'profiles', fig_name, roll_writes=50, start_file=start_file, n_files=n_files, distribution='even-write')
MPI.COMM_WORLD.barrier()
readerOne = SingleTypeReader(root_dir, 'profiles', fig_name, start_file=start_file, n_files=n_files, distribution='single', global_comm=MPI.COMM_SELF)
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
axs = [ax1, ax2]

data_cube = []

grad_field = domain.new_field()
grad_integ_field = domain.new_field()
grad_mu_field = domain.new_field()
grad_mu_integ_field = domain.new_field()
grad_ad_field = domain.new_field()
grad_rad_field = domain.new_field()
grad_rad_integ_field = domain.new_field()
delta_grad_field = domain.new_field()
dedalus_fields = [grad_field, grad_integ_field, grad_ad_field, grad_rad_field, grad_rad_integ_field, \
                  grad_mu_field, grad_mu_integ_field, delta_grad_field]

tasks, first_tasks = OrderedDict(), OrderedDict()

bases_names = ['z',]
fields = ['T', 'T_z', 'mu_z', 'bruntN2', 'bruntN2_structure', 'bruntN2_composition',\
          'F_rad', 'F_conv', 'F_rad_mu', 'F_conv_mu']
first_fields = ['T_rad_z', 'T_ad_z', 'F_rad', 'flux_of_z', 'T_z']
if not rolled_reader.idle:
    while readerOne.writes_remain():
        first_dsets, first_ni = readerOne.get_dsets(first_fields)
        for k in first_fields:
            first_tasks[k] = first_dsets[k][first_ni].squeeze()
        grad_rad = -first_tasks['T_rad_z']
        grad_ad  = -first_tasks['T_ad_z']
        system_flux = first_tasks['flux_of_z'].squeeze()[-1] #get flux at top of domain
        system_flux_prof = first_tasks['flux_of_z'].squeeze() / system_flux
        F_cond0 = first_tasks['F_rad'].squeeze()

        for f in dedalus_fields:
            f.set_scales(1)
        grad_ad_field['g'] = grad_ad
        delta_grad_field['g'] = grad_ad - grad_rad
        grad_rad_field['g'] = grad_rad
        grad_rad_field.antidifferentiate('z', ('left', 0), out=grad_rad_integ_field)
        for f in dedalus_fields:
            f.set_scales(dense_scales, keep_data=True)
        #Find Ls
        Ls = z_dense[delta_grad_field['g'] < 0][-1]
        break

    while rolled_reader.writes_remain():
        dsets, ni = rolled_reader.get_dsets(fields)
        time_data = dsets[fields[0]].dims[0]
        z = match_basis(dsets[fields[0]], 'z')

        for k in fields:
            tasks[k] = dsets[k][ni,:].squeeze()
            if len(tasks[k].shape) == 1:
                tasks[k] = np.expand_dims(tasks[k], axis=0)

        F_cond = tasks['F_rad']
        F_conv = tasks['F_conv']
        T      = tasks['T']
        T_z    = tasks['T_z']
        mu_z   = tasks['mu_z']
        grad = -T_z[0,:]
        grad_mu = -mu_z[0,:]*inv_R_in

        for f in [grad_field, grad_mu_field]:
            f.set_scales(1, keep_data=True)
        grad_field['g'] = -T_z
        grad_field.antidifferentiate('z', ('left', 0), out=grad_integ_field)
        grad_mu_field['g'] = -mu_z*inv_R_in
        grad_mu_field.antidifferentiate('z', ('left', 0), out=grad_mu_integ_field)
        for f in dedalus_fields:
            f.set_scales(dense_scales, keep_data=True)

        #departure from grad_ad: 0.1, 0.5, 0.9
        departures = []
        mu_z_max = grad_mu_field['g'].max()
        for departure_factor in [0.1, 0.5, 0.98]:
            mu_z_departure = grad_mu_field['g'] > departure_factor*mu_z_max
            if np.sum(mu_z_departure) > 0:
                z_mu_departure = z_dense[mu_z_departure].min()
            else:
                z_mu_departure = np.nan
            departures.append(z_mu_departure)

        L_d01 = departures[0]
        L_d05 = departures[1]
        L_d09 = departures[2]

        data_list = [time_data['sim_time'][ni], time_data['write_number'][ni], L_d01, L_d05, L_d09]
        data_cube.append(data_list)

        ax1.plot(z,  tasks['bruntN2_structure'][0,:],   c='b', label=r'$N^2_{\rm{structure}}$')
        ax1.plot(z,  tasks['bruntN2_composition'][0,:], c='g', label=r'$N^2_{\rm{composition}}$')
        ax1.plot(z,  tasks['bruntN2'][0,:], c='k', label=r'$N^2$')
        ax1.plot(z, -tasks['bruntN2'][0,:], c='k', ls='--')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left')
        ax1.set_ylabel(r'$N^2$')
        ax1.set_ylim(1e-2, np.max(tasks['bruntN2'])*2)

        ax2.axhline(0, c='k')
        ax2.plot(z, grad - grad_ad,     label=r'$\nabla$', c='b')
        ax2.plot(z, grad_rad - grad_ad, label=r'$\nabla_{\rm{rad}}$', c='r')
        ax2.plot(z, grad_mu/inv_R_in, label=r'$\nabla_{\rm{mu}}\cdot R_{0}$', c='g')
        y_min = np.abs(-grad_rad[z.flatten() > 1]).min()
        deltay = np.abs(grad_ad).max() - y_min
        ax2.set_ylim(- 2, 2)
        ax2.legend(loc='upper right')
        ax2.set_ylabel(r'$\nabla - \nabla_{\rm{ad}}$')

        for ax in axs:
            ax.set_xlabel('z')
            ax.set_xlim(z.min(), z.max())
            ax.axvline(L_d01, c='red')
            ax.axvline(L_d05, c='k')
            ax.axvline(L_d09, c='red')

        plt.suptitle('sim_time = {:.2f}'.format(time_data['sim_time'][ni]))

        fig.savefig('{:s}/{:s}_{:06d}.png'.format(rolled_reader.out_dir, fig_name, start_fig+time_data['write_number'][ni]), dpi=int(args['--dpi']), bbox_inches='tight')
        for ax in axs:
            ax.cla()
    data_cube = np.array(data_cube)
    write_nums = np.array(data_cube[:,1], dtype=int)
buffer = np.zeros(1, dtype=int)
if rolled_reader.idle:
    buffer[0] = 0
else:
    buffer[0] = int(write_nums.max())
rolled_reader.reader.global_comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)
global_max_write = buffer[0]
if rolled_reader.idle:
    buffer[0] = int(1e6)
else:
    buffer[0] = int(write_nums.min())
rolled_reader.reader.global_comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MIN)
global_min_write = buffer[0]
if rolled_reader.idle:
    buffer[0] = 0
else:
    buffer[0] = data_cube.shape[1]
rolled_reader.reader.global_comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)
num_fields = buffer[0]


global_data = np.zeros((int(global_max_write - global_min_write + 1), num_fields))
if not rolled_reader.idle:
    write_nums -= int(global_min_write)
    global_data[write_nums,:] = data_cube
rolled_reader.reader.global_comm.Allreduce(MPI.IN_PLACE, global_data, op=MPI.SUM)
times      = global_data[:,0]
write_nums = global_data[:,1]
L_d01s     = global_data[:,2]
L_d05s     = global_data[:,3]
L_d09s     = global_data[:,4]

if rolled_reader.reader.global_comm.rank == 0:
    fig = plt.figure()
    plt.plot(times, L_d01s - Ls, c='k', label=r'10% of max $\nabla\mu$')
    plt.plot(times, L_d05s - Ls, c='red', label=r'50% of max $\nabla\mu$')
    plt.plot(times, L_d09s - Ls, c='k', label=r'98% of max $\nabla\mu$')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel(r'$\delta_p$')
    fig.savefig('{:s}/{:s}.png'.format(rolled_reader.out_dir, 'trace_top_cz'), dpi=400, bbox_inches='tight')
    with h5py.File('{:s}/data_top_cz.h5'.format(rolled_reader.out_dir), 'w') as f:
        f['times'] = times     
        f['write_nums'] = write_nums
        f['L_d01s'] = L_d01s    
        f['L_d05s'] = L_d05s    
        f['L_d09s'] = L_d09s    
