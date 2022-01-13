"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.

Usage:
    find_top_cz.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: paper_profiles]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --publication_fig
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

import palettable.colorbrewer.qualitative as bqual

import dedalus.public as de
from plotpal.file_reader import SingleTypeReader, match_basis

resolution_regex_2d = re.compile('(.*)x(.*)')
resolution_regex_3d = re.compile('(.*)x(.*)x(.*)')
resolution_regex_3d_compound = re.compile('(.*)x(.*)x(.*)-(.*)')

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

if args['--publication_fig']: #or True:
    n_files     = 5
else:
    n_files = np.inf
start_file  = int(args['--start_file']) - 2
start_fig = int(args['--start_fig']) - 1

root_dir    = args['<root_dir>']
if root_dir is None:
    logger.error('No dedalus output dir specified, exiting')
    import sys
    sys.exit()
fig_name   = args['--fig_name']
logger.info("reading data from {}".format(root_dir))

nz_top = None
#therm_mach2 = float(root_dir.split("Ma2t")[-1].split("_")[0])
for string in root_dir.split('_'):
    val = string.split('/')[0]
    if 'Pe' in val:
        Pe_in = float(val.split('Pe')[-1])
    elif 'invR' in val:
        inv_R_in = float(val.split('invR')[-1])
    elif 'mixedICs' in val:
        continue
    elif 'Lx' in val:
        continue
    elif resolution_regex_3d_compound.match(val):
        res_strs = val.split('x')
        nx = int(res_strs[0])
        ny = int(res_strs[1])
        nz, nz_top = [int(s) for s in res_strs[2].split('-')]
    elif resolution_regex_3d.match(val):
        res_strs = val.split('x')
        nx, ny, nz = [int(s) for s in res_strs]
    elif resolution_regex_2d.match(val):
        res_strs = val.split('x')
        nx, nz = [int(s) for s in res_strs]
Lz = 3
if nz_top is None:
    z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=1)
else:
    z_basis1 = de.Chebyshev('z', nz, interval=(0, 2.2), dealias=1)
    z_basis2 = de.Chebyshev('z', nz_top, interval=(2.2, Lz), dealias=1)
    z_basis = de.Compound('z', [z_basis1, z_basis2], dealias=1)
domain = de.Domain([z_basis,], grid_dtype=np.float64, mesh=None, comm=MPI.COMM_SELF)
dense_scales=20
z_dense = domain.grid(0, scales=dense_scales)
z = domain.grid(0, scales=1)

roll_writes = 50
rolled_reader = SingleTypeReader(root_dir, 'profiles', fig_name, roll_writes=roll_writes, start_file=start_file, n_files=n_files, distribution='even-write')
MPI.COMM_WORLD.barrier()
readerOne = SingleTypeReader(root_dir, 'profiles', fig_name, start_file=1, n_files=n_files, distribution='single', global_comm=MPI.COMM_SELF)
fig = plt.figure(figsize=(7.5,3))
axL_1 = fig.add_subplot(3,2,1)
axL_2 = fig.add_subplot(3,2,3)
axL_3 = fig.add_subplot(3,2,5)
axR_1 = fig.add_subplot(3,2,2)
axR_2 = fig.add_subplot(3,2,4)
axR_3 = fig.add_subplot(3,2,6)
axs = [axL_1, axL_2, axL_3, axR_1, axR_2, axR_3]
plt.subplots_adjust(top=0.9, bottom=0, hspace=0.05, wspace=0.1, left=0, right=1)

N2_structure_field = domain.new_field()
grad_field = domain.new_field()
grad_mu_field = domain.new_field()
grad_ad_field = domain.new_field()
grad_rad_field = domain.new_field()
delta_grad_field = domain.new_field()
freq2_conv_field = domain.new_field()
freq2_conv_field_enstrophy = domain.new_field()
buoyancy_field = domain.new_field()
buoyancy_field_schwarz = domain.new_field()
dz_buoyancy_field = domain.new_field()
dz_buoyancy_field_schwarz = domain.new_field()
KE_field = domain.new_field()
dedalus_fields = [N2_structure_field, grad_field, grad_mu_field, grad_ad_field, grad_rad_field,
                    delta_grad_field, freq2_conv_field, freq2_conv_field_enstrophy,
                    buoyancy_field, buoyancy_field_schwarz, dz_buoyancy_field, dz_buoyancy_field_schwarz,
                    KE_field]

def plot_fields(tasks, ax1, ax2, ax3):

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('z')
        ax.set_xlim(z.min(), z.max())

    T_z    = tasks['T_z']
    mu_z   = tasks['mu_z']
    grad = -T_z
    grad_mu = -mu_z*inv_R_in

    for f in dedalus_fields:
        f.set_scales(1, keep_data=True)
    grad_field['g'] = -T_z
    N2_structure_field['g'] = -(grad_field['g'] - grad_ad)
    grad_mu_field['g'] = -mu_z*inv_R_in
    freq2_conv_field['g'] = 2*tasks['KE'] #need to divide by L_d05^2 later.
    freq2_conv_field_enstrophy['g'] = tasks['enstrophy']
    dz_buoyancy_field['g'] = (T_z - (-grad_ad)) - mu_z * inv_R_in
    dz_buoyancy_field.antidifferentiate('z', ('left', 0), out=buoyancy_field)
    dz_buoyancy_field_schwarz['g'] = (T_z - (-grad_ad))
    dz_buoyancy_field_schwarz.antidifferentiate('z', ('left', 0), out=buoyancy_field_schwarz)
    KE_field['g'] = tasks['KE']
    for f in dedalus_fields:
        f.set_scales(dense_scales, keep_data=True)

    N2_structure = N2_structure_field['g']
    N2_composition = grad_mu_field['g']
    N2_tot = N2_structure + N2_composition

    y_schwarz = dz_buoyancy_field_schwarz['g']
    y_ledoux = dz_buoyancy_field['g']

    #departure from grad_ad: 0.1, 0.5, 0.9
    departures = []
    mu_z_max = grad_mu_field['g'].max()
    for departure_factor in [0.02, 0.5, 0.999]:
        mu_z_departure = grad_mu_field['g'] > departure_factor*mu_z_max
        if np.sum(mu_z_departure) > 0:
            z_mu_departure = z_dense[mu_z_departure].min()
        else:
            z_mu_departure = np.nan
        departures.append(z_mu_departure)

    L_d002 = departures[0]
    L_d05 = departures[1]
    L_d0999 = departures[2]

    #Find where RZ switches from thermally stable to compositionally stable
    N2_switch_height = z_dense[N2_structure < 0][-1]

    #Find edge of OZ by a crude measure.
#        oz_bound = z_dense[F_conv_mu_field['g'] > F_conv_mu_field['g'].max()*1e-1][-1]
    if KE_field['g'].max() > 0:
        oz_bound = z_dense[KE_field['g'] > KE_field['g'].max()*1e-1][-1]
    else:
        oz_bound = 0

    #point of neutral buoyancy is CZ edge.
    cz_bound = z_dense[(z_dense > z_dense[3])*(buoyancy_field['g'] < 0)][-1]

    #point of neutral buoyancy is CZ edge. (also get it by schwarz criterion
    cz_bound_schwarz = z_dense[(z_dense > z_dense[3])*(buoyancy_field_schwarz['g'] < 0)][-1]


    #Stiffness
    freq2_conv_field['g'] /= L_d05**2

    for ax, ylims in ((ax1, (-0.05, 1.05)), (ax2, (y_ledoux.min() - 0.1, inv_R_in*100)), (ax3, (1e-6, inv_R_in*100))):
        ax.fill_between((0, cz_bound), ylims[0], ylims[1], color=bqual.Dark2_3.mpl_colors[1], alpha=0.15)
        if cz_bound != cz_bound_schwarz:
            ax.fill_between((cz_bound, cz_bound_schwarz), ylims[0], ylims[1], color=bqual.Dark2_3.mpl_colors[0], alpha=0.15)
        ax.fill_between((cz_bound_schwarz, Lz), ylims[0], ylims[1], color=bqual.Dark2_3.mpl_colors[2], alpha=0.15)

    ax1.plot(z, tasks['mu'], c='k')
    ax1.set_ylim(-0.02, 1.02)

    ax2.axhline(0, c='k', lw=0.5)
    colors = bqual.Dark2_3.mpl_colors
    ax2.plot(z_dense, y_schwarz, c=colors[2])
    ax2.plot(z_dense, y_ledoux, c=colors[0])
    ax2.plot(z_dense, -y_schwarz, c=colors[2], ls='--')
    ax2.plot(z_dense, -y_ledoux, c=colors[0], ls='--')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-1, 30*inv_R_in)

    #Add y_Slabel
    max_x = z_dense[y_schwarz < 0][-1]
    max_y = y_schwarz[z_dense > max_x + 0.05][0]
    transformed = ax2.transLimits.transform((max_x, np.log10(max_y)))
    ax2.text(transformed[0]+0.03, transformed[1]-0.05, r'$y_{\rm{S}}$', color=colors[2], transform=ax2.transAxes)

    #Add y_L label
    max_x = L_d0999
    max_y = y_ledoux[z_dense > max_x][0]
    transformed = ax2.transLimits.transform((max_x, np.log10(max_y)))
    ax2.text(transformed[0]-0.1, transformed[1] - 0.05, r'$y_{\rm{L}}$', color=colors[0], transform=ax2.transAxes)




    colors = bqual.Set1_5.mpl_colors
    ax3.set_ylim(1e-3, 30*inv_R_in)
    ax3.set_yscale('log')
    ax3.plot(z,  tasks['bruntN2'], c=colors[0], label=r'$N^2$')
    ax3.plot(z, -tasks['bruntN2'], c=colors[0], ls='--')
    ax3.plot(z_dense, freq2_conv_field['g'], c=colors[1], label=r'$f_{\rm{conv}}^2$')

    #Add N^2 label
    maxbrunt_x = z_dense[N2_tot < 0][-1]
    maxbrunt_y = N2_tot[z_dense > maxbrunt_x + 0.05][0]
    transformed = ax3.transLimits.transform((maxbrunt_x, np.log10(maxbrunt_y)))
    ax3.text(transformed[0]-0.04, transformed[1]+0.02, r'$N^2$', color=colors[0], transform=ax3.transAxes)

    #Add f_conv^2 label
    if freq2_conv_field['g'].max() > 0:
        maxf2_x = z_dense[freq2_conv_field['g'] > freq2_conv_field['g'].max()*0.3][-1]
        maxf2_y = freq2_conv_field['g'].max()
        transformed = ax3.transLimits.transform((maxf2_x, np.log10(maxf2_y)))
        ax3.text(transformed[0]+0.01, transformed[1]+0.01, r'$f_{\rm{conv}}^2$', color=colors[1], transform=ax3.transAxes)
#    ax3.legend(loc='upper left')


    
    colors = bqual.Accent_5.mpl_colors
#        ax.axvline(L_d0999, c=colors[0])
#        ax.axvline(N2_switch_height, c=colors[1])
#        ax.axvline(cz_bound, c=colors[2])
#        ax.axvline(cz_bound_schwarz, lw=0.5, c=colors[3])
#        ax.axvline(oz_bound, c=colors[4])

    for ax in [ax1, ax2]:
        ax.set_xticks(())
        ax.set_xticklabels(())


    

tasks, first_tasks = OrderedDict(), OrderedDict()

bases_names = ['z',]
fields = ['T', 'T_z', 'mu', 'mu_z', 'bruntN2', 'bruntN2_structure', 'bruntN2_composition',\
          'F_rad', 'F_conv', 'F_rad_mu', 'F_conv_mu', 'KE', 'enstrophy']
first_fields = fields + ['T_rad_z', 'T_ad_z', 'flux_of_z']
if not rolled_reader.idle:
    while readerOne.writes_remain():
        first_dsets, first_ni = readerOne.get_dsets(first_fields)
        for k in first_fields:
            first_tasks[k] = first_dsets[k][first_ni].squeeze()
        grad_rad = -first_tasks['T_rad_z']
        grad_ad  = -first_tasks['T_ad_z']
        grad_mu_init = -first_tasks['mu_z']*inv_R_in
        mu_init = first_tasks['mu']
        system_flux = first_tasks['flux_of_z'].squeeze()[-1] #get flux at top of domain
        system_flux_prof = first_tasks['flux_of_z'].squeeze() / system_flux
        F_cond0 = first_tasks['F_rad'].squeeze()

        for f in dedalus_fields:
            f.set_scales(1)
        grad_ad_field['g'] = grad_ad
        delta_grad_field['g'] = grad_ad - grad_rad
        grad_rad_field['g'] = grad_rad
        for f in dedalus_fields:
            f.set_scales(dense_scales, keep_data=True)
        #Find Ls
        Ls = z_dense[delta_grad_field['g'] < 0][-1]

        break
    
    count = 0
    while rolled_reader.writes_remain():
        plot_fields(first_tasks, axL_1, axL_2, axL_3)
        count += 1
        if count != roll_writes+1 and args['--publication_fig']:
            continue

        dsets, ni = rolled_reader.get_dsets(fields)
        time_data = dsets[fields[0]].dims[0]
        z = match_basis(dsets[fields[0]], 'z')

        for k in fields:
            tasks[k] = dsets[k][ni,:].squeeze()
        
        plot_fields(tasks, axR_1, axR_2, axR_3)

        for ax in axs:
            ax.set_xlim(0, 3)


        axL_1.set_ylabel('C')
        axL_2.set_ylabel(r'$y = \nabla - \nabla_X$')
        axL_3.set_ylabel(r'$f^2$')

        axL_3.set_xlabel(r'$z$')
        axR_3.set_xlabel(r'$z$')

        for ax in [axR_1, axR_2, axR_3]:
            ax.set_yticklabels(())

        if args['--publication_fig']:
            plt.subplots_adjust(top=1, bottom=0, hspace=0.05, wspace=0.07, left=0, right=1)
            fig.savefig('{:s}/fig2_profiles.png'.format(rolled_reader.out_dir), dpi=int(args['--dpi']), bbox_inches='tight')
            fig.savefig('{:s}/fig2_profiles.pdf'.format(rolled_reader.out_dir), dpi=int(args['--dpi']), bbox_inches='tight')
        else:
            plt.suptitle('sim_time = {:.2f}'.format(time_data['sim_time'][ni]))
            fig.savefig('{:s}/{:s}_{:06d}.png'.format(rolled_reader.out_dir, fig_name, start_fig+time_data['write_number'][ni]), dpi=int(args['--dpi']), bbox_inches='tight')
        for ax in axs:
            ax.cla()

