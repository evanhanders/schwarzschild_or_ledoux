"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.

Usage:
    find_top_cz.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: paper_profiles]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 20]
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

n_files     = 5
start_file  = int(args['--start_file']) - 2
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
    elif 'mixedICs' in val:
        continue
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
z = domain.grid(0, scales=1)

roll_writes = 50
rolled_reader = SingleTypeReader(root_dir, 'profiles', fig_name, roll_writes=roll_writes, start_file=start_file, n_files=n_files, distribution='even-write')
MPI.COMM_WORLD.barrier()
readerOne = SingleTypeReader(root_dir, 'profiles', fig_name, start_file=1, n_files=n_files, distribution='single', global_comm=MPI.COMM_SELF)
fig = plt.figure(figsize=(7.5,6))
axL_1 = fig.add_subplot(3,2,1)
axL_2 = fig.add_subplot(3,2,3)
axR_1 = fig.add_subplot(3,2,2)
axR_2 = fig.add_subplot(3,2,4)
axR_3 = fig.add_subplot(3,2,6)
axs = [axL_1, axL_2, axR_1, axR_2, axR_3]

data_cube = []

N2_structure_field = domain.new_field()
grad_field = domain.new_field()
grad_integ_field = domain.new_field()
grad_mu_field = domain.new_field()
grad_mu_integ_field = domain.new_field()
grad_ad_field = domain.new_field()
grad_rad_field = domain.new_field()
grad_rad_integ_field = domain.new_field()
delta_grad_field = domain.new_field()
KE_field = domain.new_field()
KE_int_field = domain.new_field()
F_conv_mu_field = domain.new_field()
dz_buoyancy_field = domain.new_field()
buoyancy_field = domain.new_field()
dz_buoyancy_field_schwarz = domain.new_field()
buoyancy_field_schwarz = domain.new_field()
freq2_conv_field = domain.new_field()
freq2_conv_int_field = domain.new_field()
freq2_conv_field_enstrophy = domain.new_field()
freq2_conv_int_field_enstrophy = domain.new_field()
dedalus_fields = [grad_field, grad_integ_field, grad_ad_field, grad_rad_field, grad_rad_integ_field, \
                  grad_mu_field, grad_mu_integ_field, delta_grad_field, N2_structure_field, KE_field, KE_int_field, F_conv_mu_field, \
                  dz_buoyancy_field, buoyancy_field, freq2_conv_field, freq2_conv_int_field,\
                  dz_buoyancy_field_schwarz, buoyancy_field_schwarz, freq2_conv_field_enstrophy, freq2_conv_int_field_enstrophy]

tasks, first_tasks = OrderedDict(), OrderedDict()

bases_names = ['z',]
fields = ['T', 'T_z', 'mu', 'mu_z', 'bruntN2', 'bruntN2_structure', 'bruntN2_composition',\
          'F_rad', 'F_conv', 'F_rad_mu', 'F_conv_mu', 'KE', 'enstrophy']
first_fields = ['mu', 'mu_z', 'T_rad_z', 'T_ad_z', 'F_rad', 'flux_of_z', 'T_z']
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

        axL_1.plot(z, mu_init)
        axL_2.plot(z, grad_ad, c='k')
        axL_2.plot(z, grad_rad, c='r')
        axL_2.plot(z, grad_mu_init, c='g')

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
    
    count = 0
    while rolled_reader.writes_remain():
        count += 1
        if count != roll_writes+1:
            continue

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

        for f in [grad_field, grad_mu_field, N2_structure_field, KE_field, F_conv_mu_field, dz_buoyancy_field, freq2_conv_field, dz_buoyancy_field_schwarz, freq2_conv_field_enstrophy]:
            f.set_scales(1, keep_data=True)
        grad_field['g'] = -T_z
        N2_structure_field['g'] = -(grad_field['g'] - grad_ad)
        grad_field.antidifferentiate('z', ('left', 0), out=grad_integ_field)
        grad_mu_field['g'] = -mu_z*inv_R_in
        grad_mu_field.antidifferentiate('z', ('left', 0), out=grad_mu_integ_field)
        KE_field['g'] = tasks['KE']
        KE_field.antidifferentiate('z', ('left', 0), out=KE_int_field)
        F_conv_mu_field['g'] = tasks['F_conv_mu'][0,:]
        dz_buoyancy_field['g'] = (T_z - (-grad_ad)) - mu_z * inv_R_in
        dz_buoyancy_field.antidifferentiate('z', ('left', 0), out=buoyancy_field)
        dz_buoyancy_field_schwarz['g'] = (T_z - (-grad_ad))
        dz_buoyancy_field_schwarz.antidifferentiate('z', ('left', 0), out=buoyancy_field_schwarz)
        freq2_conv_field['g'] = 2*tasks['KE'][0,:] #need to divide by L_d05^2 later.
        freq2_conv_field.antidifferentiate('z', ('left', 0), out=freq2_conv_int_field)
        freq2_conv_field_enstrophy['g'] = tasks['enstrophy'][0,:]
        freq2_conv_field_enstrophy.antidifferentiate('z', ('left', 0), out=freq2_conv_int_field_enstrophy)
        for f in dedalus_fields:
            f.set_scales(dense_scales, keep_data=True)

        axR_1.plot(z, tasks['mu'][0,:])
        axR_2.plot(z, grad_ad, c='k')
        axR_2.plot(z_dense, grad_field['g'], c='r')
        axR_2.plot(z_dense, grad_mu_field['g'], c='g')

        N2_structure = N2_structure_field['g']
        N2_composition = grad_mu_field['g']
        N2_tot = N2_structure + N2_composition

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
        mean_cz_KE = KE_int_field['g'][z_dense < L_d002][-1] / L_d002
#        oz_bound = z_dense[F_conv_mu_field['g'] > F_conv_mu_field['g'].max()*1e-1][-1]
        oz_bound = z_dense[KE_field['g'] > KE_field['g'].max()*1e-1][-1]

        #point of neutral buoyancy is CZ edge.
        cz_bound = z_dense[(z_dense > z_dense[3])*(buoyancy_field['g'] < 0)][-1]
#        try:
#            opt = brentq(interp1d(z_dense, buoyancy_field['g']), z_dense[3], oz_bound)
#            cz_bound = opt
#        except:
#            print('brentq failed to converge')
#            cz_bound = z_dense[(z_dense > z_dense[3])*(buoyancy_field['g'] > 0)][0]
#            cz_bound = z_dense[KE_field['g'] > mean_cz_KE*5e-1][-1]

        #point of neutral buoyancy is CZ edge. (also get it by schwarz criterion
        cz_bound_schwarz = z_dense[(z_dense > z_dense[3])*(buoyancy_field_schwarz['g'] < 0)][-1]


        #Stiffness
        freq2_conv_field['g'] /= L_d05**2
        freq2_conv_int_field['g'] /= L_d05**2
        fconv2 = freq2_conv_int_field.interpolate(z=L_d05)['g'].min() / L_d05 #take avg.
        fconv2_enstrophy = freq2_conv_int_field_enstrophy.interpolate(z=L_d05)['g'].min() / L_d05 #take avg.
        N2max = np.max(N2_tot)
        S_measured = N2max/fconv2
        S_measured_enstrophy = N2max/fconv2_enstrophy

        data_list = [time_data['sim_time'][ni], time_data['write_number'][ni], L_d002, L_d05, L_d0999]
        data_list += [N2_switch_height, cz_bound, oz_bound]
        data_list += [fconv2, N2max, S_measured, cz_bound_schwarz, fconv2_enstrophy, S_measured_enstrophy]
        data_cube.append(data_list)

        axR_3.plot(z,  tasks['bruntN2_structure'][0,:],   c='b', label=r'$N^2_{\rm{structure}}$')
        axR_3.plot(z,  tasks['bruntN2_composition'][0,:], c='g', label=r'$N^2_{\rm{composition}}$')
        axR_3.plot(z,  tasks['bruntN2'][0,:], c='k', label=r'$N^2$')
        axR_3.plot(z, -tasks['bruntN2'][0,:], c='k', ls='--')
        axR_3.plot(z, 2*tasks['KE'][0,:]/L_d05**2, c='orange', label=r'$f_{\rm{conv}}^2$')
        axR_3.plot(z, tasks['enstrophy'][0,:], c='purple', label=r'$\omega^2$')
        axR_3.set_yscale('log')
        axR_3.legend(loc='upper left')
        axR_3.set_ylabel(r'$N^2$')
        axR_3.set_ylim(1e-4, np.max(tasks['bruntN2'])*2)

#        ax2.axhline(0, c='k')
#        ax2.plot(z, grad - grad_ad,     label=r'$\nabla$', c='b')
#        ax2.plot(z, grad_rad - grad_ad, label=r'$\nabla_{\rm{rad}}$', c='r')
#        ax2.plot(z, grad_mu/inv_R_in, label=r'$\nabla_{\rm{mu}}\cdot R_{0}$', c='g')
#        y_min = np.abs(-grad_rad[z.flatten() > 1]).min()
#        deltay = np.abs(grad_ad).max() - y_min
#        ax2.set_ylim(- 2, 2)
#        ax2.legend(loc='upper right')
#        ax2.set_ylabel(r'$\nabla - \nabla_{\rm{ad}}$')
#
#        ax3.axhline(0, c='k')
#        ax3.plot(z, F_conv[0,:], label=r'$F_{\rm{conv}}$', c='orange')
#        ax3.plot(z, -F_conv[0,:], c='orange', ls='--')
#        ax3.plot(z, F_cond[0,:], label=r'$F_{\rm{cond}}$', c='blue')
#        ax3.plot(z, tasks['F_conv_mu'][0,:], c='green', label=r'$F_{\rm{conv},\mu}$')
#        ax3.legend()
#        ax3.set_yscale('log')
#        ax3.set_ylim(F_cond[0,:].max()/1e6, F_cond[0,:].max()*2)
#        ax3.set_ylabel('Flux')
#
#        for ax in axs:
#            ax.set_xlabel('z')
#            ax.set_xlim(z.min(), z.max())
##            ax.axvline(L_d002, c='red')
##            ax.axvline(L_d05, c='k')
#            ax.axvline(L_d0999, c='red')
#            ax.axvline(N2_switch_height, c='green')
#            ax.axvline(cz_bound, c='k')
#            ax.axvline(cz_bound_schwarz, c='k', lw=0.5)
#            ax.axvline(oz_bound, c='blue')

        plt.suptitle('sim_time = {:.2f}'.format(time_data['sim_time'][ni]))

        fig.savefig('{:s}/{:s}_{:06d}.png'.format(rolled_reader.out_dir, fig_name, start_fig+time_data['write_number'][ni]), dpi=int(args['--dpi']), bbox_inches='tight')
        for ax in axs:
            ax.cla()

