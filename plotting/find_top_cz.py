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
    
    --final_only                        If flagged, just do the final plots
"""
import re
from collections import OrderedDict
import h5py
from mpi4py import MPI
from docopt import docopt
args = docopt(__doc__)
import numpy as np
import matplotlib
#matplotlib.use('agg')
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

resolution_regex_2d = re.compile('(.*)x(.*)')
resolution_regex_3d = re.compile('(.*)x(.*)x(.*)')
resolution_regex_3d_compound = re.compile('(.*)x(.*)x(.*)-(.*)')

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
nz_top = None
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

rolled_reader = SingleTypeReader(root_dir, 'profiles', fig_name, roll_writes=50, start_file=start_file, n_files=n_files, distribution='even-write')
MPI.COMM_WORLD.barrier()
readerOne = SingleTypeReader(root_dir, 'profiles', fig_name, start_file=start_file, n_files=n_files, distribution='single', global_comm=MPI.COMM_SELF)
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
axs = [ax1, ax2, ax3]

if not args['--final_only']:
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
    yL_field = domain.new_field()
    yS_field = domain.new_field()
    dedalus_fields = [grad_field, grad_integ_field, grad_ad_field, grad_rad_field, grad_rad_integ_field, \
                      grad_mu_field, grad_mu_integ_field, delta_grad_field, N2_structure_field, KE_field, KE_int_field, F_conv_mu_field, \
                      dz_buoyancy_field, buoyancy_field, freq2_conv_field, freq2_conv_int_field,\
                      dz_buoyancy_field_schwarz, buoyancy_field_schwarz, freq2_conv_field_enstrophy, freq2_conv_int_field_enstrophy, \
                      yL_field, yS_field]

    tasks, first_tasks = OrderedDict(), OrderedDict()

    bases_names = ['z',]
    fields = ['T', 'T_z', 'mu_z', 'bruntN2', 'bruntN2_structure', 'bruntN2_composition',\
              'F_rad', 'F_conv', 'F_rad_mu', 'F_conv_mu', 'KE', 'enstrophy']
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

            for f in [grad_field, grad_mu_field, N2_structure_field, KE_field, F_conv_mu_field, dz_buoyancy_field, freq2_conv_field, dz_buoyancy_field_schwarz, freq2_conv_field_enstrophy, yL_field, yS_field]:
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
            yS_field['g'] = grad_rad - grad_ad
            yL_field['g'] = grad_rad - grad_mu_field['g'] - grad_ad
            for f in dedalus_fields:
                f.set_scales(dense_scales, keep_data=True)

            N2_structure = N2_structure_field['g']
            N2_composition = grad_mu_field['g']
            N2_tot = N2_structure + N2_composition

            #departure from grad_ad: 0.1, 0.5, 0.9
            departures = []
            mu_z_max = grad_mu_field['g'].max()
            for departure_factor in [0.02, 0.5, 0.80]:
                mu_z_departure = grad_mu_field['g'] > departure_factor*mu_z_max
                if np.sum(mu_z_departure) > 0:
                    z_mu_departure = z_dense[mu_z_departure].min()
                else:
                    z_mu_departure = np.nan
                departures.append(z_mu_departure)

            L_d002 = departures[0]
            L_d05 = departures[1]
            L_d080 = departures[2]

            yL_switch = z_dense[yL_field['g'] < 0][0]
            yS_switch = z_dense[yS_field['g'] < 0][0]

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

            data_list = [time_data['sim_time'][ni], time_data['write_number'][ni], L_d002, L_d05, L_d080]
            data_list += [N2_switch_height, cz_bound, oz_bound]
            data_list += [fconv2, N2max, S_measured, cz_bound_schwarz, fconv2_enstrophy, S_measured_enstrophy]
            data_list += [yL_switch, yS_switch]
            data_cube.append(data_list)

            ax1.plot(z,  tasks['bruntN2_structure'][0,:],   c='b', label=r'$N^2_{\rm{structure}}$')
            ax1.plot(z,  tasks['bruntN2_composition'][0,:], c='g', label=r'$N^2_{\rm{composition}}$')
            ax1.plot(z,  tasks['bruntN2'][0,:], c='k', label=r'$N^2$')
            ax1.plot(z, -tasks['bruntN2'][0,:], c='k', ls='--')
            ax1.plot(z, 2*tasks['KE'][0,:]/L_d05**2, c='orange', label=r'$f_{\rm{conv}}^2$')
            ax1.plot(z, tasks['enstrophy'][0,:], c='purple', label=r'$\omega^2$')
            ax1.set_yscale('log')
            ax1.legend(loc='upper left')
            ax1.set_ylabel(r'$N^2$')
            ax1.set_ylim(1e-4, np.max(tasks['bruntN2'])*2)

            ax2.axhline(0, c='k')
            ax2.plot(z, grad - grad_ad,     label=r'$\nabla$', c='b')
            ax2.plot(z, grad_rad - grad_ad, label=r'$\nabla_{\rm{rad}}$', c='r')
            ax2.plot(z, grad_mu/inv_R_in, label=r'$\nabla_{\rm{mu}}\cdot R_{0}$', c='g')
            y_min = np.abs(-grad_rad[z.flatten() > 1]).min()
            deltay = np.abs(grad_ad).max() - y_min
            ax2.set_ylim(- 2, 2)
            ax2.legend(loc='upper right')
            ax2.set_ylabel(r'$\nabla - \nabla_{\rm{ad}}$')

            ax3.axhline(0, c='k')
            ax3.plot(z, F_conv[0,:], label=r'$F_{\rm{conv}}$', c='orange')
            ax3.plot(z, -F_conv[0,:], c='orange', ls='--')
            ax3.plot(z, F_cond[0,:], label=r'$F_{\rm{cond}}$', c='blue')
            ax3.plot(z, tasks['F_conv_mu'][0,:], c='green', label=r'$F_{\rm{conv},\mu}$')
            ax3.legend()
            ax3.set_yscale('log')
            ax3.set_ylim(F_cond[0,:].max()/1e6, F_cond[0,:].max()*2)
            ax3.set_ylabel('Flux')

            for ax in axs:
                ax.set_xlabel('z')
                ax.set_xlim(z.min(), z.max())
    #            ax.axvline(L_d002, c='red')
    #            ax.axvline(L_d05, c='k')
                ax.axvline(L_d080, c='red')
                ax.axvline(N2_switch_height, c='green')
                ax.axvline(cz_bound, c='k')
                ax.axvline(cz_bound_schwarz, c='k', lw=0.5)
                ax.axvline(oz_bound, c='blue')

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
    L_d002s     = global_data[:,2]
    L_d05s     = global_data[:,3]
    L_d080s     = global_data[:,4]
    N2_switch    = global_data[:,5]
    cz_bound     = global_data[:,6]
    oz_bound     = global_data[:,7]
    fconv2       = global_data[:,8]
    N2max        = global_data[:,9]
    S_measured   = global_data[:,10]
    cz_bound_schwarz   = global_data[:,11]
    fconv2_enstrophy       = global_data[:,12]
    S_measured_enstrophy   = global_data[:,13]
    yL_switch              = global_data[:,14]
    yS_switch              = global_data[:,15]

if rolled_reader.reader.global_comm.rank == 0:

    if not args['--final_only']:
        fig = plt.figure()
        plt.plot(times, L_d002s - Ls, c='k', label=r'2% of max $\nabla\mu$')
        plt.plot(times, L_d05s - Ls, c='red', label=r'50% of max $\nabla\mu$')
        plt.plot(times, L_d080s - Ls, c='k', label=r'80% of max $\nabla\mu$')
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel(r'$\delta_p$')
        fig.savefig('{:s}/{:s}.png'.format(rolled_reader.out_dir, 'trace_top_cz'), dpi=400, bbox_inches='tight')

        #stiffness trace
        fig = plt.figure(figsize=(9, 5))
        ax1 = fig.add_subplot(1,3,1)
        ax1.plot(times, fconv2)
        ax1.plot(times, fconv2_enstrophy, label='enstrophy', c='indigo')
        ax1.legend()
        ax1.set_ylabel(r'$f_{\rm{conv}}^2$')
        ax1.set_yscale('log')
        ax1.grid()
        ax2 = fig.add_subplot(1,3,2)
        ax2.plot(times, N2max)
        ax2.set_ylabel(r'$N_{\rm{max}}^2$')
        ax3 = fig.add_subplot(1,3,3)
        ax3.plot(times, S_measured)
        ax3.plot(times, S_measured_enstrophy, label='enstrophy', c='indigo')
        ax3.legend()
        ax3.set_xlabel('time')
        ax3.set_ylabel('Stiffness')
        ax3.grid()
        ax3.set_yscale('log')
        fig.savefig('{:s}/{:s}.png'.format(rolled_reader.out_dir, 'stiffness_trace'), dpi=400, bbox_inches='tight')
    
    if args['--final_only']:
        with h5py.File('{:s}/data_top_cz.h5'.format(rolled_reader.out_dir), 'r') as f:
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

    #kippenhahn fig
    colors = bqual.Dark2_4.mpl_colors
    GREEN  = colors[0]
    ORANGE = colors[1]
    PURPLE = colors[2]
    PINK   = colors[3]
    kfig = plt.figure(figsize=(3.25, 2.5))
    ax = kfig.add_subplot(1,1,1)
    print(yS_switch, yL_switch)
    plt.fill_between(times, yS_switch, Lz*np.ones_like(times), facecolor=PURPLE, alpha=0.5)
    plt.fill_between(times, yL_switch, yS_switch, facecolor=GREEN, alpha=0.5)
    plt.fill_between(times, np.zeros_like(times), yL_switch, facecolor=ORANGE, alpha=0.5)
    plt.fill_between(times, yL_switch, oz_bound, facecolor=PINK, alpha=0.2, hatch='/')
    plt.plot(times, yS_switch, c=GREEN, lw=2)
    plt.plot(times, yL_switch, c=ORANGE, lw=2)
#    plt.plot(times, N2_switch, c='k')
    plt.plot(times, L_d080s, c='k', ls='--')
    end_entrainment = times[yS_switch > yL_switch*1.01][-1]/times.max()
    if end_entrainment < 0.5:
        plt.xlim(0, end_entrainment*times.max()*2)
        end_entrainment = 0.5
    len_overshoot   = (1-end_entrainment)
    plt.ylim(0, Lz)
    plt.xlabel('time')
    plt.ylabel('z')


    arrow_kwargs = { 'edgecolor' : 'k', 'facecolor' : 'k', 'width' : 0.001, 'head_width' : 0.007, 'head_length': 0.003}
    arrow_room = end_entrainment/20
    plt.arrow(x=end_entrainment/2, y=2.8/3, dx=end_entrainment/2-arrow_room, dy=0, transform=ax.transAxes, **arrow_kwargs)
    plt.arrow(x=end_entrainment/2, y=2.8/3, dx=-end_entrainment/2+arrow_room, dy=0, transform=ax.transAxes, **arrow_kwargs)
    plt.arrow(x=end_entrainment + len_overshoot/2, y=2.8/3, dx=len_overshoot/2-arrow_room, dy=0, transform=ax.transAxes, **arrow_kwargs)
    plt.arrow(x=end_entrainment + len_overshoot/2, y=2.8/3, dx=-len_overshoot/2+arrow_room, dy=0, transform=ax.transAxes, **arrow_kwargs)
    plt.text(x=end_entrainment/2, y=2.5/3, s='entrainment', ha='center', transform=ax.transAxes)
    plt.text(x=end_entrainment + len_overshoot/2, y=2.5/3, s='overshoot', ha='center', transform=ax.transAxes)




    x_line = end_entrainment*np.ones(100)
    y_line = np.linspace(2.65, 2.95, 100)/3
    plt.plot(x_line, y_line, c='k', lw=1, transform=ax.transAxes)


    plt.savefig('{:s}/{:s}.png'.format(rolled_reader.out_dir, 'kippenhahn'), dpi=400, bbox_inches='tight')
    plt.savefig('{:s}/{:s}.pdf'.format(rolled_reader.out_dir, 'kippenhahn'), dpi=400, bbox_inches='tight')
    if not args['--final_only']:
        with h5py.File('{:s}/data_top_cz.h5'.format(rolled_reader.out_dir), 'w') as f:
            f['times'] = times     
            f['write_nums'] = write_nums
            f['L_d002s'] = L_d002s    
            f['L_d05s'] = L_d05s    
            f['L_d080s'] = L_d080s    
            f['N2_switch'] = N2_switch
            f['cz_bound'] = cz_bound
            f['cz_bound_schwarz'] = cz_bound_schwarz
            f['oz_bound'] = oz_bound
            f['fconv2'] = fconv2
            f['N2max']  = N2max
            f['S_measured'] = S_measured
            f['yL_switch'] = yL_switch
            f['yS_switch'] = yS_switch
