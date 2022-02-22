"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.
"""
import re
from collections import OrderedDict
import h5py
from mpi4py import MPI
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
plt.style.use('./apj.mplstyle')

import palettable.colorbrewer.qualitative as bqual
import dedalus.public as de

evolved_time = 17000

fig = plt.figure(figsize=(7.5,3))
axL_1 = fig.add_subplot(3,2,1)
axL_2 = fig.add_subplot(3,2,3)
axL_3 = fig.add_subplot(3,2,5)
axR_1 = fig.add_subplot(3,2,2)
axR_2 = fig.add_subplot(3,2,4)
axR_3 = fig.add_subplot(3,2,6)
axs = [axL_1, axL_2, axL_3, axR_1, axR_2, axR_3]
plt.subplots_adjust(top=0.9, bottom=0, hspace=0.05, wspace=0.1, left=0, right=1)

inv_R_in = 10
nz = 512
nz_top = 64
Lz = 3
z_basis1 = de.Chebyshev('z', nz, interval=(0, 2.2), dealias=1)
z_basis2 = de.Chebyshev('z', nz_top, interval=(2.2, Lz), dealias=1)
z_basis = de.Compound('z', [z_basis1, z_basis2], dealias=1)
domain = de.Domain([z_basis,], grid_dtype=np.float64, mesh=None, comm=MPI.COMM_SELF)
dense_scales=20
z = domain.grid(0, scales=1)
z_dense = domain.grid(0, scales=dense_scales)
roll_writes = 100
half_roll = int(roll_writes/2)

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
yL_field = domain.new_field()
yS_field = domain.new_field()
dedalus_fields = [N2_structure_field, grad_field, grad_mu_field, grad_ad_field, grad_rad_field,
                    delta_grad_field, freq2_conv_field, freq2_conv_field_enstrophy,
                    buoyancy_field, buoyancy_field_schwarz, dz_buoyancy_field, dz_buoyancy_field_schwarz,
                    KE_field, yL_field, yS_field]



with h5py.File('./figure_data/merged_profiles.h5', 'r') as f:

    start_ind = 0
    evolved_ind = np.argmin(np.abs(f['sim_time'][()] - evolved_time))

    def plot_fields(tasks, ax1, ax2, ax3, first=False):

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
        yL_field['g'] = grad_rad_field['g'] - grad_ad_field['g'] - grad_mu_field['g']
        yS_field['g'] = grad_rad_field['g'] - grad_ad_field['g']
        for f in dedalus_fields:
            f.set_scales(dense_scales, keep_data=True)

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
    #        oz_bound = z_dense[F_conv_mu_field['g'] > F_conv_mu_field['g'].max()*1e-1][-1]
        if KE_field['g'].max() > 0:
            oz_bound = z_dense[KE_field['g'] > KE_field['g'].max()*1e-1][-1]
        else:
            oz_bound = 0

        #point of neutral buoyancy is CZ edge.
        cz_bound = z_dense[(z_dense > z_dense[3])*(buoyancy_field['g'] < 0)][-1]
        yL_switch = z_dense[yL_field['g'] < 0][0]

        #point of neutral buoyancy is CZ edge. (also get it by schwarz criterion
        cz_bound_schwarz = z_dense[(z_dense > z_dense[3])*(buoyancy_field_schwarz['g'] < 0)][-1]
        yS_switch = z_dense[yS_field['g'] < 0][0]


        #Stiffness
        freq2_conv_field['g'] /= L_d05**2

        for ax, ylims in ((ax1, (-0.05, 1.05)), (ax2, (yL_field['g'].min() - 0.1, inv_R_in*100)), (ax3, (1e-6, inv_R_in*100))):
            ax.fill_between((0, yL_switch), ylims[0], ylims[1], color=bqual.Dark2_3.mpl_colors[1], alpha=0.15)
            if yS_switch != yL_switch:
                ax.fill_between((yL_switch, yS_switch), ylims[0], ylims[1], color=bqual.Dark2_3.mpl_colors[0], alpha=0.15)
            ax.fill_between((yS_switch, Lz), ylims[0], ylims[1], color=bqual.Dark2_3.mpl_colors[2], alpha=0.15)
            if not first:
    #            print(ax.fill_between((yL_switch, oz_bound), ylims[0], ylims[1], hatch='//', edgecolor=(0, 0, 0, 0.2), facecolor="none", zorder=2))
                print(ax.add_patch(matplotlib.patches.Rectangle((yL_switch, ylims[0]), oz_bound-yL_switch, ylims[1]-ylims[0], fill=False, hatch='//', alpha=0.1)))

        ax1.plot(z, tasks['mu'], c='k')
        ax1.set_ylim(-0.05, 1.05)

        ax2.axhline(0, c='k', lw=0.5)
        colors = bqual.Dark2_3.mpl_colors
        if first:
            dashes = (4, 3)
        else:
            dashes = (3.7, 2.1)
        ax2.plot(z_dense, yS_field['g'], c=colors[2])
        ax2.plot(z_dense, yL_field['g'], c=colors[0])
        ax2.plot(z_dense, -yS_field['g'], c=colors[2], ls='--', dashes=dashes)
        ax2.plot(z_dense, -yL_field['g'], c=colors[0], ls='--', dashes=dashes)
        ax2.set_yscale('log')
        ax2.set_ylim(3e-1, 50*inv_R_in)

        #Add y_Slabel
    #    print(max_x)
        max_x = z_dense[yS_field['g'] > 0][-1]
        max_y = np.max(yS_field['g'])
    #    max_y = -yS_field['g'][z_dense > max_x + 0.05][0]
        transformed = ax2.transLimits.transform((max_x, np.log10(max_y)))
        if first:
            ax2.text(transformed[0]+0.03, transformed[1] - 0.05, r'$\mathcal{Y}_{\rm{S}}$', color=colors[2], transform=ax2.transAxes)
        else:
            ax2.text(transformed[0]+0.03, transformed[1] - 0.05, r'$\mathcal{Y}_{\rm{S}}$', color=colors[2], transform=ax2.transAxes)

        #Add y_L label
        max_x = L_d0999
        max_y = -yL_field['g'][z_dense > max_x][0]
        transformed = ax2.transLimits.transform((max_x, np.log10(max_y)))
        if first:
            ax2.text(transformed[0]-0.105, transformed[1] - 0.075, r'$\mathcal{Y}_{\rm{L}}$', color=colors[0], transform=ax2.transAxes)
        else:
            ax2.text(transformed[0]-0.095, transformed[1] - 0.05, r'$\mathcal{Y}_{\rm{L}}$', color=colors[0], transform=ax2.transAxes)


        if first:
            ax2.plot(-1e-16*np.ones(2), 1e-16*np.ones(2), c='k', ls='--', label='negative')
            ax2.plot(-1e-16*np.ones(2), 1e-16*np.ones(2), c='k', label='positive')
            ax2.legend(loc='upper left', ncol=2, framealpha=0.8, borderpad=0.2, borderaxespad=0.2, fancybox=True, labelspacing=0.1, fontsize=9, columnspacing=1, handletextpad=0.4, handlelength=1.72)

        y_L_max_x = max_x




        colors = bqual.Set1_5.mpl_colors
        ax3.set_ylim(3e-1, 50*inv_R_in)
        ax3.set_yscale('log')
        ax3.plot(z,  tasks['bruntN2'], c=colors[0], label=r'$N^2$')
        ax3.plot(z, -tasks['bruntN2'], c=colors[0], ls='--')
        ax3.plot(z_dense, 1e3*freq2_conv_field['g'], c=colors[1], label=r'$10^3\,f_{\rm{conv}}^2$')

        #Add N^2 label
        maxbrunt_x = y_L_max_x
        maxbrunt_y = N2_tot[(z_dense > maxbrunt_x)*(z_dense < maxbrunt_x + 0.25)].max()
        transformed = ax3.transLimits.transform((maxbrunt_x, np.log10(maxbrunt_y)))
        if first:
            ax3.text(transformed[0]-0.105, transformed[1]-0.075, r'$N^2$', color=colors[0], transform=ax3.transAxes)
        else:
            ax3.text(transformed[0]-0.095, transformed[1]-0.1, r'$N^2$', color=colors[0], transform=ax3.transAxes)

        #Add f_conv^2 label
        if freq2_conv_field['g'].max() > 0:
            maxf2_x = z_dense[freq2_conv_field['g'] > freq2_conv_field['g'].max()*0.3][-1]
            maxf2_y = 1e3*freq2_conv_field['g'].max()
            transformed = ax3.transLimits.transform((maxf2_x, np.log10(maxf2_y)))
            ax3.text(transformed[0]+0.01, transformed[1]-0.07, r'$10^3\,f_{\rm{conv}}^2$', color=colors[1], transform=ax3.transAxes)
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

    N = len(f['sim_time'][()])
    for k in f.keys():
        if k in ['sim_time', 'z']:
            continue
        else:
            #roll
            if evolved_ind < half_roll:
                tasks[k] = np.mean(f[k][:evolved_ind+half_roll,:], axis=0).squeeze()
            elif evolved_ind > N-half_roll:
                tasks[k] = np.mean(f[k][evolved_ind-half_roll:,:], axis=0).squeeze()
            else:
                tasks[k] = np.mean(f[k][evolved_ind-half_roll:evolved_ind+half_roll,:], axis=0).squeeze()
            first_tasks[k] = f[k][0,:].squeeze()


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

    plot_fields(first_tasks, axL_1, axL_2, axL_3, first=True)
    plot_fields(tasks, axR_1, axR_2, axR_3)

    for ax in axs:
        ax.set_xlim(0, 3)

    axL_1.set_ylabel(r'$\mu$')
    axL_2.set_ylabel(r'$\mathcal{Y}$')
    axL_3.set_ylabel(r'$f^2$')

    axL_3.set_xlabel(r'$z$')
    axR_3.set_xlabel(r'$z$')

    axL_1.text(x=0.5, y=1.1, s='Initial ($t = 0$)', ha='center', transform=axL_1.transAxes)
    axR_1.text(x=0.5, y=1.1, s='Evolved ($t = {{{:.0f}}}$)'.format(evolved_time), ha='center', transform=axR_1.transAxes)

    for ax in [axR_1, axR_2, axR_3]:
        ax.set_yticklabels(())

    print('saving fig')
    fig.savefig('{:s}/fig2_profiles.png'.format('./'), dpi=400, bbox_inches='tight')
    fig.savefig('{:s}/fig2_profiles.pdf'.format('./'), dpi=400, bbox_inches='tight')

    for ax in axs:
        ax.cla()

