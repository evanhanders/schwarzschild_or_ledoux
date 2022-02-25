"""
Post-processing script that turns 2D profile data (time, z) into 1D time data.
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

import dedalus.public as de
from plotpal.file_reader import SingleTypeReader, match_basis

import palettable.colorbrewer.qualitative as bqual

profile_data_file = './figure_data/merged_profiles.h5'
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

with h5py.File(profile_data_file, 'r') as f:
    grad_rad = -f['T_rad_z'][0,:].squeeze()
    grad_ad  = -f['T_ad_z'][0,:].squeeze()
    system_flux = f['flux_of_z'][0,:].squeeze()[-1] #get flux at top of domain
    system_flux_prof = f['flux_of_z'][0,:].squeeze() / system_flux
    F_cond0 = f['F_rad'][0,:].squeeze()

    for fd in dedalus_fields:
        fd.set_scales(1)
    grad_ad_field['g'] = grad_ad
    delta_grad_field['g'] = grad_ad - grad_rad
    grad_rad_field['g'] = grad_rad
    grad_rad_field.antidifferentiate('z', ('left', 0), out=grad_rad_integ_field)
    for fd in dedalus_fields:
        fd.set_scales(dense_scales, keep_data=True)
    #Find Ls
    Ls = z_dense[delta_grad_field['g'] < 0][-1]
    N = len(f['sim_time'][()])
    for i in range(N):
        if i % 10 == 0:
            print('reading write {}'.format(i+1))
        for k in f.keys():
            if k in ['sim_time', 'z']:
                continue
            else:
                #roll
                if i < half_roll:
                    tasks[k] = np.mean(f[k][:i+half_roll,:], axis=0).squeeze()
                elif i > N-half_roll:
                    tasks[k] = np.mean(f[k][i-half_roll:,:], axis=0).squeeze()
                else:
                    tasks[k] = np.mean(f[k][i-half_roll:i+half_roll,:], axis=0).squeeze()

        F_cond = tasks['F_rad']
        F_conv = tasks['F_conv']
        T      = tasks['T']
        T_z    = tasks['T_z']
        mu_z   = tasks['mu_z']
        grad = -T_z
        grad_mu = -mu_z*inv_R_in

        for fd in [grad_field, grad_mu_field, N2_structure_field, KE_field, F_conv_mu_field, dz_buoyancy_field, freq2_conv_field, dz_buoyancy_field_schwarz, freq2_conv_field_enstrophy, yL_field, yS_field]:
            fd.set_scales(1, keep_data=True)
        grad_field['g'] = -T_z
        N2_structure_field['g'] = -(grad_field['g'] - grad_ad)
        grad_field.antidifferentiate('z', ('left', 0), out=grad_integ_field)
        grad_mu_field['g'] = -mu_z*inv_R_in
        grad_mu_field.antidifferentiate('z', ('left', 0), out=grad_mu_integ_field)
        KE_field['g'] = tasks['KE']
        KE_field.antidifferentiate('z', ('left', 0), out=KE_int_field)
        F_conv_mu_field['g'] = tasks['F_conv_mu']
        dz_buoyancy_field['g'] = (T_z - (-grad_ad)) - mu_z * inv_R_in
        dz_buoyancy_field.antidifferentiate('z', ('left', 0), out=buoyancy_field)
        dz_buoyancy_field_schwarz['g'] = (T_z - (-grad_ad))
        dz_buoyancy_field_schwarz.antidifferentiate('z', ('left', 0), out=buoyancy_field_schwarz)
        freq2_conv_field['g'] = 2*tasks['KE'] #need to divide by L_d05^2 later.
        freq2_conv_field.antidifferentiate('z', ('left', 0), out=freq2_conv_int_field)
        freq2_conv_field_enstrophy['g'] = tasks['enstrophy']
        freq2_conv_field_enstrophy.antidifferentiate('z', ('left', 0), out=freq2_conv_int_field_enstrophy)
        yS_field['g'] = grad_rad - grad_ad
        yL_field['g'] = grad_rad - grad_mu_field['g'] - grad_ad
        for fd in dedalus_fields:
            fd.set_scales(dense_scales, keep_data=True)

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

        data_list = [f['sim_time'][i], i+1, L_d002, L_d05, L_d080]
        data_list += [N2_switch_height, cz_bound, oz_bound]
        data_list += [fconv2, N2max, S_measured, cz_bound_schwarz, fconv2_enstrophy, S_measured_enstrophy]
        data_list += [yL_switch, yS_switch]
        data_cube.append(data_list)

    data_cube = np.array(data_cube)
    write_nums = np.array(data_cube[:,1], dtype=int)

global_data = data_cube
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

with h5py.File('./figure_data/scalar_data.h5', 'w') as f:
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
