"""
Dedalus script for a one-layer double-diffusive convection / semiconvection simulation.
The domain spans z = [0, 1].

There are 6 control parameters:
    Pe      - The approximate freefall peclet number of convection (sqrt(Ra*Pr))
    Pr      - The Prandtl number = (viscous diffusivity / thermal diffusivity)
    tau     - The diffusivity ratio = (compositional diffusivity / thermal diffusivity)
    inv_R   - The inverse density ratio - Ledoux stable if inv_R > 1. ODDC unstable if 1 < inv_R < (Pr + 1)/(Pr + tau)
    aspect  - The aspect ratio (Lx = aspect * Lz)

by default, tau = (kappa composition) / (kappa thermal) is set equal to Pr.

Usage:
    triLayer_model.py [options] 
    triLayer_model.py <config> [options] 

Options:
    --Pe=<Peclet>              Freefall peclet number [default: 1e3]
    --Pr=<Prandtl>             Prandtl number = nu/kappa [default: 0.5]
    --inv_R=<inv_d_ratio>      The inverse effective density ratio [default: 4]
    --tau=<tau>                Diffusivity ratio. If not set, tau = Pr
    --tau_k0=<tau>             Diffusivity ratio for k = 0. If not set, tau = Pr
    --aspect=<aspect>          Aspect ratio of domain [default: 2.5]
    --f0=<factor>              Factor by which to reduce diffusivity on m=0 mode [default: 1]
    --2D                       If flagged, just do a 2D problem

    --nz=<nz>                  Vertical resolution   [default: 128]
    --nx=<nx>                  Horizontal (x) resolution [default: 128]
    --ny=<ny>                  Horizontal (y) resolution (sets to nx by default)
    --RK222                    Use RK222 timestepper (default: RK443)
    --SBDF2                    Use SBDF2 timestepper (default: RK443)
    --safety=<s>               CFL safety factor [default: 0.75]
    --mesh=<m>                 Processor distribution mesh (e.g., "4,4")

    --no_slip                  Use no-slip upper and lower boundary (instead of stress-free)

    --run_time_wall=<time>     Run time, in hours [default: 119.5]
    --run_time_ff=<time>       Run time, in freefall times [default: 3e4]

    --restart=<restart_file>   Restart from checkpoint
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --label=<label>            Optional additional case name label
    --root_dir=<dir>           Root directory for output [default: ./]

    --plot_model               If flagged, create and plt.show() some plots of the 1D atmospheric structure.
"""
import logging
import os
import sys
import time
from collections import OrderedDict
from configparser import ConfigParser
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from mpi4py import MPI
from scipy.special import erf

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

logger = logging.getLogger(__name__)
args = docopt(__doc__)

#Read config file
if args['<config>'] is not None: 
    config_file = Path(args['<config>'])
    config = ConfigParser()
    config.read(str(config_file))
    for n, v in config.items('parameters'):
        for k in args.keys():
            if k.split('--')[-1].lower() == n:
                if v == 'true': v = True
                args[k] = v

def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.

    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.scales
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)

def global_noise(domain, seed=42, **kwargs):
    """
    Create a field fielled with random noise of order 1.  Modify seed to
    get varying noise, keep seed the same to directly compare runs.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    filter_field(noise_field, **kwargs)
    return noise_field

def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

def set_equations(problem):
    twoD = args['--2D']
    threeD = not(twoD)
    if twoD:
        kx_0  = "nx == 0"
        kx_n0 = "nx != 0"
    else:
        kx_0  = "(nx == 0) and (ny == 0)"
        kx_n0 = "(nx != 0) or  (ny != 0)"
    equations = ( (True,      "True", "T1_z - dz(T1) = 0"),
                  (True,      "True", "mu1_z - dz(mu1) = 0"),
                  (not(twoD), "True", "ωx - dy(w) + dz(v) = 0"),
                  (True,      "True", "ωy - dz(u) + dx(w) = 0"),
                  (not(twoD), "True", "ωz - dx(v) + dy(u) = 0"),
                  (True,      kx_n0,  "dx(u) + dy(v) + dz(w) = 0"), #Incompressibility
                  (True,      kx_0,   "p = 0"), #Incompressibility
                  (True,      kx_n0,  "dt(u) + (Pr/Pe0)*(dy(ωz) - dz(ωy))     + dx(p)                   = v*ωz - w*ωy "), #momentum-x
                  (True,      kx_0,   "dt(u) + (Pr/Pe0)*(dy(ωz) - dz(ωy))*f0  + dx(p)                   = v*ωz - w*ωy "), #momentum-x
                  (threeD,    kx_n0,  "dt(v) + (Pr/Pe0)*(dz(ωx) - dx(ωz))     + dy(p)                   = w*ωx - u*ωz "), #momentum-x
                  (threeD,    kx_0,   "dt(v) + (Pr/Pe0)*(dz(ωx) - dx(ωz))*f0  + dy(p)                   = w*ωx - u*ωz "), #momentum-x
                  (True,      kx_n0,  "dt(w) + (Pr/Pe0)*(dx(ωy) - dy(ωx))     + dz(p) - T1 + inv_R*mu1  = u*ωy - v*ωx "), #momentum-z
                  (True,      kx_0,   "w = 0"), #momentum-z
                  (True,      kx_n0, "dt(T1)  + w*(T0_z - T_ad_z) - (1/Pe0)*Lap(T1, T1_z) = -UdotGrad(T1, T1_z)"), #energy eqn k != 0
                  (True,      kx_0,  "dt(T1)  + w*(T0_z - T_ad_z) - (1/Pe0)*dz(f0*T1_z)     = -UdotGrad(T1, T1_z) + Q + (1/Pe0)*dz(f0*T0_z)"), #energy eqn k = 0
                  (True,      kx_n0, "dt(mu1) + w*mu0_z - (tau/Pe0)*Lap(mu1, mu1_z)       = -UdotGrad(mu1, mu1_z)"), #composition eqn k != 0
                  (True,      kx_0,  "dt(mu1) + w*mu0_z - (tau_k0/Pe0)*Lap(f0*mu1, f0*mu1_z)    = -UdotGrad(mu1, mu1_z) + (tau_k0/Pe0)*dz(f0*mu0_z)"), #composition eqn k = 0
                )
    for solve, cond, eqn in equations:
        if solve:
            logger.info('solving eqn {} under condition {}'.format(eqn, cond))
            problem.add_equation(eqn, condition=cond)

    no_slip = args['--no_slip']
    stress_free = not(no_slip)

    boundaries = ( (True,                  " left(T1_z) = 0", "True"),
                   (True,                  "right(T1) = 0", "True"),
                   (True,                  " left(mu1) = 0", "True"),
                   (True,                  "right(mu1) = 0", "True"),
                   (no_slip,               " left(u) = 0", "True"),
                   (no_slip,               "right(u) = 0", "True"),
                   (threeD*no_slip,        " left(v) = 0", "True"),
                   (threeD*no_slip,        "right(v) = 0", "True"),
                   (threeD*stress_free,    " left(ωx) = 0", "True"),
                   (threeD*stress_free,    "right(ωx) = 0", "True"),
                   (stress_free,           " left(ωy) = 0", "True"),
                   (stress_free,           "right(ωy) = 0", "True"),
                   (True,                  " left(w) = 0", kx_n0),
                   (True,                  "right(w) = 0", kx_n0),
                 )
    for solve, bc, cond in boundaries:
        if solve: 
            logger.info('solving bc {} under condition {}'.format(bc, cond))
            problem.add_bc(bc, condition=cond)

    return problem

def set_subs(problem):
    twoD = args['--2D']
    # Set up useful algebra / output substitutions
    if twoD:
        problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
        problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
        problem.substitutions['v'] = '0'
        problem.substitutions['dy(A)'] = '0'
        problem.substitutions['ωx'] = problem.substitutions['ωz'] = '0'
    else:
        problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
        problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz/Ly'
    problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
    problem.substitutions['Lap(A, A_z)']                   = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
    problem.substitutions['UdotGrad(A, A_z)']              = '(u*dx(A) + v*dy(A) + w*A_z)'
    problem.substitutions['GradAdotGradB(A, B, A_z, B_z)'] = '(dx(A)*dx(B) + dy(A)*dy(B) + A_z*B_z)'
    problem.substitutions['enstrophy'] = '(ωx**2 + ωy**2 + ωz**2)'
    problem.substitutions['vel_rms']   = 'sqrt(u**2 + v**2 + w**2)'
    problem.substitutions['Re']        = '((Pe0/Pr)*vel_rms)'
    problem.substitutions['Pe']        = '(Pe0*vel_rms)'
    problem.substitutions['T_z']       = '(T0_z + T1_z)'
    problem.substitutions['T']         = '(T0 + T1)'
    problem.substitutions['mu_z']      = '(mu0_z + mu1_z)'
    problem.substitutions['mu']        = '(mu0   + mu1)'

    problem.substitutions['bruntN2_structure']   = 'T_z - T_ad_z'
    problem.substitutions['bruntN2_composition'] = '-mu_z*inv_R' #dR - density ratio
    problem.substitutions['bruntN2']             = 'bruntN2_structure + bruntN2_composition'

    #Fluxes
    problem.substitutions['F_rad']       = '-(f0/Pe0)*T_z'
    problem.substitutions['F_rad_mu']    = '-(f0*tau_k0/Pe0)*mu_z'
    problem.substitutions['T_rad_z']     = '-flux_of_z/(f0/Pe0)'
    problem.substitutions['T_rad_z_IH']  = '-right(flux_of_z)/(f0/Pe0)'
    problem.substitutions['F_conv']      = 'w*T'
    problem.substitutions['F_conv_mu']   = 'w*mu'
    problem.substitutions['tot_flux']    = '(F_conv + F_rad)'
    return problem

def initialize_output(solver, data_dir, mode='overwrite', output_dt=2, iter=np.inf):
    #TODO: Revisit outputs
    twoD = args['--2D']
    Lx = solver.problem.parameters['Lx']
    Ly = solver.problem.parameters['Ly']
    analysis_tasks = OrderedDict()
    slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=40, mode=mode, iter=iter)
    if twoD:
        slices.add_task("T", name='T')
        slices.add_task("mu", name='mu')
        slices.add_task("ωy", name='vorticity')
        slices.add_task("u", name='u')
        slices.add_task("w", name='w')
    else:
        slices.add_task("interp(T1, y={})".format(Ly/2), name="T1_y_mid")
        slices.add_task("interp(T1, x={})".format(Lx/2), name="T1_x_mid")
        slices.add_task("interp(T1, z=0.2)",  name="T1_z_0.2")
        slices.add_task("interp(T1, z=0.5)",  name="T1_z_0.5")
        slices.add_task("interp(T1, z=1)",    name="T1_z_1")
        slices.add_task("interp(T1, z=1.2)",  name="T1_z_1.2")
        slices.add_task("interp(T1, z=1.5)",  name="T1_z_1.5")
        slices.add_task("interp(T1, z=1.8)",  name="T1_z_1.8")
        slices.add_task("interp(w, y={})".format(Ly/2), name="w_y_mid")
        slices.add_task("interp(w, x={})".format(Lx/2), name="w_x_mid")
        slices.add_task("interp(w, z=0.2)",   name="w_z_0.2")
        slices.add_task("interp(w, z=0.5)",   name="w_z_0.5")
        slices.add_task("interp(w, z=1)",     name="w_z_1")
        slices.add_task("interp(w, z=1.2)",   name="w_z_1.2")
        slices.add_task("interp(w, z=1.5)",   name="w_z_1.5")
        slices.add_task("interp(w, z=1.8)",   name="w_z_1.8")
    analysis_tasks['slices'] = slices

    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=40, mode=mode)
    profiles.add_task("plane_avg(mu)", name='mu')
    profiles.add_task("plane_avg(mu_z)", name='mu_z')
    profiles.add_task("plane_avg(T)", name='T')
    profiles.add_task("plane_avg(T_z)", name='T_z')
    profiles.add_task("plane_avg(T1)", name='T1')
    profiles.add_task("plane_avg(sqrt((T1 - plane_avg(T1))**2))", name='T1_fluc')
    profiles.add_task("plane_avg(T1_z)", name='T1_z')
    profiles.add_task("plane_avg(u)", name='u')
    profiles.add_task("plane_avg(w)", name='w')
    profiles.add_task("plane_avg(vel_rms)", name='vel_rms')
    profiles.add_task("plane_avg(vel_rms**2 / 2)", name='KE')
    profiles.add_task("plane_avg(sqrt((v*ωz - w*ωy)**2 + (u*ωy - v*ωx)**2))", name='advection')
    profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
    profiles.add_task("plane_avg(bruntN2)", name="bruntN2")
    profiles.add_task("plane_avg(bruntN2_structure)", name="bruntN2_structure")
    profiles.add_task("plane_avg(bruntN2_composition)", name="bruntN2_composition")
    profiles.add_task("plane_avg(flux_of_z)", name="flux_of_z")
    profiles.add_task("plane_avg(T_rad_z)", name="T_rad_z")
    profiles.add_task("plane_avg(T_rad_z)", name="T_rad_z_IH")
    profiles.add_task("plane_avg(T_ad_z)", name="T_ad_z")
    profiles.add_task("plane_avg(F_rad)", name="F_rad")
    profiles.add_task("plane_avg(F_rad_mu)", name="F_rad_mu")
    profiles.add_task("plane_avg(F_rad_mu + F_conv_mu)", name="F_mu_tot")
    profiles.add_task("plane_avg(F_conv)", name="F_conv")
    profiles.add_task("plane_avg(F_conv_mu)", name="F_conv_mu")
    profiles.add_task("plane_avg(w * vel_rms**2 / 2)", name="F_KE")
    profiles.add_task("plane_avg(w**3 / 2)", name="F_KE_vert")
    profiles.add_task("plane_avg(w * p)", name="F_KE_p")
    analysis_tasks['profiles'] = profiles

    scalars = solver.evaluator.add_file_handler(data_dir+'scalars', sim_dt=output_dt*5, max_writes=np.inf, mode=mode)
    scalars.add_task("vol_avg(cz_mask*vel_rms**2)/vol_avg(cz_mask)", name="cz_vel_squared")
    scalars.add_task("vol_avg((1-cz_mask)*bruntN2)/vol_avg(1-cz_mask)", name="rz_brunt_squared")
    analysis_tasks['scalars'] = scalars

    checkpoint_min = 60
    checkpoint = solver.evaluator.add_file_handler(data_dir+'checkpoint', wall_dt=checkpoint_min*60, sim_dt=np.inf, iter=np.inf, max_writes=1, mode=mode)
    checkpoint.add_system(solver.state, layout = 'c')
    analysis_tasks['checkpoint'] = checkpoint

    if not twoD:
        volumes = solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=100*output_dt, max_writes=5, mode=mode, iter=iter)
        volumes.add_task("w")
        volumes.add_task("T1")
        volumes.add_task("enstrophy")
        analysis_tasks['volumes'] = volumes

    return analysis_tasks

def run_cartesian_instability(args):
    #############################################################################################
    ### 1. Read in command-line args, set up data directory
    if args['--tau'] is None:
        args['--tau'] = args['--Pr']
    if args['--tau_k0'] is None:
        args['--tau_k0'] = args['--tau']
    twoD = args['--2D']
    if args['--ny'] is None: args['--ny'] = args['--nx']
    data_dir = args['--root_dir'] + '/' + sys.argv[0].split('.py')[0]
    data_dir += "_Pe{}_Pr{}_tau{}_tauk0{}_invR{}_f{}_a{}".format(args['--Pe'], args['--Pr'], args['--tau'], args['--tau_k0'], args['--inv_R'], args['--f0'],  args['--aspect'])
    if twoD:
        data_dir += '_{}x{}'.format(args['--nx'], args['--nz'])
    else:
        data_dir += '_{}x{}x{}'.format(args['--nx'], args['--ny'], args['--nz'])
    if args['--no_slip']:
        data_dir += '_noslip'
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}'.format(data_dir)):
            os.makedirs('{:s}'.format(data_dir))
    logger.info("saving run in: {}".format(data_dir))

    if not twoD:
        mesh = args['--mesh']
        ncpu = MPI.COMM_WORLD.size
        if mesh is not None:
            mesh = mesh.split(',')
            mesh = [int(mesh[0]), int(mesh[1])]
        else:
            log2 = np.log2(ncpu)
            if log2 == int(log2):
                mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
            logger.info("running on processor mesh={}".format(mesh))
    else:
        mesh = None

    ########################################################################################
    ### 2. Organize simulation parameters
    aspect   = float(args['--aspect'])
    nx = int(args['--nx'])
    ny = int(args['--ny'])
    nz = int(args['--nz'])
    Pe0   = float(args['--Pe'])
    Pr    = float(args['--Pr'])
    tau   = float(args['--tau'])
    tau_k0   = float(args['--tau_k0'])
    inv_R = float(args['--inv_R'])
    f0    = float(args['--f0'])

    Lz    = 3
    Lx    = aspect * Lz
    Ly    = Lx
    ell   = ((Pr/Pe0)*(1/Pe0))**(1/4)

    logger.info("Running two-layer instability with the following parameters:")
    logger.info("   Pe = {:.3e}, inv_R = {:.3e}, resolution = {}x{}x{}, aspect = {}".format(Pe0, inv_R, nx, ny, nz, aspect))
    logger.info("   Pr = {:2g}, tau = {:2g}".format(Pr, tau))
    logger.info("   f0 = {:.3e}, ell = {:.3e}".format(f0, ell))
    
    ###########################################################################################################3
    ### 3. Setup Dedalus domain, problem, and substitutions/parameters
    x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=3/2)
    if not twoD:
        y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
        bases = [x_basis, y_basis, z_basis]
    else:
        bases = [x_basis, z_basis]
    domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)
    reducer = flow_tools.GlobalArrayReducer(domain.distributor.comm_cart)
    z = domain.grid(-1)
    z_de = domain.grid(-1, scales=domain.dealias)

    #Establish variables and setup problem
    variables = ['T1', 'T1_z', 'mu1', 'mu1_z', 'p', 'u', 'v', 'w', 'ωx', 'ωy', 'ωz']
    if twoD:
        [variables.remove(v) for v in ['v', 'ωx', 'ωz']]
    problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

    if twoD:
        z_slice = (slice(0, 1), slice(None))
    else:
        z_slice = (slice(0, 1), slice(0, 1), slice(None))

    # Set up background / initial state vs z.
    mu0   = domain.new_field()
    mu0_z = domain.new_field()
    T0   = domain.new_field()
    T0_z = domain.new_field()
    T0_zz = domain.new_field()
    T_ad_z = domain.new_field()
    T_rad_z0 = domain.new_field()
    Q = domain.new_field()
    flux_of_z = domain.new_field()
    cz_mask = domain.new_field()
    f_field = domain.new_field()
    for f in [T0, T0_z, T_ad_z, Q, flux_of_z, T_rad_z0, cz_mask, mu0, mu0_z, f_field]:
        f.set_scales(domain.dealias)
    for f in [T_ad_z, mu0, mu0_z, T0_z, f_field]:
        if twoD:
            f.meta['x']['constant'] = True
        else:
            f.meta['x', 'y']['constant'] = True

    grad_ad = 5 * (inv_R - 2)
    grad_rad_cz = grad_ad + 1
    grad_rad_rz = grad_ad - inv_R
    F = grad_rad_cz/Pe0
    F_conv = (grad_rad_cz - grad_ad)/Pe0
    
    T_rad_z0['g'] = - (grad_rad_cz + (grad_rad_rz-grad_rad_cz)*zero_to_one(z_de, 2, width=0.05))
    T_ad_z['g']   = -grad_ad
    f_field['g']  = (-Pe0*F/T_rad_z0).evaluate()['g']
    print(f_field.interpolate(z=2.2)['g'].min(), f_field['g'].min())

    cz_mask['g'] = one_to_zero(z_de, 1, width=0.05)
    Q['g'] = 0 #(5/Pe0)*(1 - f0)*one_to_zero(z_de, 0.3, width=0.05)*zero_to_one(z_de, 0.1, width=0.05)
#    Q.antidifferentiate('z', ('left', (0)), out=flux_of_z)
    Q.antidifferentiate('z', ('left', (F)), out=flux_of_z)

    delta_Tz = T_rad_z0['g'] - T_ad_z['g']
    #Erf has a width that messes up the transition; bump up T0_zz so it transitions to grad_rad at top.
    T0_z['g'] = T_rad_z0['g']  # T_ad_z['g'] + delta_Tz*zero_to_one(z_de, 1, width=0.05)
    T0_z.antidifferentiate('z', ('right', 1), out=T0)
    T0_z.differentiate('z', out=T0_zz)

    mu0_z['g'] = -1*zero_to_one(z_de, 1, width=0.05)*one_to_zero(z_de, 2, width=0.05)
    mu0_z.antidifferentiate('z', ('right', 0), out=mu0)

    max_brunt = (inv_R - 1)

    #Plug in default parameters
    problem.parameters['inv_R']     = inv_R
    problem.parameters['Pe0']       = Pe0
    problem.parameters['Pr']        = Pr 
    problem.parameters['tau']       = tau
    problem.parameters['tau_k0']       = tau_k0
    problem.parameters['f0']        = f_field
    problem.parameters['Lx']        = Lx
    problem.parameters['Ly']        = Ly
    problem.parameters['Lz']        = Lz
    problem.parameters['mu0']       = mu0
    problem.parameters['mu0_z']     = mu0_z
    problem.parameters['T0']        = T0
    problem.parameters['T0_z']      = T0_z
    problem.parameters['T0_zz']     = T0_zz
    problem.parameters['T_ad_z']    = T_ad_z
    problem.parameters['Q']         = Q
    problem.parameters['flux_of_z'] = flux_of_z
    problem.parameters['cz_mask']   = cz_mask
    problem.parameters['max_brunt'] = max_brunt

    problem = set_subs(problem)
    problem = set_equations(problem)

    if args['--RK222']:
        logger.info('using timestepper RK222')
        ts = de.timesteppers.RK222
    elif args['--SBDF2']:
        logger.info('using timestepper SBDF2')
        ts = de.timesteppers.SBDF2
    else:
        logger.info('using timestepper RK443')
        ts = de.timesteppers.RK443
    solver = problem.build_solver(ts)
    logger.info('Solver built')

    ###########################################################################
    ### 4. Set initial conditions or read from checkpoint.
    mode = 'overwrite'
    if args['--restart'] is None:
        T1 = solver.state['T1']
        T1_z = solver.state['T1_z']
        z_de = domain.grid(-1, scales=domain.dealias)
        for f in [T1, T1_z]:
            f.set_scales(domain.dealias, keep_data=True)

        noise = global_noise(domain, int(args['--seed']))
        #TT
#        T1['g'] = 1e-6*np.sin(np.pi*z_de)*noise['g']
        #FT
        T1['g'] = 1e-6*np.cos((np.pi/2)*z_de/Lz)*noise['g']
        T1.differentiate('z', out=T1_z)
        dt = None
    else:
#        write, dt = solver.load_state(args['--restart'], -1) 
        mode = 'append'
        #For some reason some of the state fields are missing from checkpoints (Tz); copy+paste and modify from coer/solvers.py
        import pathlib
        path = pathlib.Path(args['--restart'])
        index = -1
        logger.info("Loading solver state from: {}".format(path))
        with h5py.File(str(path), mode='r') as file:
            # Load solver attributes
            write = file['scales']['write_number'][index]
            try:
                dt = file['scales']['timestep'][index]
            except KeyError:
                dt = None
            solver.iteration = solver.initial_iteration = file['scales']['iteration'][index]
            solver.sim_time = solver.initial_sim_time = file['scales']['sim_time'][index]
            # Log restart info
            logger.info("Loading iteration: {}".format(solver.iteration))
            logger.info("Loading write: {}".format(write))
            logger.info("Loading sim time: {}".format(solver.sim_time))
            logger.info("Loading timestep: {}".format(dt))
            # Load fields
            for field in solver.state.fields:
                if field.name not in file['tasks'].keys():
                    logger.info("can't find {}".format(field))
                    continue
                dset = file['tasks'][field.name]
                # Find matching layout
                for layout in solver.domain.dist.layouts:
                    if np.allclose(layout.grid_space, dset.attrs['grid_space']):
                        break
                else:
                    raise ValueError("No matching layout")
                # Set scales to match saved data
                scales = dset.shape[1:] / layout.global_shape(scales=1)
                scales[~layout.grid_space] = 1
                # Extract local data from global dset
                dset_slices = (index,) + layout.slices(tuple(scales))
                local_dset = dset[dset_slices]
                # Copy to field
                field_slices = tuple(slice(n) for n in local_dset.shape)
                field.set_scales(scales, keep_data=False)
                field[layout][field_slices] = local_dset
                field.set_scales(solver.domain.dealias, keep_data=True)
        solver.state['T1'].differentiate('z', out=solver.state['T1_z'])

    ###########################################################################
    ### 5. Set simulation stop parameters, output, and CFL
    t_ff    = 1
    t_therm = Pe0
    if max_brunt > 0:
        t_brunt   = np.sqrt(1/max_brunt)
    else:
        t_brunt = np.inf
    max_dt    = np.min((0.5*t_ff, t_brunt))
    logger.info('buoyancy and brunt times are: {:.2e} / {:.2e}; max_dt: {:.2e}'.format(t_ff, t_brunt, max_dt))
    if dt is None:
        dt = max_dt

    cfl_safety = float(args['--safety'])
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.25, max_dt=max_dt, threshold=0.2)
    if twoD:
        CFL.add_velocities(('u', 'w'))
    else:
        CFL.add_velocities(('u', 'v', 'w'))

    run_time_ff   = float(args['--run_time_ff'])
    run_time_wall = float(args['--run_time_wall'])
    solver.stop_sim_time  = run_time_ff*t_ff
    solver.stop_wall_time = run_time_wall*3600.
 
    ###########################################################################
    ### 6. Setup output tasks; run main loop.
    analysis_tasks = initialize_output(solver, data_dir, mode=mode, output_dt=t_ff)

    dense_scales = 20
    dense_x_scales = 1#mesh[0]/nx
    dense_y_scales = 1#mesh[1]/ny
    z_dense = domain.grid(-1, scales=dense_scales)
    dense_handler = solver.evaluator.add_dictionary_handler(sim_dt=1, iter=np.inf)
    dense_handler.add_task("plane_avg(-T_z)", name='grad', scales=(dense_x_scales, dense_y_scales, dense_scales), layout='g')

    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re", name='Re')
    flow.add_property("Pe", name='Pe')
    flow.add_property("cz_mask*sqrt(vel_rms**2)/(inv_R-1)", name='inv_stiffness')
    flow.properties.add_task("plane_avg(T1_z)", name='mean_T1_z', scales=domain.dealias, layout='g')
    flow.properties.add_task("plane_avg(right(T))", name='right_T', scales=domain.dealias, layout='g')


    Hermitian_cadence = 100

    def main_loop(dt):
        Re_avg = 0
        try:
            logger.info('Starting loop')
            start_iter = solver.iteration
            start_time = time.time()
            while solver.ok and np.isfinite(Re_avg):
                effective_iter = solver.iteration - start_iter
                solver.step(dt)

                if effective_iter % Hermitian_cadence == 0:
                    for f in solver.state.fields:
                        f.require_grid_space()

                if effective_iter % 10 == 0:
                    Re_avg = flow.grid_average('Re')

                    log_string =  'Iteration: {:7d}, '.format(solver.iteration)
                    log_string += 'Time: {:8.3e} ({:8.3e} therm), dt: {:8.3e}, '.format(solver.sim_time/t_ff, solver.sim_time/Pe0,  dt/t_ff)
                    log_string += 'Pe: {:8.3e}/{:8.3e}, '.format(flow.grid_average('Pe'), flow.max('Pe'))
                    log_string += 'stiffness: {:8.3e}/{:8.3e}, '.format(1/flow.grid_average('inv_stiffness'), 1/flow.min('inv_stiffness'))
                    logger.info(log_string)

                dt = CFL.compute_dt()
                    
        except:
            raise
            logger.error('Exception raised, triggering end of main loop.')
        finally:
            end_time = time.time()
            main_loop_time = end_time-start_time
            n_iter_loop = solver.iteration-start_iter
            logger.info('Iterations: {:d}'.format(n_iter_loop))
            logger.info('Sim end time: {:f}'.format(solver.sim_time))
            logger.info('Run time: {:f} sec'.format(main_loop_time))
            logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
            logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
            try:
                final_checkpoint = solver.evaluator.add_file_handler(data_dir+'final_checkpoint', wall_dt=np.inf, sim_dt=np.inf, iter=1, max_writes=1)
                final_checkpoint.add_system(solver.state, layout = 'c')
                solver.step(1e-5*dt) #clean this up in the future...works for now.
                post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
            except:
                raise
                print('cannot save final checkpoint')
            finally:
                logger.info('beginning join operation')
                for key, task in analysis_tasks.items():
                    logger.info(task.base_path)
                    post.merge_analysis(task.base_path)
            domain.dist.comm_cart.Barrier()
        return Re_avg

    Re_avg = main_loop(dt)
    if np.isnan(Re_avg):
        return False, data_dir
    else:
        return True, data_dir

if __name__ == "__main__":
    ended_well, data_dir = run_cartesian_instability(args)
    if MPI.COMM_WORLD.rank == 0:
        print('ended with finite Re? : ', ended_well)
        print('data is in ', data_dir)
