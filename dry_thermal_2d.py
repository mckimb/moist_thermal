"""
Dedalus script for simulating a 2D, azimuthally-symmetric, dry thermal in a
Boussinesq atmosphere. Equations are found in Lecoanet & Jeevanjee (2019)
http://nadirjeevanjee.com/papers/17thermals.pdf. Note: nproc is the
number of processors to run the script.

Usage:
    dry_thermal.py [options]
    $ mpiexec -n nproc python3 moist_thermal.py

Options:
    --Reynolds=<Re>       Reynolds number [default: 6e2]
    --Prandtl=<Pr>        Prandtl number [default: 1]
    --nz=<n>              Number of z coefficients [default: 256]
    --nr=<n>              Number of r coefficients [default: 128]
    --aspect=<a>          Domain runs from [0, Lr], with Lr = aspect*Lz [default: 0.25]
    --label=<l>           an optional string at the end of the output directory for easy identification
    --rk443               If flagged, timestep using RK443. Else, SBDF2.
    --Lz=<L>              The depth of the domain, in thermal diameters [default: 20]
    --safety=<s>          Safety factor base [default: 0.1]
    --out_cadence=<o>     Time cadence of output saves in freefall times [default: 0.1]
    --restart=<file>      Name of file to restart from, if starting from checkpoint
"""

import os
import time
import sys
import numpy as np
from mpi4py import MPI
from scipy.special import erf
from dedalus import public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
args = docopt(__doc__)
from logic.checkpointing import Checkpoint

################
# Read in args
################
Re          = float(args['--Reynolds'])
Pr          = float(args['--Prandtl'])
nz          = int(args['--nz'])
nr          = int(args['--nr'])
aspect      = float(args['--aspect'])
Lz          = float(args['--Lz'])

########################
# Set up atmosphere info
########################
Lr        = aspect*Lz
radius    = 0.5
delta_r   = radius/5

#(r0, z0) is the midpoint of the (spherical) thermal
r0        = 0
z0        = 3*radius

###################
# Set up output dir
###################
data_dir = './'+sys.argv[0].split('.py')[0]
data_dir += '_Re{:s}_Xi{:s}_aspect{:s}_Lz{:s}'.format(args['--Reynolds'], args['--Xi'], args['--aspect'], args['--Lz'])
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))
logger.info('saving files in {:s}'.format(data_dir))

#####################
# Dedalus simulation
#####################
r_basis = de.Chebyshev('r', nr, interval=(0, Lr), dealias=3/2)
z_basis = de.Fourier(  'z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([z_basis, r_basis], grid_dtype=np.float64,mesh=[1])
z = domain.grid(0)
r = domain.grid(1)

problem = de.IVP(domain, variables=['u', 'w', 'ur', 'wr', 'rho1', 'rho1r', 'p1'])
problem.meta['u', 'ur', 'w', 'wr', 'rho1', 'rho1r', 'p1']['r']['dirichlet'] = True

problem.parameters['pi']         = np.pi
problem.parameters['Lz']         = Lz
problem.parameters['Lr']         = Lr
problem.parameters['Re']         = Re
problem.parameters['Pr']         = Pr
problem.parameters['nu']         = radius/Re
problem.parameters['kappa']      = radius/(Re*Pr)
problem.parameters['Qkappa']     = radius/(Re*QPr)

# Cylindrical operators
problem.substitutions['UdotGradU_r']        = 'u*ur + w*dz(u)'
problem.substitutions['UdotGradU_z']        = 'u*wr + w*dz(w)'
problem.substitutions['UdotGradrho1']       = 'u*rho1r + w*dz(rho1)'
problem.substitutions['LapU_r']             = 'ur/r + dr(ur) + dz(dz(u)) - u/(r*r)'
problem.substitutions['LapU_z']             = 'wr/r + dr(wr) + dz(dz(w))'
problem.substitutions['Laprho1']            = 'rho1r/r + dr(rho1r) + dz(dz(rho1))'
problem.substitutions['DivU']               = 'u/r + ur + dz(w)'

# Vorticity substitution
problem.substitutions['V']                  = 'dz(u) - wr'
problem.substitutions['Vr']                 = 'dz(ur) - dr(wr)'

# Equations
problem.add_equation("rho1r - dr(rho1) = 0")
problem.add_equation("ur  - dr(u)  = 0")
problem.add_equation("wr  - dr(w)  = 0")
problem.add_equation("DivU = 0")        # Continuity
problem.add_equation("dt(u) + dr(p1) + LapU_r/Re = UdotGradU_r")          # Momentum-r
problem.add_equation("dt(w) + dz(p1) + LapU_z/Re + rho1 = UdotGradU_z")   # Momentum-z
problem.add_equation("dt(rho1) + Laprho1/(Re*Pr) = UdotGradrho1")           # Buoyancy

# Boundary and Regularity Conditions
problem.add_bc("right(rho1r)    = 0")
problem.add_bc("right(V)  = 0", condition="nz != 0")
problem.add_bc("right(w)    = 0")
problem.add_bc("left(p1)  = 0", condition="nz == 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(wr) = 0")
problem.add_bc("left(rho1r) = 0")

#########################
# Initialization of run
#########################
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Set up initial conditions
rho1 = solver.state['rho1']
rho1r = solver.state['rho1r']
w = solver.state['w']

logger.info('checkpointing in {}'.format(data_dir))
checkpoint = Checkpoint(data_dir)
mode = 'overwrite'
restart = args['--restart']

# Simulation termination parameters
solver.stop_sim_time = 20.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

#################################
# Initial conditions from Daniel
#################################
noise = domain.new_field()
noise.meta['r']['dirichlet'] = True
noise.set_scales(domain.dealias)

amp = 0.01/0.5556*0.25
cutoff = 128 # must be an even divisor of N

if restart is None:
  from scipy.special import erf
  r_IC = np.sqrt((z - z0)**2 + (r - r0)**2)
  rho1['g'] = ( erf( (r_IC - radius)/delta_r) - 1 )/2
  rho1.differentiate('r',out=rho1r)

  cshape = domain.dist.coeff_layout.global_shape(scales=1)
  clocal = domain.dist.coeff_layout.local_shape(scales=1)
  cslices = domain.dist.coeff_layout.slices(scales=1)
  rand = np.random.RandomState(seed=42)

  amp_noise = np.zeros(clocal)
  amp_noise_rnd = rand.standard_normal((cutoff,cutoff))
  if np.alltrue([s.start<cutoff for s in cslices]):
    amp_noise[:cutoff,:] = amp_noise_rnd[cslices]

  amp_phase = np.zeros(clocal)
  amp_phase_rnd = rand.uniform(0,2*np.pi,(cutoff,cutoff))
  if np.alltrue([s.start<cutoff for s in cslices]):
    amp_phase[:cutoff,:] = amp_phase_rnd[cslices]

  kr = r_basis.elements.reshape(1,nr)[:,cslices[1]]
  kz = z_basis.elements.reshape(int(nz/2),1)[cslices[0],:] # divide by 2 because k<0 fourier modes are discarded
  k = np.sqrt(1+kz**2+kr**2)

  logger.info('shapes:')
  logger.info(amp_phase.shape)
  logger.info(amp_noise.shape)
  logger.info(k.shape)
  logger.info(noise['c'].shape)

  noise['c'] = k**(-1/3)*amp_noise*np.sin(amp_phase)
  rho1['g'] *= (1 + amp*noise['g'])

  # initial timestep
  start_dt = 1e-3

#################################
# Processing and Postprocessing
#################################
# Analysis & outputs
slices   = solver.evaluator.add_file_handler('{:s}/slices'.format(data_dir),   sim_dt=out_cadence, max_writes=20, mode='overwrite')
profiles = solver.evaluator.add_file_handler('{:s}/profiles'.format(data_dir), sim_dt=out_cadence, max_writes=20, mode='overwrite')
scalars  = solver.evaluator.add_file_handler('{:s}/scalars'.format(data_dir),   sim_dt=out_cadence, max_writes=1e4, mode='overwrite')

for f in ['rho1', 'rho1r', 'u', 'ur', 'w', 'wr', 'V', 'Vr', 'p1']:
    slices.add_task(f, name=f, layout='g')
    profiles.add_task('integ({}, "z")/Lz'.format(f), name='{}_z_avg'.format(f), layout='g')
    profiles.add_task('integ({}, "r")/Lr'.format(f), name='{}_r_avg'.format(f), layout='g')
    scalars.add_task( 'integ({}, "r", "z")/Lz/Lr'.format(f), name='{}_avg'.format(f), layout='g')

for f, nm in [('-rho1','tot_buoy'),('V', 'tot_circ')]:
    scalars.add_task('integ({},"r","z")'.format(f), name='{}'.format(nm), layout='g')

# CFL
safety_factor = float(args['--safety'])
CFL = flow_tools.CFL(solver, initial_dt=start_dt, cadence=1, safety=safety_factor,
                     max_change=1.5, min_change=0.5, max_dt=0.02, threshold=0.05)
CFL.add_velocities(('w', 'u'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property('sqrt(u**2 + w**2)', name='v_rms')
flow.add_property('sqrt(u**2 + w**2)/nu', name='Re')
flow.add_property('integ(V,"r","z")', name='circ')

# Main loop
dt = start_dt
logger.info('Starting loop')
start_time = time.time()
try:
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: {:.2e}, Time: {:.2e}, dt: {:.2e}'.format(solver.iteration, solver.sim_time, dt) +\
                        'Max Re = {:.2e}, Circ = {:.2e}'.format(flow.max('Re'), flow.max('circ')))
        if np.isnan(flow.max('v_rms')):
            logger.info('NaN, breaking.')
            break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
    final_checkpoint.set_checkpoint(solver, wall_dt=1, mode="append")
    solver.step(dt/1000) #clean this up in the future...works for now.
    for t in [checkpoint, final_checkpoint]:
        post.merge_process_files(t.checkpoint_dir, cleanup=False)
    for t in [slices, profiles, scalars]:
        post.merge_process_files(t.base_path, cleanup=False)
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Iter/sec: %.2f ' %(solver.iteration/(end_time-start_time)))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
