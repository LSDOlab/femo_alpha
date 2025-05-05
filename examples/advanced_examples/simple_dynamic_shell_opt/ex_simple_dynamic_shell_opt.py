'''Shell thickness optimization of a cantilever plate '''
'''
Thickness optimization of cantilever plate with Reissner-Mindlin shell elements.
This example uses pre-built Reissner-Mindlin shell model in FEMO

Author: Ru Xiang
Date: 2024-06-20
'''

import dolfinx
from dolfinx.fem import Constant
from mpi4py import MPI
import csdl_alpha as csdl
import numpy as np

from femo_alpha.dynamic_rm_shell.plate_sim import PlateSim
from femo_alpha.dynamic_rm_shell.state_operation_dynamic import StateOperation
from femo_alpha.dynamic_rm_shell.total_strain_energy_operation import TotalStrainEnergyOperation
from femo_alpha.dynamic_rm_shell.volume_operation import VolumeOperation
from femo_alpha.fea.utils_dolfinx import createCustomMeasure

run_verify_forward_eval = False
run_check_derivatives = True
run_optimization = False
element_wise_material = False

'''
1. Define the mesh
'''

plate = [#### quad mesh ####
        "plate_2_10_quad_1_5.xdmf", # trigger PETSc error in evaluation
        "plate_2_10_quad_2_10.xdmf",
        "plate_2_10_quad_4_20.xdmf",
        "plate_2_10_quad_8_40.xdmf",
        "plate_2_10_quad_10_50.xdmf",]

filename = "./plate_meshes/"+plate[1]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

# Define physical parameters
E_val = 1e8  # Young's modulus
nu_val = 0.3  # Poisson ratio
h_val = 0.1  # thickness
width = 2.  # plate width
length = 10.  # plate length
rho_val = 10.0
# f_d = 5.  # force magnitude


Nsteps  = 2


# # Time-stepping parameters
# T       = 5.
# dt = T/Nsteps
# p0 = 5.

# f_0 = Constant(mesh, (0.0,0.0,-p0))


# gust loads (1-cosine)
V_inf = 50.  # freestream velocity magnitude in m/s
V_p = 50. # peak velocity of the gust

# Time-stepping parameters
l_chord = 1.2 # unit: m, chord length
GGLc = 5 # gust gradient length in chords
T0 = 0.02 # static before the gust
T1 = GGLc*l_chord/V_inf # gust duration 0.12s
# T2 = 0.36 # calm down
T2 = 2.86
T = T0+T1+T2 # total time
# Nsteps  = 1
dt = T/Nsteps
def V_g(t):
    V_g = 0.
    if T0 <= t <= T0+T1:
        # V_g = V_p*np.sin(2*np.pi*t/T1)**2
        V_g = V_p*(1-np.cos(2*np.pi*(t-T0)/T1))
    return V_g
    # return 1/2*V_p*(1-np.cos(2*np.pi*t/GGLc))

t = np.linspace(0,T,Nsteps+1)
force_history = np.zeros((Nsteps+1, nn*3))
for i in range(Nsteps+1):
    ti = t[i]
    V_gi = V_g(ti)
    force_vector = np.zeros((nn, 3))
    force_vector[:, 2] = V_gi*0.1
    force_vector = force_vector.flatten()
    force_history[i,:] = force_vector

plate_sim = PlateSim(mesh, E_val, nu_val, rho_val, dt, Nsteps)

'''
3. Set up csdl recorder and run the simulation
'''
recorder = csdl.Recorder(inline=True)
recorder.start()


## Constant loads
# force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
# force_vector.value[:, 2] = -p0 # body force per node
# force_vector = force_vector.flatten()
# force_history = csdl.expand(force_vector, out_shape=(plate_sim.time_levels,nn*3), action='i->ji')
# force_history.add_name('force_history')


# Time-dependent loads
force_history = csdl.Variable(value=force_history, name='force_history')
force_history.add_name('force_history')

if element_wise_material:
    thickness = csdl.Variable(value=h_val*np.ones(nel), name='thickness')
else:
    thickness = csdl.Variable(value=h_val*np.ones(nn), name='thickness')


state_operation = StateOperation(plate_sim=plate_sim, gradient_mode='petsc',
                                # debug_mode=True,
                                record=True, path='./records/')
input_vars = csdl.VariableGroup()
input_vars.thickness = thickness
input_vars.force_history = force_history

disp_history = state_operation.evaluate(input_vars)
disp_history.add_name('disp_history')

total_strain_energy_operation = TotalStrainEnergyOperation(plate_sim=plate_sim)
input_vars.disp_history = disp_history
total_strain_energy = total_strain_energy_operation.evaluate(input_vars)
total_strain_energy.add_name('total_strain_energy')


volume_operation = VolumeOperation(plate_sim=plate_sim)
volume = volume_operation.evaluate(input_vars)
volume.add_name('volume')

# shell_outputs = shell_model.evaluate(force_vector, thickness, E, nu, density,
#                                         node_disp,
#                                         debug_mode=False,
#                                         is_pressure=False)
# disp_solid = shell_outputs.disp_solid
# total_strain_energy = shell_outputs.total_strain_energy
# aggregated_stress = shell_outputs.aggregated_stress
# mass = shell_outputs.mass
# F_solid = shell_outputs.F_solid

if run_verify_forward_eval:
    print("total total_strain_energy:", total_strain_energy.value)
    print("Mass:", volume.value*rho_val)
    import matplotlib.pyplot as plt
    plt.plot(np.linspace(0, T, Nsteps+1), plate_sim.tip_disp_history)
    plt.legend(["dt="+str(dt)])
    plt.show()

if run_check_derivatives:
    sim = csdl.experimental.PySimulator(recorder)
    sim.check_totals([total_strain_energy],[thickness])
    # sim.check_totals([aggregated_stress],[force_vector])

if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import PySLSQP
    thickness.set_as_design_variable(upper=0.2, lower=2E-2)

    mass_0 = rho_val*h_val*width*length
    mass = volume*rho_val
    mass.add_name('mass')
    mass.set_as_constraint(lower=mass_0, upper=mass_0)
    total_strain_energy.set_as_objective()
    # aggregated_stress.set_as_objective(scaler=1e-4)


    sim = csdl.experimental.PySimulator(recorder)

    prob = CSDLAlphaProblem(problem_name='dynamic_plate_thickness_min_total_strain_energy', simulator=sim)
    optimizer = PySLSQP(prob, solver_options={'maxiter':200, 'acc':1e-8})

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()
    print("Optimization results:")
    print(" "*4, total_strain_energy.names, total_strain_energy.value)
    print(" "*4, mass.names, mass.value)

    FOLDER_NAME = "optimization_results/"

    xdmf_file_thickness = dolfinx.io.XDMFFile(plate_sim.comm, FOLDER_NAME+"thickness.xdmf", "w")
    xdmf_file_thickness.write_mesh(plate_sim.mesh)
    xdmf_file_thickness.write_function(plate_sim.t)

    # run plate_sim with optimized thickness and save dynamic displacements
    plate_sim.reset_solution_vectors()
    svk_res = plate_sim.SVK_residual()
    plate_sim.solve_dynamic_problem(svk_res, saving_outputs=True, PATH=FOLDER_NAME)
    import matplotlib.pyplot as plt
    plt.plot(np.linspace(0, T, Nsteps+1), plate_sim.tip_disp_history)
    plt.legend(["dt="+str(dt)])
    plt.show()

recorder.stop()

# '''
# 4. post-processing
# '''
# w = shell_model.fea.states_dict['disp_solid']['function']
# u_mid = w.sub(0).collapse().x.array
# theta = w.sub(1).collapse().x.array

# print("Tip deflection:", max(abs(u_mid)))
# print("Total strain energy:", total_strain_energy.value)
# print("Mass:", mass.value)
# print("Aggregated stress:", aggregated_stress.value)

# print("  Number of elements = "+str(nel))
# print("  Number of vertices = "+str(nn))






