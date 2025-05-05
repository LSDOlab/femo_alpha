'''Shell thickness optimization of a cantilever plate '''
'''
Thickness optimization of cantilever plate with Reissner-Mindlin shell elements.
This example uses pre-built Reissner-Mindlin shell model in FEMO

Author: Ru Xiang
Date: 2024-06-20
'''

import dolfinx
from mpi4py import MPI
import csdl_alpha as csdl
import numpy as np

from femo_alpha.rm_shell.rm_shell_model import RMShellModel
from femo_alpha.fea.utils_dolfinx import createCustomMeasure

run_verify_forward_eval = True
run_check_derivatives = False
run_optimization = False
element_wise_material = False

'''
1. Define the mesh
'''

plate = [#### quad mesh ####
        "plate_2_10_quad_4_20.xdmf",
        "plate_2_10_quad_8_40.xdmf",
        "plate_2_10_quad_10_50.xdmf",]

filename = "./plate_meshes/"+plate[0]
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
f_d = 5.  # force magnitude

'''
2. Define the boundary conditions
'''
# clamped root boundary condition
DOLFIN_EPS = 3E-16
def ClampedBoundary(x):
    return np.less(x[0], 0.0+DOLFIN_EPS)

'''
3. Set up csdl recorder and run the simulation
'''
recorder = csdl.Recorder(inline=True)
recorder.start()

force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
force_vector.value[:, 2] = f_d*0.00868/0.04627 # body force per node

pressure_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
pressure_vector.value[:, 2] = f_d # body force per unit surface area

node_disp = csdl.Variable(value=np.zeros((nn, 3)), name='node_disp')
node_disp.add_name('node_disp')

if element_wise_material:
    thickness = csdl.Variable(value=h_val*np.ones(nel), name='thickness')
    E = csdl.Variable(value=E_val*np.ones(nel), name='E')
    nu = csdl.Variable(value=nu_val*np.ones(nel), name='nu')
    density = csdl.Variable(value=rho_val*np.ones(nel), name='density')
else:
    thickness = csdl.Variable(value=h_val*np.ones(nn), name='thickness')
    E = csdl.Variable(value=E_val*np.ones(nn), name='E')
    nu = csdl.Variable(value=nu_val*np.ones(nn), name='nu')
    density = csdl.Variable(value=rho_val*np.ones(nn), name='density')
        
# All FEA variables will be saved to xdmf files if record=True
shell_model = RMShellModel(mesh, shell_bc_func=ClampedBoundary, 
                           element_wise_material=element_wise_material,
                           record=True)
shell_outputs = shell_model.evaluate(pressure_vector, thickness, E, nu, density,
                                        node_disp,
                                        debug_mode=False,
                                        is_pressure=True)
# shell_outputs = shell_model.evaluate(force_vector, thickness, E, nu, density,
#                                         node_disp,
#                                         debug_mode=False,
#                                         is_pressure=False)
disp_solid = shell_outputs.disp_solid
compliance = shell_outputs.compliance
aggregated_stress = shell_outputs.aggregated_stress
mass = shell_outputs.mass
von_mises = shell_outputs.stress
F_solid = shell_outputs.F_solid

if run_verify_forward_eval:
    Ix = width*h_val**3/12

    print("Euler-Beinoulli Beam theory deflection:",
        float(f_d*width*length**4/(8*E_val*Ix)))
    print("Reissner-Mindlin FE deflection:", max(disp_solid.value))
    print("Compliance:", compliance.value)
    print("aggreated_stress:", aggregated_stress.value)

if run_check_derivatives:
    sim = csdl.experimental.PySimulator(recorder)
    sim.check_totals([compliance],[thickness])
    # sim.check_totals([aggregated_stress],[force_vector])

if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import PySLSQP
    thickness.set_as_design_variable(upper=0.2, lower=2E-2)

    mass_0 = rho_val*h_val*width*length
    mass.set_as_constraint(lower=mass_0, upper=mass_0)
    # compliance.set_as_objective()
    aggregated_stress.set_as_objective(scaler=1e-4)

    sim = csdl.experimental.PySimulator(recorder)

    prob = CSDLAlphaProblem(problem_name='plate_thickness_min_stress', simulator=sim)
    optimizer = PySLSQP(prob, solver_options={'maxiter':200, 'acc':1e-9})
    # optimizer = PySLSQP(prob, ftol=1e-9, maxiter=1000, outputs=['x'])

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()
    print("Optimization results:")
    print(" "*4, compliance.names, compliance.value)
    print(" "*4, mass.names, mass.value)

recorder.stop()

'''
4. post-processing
'''
w = shell_model.fea.states_dict['disp_solid']['function']
u_mid = w.sub(0).collapse().x.array
theta = w.sub(1).collapse().x.array

print("Tip deflection:", max(abs(u_mid)))
print("Compliance:", compliance.value)
print("Mass:", mass.value)
print("Aggregated stress:", aggregated_stress.value)
print("Von Mises stress:", max(von_mises.value))

print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))






