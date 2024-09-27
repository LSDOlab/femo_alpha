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
element_wise_material = True

'''
1. Define the mesh
'''

plate = [#### quad mesh ####
        "pendulum_shell_mesh_2_10.xdmf",
        "plate_2_10_quad_1_5.xdmf",
        "plate_2_10_quad_2_10.xdmf",
        "plate_2_10_quad_4_20.xdmf",
        "plate_2_10_quad_8_40.xdmf",
        "plate_2_10_quad_10_50.xdmf",]

filename = "./plate_meshes/"+plate[0]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

# E_val = 4.32e8
# nu_val = 0.0
# h_val = 0.2
# rho_val = 1.0
# width = 2.
# length = 10.

E_val = 1e5 # unit: pa
nu_val = 0.
h_val = 0.01
width = 2.
length = 10.
# rho_val = 1e-3
rho_val = 10.

E_val = 70e9
nu_val = 0.3
rho_val = 2700    # Density of aluminum
# density = 19.3e3
# density = 1e5
h_val = 0.001
f_d = 10.*h_val
'''
2. Define the boundary conditions
'''
# clamped root boundary condition
DOLFIN_EPS = 3E-16
# def ClampedBoundary(x):
#     return np.less(x[0], 0.0+DOLFIN_EPS)

def ClampedBoundary(x):
    return np.greater(x[2], 0.5-DOLFIN_EPS)

'''
3. Set up csdl recorder and run the simulation
'''

# force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
# force_vector.value[:, 2] = f_d*0.00868/0.04627*105/nn # body force per node
# pressure_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
# pressure_vector.value[:, 2] = f_d # body force per unit surface area

# node_disp = csdl.Variable(value=np.zeros((nn, 3)), name='node_disp')
# node_disp.add_name('node_disp')

# if element_wise_material:
#     thickness = csdl.Variable(value=h_val*np.ones(nel), name='thickness')
#     E = csdl.Variable(value=E_val*np.ones(nel), name='E')
#     nu = csdl.Variable(value=nu_val*np.ones(nel), name='nu')
#     density = csdl.Variable(value=rho_val*np.ones(nel), name='density')
# else:
#     thickness = csdl.Variable(value=h_val*np.ones(nn), name='thickness')
#     E = csdl.Variable(value=E_val*np.ones(nn), name='E')
#     nu = csdl.Variable(value=nu_val*np.ones(nn), name='nu')
#     density = csdl.Variable(value=rho_val*np.ones(nn), name='density')
        
# # All FEA variables will be saved to xdmf files if record=True
# shell_model = RMShellModel(mesh, element_type="CG2CG1",
#                         shell_bc_func=ClampedBoundary, 
#                         element_wise_material=element_wise_material,
#                         record=True)
# shell_outputs = shell_model.evaluate(pressure_vector, thickness, E, nu, density,
#                                         node_disp,
#                                         debug_mode=False,
#                                         is_pressure=True)
# # shell_outputs = shell_model.evaluate(force_vector, thickness, E, nu, density,
# #                                         node_disp,
# #                                         debug_mode=False,
# #                                         is_pressure=False)
# disp_solid = shell_outputs.disp_solid
# compliance = shell_outputs.compliance
# aggregated_stress = shell_outputs.aggregated_stress
# mass = shell_outputs.mass
# F_solid = shell_outputs.F_solid

from shell_analysis_fenicsx import ShellElement, MaterialModel, ElasticModel, solveNonlinear
from ufl import Constant
from dolfinx.fem import Function
import ufl
from dolfinx.fem import locate_dofs_geometrical, dirichletbc
def initial_solve(mesh, E_val, nu_val, h_val):
    # E = Constant(mesh,E_val) # Young's modulus
    # nu = Constant(mesh,nu_val) # Poisson ratio
    # h = Constant(mesh,h_val) # Shell thickness
    E = E_val # Young's modulus
    nu = nu_val # Poisson ratio
    h = h_val # Shell thickness
    f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area

    element_type = "CG2CG1"
    #element_type = "CG2CR1"

    element = ShellElement(
                    mesh,
                    element_type,
    #                inplane_deg=3,
    #                shear_deg=3
                    )
    W = element.W
    wfunc = Function(W)
    dx_inplane, dx_shear = element.dx_inplane, element.dx_shear


    #### Compute the CLT model from the material properties (for single-layer material)
    material_model = MaterialModel(E=E,nu=nu,h=h)
    elastic_model = ElasticModel(mesh,wfunc,material_model.CLT)
    elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
    F = elastic_model.weakFormResidual(elastic_energy, f)

    ######### Set the BCs to have all the dofs equal to 0 on the left edge ##########
    # Define BCs geometrically
    locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),
                                        lambda x: np.isclose(x[2], 0.5 ,atol=1e-6))
    locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
                                        lambda x: np.isclose(x[2], 0.5 ,atol=1e-6))
    ubc=  Function(W)
    with ubc.vector.localForm() as uloc:
        uloc.set(0.)

    bcs = [dirichletbc(ubc, locate_BC1, W.sub(0)),
            dirichletbc(ubc, locate_BC2, W.sub(1)),
        ]


    ########## Solve with Newton solver wrapper: ##########
    solveNonlinear(F,wfunc,bcs)
    disp_solid = wfunc.x.array

    u_mid, _ = wfunc.split()
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_mid)
    return disp_solid

disp_solid = initial_solve(mesh, E_val, nu_val, h_val)
disp_solid[:] = 0.
recorder = csdl.Recorder(inline=True)
recorder.start()

force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
force_vector.value[:, 2] = f_d*0.00868/0.04627*105/nn # body force per node

# node_disp = csdl.Variable(value=np.zeros((nn, 3)), name='node_disp')
# node_disp.add_name('node_disp')

thickness = csdl.Variable(value=h_val*np.ones(nel), name='thickness')
E = csdl.Variable(value=E_val*np.ones(nel), name='E')
nu = csdl.Variable(value=nu_val*np.ones(nel), name='nu')
density = csdl.Variable(value=rho_val*np.ones(nel), name='density')

shell_model = RMShellModel(mesh, element_type="CG2CG1",
                        shell_bc_func=ClampedBoundary, 
                        element_wise_material=element_wise_material,
                        record=True, dynamic_only=True)

# w = dolfinx.fem.Function(shell_model.shell_pde.W)
w = csdl.Variable(value=disp_solid, name='disp_solid')
wdot = csdl.Variable(value=0., shape=disp_solid.shape, name='wdot')
wddot = csdl.Variable(value=0., shape=disp_solid.shape, name='wwdot')
# wdot.value[:] = 0.1
# wddot.value[:] = 0.1
# w.value[:] = 0.1
dynamic_outputs = shell_model.evaluate_dynamic_residual(disp_solid=w, wdot=wdot, wddot=wddot, 
                                        force_vector=force_vector, thickness=thickness, 
                                        E=E, nu=nu, density=density)
dynamic_residual = dynamic_outputs.dynamic_residual
# # exit()
# print(shell_model.dynamic_fea.states_dict.keys())
# print(shell_model.dynamic_fea.outputs_dict.keys())
# print(shell_model.dynamic_fea.outputs_field_dict.keys())
# print(shell_model.dynamic_fea.outputs_residual_dict.keys())
# print(shell_model.dynamic_fea.inputs_dict.keys())
# exit()

# K_mat, M_mat = shell_model.evaluate_modal_fea(E_val, nu_val, h_val, rho_val)
# eta_m = 0.02
# eta_k = 0.05
# C_mat = eta_m*M_mat + eta_k*K_mat
# res = np.matmul(K_mat, disp_solid.value) \
#     + np.matmul(M_mat, wddot.value) \
#     + np.matmul(C_mat, wdot.value) 
#     # - F_solid.value

# print(K_mat[:10,:10])
# print(M_mat[:10,:10])

# print("Dynamic residual by matvec product:", res[:10])
print("Dynamic residual:", dynamic_residual.value[:10])

# csdl.derivative_utils.verify_derivatives([dynamic_residual], [w,wdot,wddot], 1.e-1, raise_on_error=False)

from femo_alpha.fea.utils_dolfinx import assemble, computePartials
w_init = w.value.copy()
fd_step = 1e-4
fd_dRdw = np.zeros((w.shape[0],w.shape[0]))

# for i in range(w.shape[0]):
#     w.value[:] = w_init
#     w.value[i] += fd_step
#     dynamic_outputs_i = shell_model.evaluate_dynamic_residual(disp_solid=w, wdot=wdot, wddot=wddot, 
#                                         force_vector=force_vector, thickness=thickness, 
#                                         E=E, nu=nu, density=density)
#     fd_dRdw[i,:] = (dynamic_outputs_i.dynamic_residual.value - dynamic_residual.value) / fd_step

dRdw = assemble(
                computePartials(
                    shell_model.dynamic_fea.outputs_residual_dict["dynamic_residual"]["form"], 
                    shell_model.dynamic_fea.inputs_dict["disp_solid"]["function"]
                ),
                dim=2,
            )

# wdot_init = wdot.value.copy()
# fd_step = 1e-4
# fd_dRdwdot = np.zeros((wdot.shape[0],wdot.shape[0]))
# for i in range(wdot.shape[0]):
#     wdot.value[:] = wdot_init
#     wdot.value[i] += fd_step
#     dynamic_outputs_i = shell_model.evaluate_dynamic_residual(disp_solid=w, wdot=wdot, wddot=wddot, 
#                                         force_vector=force_vector, thickness=thickness, 
#                                         E=E, nu=nu, density=density)
#     fd_dRdwdot[i,:] = (dynamic_outputs_i.dynamic_residual.value - dynamic_residual.value) / fd_step

dRdwdot = assemble(
                computePartials(
                    shell_model.dynamic_fea.outputs_residual_dict["dynamic_residual"]["form"], 
                    shell_model.dynamic_fea.inputs_dict["wdot"]["function"]
                ),
                dim=2,
            )

# wddot_init = wddot.value.copy()
# fd_step = 1e-4
# fd_dRdwddot = np.zeros((wddot.shape[0],wddot.shape[0]))
# for i in range(wddot.shape[0]):
#     wddot.value[:] = wddot_init
#     wddot.value[i] += fd_step
#     dynamic_outputs_i = shell_model.evaluate_dynamic_residual(disp_solid=w, wdot=wdot, wddot=wddot, 
#                                         force_vector=force_vector, thickness=thickness, 
#                                         E=E, nu=nu, density=density)
#     fd_dRdwddot[i,:] = (dynamic_outputs_i.dynamic_residual.value - dynamic_residual.value) / fd_step

dRdwddot = assemble(
                computePartials(
                    shell_model.dynamic_fea.outputs_residual_dict["dynamic_residual"]["form"], 
                    shell_model.dynamic_fea.inputs_dict["wddot"]["function"]
                ),
                dim=2,
            )

# print("-"*40)
# print("dRdw:")
# print("FD norm:", np.linalg.norm(fd_dRdw))
# print("Analytical norm:", np.linalg.norm(dRdw))

# print("-"*40)
# print("dRdwdot:")
# # print("FD norm:", np.linalg.norm(fd_dRdwdot))
# print("Analytical norm:", np.linalg.norm(dRdwdot))
# print("-"*40)
# print("dRdwddot:")
# # print("FD norm:", np.linalg.norm(fd_dRdwddot))
# print("Analytical norm:", np.linalg.norm(dRdwddot))
print("dRdw")
print(dRdw[:10,:10])
print("dRdwdot")
print(dRdwdot[:10,:10])
print("dRdwddot")
print(dRdwddot[:10,:10])

recorder.stop()
# wdot.value[:] = 0.1
# dynamic_outputs = shell_model.evaluate_dynamic_residual(disp_solid, wdot, wddot, 
#                                         force_vector, thickness, E, nu, density)
# dynamic_residual = dynamic_outputs.dynamic_residual
# print("Dynamic residual:", dynamic_residual.value[:10])
exit()









if run_verify_forward_eval:
    Ix = width*h_val**3/12
    print("Euler-Beinoulli Beam theory deflection:",
        float(f_d*width*length**4/(8*E_val*Ix)))
    print("Reissner-Mindlin FE deflection:", max(disp_solid.value))
    print(disp_solid.value)
    np.savetxt("disp_solid.txt", 0.1*disp_solid.value)


if run_check_derivatives:
    sim = csdl.experimental.PySimulator(recorder)
    sim.check_totals([aggregated_stress],[thickness])
    # sim.check_totals([aggregated_stress],[force_vector])

if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import SLSQP
    thickness.set_as_design_variable(upper=10, lower=1E-2)

    mass_0 = rho_val*h_val*width*length
    mass.set_as_constraint(lower=mass_0, upper=mass_0)
    compliance.set_as_objective()
    sim = csdl.experimental.PySimulator(recorder)

    prob = CSDLAlphaProblem(problem_name='plate_thickness', simulator=sim)

    optimizer = SLSQP(prob, ftol=1e-9, maxiter=1000, outputs=['x'])

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

print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))






