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

from evaluate_modal_fea import evaluate_modal_fea


'''
3. Set up csdl recorder and run the simulation
'''
recorder = csdl.Recorder(inline=True)
recorder.start()

'''
1. Define the mesh
'''

plate = [#### quad mesh ####
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

E_val = 1e5 # unit: pa
nu_val = 0.
h_val = 0.01
width = 2.
length = 10.
rho_val = 1e-3
f_d = 10.*h_val

# # '''
# # 2. Define the boundary conditions
# # '''
# # # clamped root boundary condition
# # DOLFIN_EPS = 3E-16
# # def ClampedBoundary(x):
# #     return np.less(x[0], 0.0+DOLFIN_EPS)

# # # All FEA variables will be saved to xdmf files if record=True
# # shell_model = RMShellModel(mesh, element_type="CG2CG1",
# #                         shell_bc_func=ClampedBoundary, 
# #                         element_wise_material=element_wise_material,
# #                         PENALTY_BC=False,
# #                         record=True)




# from dolfinx.fem import Constant
# import shell_analysis_fenicsx as shl
# from dolfinx.fem import Function
# import ufl



# E = Constant(mesh,E_val) # Young's modulus
# nu = Constant(mesh,nu_val) # Poisson ratio
# h = Constant(mesh,h_val) # Shell thickness
# rho = Constant(mesh,rho_val)
# f_0 = Constant(mesh, (0.0,0.0,0.0))

# element_type = "CG2CG1"
# element = shl.ShellElement(
#                 mesh,
#                 element_type,
#                 )
# W = element.W

# w = Function(W)
# dx_inplane, dx_shear = element.dx_inplane, element.dx_shear
# # print(E_val, nu_val, h_val, rho_val)
# # print("w: ", w.x.array)
# # exit()
# #### Compute the CLT model from the material properties (for single-layer material)
# material_model = shl.MaterialModel(E=E,nu=nu,h=h)
# elastic_model = shl.ElasticModel(mesh, w, material_model.CLT)
# elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane, dx_shear)

# dWint = elastic_model.weakFormResidual(elastic_energy, f_0)
# dWmass = elastic_model.inertialResidual(rho, h)

# K_form = ufl.derivative(dWint, w)
# M_form = ufl.derivative(dWmass, w)
# # K = assemble_matrix(form(K_form), bcs)
# K = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(K_form))
# M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(M_form))
# K.assemble()
# M.assemble()

# K_dense = K.convert("dense")
# M_dense = M.convert("dense")

# K_mat = K_dense.getDenseArray()
# M_mat = M_dense.getDenseArray()


# disp_solid = csdl.Variable(value=0., shape=(len(w.x.array),), name='disp_solid')
# wdot = csdl.Variable(value=0., shape=disp_solid.shape, name='wdot')
# wddot = csdl.Variable(value=0., shape=disp_solid.shape, name='wwdot')
# wdot.value[:] = 0.1
# wddot.value[:] = 0.1
# disp_solid.value[:] = 0.1
# eta_m = 0.02
# eta_k = 0.05
# C_mat = eta_m*M_mat + eta_k*K_mat
# res = np.matmul(K_mat, disp_solid.value) \
#     + np.matmul(M_mat, wddot.value) \
#     + np.matmul(C_mat, wdot.value) 
#     # - F_solid.value


K_mat, M_mat = evaluate_modal_fea(mesh, E_val, nu_val, h_val, rho_val)
print(K_mat[:10,:10])
print(M_mat[:10,:10])

disp_solid = csdl.Variable(value=0., shape=(K_mat.shape[0],), name='disp_solid')
wdot = csdl.Variable(value=0., shape=disp_solid.shape, name='wdot')
wddot = csdl.Variable(value=0., shape=disp_solid.shape, name='wwdot')
wdot.value[:] = 0.1
wddot.value[:] = 0.1
disp_solid.value[:] = 0.1
eta_m = 0.02
eta_k = 0.05
C_mat = eta_m*M_mat + eta_k*K_mat
res = np.matmul(K_mat, disp_solid.value) \
    + np.matmul(M_mat, wddot.value) \
    + np.matmul(C_mat, wdot.value) 
    # - F_solid.value

print("Dynamic residual by matvec product:", res[:10])
