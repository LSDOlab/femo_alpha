
import dolfinx
from dolfinx.fem import Constant
import shell_analysis_fenicsx as shl
from dolfinx.fem import Function
import ufl


def evaluate_modal_fea(mesh, E_val, nu_val, h_val, rho_val):
    E = Constant(mesh,E_val) # Young's modulus
    nu = Constant(mesh,nu_val) # Poisson ratio
    h = Constant(mesh,h_val) # Shell thickness
    rho = Constant(mesh,rho_val)
    f_0 = Constant(mesh, (0.0,0.0,0.0))

    element_type = "CG2CG1"
    element = shl.ShellElement(
                    mesh,
                    element_type,
                    )
    W = element.W

    w = Function(W)
    dx_inplane, dx_shear = element.dx_inplane, element.dx_shear
    # print(E_val, nu_val, h_val, rho_val)
    # print("w: ", w.x.array)
    # exit()
    #### Compute the CLT model from the material properties (for single-layer material)
    material_model = shl.MaterialModel(E=E,nu=nu,h=h)
    elastic_model = shl.ElasticModel(mesh, w, material_model.CLT)
    elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane, dx_shear)

    dWint = elastic_model.weakFormResidual(elastic_energy, f_0)
    dWmass = elastic_model.inertialResidual(rho, h)

    K_form = ufl.derivative(dWint, w)
    M_form = ufl.derivative(dWmass, w)
    # K = assemble_matrix(form(K_form), bcs)
    K = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(K_form))
    M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(M_form))
    K.assemble()
    M.assemble()

    K_dense = K.convert("dense")
    M_dense = M.convert("dense")

    K_mat = K_dense.getDenseArray()
    M_mat = M_dense.getDenseArray()

    return K_mat, M_mat
