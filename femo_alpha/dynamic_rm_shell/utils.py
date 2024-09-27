import numpy as np
import petsc4py
from petsc4py import PETSc
# from dolfinx import *

from dolfinx.fem import form
from dolfinx.fem.petsc import assemble_matrix

def stack_array_into_vector(inp_arr):
    return np.ravel(inp_arr, order='F')

def reshape_vector_into_array(inp_vec, n_cols):
    n_rows = int(inp_vec.shape[0]/n_cols)
    if inp_vec.shape[0] - n_rows*n_cols != 0:
        raise ValueError("`inp_vec` of shape {} cannot be reshaped into {} rows and {} columns".format(inp_vec.shape, n_rows, n_cols))
    return np.reshape(inp_vec, (n_rows, n_cols), order='F')
    
def apply_hom_DirichletBCs_to_matrix(mat, bc_dof_idxs):
    mat[bc_dof_idxs, :] = 0.  # first set the rows of the matrix to 0
    # then set the corresponding diagonal entries to 1.
    for bc_idx in bc_dof_idxs:
        mat[bc_idx, bc_idx] = 1.
    return mat

def create_dense_petsc_vec(inp_comm, length):
    vec = PETSc.Vec(inp_comm)
    vec.create(comm=inp_comm)
    vec.setSizes(length)
    vec.setUp()
    return vec

def populate_dense_petsc_vec(vec, inp_np_arr):
    vec.setValues(range(inp_np_arr.size), inp_np_arr)
    vec.assemble()
    vec.ghostUpdate()

def create_unassembled_dense_petsc_mat(inp_comm, shape):
    mat = PETSc.Mat(inp_comm)
    # mat.createAIJ([[shape[0],None],[None,shape[1]]],
                        #  comm=comm)
    mat.createAIJ([[shape[0],None],[None,shape[1]]],
                         comm=inp_comm)
    # mat.setPreallocationNNZ([shape[0],
                                    # shape[1])  # assume that matrix is full
    mat.setUp()
    return mat

def populate_petsc_mat_with_array(petsc_mat, np_array):
    # PETSc default is to store values rowwise
    # first flatten the input numpy array rowwise
    np_array_flat = np_array.flatten(order='C')
    petsc_mat.setValues(list(range(np_array.shape[0])),list(range(np_array.shape[1])),np_array_flat,addv=PETSc.InsertMode.INSERT)
    # petsc_mat[:,]
    return petsc_mat

def assemble_populated_petsc_mat(petsc_mat):
    petsc_mat.assemblyBegin()
    petsc_mat.assemblyEnd()
    # return PETScMatrix(petsc_mat)
    return petsc_mat

def create_dense_np_mat_from_form(inp_form):
    petsc_mat = assemble_matrix(form(inp_form))
    petsc_mat.assemble()
    petsc_mat.convert("dense")
    np_mat = petsc_mat.getDenseArray()
    return np_mat


# helper function to do MatMultTranspose() without all the setup steps 
# for the results vector
def multTranspose(M,b):
    """
    Returns ``M^T*b``, where ``M`` and ``b`` are DOLFIN ``GenericTensor`` and
    ``GenericVector`` objects. This function is copied from `tIGAr/common.py` 
    for convenience; the original can be found here:
        https://github.com/david-kamensky/tIGAr/blob/master/tIGAr/common.py#L97
    """
    totalDofs = M.getSizes()[1][1]
    comm = M.getComm()
    MTbv = create_dense_petsc_vec(comm, totalDofs)
    # print("M shape: {}".format(M.getSizes()))
    # print("b shape: {}".format(b.getSizes()))
    # print("MTbv shape: {}".format(MTbv.getSizes()))
    M.multTranspose(b,MTbv)
    return MTbv

def mult(M, b):
    """
    Returns ``M*b``, where ``M`` and ``b`` are DOLFIN ``GenericTensor`` and
    ``GenericVector`` objects.
    """
    totalDofs = M.getSizes()[0][0]
    comm = M.getComm()
    Mbv = create_dense_petsc_vec(comm, totalDofs)
    # print(M.getSizes())
    # print(b.getSizes())
    # print(Mbv.getSizes())
    M.mult(b, Mbv)
    return Mbv

def convert_np_array_to_petsc_mat(comm, np_array):
    petsc_mat_empty = create_unassembled_dense_petsc_mat(comm, np_array.shape)
    petsc_mat_pop = populate_petsc_mat_with_array(petsc_mat_empty, np_array)
    petsc_mat = assemble_populated_petsc_mat(petsc_mat_pop)
    return petsc_mat

def create_mumps_solver(inp_comm, mat):
    # we plug everything into the linear problem solution class
    solver = PETSc.KSP().create(inp_comm)
    solver.setOperators(mat)
    solver.setType("preonly")

    # set LU w/ MUMPS
    pc = solver.getPC()
    pc.setType("lu")
    pc.setFactorSolverType('mumps')

    # solve
    solver.setUp()
    return solver

def convert_petsc_vec_list_to_np_vec_list(inp_petsc_vec_list):
    np_vec_list = []
    for petsc_vec in inp_petsc_vec_list:
        np_vec_list += [petsc_vec.getValues(range(petsc_vec.getSize()))]
        # print("petsc vec length: {}".format(petsc_vec.getSize()))
        # print("numpy vec length: {}".format(petsc_vec.getValues(range(petsc_vec.getSize())).shape))
    return np_vec_list