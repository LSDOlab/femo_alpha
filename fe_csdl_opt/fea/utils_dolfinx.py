"""
Reusable functions for the PETSc and UFL operations
"""

import dolfinx
import dolfinx.io
from ufl import Identity, dot, dx, derivative
from dolfinx.mesh import create_unit_square
from dolfinx.fem import form, assemble_scalar, Function
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix,
                        NonlinearProblem, apply_lifting, set_bc)
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from scipy.spatial import KDTree
from mpi4py import MPI
import numpy as np
from scipy.spatial import KDTree


def findNodeIndices(node_coordinates, coordinates):
    """
    Find the indices of the closest nodes, given the `node_coordinates`
    for a set of nodes and the `coordinates` for all of the vertices
    in the mesh, by using scipy.spatial.KDTree
    """
    tree = KDTree(coordinates)
    dist, node_indices = tree.query(node_coordinates)
    return node_indices

def createUnitSquareMesh(n):
    """
    Create unit square mesh for test purposes
    """
    return create_unit_square(MPI.COMM_WORLD, n, n)

def getFormArray(F):
    """
    Compute the array representation of the Form
    """
    return assemble_vector(form(F)).getArray()

def getFuncArray(v):
    """
    Compute the array representation of the Function
    """
    return v.vector.getArray()

def setFuncArray(v, v_array):
    """
    Set the fuction based on the array
    """
    v.vector[:] = v_array
    v.vector.assemble()
    v.vector.ghostUpdate()

def assembleMatrix(M, bcs=[]):
    """
    Compute the array representation of the matrix form
    """
    M_ = assemble_matrix(form(M), bcs=bcs)
    M_.assemble()
    return M_

def assembleSystem(J, F, bcs=[]):
    """
    Compute the array representations of the linear system
    """
    a = form(J)
    L = form(F)
    A = assemble_matrix(a, bcs=bcs)
    A.assemble()
    L = form(F)
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    return A, b

def assemble(f, dim=0, bcs=[]):
    if dim == 0:
        return assembleScalar(f)
    elif dim == 1:
        return assembleVector(f)
    elif dim == 2:
        M = assembleMatrix(f, bcs=bcs)
        return convertToDense(M.copy())
    else:
        return TypeError("Invalid type for assembly.")

def assembleScalar(c):
    """
    Compute the array representation of the scalar form
    """
    return assemble_scalar(form(c))


def assembleVector(v):
    """
    Compute the array representation of the vector form
    """
    return assemble_vector(form(v)).array


def errorNorm(v, v_ex):
    """
    Calculate the L2 norm of two functions
    """
    comm = MPI.COMM_WORLD
    error = form((v - v_ex)**2 * dx)
    E = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))
    return E


def transpose(A):
    """
    Transpose for matrix of DOLFIN type
    """
    return A.transpose(PETSc.Mat(MPI.COMM_WORLD))


def computeMatVecProductFwd(A, x):
    """
    Compute y = A * x
    A: PETSc matrix
    x: ufl function
    """
    y = A*x.vector
    y.assemble()
    return y.getArray()


def computeMatVecProductBwd(A, R):
    """
    Compute y = A.T * R
    A: PETSc matrix
    R: ufl function
    """
    row, col = A.getSizes()
    y = PETSc.Vec().create()
    y.setSizes(col)
    y.setUp()
    A.multTranspose(R.vector,y)
    y.assemble()
    return y.getArray()


def convertToDense(A_petsc):
    """
    Convert the PETSc matrix to a dense numpy array
    (super unefficient, only used for debugging purposes)
    """
    A_petsc.assemble()
    A_dense = A_petsc.convert("dense")
    return A_dense.getDenseArray()


def update(v, v_values):
    """
    Update the nodal values in every dof of the DOLFIN function `v`
    according to `v_values`.
    -------------------------
    v: dolfin function
    v_values: numpy array
    """
    if len(v_values) == 1:
        v.vector.set(v_values)
    else:
        setFuncArray(v, v_values)

I = Identity(2)
def gradx(f,uhat):
    """
    Convert the differential operation from the reference domain
    to the measure in the deformed configuration based on the mesh
    movement of `uhat`
    --------------------------
    f: DOLFIN function for the solution of the physical problem
    uhat: DOLFIN function for mesh movements
    """
    return dot(grad(f), inv(I + grad(uhat)))


def J(uhat):
    """
    Compute the determinant of the deformation gradient used in the
    integration measure of the deformed configuration wrt the the
    reference configuration.
    ---------------------------
    uhat: DOLFIN function for mesh movements
    """
    return det(I + grad(uhat))


def computePartials(form, function):
    return derivative(form, function)

def createFunction(function):
    return Function(function.function_space)

def solveNonlinear(F, w, bcs=[],
                    abs_tol=1e-50,
                    rel_tol=1e-30,
                    max_it=3,
                    error_on_nonconvergence=False,
                    report=False):

    """
    Wrap up the nonlinear solver for the problem F(w)=0 and
    returns the solution
    """

    problem = NonlinearProblem(F, w, bcs)

    # Set the initial guess of the solution
    with w.vector.localForm() as w_local:
        w_local.set(0.9)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    if report == True:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    # Set the Newton solver options
    solver.atol = abs_tol
    solver.rtol = rel_tol
    solver.max_it = max_it
    solver.error_on_nonconvergence = error_on_nonconvergence
    opts = PETSc.Options()
    opts["nls_solve_pc_factor_mat_solver_type"] = "mumps"
    solver.solve(w)


def solveKSP(A, b, x):
    """
    Wrap up the KSP solver for the linear system Ax=b
    """
    ######### Set up the KSP solver ###############

    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)

    # additive Schwarz method
    pc = ksp.getPC()
    pc.setType("asm")

    ksp.setFromOptions()
    ksp.setUp()

    localKSP = pc.getASMSubKSP()[0]
    localKSP.setType(PETSc.KSP.Type.GMRES)
    localKSP.getPC().setType("lu")
    localKSP.setTolerances(1.0e-12)
    #ksp.setGMRESRestart(30)
    ksp.setConvergenceHistory()
    ksp.solve(b, x)
    history = ksp.getConvergenceHistory()

def solveKSP_mumps(A, b, x):
    """
    Implementation of KSP solution of the linear system Ax=b using MUMPS
    """

    # setup petsc for pre-only solve
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")

    # set LU w/ MUMPS
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType('mumps')

    # solve
    ksp.setUp()
    ksp.solve(b, x)