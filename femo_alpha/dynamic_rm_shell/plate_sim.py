import mpi4py
mpi4py.rc.threads = False
from mpi4py import MPI 
from time import perf_counter

import numpy as np
import scipy.sparse as scsp
from logging import WARNING
from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, FunctionSpace, VectorFunctionSpace)
from dolfinx.fem.petsc import _assemble_matrix_mat
# from dolfinx.mesh import locate_entities
from dolfinx.fem.petsc import create_matrix
from ufl import adjoint
# from petsc4py import PETSc
from shell_analysis_fenicsx import *
from femo_alpha.dynamic_rm_shell.nonlinear_utils import *
import basix
import scipy.sparse as sp

def extractTipDispDolfinx(w, x_tip=[10.,0.,0.], cell_tip=18):
    return w.sub(0).eval(x_tip, cell_tip)[-1]

class BC_list(object):
    """
    Setting Dirichlet boundary condition
    """
    def __init__(self, W_space, direction=0, custom_bc_func=None):
        self.W = W_space
        self.direction = direction
        self.custom_bc_func = custom_bc_func

    def return_affected_dof_tuple(self):
        if self.custom_bc_func is not None:
            locate_BC1 = locate_dofs_geometrical((self.W.sub(0), self.W.sub(0).collapse()[0]),
                                        self.custom_bc_func)
            locate_BC2 = locate_dofs_geometrical((self.W.sub(1), self.W.sub(1).collapse()[0]),
                                        self.custom_bc_func)
        else:
            locate_BC1 = locate_dofs_geometrical((self.W.sub(0), self.W.sub(0).collapse()[self.direction]),
                                            lambda x: np.isclose(x[0], 0. ,atol=1e-6))
            locate_BC2 = locate_dofs_geometrical((self.W.sub(1), self.W.sub(1).collapse()[self.direction]),
                                            lambda x: np.isclose(x[0], 0. ,atol=1e-6))
        return locate_BC1, locate_BC2

    def return_bc_list(self):
        ######### Set the BCs to have all the dofs equal to 0 on the left edge ##########
        # Define BCs geometrically
        locate_BC1, locate_BC2 = self.return_affected_dof_tuple()
        ubc = Function(self.W)
        with ubc.vector.localForm() as uloc:
            uloc.set(0.)

        bcs = [dirichletbc(ubc, locate_BC1, self.W.sub(0)),
            dirichletbc(ubc, locate_BC2, self.W.sub(1)),
            ]
        return bcs

class PlateSim(object):
    """
    Implements St. Venant--Kirchhoff constitutive model for 
    isogeometric shell.
    """
    def __init__(self, mesh, E, nu, rho, dt, Nsteps, element_wise_thickness=False,
                custom_bc_func=None, add_self_weight=False, g_factor=None,
                quad_deg=3, comm=MPI.COMM_WORLD):
        self.mesh = mesh
        self.E = E
        self.nu = nu
        self.rho = rho
        self.dt = dt
        self.Nsteps = Nsteps
        self.time_levels = Nsteps + 1  # this variable includes the initial condition as a time level
        self.comm = comm

        self.nn = mesh.topology.index_map(0).size_local
        self.nel = mesh.topology.index_map(mesh.topology.dim).size_local

        self.ALPHA = 1

        self.quad_deg = quad_deg

        # Create function spaces and functions of unknowns
        element_type = "CG2CG1"
        self.element = ShellElement(
                self.mesh,
                element_type,
                inplane_deg=self.quad_deg,
                shear_deg=self.quad_deg,
                )
        self.W = self.element.W

        # define thickness function space
        self.element_wise_thickness = element_wise_thickness
        if element_wise_thickness:
            self.W_t = FunctionSpace(self.mesh, ("DG", 0))
        else:
            self.W_t = FunctionSpace(self.mesh, ("CG", 1))
        self.t = Function(self.W_t)

        # define force function space
        self.W_f = VectorFunctionSpace(self.mesh, ("CG", 1))
        self.f = Function(self.W_f)
        # self.f_history = np.zeros((self.time_levels, self.nn*3))

        self.tip_disp_history = np.zeros((self.time_levels,))
        self.opt_iter = 0
        
        # compute "average" cell size as area/num_cells
        
        # define von Mises stress function space
        self.W_s = FunctionSpace(self.mesh, ("DG", 1))
        self.vmstress = Function(self.W_s)
        
        self.x_tip = None
        self.cell_tip = None

        # define current quantities
        self.w = Function(self.W)
        self.u, self.theta = split(self.w)
        self.dw = TestFunction(self.W)
        self.du_mid, self.dtheta = split(self.dw)

        # Quantities from the previous time step
        self.w_old = Function(self.W)
        self.u_old, self.theta_old = split(self.w_old)
        self.wdot_old = Function(self.W)
        self.udot_old, self.thetadot_old = split(self.wdot_old)

        # Set up the time integration scheme
        # TODO: Check whether this has to be defined in a separate function
        #self.u_mid = Constant(self.mesh, 0.5)*(self.u_old + self.u)
        #self.theta_mid = Constant(self.mesh, 0.5)*(self.theta_old + self.theta)
        self.w_mid = Constant(self.mesh, 0.5)*(self.w_old + self.w)
        self.udot = Constant(self.mesh, 2/self.dt)*self.u - Constant(self.mesh, 2/self.dt)*self.u_old - self.udot_old
        self.uddot = (self.udot - self.udot_old)/self.dt
        self.thetadot = Constant(self.mesh, 2/self.dt)*self.theta - \
            Constant(self.mesh, 2/self.dt)*self.theta_old - self.thetadot_old
        self.thetaddot = (self.thetadot - self.thetadot_old)/self.dt


        self.bc_obj = BC_list(self.W, custom_bc_func=custom_bc_func)
        self.bc_list = self.bc_obj.return_bc_list()
        self.add_self_weight = add_self_weight
        self.g_factor = g_factor

        self.fe_dofs = self.w.vector.size
        self.num_var = self.t.vector.size

        # self.nodal_disp_map = self.construct_nodal_disp_map()


    def set_up_tip_dofs(self, x_tip, cell_tip):
        self.x_tip = x_tip
        self.cell_tip = cell_tip

    def update_t(self, t_array):
        self.t.vector.setValues(range(t_array.size), t_array)
        self.t.vector.assemble()
        self.t.vector.ghostUpdate()
    
    def update_f(self, f_array):
        self.f.vector.setValues(range(f_array.size), f_array)
        self.f.vector.assemble()
        self.f.vector.ghostUpdate()

    def update_f_history(self, f_history_array):
        self.f_history = f_history_array
       
    def update_materialmodel(self):
        self.material_model = MaterialModel(E=self.E,nu=self.nu,h=self.t)

    @staticmethod
    def set_solution_vector(func_obj, arr):
        func_obj.vector.setValues(range(arr.size), arr)
        func_obj.vector.assemble()
        func_obj.vector.ghostUpdate()

    def set_solution_vectors_for_adjoint(self, w, w_old, wdot_old):
        self.set_solution_vector(self.w, w)
        self.set_solution_vector(self.w_old, w_old)
        self.set_solution_vector(self.wdot_old, wdot_old)

    def reset_solution_vectors(self):
        self.set_solution_vector(self.w, np.array([0.]*self.fe_dofs))
        self.set_solution_vector(self.w_old, np.array([0.]*self.fe_dofs))
        self.set_solution_vector(self.wdot_old, np.array([0.]*self.fe_dofs))

    def SVK_residual(self):
        w_temp = Function(self.W)
        self.update_materialmodel()
        elastic_model = DynamicElasticModel(self.mesh, w_temp, self.material_model.CLT)
        dx_inplane, dx_shear = self.element.dx_inplane, self.element.dx_shear
        elastic_energy = elastic_model.elasticEnergy(self.E, self.t, dx_inplane, dx_shear)

        dWint = elastic_model.weakFormResidual(self.ALPHA, elastic_energy, w_temp, self.dw, self.f)
        dWint_mid = ufl.replace(dWint, {w_temp: self.w_mid})

        # Inertial contribution to the residual:
        dWmass = elastic_model.inertialResidual(self.rho,self.t,self.uddot,self.thetaddot)

        # Add weight force 
        if self.add_self_weight:
            if self.g_factor is None:
                g_factor = -1.0 # assume in negative z-direction
            else:
                g_factor = self.g_factor
            g = Constant(self.mesh, g_factor*9.81)
            f_d = ufl.as_vector([0.,0.,self.rho*self.t*g])
            dWext = inner(f_d,self.du_mid)*dx
        else:
            dWext = 0.0
        res = dWmass + dWint_mid - dWext
        return res

    def constructStrainEnergy(self, w):
        elastic_model = ElasticModel(self.mesh, w, self.material_model.CLT)
        elastic_energy = elastic_model.elasticEnergy(self.E, self.t, self.element.dx_inplane, self.element.dx_shear)
        return elastic_energy

    def assembleStrainEnergy(self, w):
        elastic_energy = self.constructStrainEnergy(w)
        return assemble_scalar(form(elastic_energy))

    def wdot_vector(self):
        return 2/self.dt*self.w.vector - 2/self.dt*self.w_old.vector - self.wdot_old.vector

    def compute_wdot_vector_at_time_n(self, w_arr):
        # compute wdot at the final time 
        coeff = 2/self.dt

        coeff_vec = np.empty((w_arr.shape[1]))
        coeff_vec[::2] = 2.
        coeff_vec[1::2] = -2.
        # reset first entry to 1
        coeff_vec[0] = 1.
        # set last entry to plus/minus 1 by dividing by 2
        coeff_vec[-1] = coeff_vec[-1]/2.
        # traverse through w_arr's columns in reverse order (latest solution first)
        return coeff*w_arr[:, ::-1]@coeff_vec
    
    def update_wdot_vector_at_time(self, w_vec, w_old_vec, wdot_old_vec):
        return 2/self.dt*w_vec - 2/self.dt*w_old_vec - wdot_old_vec

    def update_wdot_vector_at_time_reverse(self, w_vec, w_old_vec, wdot_vec):
        # this function is similar to self.update_wdot_vector_at_time, 
        # except that it's used when marching backwards in time (for adjoint problems)
        return 2/self.dt*w_vec - 2/self.dt*w_old_vec - wdot_vec

    def compute_solution_vectors_at_time(self, disp_arr, i, backwards_timestepping=False):
        if backwards_timestepping:
            if i <= 1:
                wdot_old = np.zeros_like(self.wdot_old.vector)
            elif i == disp_arr.shape[1]-1:        
                wdot_old = self.compute_wdot_vector_at_time_n(disp_arr[:, :i])
                # print("i = {} == disp_arr.shape[1]-1".format(i))
            else:
                wdot_old = self.update_wdot_vector_at_time_reverse(disp_arr[:, i], disp_arr[:, i-1], self.wdot_old.vector)
                # print("i idx: {}".format(i))
                # wdot_old = self.compute_wdot_vector_at_time_n(disp_arr[:, :i])
        else:
            if i > 1:
                wdot_old = self.update_wdot_vector_at_time(disp_arr[:, i-1], disp_arr[:, i-2], self.wdot_old.vector)
                # print("wdot, i = {} > 1".format(i))
            else:
                wdot_old = np.zeros_like(self.wdot_old.vector) # self.update_wdot_vector_at_time(disp_arr[:, i], np.zeros_like(disp_arr[:, i-1]), np.zeros_like(self.wdot_old.vector))
                # print("wdot, i = {}".format(i))
        w_cur = disp_arr[:, i]
        if i > 0:
            w_old = disp_arr[:, i-1]
        else:
            w_old = np.zeros_like(w_cur)
        return w_cur, w_old, wdot_old

    def update_nsteps(self, Nsteps):
        self.Nsteps = Nsteps
        self.time_levels = Nsteps+1
        self.tip_disp_history = np.zeros((self.time_levels,))

    def solve_dynamic_problem(self, residual, saving_outputs=False, PATH=None, POD_matrix=None, POD_mean=None, timing=False):
        if timing:
            time_start = perf_counter()
        
        # compute and store strain energies if plot_strainenergy==True
        if saving_outputs:
            if PATH is None:
                PATH = "solutions/"
            strain_energy_list = np.zeros((self.time_levels,))
            xdmf_file = XDMFFile(self.comm, PATH+"sim_results/displacement.xdmf", "w")
            xdmf_file.write_mesh(self.mesh)
            xdmf_file_force = XDMFFile(self.comm, PATH+"sim_results/force.xdmf", "w")
            xdmf_file_force.write_mesh(self.mesh)
            xdmf_file_vmstress = XDMFFile(self.comm, PATH+"sim_results/von_mises_stress.xdmf", "w")
            xdmf_file_vmstress.write_mesh(self.mesh)
            xdmf_file_rotation = XDMFFile(self.comm, PATH+"sim_results/rotation.xdmf", "w")
            xdmf_file_rotation.write_mesh(self.mesh)

        time = 0.0
        w_output = np.zeros((self.fe_dofs, self.time_levels))
        for i in range(0, self.time_levels):
            print("------- Time step "+str(i)+"/"+str(self.Nsteps)
                +" , t = "+str(np.round(time, 8))+" -------")
            
            if i == 0:
                # set initial conditions (zero displacement and displacement rate)
                self.set_solution_vector(self.w, np.zeros_like(self.w.vector[:]))
                self.set_solution_vector(self.wdot_old, np.zeros_like(self.w.vector[:]))
                w_output[:, i] = self.w.x.array
            else:
                # update the force vector based on the force history
                if i < self.f_history.shape[0]:
                    self.update_f(self.f_history[i, :])
                else:
                    # assume constant loads for the rest of the time steps
                    self.update_f(self.f_history[-1, :])

                # the updated force will be automatically used in the residual
                solveNonlinear_mod(residual, self.w, self.bc_list, abs_tol=1e-8, log=False)

                # Advance to the next time step
                # ** since u_dot, theta_dot are not functions, we cannot directly
                # ** interpolate them onto wdot_old.

                w_output[:, i] = self.w.x.array
                if self.x_tip is not None and self.cell_tip is not None:
                    self.tip_disp_history[i] = extractTipDispDolfinx(self.w, 
                                                x_tip=self.x_tip, 
                                                cell_tip=self.cell_tip)    
                else:
                    self.tip_disp_history[i] = extractTipDispDolfinx(self.w)

                self.set_solution_vector(self.wdot_old, self.update_wdot_vector_at_time(w_output[:, i], w_output[:, i-1], self.wdot_old.vector))# self.wdot_vector())

            self.set_solution_vector(self.w_old, self.w.vector)

            strain_energy = self.assembleStrainEnergy(self.w)

            if saving_outputs:
                strain_energy_list[i] = strain_energy
                xdmf_file_force.write_function(self.f, time)
                xdmf_file.write_function(self.w.sub(0), time)
                project(self.von_Mises_stress(), self.vmstress, lump_mass=False)
                xdmf_file_vmstress.write_function(self.vmstress, time)
                xdmf_file_rotation.write_function(self.w.sub(1), time)

            print("Strain energy:", strain_energy)

            w_output[:, i] = self.w.x.array

            time += self.dt

        if timing:
            time_end = perf_counter()
            print("Dynamic simulation wall time: {}".format(time_end-time_start))

        if saving_outputs:
            np.save(PATH+"records/strain_energy_opt_"+str(self.opt_iter), strain_energy_list, allow_pickle=False)
            np.save(PATH+"records/tip_disp_opt_"+str(self.opt_iter), self.tip_disp_history, allow_pickle=False)
            self.opt_iter += 1
        return w_output
    
    def dRdw(self, svk_res, w_cur, w_old, wdot_old, adjoint=True):
        """
        This function assembles the derivative matrix dR/dw, where R is the residual 
        vector of all time steps and w contains all displacements and rotations.
        Inputs:
        w_cur: Solution vector at current time step
        w_old: Solution vector at previous time step
        wdot: Solution time derivative at current time step
        """
        self.set_solution_vectors_for_adjoint(w_cur, w_old, wdot_old)
        # -> take derivatives of R w.r.t. current u and u_old

        # raise ValueError()
        
        if adjoint:
            dresdw = form(ufl.adjoint(ufl.derivative(svk_res, self.w)))
            dresdw_old = form(ufl.adjoint(ufl.derivative(svk_res, self.w_old)))
        else:
            dresdw = form(ufl.derivative(svk_res, self.w))
            dresdw_old = form(ufl.derivative(svk_res, self.w_old))            

        return dresdw, dresdw_old

    def dRdwdot(self, svk_res, w_cur, w_old, wdot_old):
        self.set_solution_vectors_for_adjoint(w_cur, w_old, wdot_old)
        dresdwdot_old = form(ufl.derivative(svk_res, self.wdot_old))

        # compute current wdot
        # wdot_cur = Constant(self.mesh, 2/self.dt)*self.w - Constant(self.mesh, 2/self.dt)*self.w_old - self.wdot_old
        # inp_arg = Function(self.W)
        # dudot_arg, dthetadot_arg = split(inp_arg)

        # dresdudot_cur = form(ufl.derivative(svk_res, self.udot, dudot_arg))
        # dresdthetadot_cur = form(ufl.derivative(svk_res, self.thetadot, dthetadot_arg))
        # return dresdudot_cur, dresdthetadot_cur, dresdwdot_old
        return dresdwdot_old
        # return dresdwdot_cur, dresdwdot_old

    def dRdt(self, svk_res, w_cur, w_old, wdot_old, adjoint=True):
        self.set_solution_vectors_for_adjoint(w_cur, w_old, wdot_old)

        if adjoint:
            dresdt = form(ufl.adjoint(ufl.derivative(svk_res, self.t)))
        else:
            dresdt = form(derivative(svk_res, self.t))
        # dresdt_mat = create_matrix(dresdt)

        # # construct derivative matrices
        # dresdt_mat.zeroEntries()
        # _assemble_matrix_mat(dresdt_mat, dresdt, bcs=self.bc_list)
        # dresdt_mat.assemble()

        return dresdt

    def create_mat_from_form(self, inp_form, apply_bcs=False):
        mat = create_matrix(inp_form)
        mat.zeroEntries()
        if apply_bcs:
            _assemble_matrix_mat(mat, inp_form)#, bcs=self.bc_list)
        else:
            _assemble_matrix_mat(mat, inp_form, bcs=self.bc_list)
        mat.assemble()
        return mat

    def pnorm_stress(self,m=1e-6,rho=100,alpha=None,regularization=False):
        """
        Compute the p-norm of the stress
        `rho` is the Constraint aggregation factor
        """
        vm_stress = self.von_Mises_stress()
        dxx = ufl.Measure('dx', domain=self.mesh, 
                                 metadata={'quadrature_degree':4})
        pnorm = (m*vm_stress)**rho*dxx
        if regularization:
            regularization = 0.5*Constant(self.mesh, 1e3)*self.t**rho*dxx
            pnorm += regularization
        if alpha == None:
            ##### alpha is a parameter based on the surface area
            alpha_form = Constant(self.mesh,1.0)*dxx
            alpha = assemble_scalar(form(alpha_form))
        return 1/alpha*pnorm

    def von_Mises_stress(self):
        shell_stress_RM = ShellStressRM(self.mesh, self.w, self.t, self.E, self.nu)
        # stress on the top surface
        vm_stress = shell_stress_RM.vonMisesStress(self.t/2)
        return vm_stress
    
    def construct_force_to_pressure_map(self):
        # Define variational problem for projection
        w = TestFunction(self.W_f)
        Pv = TrialFunction(self.W_f)

        a = inner(Pv,w)*dx #lhs(res)
        # Assemble linear system
        A = assemble_matrix(form(a))
        A.assemble()
        # convert mass matrix to sparse Python array
        A_csr = A.getValuesCSR()
        A_sp = sp.csr_matrix((A_csr[2], A_csr[1], A_csr[0]))
        
        # eliminate zeros that are present in mass matrix
        A_sp.eliminate_zeros()
        return A_sp
    
    def construct_nodal_disp_map(self):
        deriv_us_to_ua_coord_list = []
        Q_map = self.construct_CG2_CG1_interpolation_map()
        disp_extraction_mats = self.construct_disp_extraction_mats()
        for i in range(3):
            deriv_us_to_ua_coord_list += [sp.csr_matrix(
                                            Q_map@disp_extraction_mats[i])]
        disp_extraction_mats = sp.vstack(deriv_us_to_ua_coord_list)
        # print(disp_extraction_mats.shape)
        return disp_extraction_mats

    def construct_disp_extraction_mats(self):
        # first we construct the extraction matrix for all displacements
        disp_space, solid_disp_idxs = self.W.sub(0).collapse()
        num_disp_dofs = len(solid_disp_idxs)
        solid_rot_idxs = self.W.sub(1).collapse()[1]
        num_rot_dofs = len(solid_rot_idxs)
        # __init__ sparse mapping matrix
        disp_mat = sp.lil_matrix((num_disp_dofs, num_disp_dofs+num_rot_dofs))
        # set relevant entries to 1
        disp_mat[list(range(num_disp_dofs)), solid_disp_idxs] = 1
        # convert sparse matrix to CSR format (for faster matrix-vector products)
        disp_mat.tocsr()

        # afterwards we generate the extraction matrices for the 3 displacement components
        disp_component_extraction_mats = []
        for i in range(3):
            solid_disp_coord_idxs = disp_space.sub(i).collapse()[1]
            num_disp_coord_dofs = len(solid_disp_coord_idxs)
            # __init__ sparse mapping matrix
            disp_coord_mat = sp.lil_matrix((num_disp_coord_dofs, disp_mat.shape[0]))
            # set relevant entries to 1
            disp_coord_mat[list(range(num_disp_coord_dofs)), solid_disp_coord_idxs] = 1
            # convert sparse matrix to CSR format (for faster matrix-vector products)
            disp_coord_mat.tocsr()

            # we multiply each coordinate extraction matrix with the displacement extraction matrix
            disp_coord_mat = disp_coord_mat@disp_mat

            disp_component_extraction_mats += [disp_coord_mat]

        return disp_component_extraction_mats

    def construct_CG2_CG1_interpolation_map(self):
        CG2_space = self.W.sub(0).collapse()[0]
        phys_coord_array = self.mesh.geometry.x
        mesh_bbt = dolfinx.geometry.BoundingBoxTree(self.mesh,
                                                    self.mesh.topology.dim)

        # create basix element
        ct = basix.cell.string_to_type(self.mesh.topology.cell_type.name)
        c_element = basix.create_element(basix.ElementFamily.P, ct, 2,
                                        basix.LagrangeVariant.equispaced)
        num_cg2_dofs = CG2_space.tabulate_dof_coordinates().shape[0]

        for i in range(self.mesh.topology.index_map(0).size_local):
            x_point = np.array(phys_coord_array[i, :])
            x_point_eval = self.eval_fe_basis_all_dolfinx(x_point,
                                CG2_space.dofmap.list, mesh_bbt, c_element, i,
                                mat_shape=(self.mesh.geometry.x.shape[0], num_cg2_dofs))
            if i == 0:
                sample_mat = x_point_eval
            else:
                sample_mat += x_point_eval

        return sample_mat

    def eval_fe_basis_all_dolfinx(self, x, dofmap_adjacencylist, mesh_bbt, basix_element, x_idx, mat_shape):
        mesh_cur = self.mesh
        cell_candidates = dolfinx.geometry.compute_collisions(mesh_bbt, x)
        cell_ids = dolfinx.geometry.compute_colliding_cells(mesh_cur, cell_candidates, x)
        geom_dofs = dofmap_adjacencylist.links(cell_ids[0])
        cell_vertex_ids = mesh_cur.geometry.dofmap.links(cell_ids[0])
        x_ref = mesh_cur.geometry.cmap.pull_back(x[None, :], mesh_cur.geometry.x[cell_vertex_ids])

        c_tab = basix_element.tabulate(0, x_ref)[0, 0, :, 0]
        # NOTE: c_tab contains the DoF values on the cell cell_ids[0]; geom_dofs contains their global DoF numbers

        # basis_vec = np.zeros((mesh_cur.geometry.dofmap.num_nodes,))
        basis_vec = sp.csr_array((c_tab, (c_tab.shape[0]*[int(x_idx)], geom_dofs)), shape=mat_shape)

        return basis_vec

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    # read in mesh
    beam = [#### quad mesh ####
        "plate_2_10_quad_1_5.xdmf",
        "plate_2_10_quad_2_10.xdmf",
        "plate_2_10_quad_4_20.xdmf",
        "plate_2_10_quad_8_40.xdmf",
        "plate_2_10_quad_10_50.xdmf",
        "plate_2_10_quad_20_100.xdmf",
        "plate_2_10_quad_40_200.xdmf",
        "plate_2_10_quad_80_400.xdmf",]

    filename = "clamped-RM-plate/"+beam[1]
    with dolfinx.io.XDMFFile(comm, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # Define physical parameters
    E = 1e7  # Young's modulus
    nu = 0.3  # Poisson ratio
    h = 0.1  # thickness
    width = 2.  # plate width
    length = 5.  # plate length

    rho = 10. 

    # Time-stepping parameters
    T       = 20.0
    Nsteps  = 200
    dt = T/Nsteps
    p0 = 1.

    f_0 = Constant(mesh, (0.0,0.0,-p0))
    #f_t = [f_0]*(Nsteps+1)

    # define PlateSim object
    plate_sim = PlateSim(mesh, E, nu, rho, dt, Nsteps, f_0, comm=comm)
    # set thickness to standard value
    t_array = [h]*plate_sim.num_var
    plate_sim.update_t(np.array(t_array))
    plate_sim.update_materialmodel()
    plate_sim.solve_dynamic_problem(plate_sim.SVK_residual(), saving_outputs=True)

    # now we update the thickness and rerun the same simulation
    # t_array = [1.3*h]*plate_sim.num_var
    
    # plate_sim.update_t(np.array(t_array))
    # plate_sim.reset_solution_vectors()
    # svk_res = plate_sim.SVK_residual()
    # plate_sim.solve_dynamic_problem(svk_res, saving_outputs=False)
    
