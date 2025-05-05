
from femo_alpha.dynamic_rm_shell.utils import (create_dense_petsc_vec, 
                                                populate_dense_petsc_vec,
                                                convert_petsc_vec_list_to_np_vec_list, 
                                                create_mumps_solver,
                                                create_dense_np_mat_from_form,                                  
                                                stack_array_into_vector, 
                                                reshape_vector_into_array,
                                                apply_hom_DirichletBCs_to_matrix
                                                    )
from femo_alpha.dynamic_rm_shell.plate_sim import PlateSim
import csdl_alpha as csdl
import numpy as np
import dolfinx
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.fem import Constant
from dolfinx.fem.petsc import assemble_matrix

class StateOperation(csdl.experimental.CustomImplicitOperation):
    '''
    input: input variable
    output: state variable
    '''

    def __init__(self, plate_sim, gradient_mode='numpy', debug_mode=False, record=True, path="./"):
        super().__init__()
        '''
        Initialize the StateOperation object.
        Parameters:
        -----------
        plate_sim (FEA): An instance of the FEA class.
        debug_mode (bool, optional): If set to True, the debug mode is enabled. 
                    Defaults to False.

        Inputs: thickness
        States: disp_solid history
        '''
        # define any checks for the parameters
        csdl.check_parameter(plate_sim, 'plate_sim')

        # assign parameters to the class
        self.plate_sim = plate_sim
        self.gradient_mode = gradient_mode
        self.debug_mode = debug_mode
        self.state_name = 'disp_history'

        self.path = path
        # self.args_dict = ['thickness', 'force_vector']
        self.args_dict = ['thickness', 'force_history']
        self.input_name = 'thickness'

        self.record = record
        self.eval_iter = 0
        if self.record:
            self.thickness_recorder = dolfinx.io.XDMFFile(MPI.COMM_WORLD, path+"records/thickness.xdmf", "w")
            self.thickness_recorder.write_mesh(plate_sim.mesh)
            
        # store indices of DoFs that are defined through BCs
        u_bc_dofs, rot_bc_dofs = self.plate_sim.bc_obj.return_affected_dof_tuple()  # return dofs set by BCs
        self.bc_dofs = np.union1d(u_bc_dofs[0], rot_bc_dofs[0])  # combine BC indices


    def evaluate(self, inputs: csdl.VariableGroup):
        '''

        Evaluate the state operation

        parameters:
        -----------
        inputs (dict): A dictionary of input variables (csdl.Variable).

        returns:
        --------
        state (csdl.Variable): The state variable.
        '''
        if self.debug_mode is True:
            print('=' * 15 + str(self.state_name) + '=' * 15)
            print('CSDL: Running evaluate()...')
            print('=' * 40)

        # assign method inputs to input dictionary
        for arg_name in self.args_dict:  
            if getattr(inputs, arg_name) is not None:
                self.declare_input(arg_name, getattr(inputs, arg_name))
            else:
                raise ValueError(
                    f'Variable {arg_name} not found in the FEA model.')

        # declare output variables
        state = self.create_output(
            self.state_name,
            shape=(self.plate_sim.fe_dofs*self.plate_sim.time_levels,),
        )
        state.add_name(self.state_name)

        # declare any derivative parameters
        self.declare_derivative_parameters(self.state_name, '*', dependent=True)

        return state

    def solve_residual_equations(self, input_vals, output_vals):
        '''
        Solve the residual equations using FEMO solver.

        parameters:
        -----------
        output_vals (dict): A dictionary of output variable values.
        '''
        if self.debug_mode is True:
            print('=' * 15 + str(self.state_name) + '=' * 15)
            print('CSDL: Running solve_residual_equations()...')
            print('=' * 40)

        # solve the residual equation
        self.plate_sim.update_t(input_vals['thickness'])
        # self.plate_sim.update_f(input_vals['force_vector'])
        self.plate_sim.update_f_history(input_vals['force_history'])
        self.plate_sim.update_materialmodel()
        self.plate_sim.reset_solution_vectors()  # sets initial condition to zero
        svk_res = self.plate_sim.SVK_residual()

        # record design variables
        if self.record:
            self.thickness_recorder.write_function(self.plate_sim.t, 
                                                   self.eval_iter)
        self.eval_iter += 1
        sol_output = self.plate_sim.solve_dynamic_problem(svk_res,
                                                saving_outputs=self.record,
                                                PATH = self.path,
                                                timing=True)

        output_vals[self.state_name] = stack_array_into_vector(sol_output)
        if output_vals[self.state_name].shape[0] != stack_array_into_vector(sol_output).shape[0]:
            print(output_vals[self.state_name].shape)
            print(stack_array_into_vector(sol_output).shape)
            raise ValueError("non-matching array shapes")

        

    def compute_jacvec_product(self, input_vals, output_vals, 
                               d_inputs, d_outputs, d_residuals, mode):
        '''
        Compute the product of the Jacobian matrix and a vector.

        parameters:
        -----------
        input_vals (dict): A dictionary of input variable values.
        output_vals (dict): A dictionary of output variable values.
        d_inputs (dict): A dictionary of input variable deltas.
        d_outputs (dict): A dictionary of output variable deltas.
        d_residuals (dict): A dictionary of residual variable deltas.
        mode (str): The mode of the operation. It could be either 'fwd' or 'rev'.
        '''
        if self.debug_mode is True:
            print('=' * 15 + str(self.state_name) + '=' * 15)
            print('CSDL: Running compute_jacvec_product()...')
            print('=' * 40)

        state_name = self.state_name
        input_name = self.input_name
        # for mode = fwd
        # d_inputs --> d_residuals

        # set thickness vector and other preprocessing
        self.plate_sim.update_t(input_vals[input_name])
        self.plate_sim.update_materialmodel()
        self.plate_sim.reset_solution_vectors()
        svk_res = self.plate_sim.SVK_residual()
        disp_arr = reshape_vector_into_array(output_vals[state_name], self.plate_sim.time_levels)
        
        vec_len = int(d_outputs[state_name].shape[0]/self.plate_sim.time_levels)


        if state_name in d_residuals:
            # generate PETSc derivative matrices through FEniCS (once for every time step)
            dRdw_cur, dRdw_old = self.plate_sim.dRdw(svk_res, np.zeros((vec_len,)), np.zeros((vec_len,)), np.zeros((vec_len,)), adjoint=False)
            dRdwdot_old = self.plate_sim.dRdwdot(svk_res, np.zeros((vec_len,)), np.zeros((vec_len,)), np.zeros((vec_len,)))

            # convert PETSc derivative matrices to Numpy and apply derivatives
            dRdw_cur_mat = assemble_matrix(dRdw_cur)
            dRdw_cur_mat.assemble()
            dRdw_cur_mat.zeroRows(self.bc_dofs)

            dRdw_old_mat = assemble_matrix(dRdw_old)
            dRdw_old_mat.assemble()
            dRdw_old_mat.zeroRows(self.bc_dofs, diag=0.)

            dRdwdot_old_mat = assemble_matrix(dRdwdot_old)
            dRdwdot_old_mat.assemble()
            dRdwdot_old_mat.zeroRows(self.bc_dofs, diag=0.)

        if self.gradient_mode == 'numpy':
            dRdw_cur_mat.convert("dense")
            dRdw_cur_mat_np = dRdw_cur_mat.getDenseArray()

            dRdw_old_mat.convert("dense")
            dRdw_old_mat_np = dRdw_old_mat.getDenseArray()

            dRdwdot_old_mat.convert("dense")
            dRdwdot_old_mat_np = dRdwdot_old_mat.getDenseArray()

        elif self.gradient_mode == 'petsc':
            if mode == 'fwd':
                # create PETSc vectors and preprocess
                if state_name in 'd_residuals':
                    d_disp_cur_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                    d_disp_old_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                    du_cur_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                    du_old_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                    dudot_old_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                    dRdw_sum_vec = create_dense_petsc_vec(self.plate_sim.comm, vec_len)

                    dRdw_old_mat.axpy(2./self.plate_sim.dt, dRdwdot_old_mat) # add together dRdw_old and dRdwdot_old
                if 'thickness' in d_inputs:
                    dt_input_petsc = create_dense_petsc_vec(self.plate_sim.comm, self.plate_sim.num_var)
                    dt_output_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
            elif mode == 'rev':
                if 'thickness' in d_inputs:
                    dRidwi_minus_j_vec = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                    dRidwi_minus_1_mat = dRdw_old_mat.duplicate(copy=True)
                    dRidwi_minus_1_mat.axpy(2./self.plate_sim.dt, dRdwdot_old_mat) # add together dRdw_old and dRdwdot_old

                    dt_output_petsc = create_dense_petsc_vec(self.plate_sim.comm, self.plate_sim.num_var)

        # for mode = fwd
        # d_inputs --> d_residuals
        if mode == 'fwd':
            d_res_list = []
            if state_name in d_residuals:
                d_output = d_outputs[state_name]
                d_disp_arr = reshape_vector_into_array(d_output, self.plate_sim.time_levels)

                for i in range(0, self.plate_sim.time_levels):
                    if i%5 == 0:
                        print("compute jacvec product i index, mode fwd: {}".format(i))

                    d_res_store = np.zeros_like(d_disp_arr[:, i])
                    w_cur, w_old, wdot_old = self.plate_sim.compute_solution_vectors_at_time(disp_arr, i)
                    
                    if state_name in d_outputs:
                        if i == 0:
                            # we treat the initial (Dirichlet) condition separately, in numpy
                            # dRdw = np.eye(self.plate_sim.fe_dofs)

                            # du_total = dRdw@d_disp_arr[:, i]
                            du_total = d_disp_arr[:, i]
                        else:
                            if self.gradient_mode == 'numpy':
                                du_cur = dRdw_cur_mat_np@d_disp_arr[:, i]  # component of current time level
                                du_old = (dRdw_old_mat_np + (2/self.plate_sim.dt)*dRdwdot_old_mat_np)@d_disp_arr[:, i-1]  # component of previous time level

                                # loop over not-yet-included components of all past time levels (necessary because of chain rule terms)
                                dRdw_sum_coeff_vec = np.zeros_like(d_disp_arr[:, i])
                                if i >= 2:
                                    for j in range(2, i):
                                        # print("j loop index: {}".format(j))
                                        dRdw_sum_coeff_vec += ((-1.)**j)*d_disp_arr[:, i-j]

                                # add summed components to du_old vector
                                du_old += (4/self.plate_sim.dt)*dRdwdot_old_mat_np@dRdw_sum_coeff_vec

                                du_cur_np = du_cur
                                du_old_np = du_old

                                du_total = du_cur_np + du_old_np
                            elif self.gradient_mode == 'petsc':                             
                                populate_dense_petsc_vec(d_disp_old_petsc, d_disp_arr[:, i-1])
                                populate_dense_petsc_vec(d_disp_cur_petsc, d_disp_arr[:, i])

                                dRdw_cur_mat.mult(d_disp_cur_petsc, du_cur_petsc)
                                dRdw_old_mat.mult(d_disp_old_petsc, du_old_petsc)

                                # loop over not-yet-included components of all past time levels (necessary because of chain rule terms)
                                dRdw_sum_coeff_vec = np.zeros_like(d_disp_arr[:, i])
                                if i >= 2:
                                    for j in range(2, i):
                                        dRdw_sum_coeff_vec += ((-1.)**j)*d_disp_arr[:, i-j]

                                populate_dense_petsc_vec(dRdw_sum_vec, dRdw_sum_coeff_vec)
                                dRdwdot_old_mat.mult(dRdw_sum_vec, dudot_old_petsc)

                                # add summed components to du_old vector
                                # du_old_petsc.axpy(4./self.plate_sim.dt, dudot_old_petsc)
                                # # add du_old and du_cur together
                                # du_cur_petsc.axpy(1., du_old_petsc)
                                du_cur_petsc.maxpy([4./self.plate_sim.dt, 1.], [dudot_old_petsc, du_old_petsc])
                                du_total = du_cur_petsc.getValues(range(vec_len))

                        d_res_store += du_total

                    if 'thickness' in d_inputs:
                        # We explicitly neglect the influence of the initial condition here, 
                        # since the initial condition's jacobian dRdt = 0
                        if i == 0:
                            # Residual of initial condition does not depend on `t`, and hence there is no contribution
                            dt_np = np.zeros_like(d_res_store)
                        else:                            
                            print(" i: {}".format(i))
                            # compute dRdt (has to be done every time step even for linear models, we can't pre-construct it)
                            dRdt = self.plate_sim.dRdt(svk_res, w_cur, w_old, wdot_old, adjoint=False)

                            # convert dRdt to Numpy
                            dRdt_mat = assemble_matrix(dRdt)
                            dRdt_mat.assemble()
                            dRdt_mat.zeroRows(self.bc_dofs, diag=0.)  # apply boundary conditions

                            if self.gradient_mode == 'numpy':
                                dRdt_mat.convert("dense")
                                dRdt_mat_np = dRdt_mat.getDenseArray()

                                dt_np = dRdt_mat_np@d_inputs['thickness']
                            elif self.gradient_mode == 'petsc':
                                populate_dense_petsc_vec(dt_input_petsc, d_inputs['thickness'])
                                dRdt_mat.mult(dt_input_petsc, dt_output_petsc)
                                dt_np = dt_output_petsc.getValues(range(vec_len))

                        d_res_store += dt_np

                    d_res_list += [d_res_store]

                d_residuals[state_name] = np.concatenate(d_res_list)
                if d_residuals[state_name].shape[0] != np.concatenate(d_res_list).shape[0]:
                    print(d_residuals[state_name].shape)
                    print(np.concatenate(d_res_list).shape)
                    raise ValueError("non-matching array shapes")
        
        # for mode = rev
        # d_residuals --> d_inputs
        elif mode == 'rev':
            if state_name in d_residuals:
                # retrieve adjoint solution list
                dres_list = np.array(d_residuals[state_name])
                dres_arr = reshape_vector_into_array(dres_list, self.plate_sim.time_levels)

                if self.gradient_mode == 'numpy':
                    d_outputs_list = [np.zeros((dres_arr.shape[0],)) for i in range(self.plate_sim.time_levels)]

                elif self.gradient_mode == 'petsc':
                    # create PETSc vector that will contain instantaneous adjoint solutions
                    dres_petsc_cur = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                    d_outputs_list_petsc = [create_dense_petsc_vec(self.plate_sim.comm, vec_len) for i in range(self.plate_sim.time_levels)]
    
                for i in range(0, self.plate_sim.time_levels):
                    if i%5 == 0:
                        print("compute jacvec product i index, mode rev: {}".format(i))

                    # populate_dense_petsc_vec(dres_petsc_cur, dres_arr[:, i])  # set current adjoint solution

                    w_cur, w_old, wdot_old = self.plate_sim.compute_solution_vectors_at_time(disp_arr, i)

                    if state_name in d_outputs:

                        if i == 0:
                            if self.gradient_mode == 'petsc':
                                # d_outputs_list_petsc[i].zeroValues()
                                d_outputs_list_petsc[i].zeroEntries()
                        else:
                            # after dRdw_cur, dRdw_old and dRdwdot_old have been constructed we can set up the linear system and store relevant vectors
                            for j in range(0, i+1, 1):

                                if self.gradient_mode == 'numpy':
                                    # precompute (transpose) matrix-vector product that we can reuse for all j != i-1
                                    dRidwi_minus_j_vec = dRdwdot_old_mat_np.T@dres_arr[:, i]
                                    if j == i:
                                        d_outputs_list[j] += dRdw_cur_mat_np.T@dres_arr[:, i]
                                    elif j == i-1 and j > 0:
                                        dRidwi_minus_1 = dRdw_old_mat_np + (2/self.plate_sim.dt)*dRdwdot_old_mat_np
                                        d_outputs_list[j] += dRidwi_minus_1.T@dres_arr[:, i]
                                    elif j == i-1 and j == 0:
                                        d_outputs_list[j] += dRdw_old_mat_np.T@dres_arr[:, i]
                                    elif j < i-1 and j == 0:
                                        d_outputs_list[j] += ((-1.)**j)*(2/self.plate_sim.dt)*dRidwi_minus_j_vec
                                    else:
                                        if np.mod(i, 2) != 0:
                                            d_outputs_list[j] += ((-1.)**j)*(4/self.plate_sim.dt)*dRidwi_minus_j_vec
                                        else:
                                            d_outputs_list[j] += ((-1.)**(j+1))*(4/self.plate_sim.dt)*dRidwi_minus_j_vec

                                elif self.gradient_mode == 'petsc':
                                    populate_dense_petsc_vec(dres_petsc_cur, dres_arr[:, i])  # set current adjoint solution

                                    # precompute (transpose) matrix-vector product that we can reuse for all j != i-1
                                    dRdwdot_old_mat.multTranspose(dres_petsc_cur, dRidwi_minus_j_vec)
                                    # dRidwi_minus_j_vec = dRdwdot_old_mat_np.T@dres_arr[:, i]
                                    if j == i:
                                        # d_outputs_list[j] += dRdw_cur_mat_np.T@dres_arr[:, i]
                                        dRdw_cur_mat.multTransposeAdd(dres_petsc_cur, d_outputs_list_petsc[j], d_outputs_list_petsc[j])
                                    elif j == i-1 and j > 0:
                                        # d_outputs_list[j] += dRidwi_minus_1.T@dres_arr[:, i]
                                        dRidwi_minus_1_mat.multTransposeAdd(dres_petsc_cur, d_outputs_list_petsc[j], d_outputs_list_petsc[j])
                                    elif j == i-1 and j == 0:
                                        # d_outputs_list[j] += dRdw_old_mat_np.T@dres_arr[:, i]
                                        dRdw_old_mat.multTransposeAdd(dres_petsc_cur, d_outputs_list_petsc[j], d_outputs_list_petsc[j])
                                    elif j < i-1 and j == 0:
                                        # d_outputs_list[j] += ((-1.)**j)*(2/self.plate_sim.dt)*dRidwi_minus_j_vec
                                        d_outputs_list_petsc[j].axpy(((-1.)**j)*(2/self.plate_sim.dt), dRidwi_minus_j_vec)
                                    else:
                                        if np.mod(i, 2) != 0:
                                            # d_outputs_list[j] += ((-1.)**j)*(4/self.plate_sim.dt)*dRidwi_minus_j_vec
                                            d_outputs_list_petsc[j].axpy(((-1.)**j)*(4/self.plate_sim.dt), dRidwi_minus_j_vec)
                                        else:
                                            # d_outputs_list[j] += ((-1.)**(j+1))*(4/self.plate_sim.dt)*dRidwi_minus_j_vec
                                            d_outputs_list_petsc[j].axpy(((-1.)**(j+1))*(4/self.plate_sim.dt), dRidwi_minus_j_vec)

                    if 'thickness' in d_inputs:
                        # compute dRdt
                        if i == 0:
                            # Residual of initial condition does not depend on `t`, and hence there is no contribution
                            dt_np = np.zeros_like(d_inputs['thickness'])
                        else:
                            dRdt = self.plate_sim.dRdt(svk_res, w_cur, w_old, wdot_old, adjoint=False)

                            dRdt_mat = assemble_matrix(dRdt)
                            dRdt_mat.assemble()
                            dRdt_mat.zeroRows(self.bc_dofs, diag=0.)  # apply boundary conditions
                            
                            if self.gradient_mode == 'numpy':
                                dRdt_mat.convert("dense")
                                dRdt_mat_np = dRdt_mat.getDenseArray()
                                dt_np = dRdt_mat_np.T@dres_arr[:, i]
                            elif self.gradient_mode == 'petsc':
                                populate_dense_petsc_vec(dres_petsc_cur, dres_arr[:, i])
                                dRdt_mat.multTranspose(dres_petsc_cur, dt_output_petsc)
                                dt_np = dt_output_petsc.getValues(range(self.plate_sim.num_var))

                        d_inputs['thickness'] += dt_np

                        if d_inputs['thickness'].shape[0] != dt_np.shape[0]:
                            print(d_inputs['thickness'].shape)
                            print(self.plate_sim.dt.array.shape)
                            raise ValueError("non-matching array shapes")

            # # [RX] this would trigger an error in CSDL
            # if state_name in d_outputs:
            #     # compute dRdw
            #     # d_outputs[state_name] += du.array
            #     d_outputs[state_name] = np.concatenate(d_outputs_list)
            #     if d_outputs[state_name].shape[0] != np.concatenate(d_outputs_list).shape[0]:
            #         print(d_outputs[state_name].shape)
            #         print(np.concatenate(d_outputs_list).shape)
            #         raise ValueError("non-matching array shapes")
        else:
            raise ValueError("mode must be either 'fwd' or 'rev'.")
                                

    def apply_inverse_jacobian(self, input_vals, output_vals, 
                               d_outputs, d_residuals, mode):
        '''
        Solve linear system. Invoked when solving coupled linear system; 
        i.e. when solving Newton system to update implicit state variables, 
        and when computing total derivatives

        '''
        if self.debug_mode is True:
            print('=' * 15 + str(self.state_name) + '=' * 15)
            print('CSDL: Running apply_inverse_jacobian()...')
            print('=' * 40)

        # update current thicknesses
        self.plate_sim.update_t(input_vals['thickness'])
        # self.plate_sim.update_f(input_vals['force_vector'])
        self.plate_sim.update_materialmodel()
        self.plate_sim.reset_solution_vectors()
        svk_res = self.plate_sim.SVK_residual()
        
        state_name = self.state_name

        vec_len = int(d_residuals[state_name].shape[0]/self.plate_sim.time_levels)

        # store current solution history
        self.disp_arr = disp_arr = reshape_vector_into_array(output_vals[self.state_name], self.plate_sim.time_levels)



        # For linear models we can pre-construct the derivative matrices for each time step, 
        # since these do not depend on the system state!
        dRdw_cur, dRdw_old = self.plate_sim.dRdw(svk_res, np.zeros((vec_len,)), np.zeros((vec_len,)), np.zeros((vec_len,)), adjoint=False)
        dRdwdot_old = self.plate_sim.dRdwdot(svk_res, np.zeros((vec_len,)), np.zeros((vec_len,)), np.zeros((vec_len,)))

        dRdw_cur_mat = assemble_matrix(dRdw_cur)#, bcs=self.plate_sim.bc_list)
        dRdw_cur_mat.assemble()
        dRdw_cur_mat.zeroRows(self.bc_dofs)

        dRdw_old_mat = assemble_matrix(dRdw_old)
        dRdw_old_mat.assemble()
        dRdw_old_mat.zeroRows(self.bc_dofs, diag=0.)

        dRdwdot_old_mat = assemble_matrix(dRdwdot_old)
        dRdwdot_old_mat.assemble()
        dRdwdot_old_mat.zeroRows(self.bc_dofs, diag=0.)

        if self.gradient_mode == 'numpy':
            dRdw_cur_mat.convert("dense")
            dRdw_cur_mat_np = dRdw_cur_mat.getDenseArray()
            dRdw_cur_mat_np = apply_hom_DirichletBCs_to_matrix(dRdw_cur_mat_np, self.bc_dofs)

            dRdw_old_mat.convert("dense")
            dRdw_old_mat_np = dRdw_old_mat.getDenseArray()
            dRdw_old_mat_np[self.bc_dofs, :] = 0.

            dRdwdot_old_mat.convert("dense")
            dRdwdot_old_mat_np = dRdwdot_old_mat.getDenseArray()
            dRdwdot_old_mat_np[self.bc_dofs, :] = 0.

        elif self.gradient_mode == 'petsc':
            d_res_cur_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)

            # define MUMPS solver
            solver = create_mumps_solver(self.plate_sim.comm, dRdw_cur_mat)
            if mode == 'fwd':
                # construct PETSc versions of vectors
                d_res_cur_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)

                dRdw_old_vec_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                dRdw_sum_coeff_vec_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                dRdw_sum_mult_dRdwdot_old_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)

                petsc_previous_sol = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                petsc_sol = create_dense_petsc_vec(self.plate_sim.comm, vec_len)

                dRdw_old_mat.axpy(2./self.plate_sim.dt, dRdwdot_old_mat) # add together dRdw_old and dRdwdot_old
            elif mode == 'rev':
                rhs_add_vec_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                # cur_adjoint_sol_petsc = create_dense_petsc_vec(self.plate_sim.comm, vec_len)
                dRidwi_minus_j_vec = create_dense_petsc_vec(self.plate_sim.comm, vec_len)

                dRidwi_minus_1_mat = dRdw_old_mat.duplicate(copy=True)
                dRidwi_minus_1_mat.axpy(2./self.plate_sim.dt, dRdwdot_old_mat) # add together dRdw_old and dRdwdot_old


        # for mode = fwd
        # d_residuals --> d_outputs
        if mode == 'fwd':
            # forward mode = direct method, also known as the "tangent linear model"
            d_res_arr = reshape_vector_into_array(d_residuals[state_name], self.plate_sim.time_levels)
            direct_sol_list = [None for i in range(self.plate_sim.time_levels)]
            # loop over time steps
            for i in range(0, self.plate_sim.time_levels):
                # if i%5 == 0:
                print("solve_linear i index, mode fwd: {}".format(i))

                # compute the current state of the system (system states and state time derivatives)
                w_cur, w_old, wdot_old = self.plate_sim.compute_solution_vectors_at_time(self.disp_arr, i)
                
                if i == 0:
                    # we treat the initial (Dirichlet) condition separately
                    direct_sol_list[i] = d_res_arr[:, i]
                else:
                    if self.gradient_mode == 'numpy':
                        # dRdw_cur, dRdw_old = self.plate_sim.dRdw(svk_res, w_cur, w_old, wdot_old, adjoint=False)
                        # dRdwdot_old = self.plate_sim.dRdwdot(svk_res, w_cur, w_old, wdot_old)

                        dRdw_old_vec = -(dRdw_old_mat_np + (2/self.plate_sim.dt)*dRdwdot_old_mat_np)@direct_sol_list[i-1]  # multiply dRdw_old with previous solution

                        dRdw_sum_coeff_vec = np.zeros_like(direct_sol_list[i-1])
                        # loop over past solutions to take into account their influence on the current solution through the chain rule
                        if i >= 2:
                            for j in range(2, i):
                                dRdw_sum_coeff_vec += ((-1.)**j)*direct_sol_list[i-j]
                        
                        d_res_vec_np = d_res_arr[:, i] + dRdw_old_vec + (4/self.plate_sim.dt)*dRdwdot_old_mat_np@dRdw_sum_coeff_vec                   

                        direct_sol_np = np.linalg.solve(dRdw_cur_mat_np, d_res_vec_np)
                        direct_sol_list[i] = direct_sol_np

                    elif self.gradient_mode == 'petsc':
                        print("Running gradient mode PETSc")
                        # populate PETSc versions of vectors
                        populate_dense_petsc_vec(d_res_cur_petsc, d_res_arr[:, i])
                        populate_dense_petsc_vec(petsc_previous_sol, direct_sol_list[i-1])
                        
                        dRdw_old_mat.mult(petsc_previous_sol, dRdw_old_vec_petsc)  # multiply with solution of previous time step

                        dRdw_sum_coeff_vec = np.zeros_like(direct_sol_list[i-1])
                        # loop over past solutions to take into account their influence on the current solution through the chain rule
                        if i >= 2:
                            for j in range(2, i):
                                dRdw_sum_coeff_vec += ((-1.)**j)*direct_sol_list[i-j]
                        
                        populate_dense_petsc_vec(dRdw_sum_coeff_vec_petsc, dRdw_sum_coeff_vec)
                        
                        dRdwdot_old_mat.mult(dRdw_sum_coeff_vec_petsc, dRdw_sum_mult_dRdwdot_old_petsc)

                        # combine all vector terms
                        # d_res_cur_petsc.axpy(-1., dRdw_old_vec_petsc)
                        d_res_cur_petsc.maxpy([-1., 4./self.plate_sim.dt], [dRdw_old_vec_petsc, dRdw_sum_mult_dRdwdot_old_petsc])

                        # we plug everything into the linear problem solution class
                        solver.solve(d_res_cur_petsc, petsc_sol)

                        # convert solution to numpy for OpenMDAO
                        timestep_petsc_sol = petsc_sol.getValues(range(vec_len))

                        direct_sol_list[i] = timestep_petsc_sol

            d_outputs[state_name] = np.concatenate(direct_sol_list)
            if d_outputs[state_name].shape[0] != np.concatenate(direct_sol_list).shape[0]:
                print(d_outputs[state_name].shape)
                print(np.concatenate(direct_sol_list).shape)
                raise ValueError("non-matching array shapes")
        
            
        # for mode = rev:
        # d_outputs --> d_residuals
        elif mode == 'rev':
            # reverse mode = adjoint method
            # create list of lists that will store the right-hand side vector from off-diagonal terms
            if self.gradient_mode == 'numpy':
                rhs_vec_lists = [[np.zeros((self.plate_sim.fe_dofs,)) for j in range(self.plate_sim.time_levels)] for i in range(self.plate_sim.time_levels)]
                adjoint_sol_list = [np.zeros((self.plate_sim.fe_dofs,)) for i in range(self.plate_sim.time_levels)]
            elif self.gradient_mode == 'petsc':
                rhs_vec_lists_petsc = [[create_dense_petsc_vec(self.plate_sim.comm, vec_len) for j in range(self.plate_sim.time_levels)] for i in range(self.plate_sim.time_levels)]
                adjoint_sol_list_petsc = [create_dense_petsc_vec(self.plate_sim.comm, vec_len) for j in range(self.plate_sim.time_levels)]

            dr_arr = reshape_vector_into_array(d_outputs[state_name], self.plate_sim.time_levels)

            # loop over time steps starting at last time step, solve each one in succession
            for i in range(self.plate_sim.time_levels-1, -1, -1):
                print("apply inverse jacobian i index, mode rev: {}".format(i))
                # compute wdot at current time level
                # w_cur, w_old, wdot_old = self.plate_sim.compute_solution_vectors_at_time(self.disp_arr, i, backwards_timestepping=True)

                if self.gradient_mode == 'numpy':
                    # we can set up the linear system and store relevant vectors
                    if i == self.plate_sim.time_levels - 1:
                        rhs_add_vec = np.zeros_like(dr_arr[:, i])
                    else:
                        rhs_add_vec = np.zeros_like(dr_arr[:, i])
                        for j in range(i+1, self.plate_sim.time_levels):  # loop from j = i + 1 to j = N
                            rhs_add_vec -= rhs_vec_lists[i][j]

                    # calculate the adjoint solution of the current time step
                    if i == 0:
                        adj_sol_np = np.zeros_like(dr_arr[:, i])
                    else:
                        adj_sol_np = np.linalg.solve(dRdw_cur_mat_np.T, dr_arr[:, i] + rhs_add_vec)

                    adjoint_sol_list[i] = adj_sol_np

                    # calculate the vectors corresponding to dR^{i}dw^{j}

                    # precompute (transpose) matrix-vector product that we can reuse for all j != i-1
                    dRidwi_minus_j_vec = dRdwdot_old_mat_np.T@adjoint_sol_list[i]
                    dRidwi_minus_1 = dRdw_old_mat_np + (2/self.plate_sim.dt)*dRdwdot_old_mat_np
                    for j in range(i-1, -1, -1):
                        if j == i-1 and j > 0:
                            rhs_vec_lists[j][i] += dRidwi_minus_1.T@adjoint_sol_list[i]
                        elif j == i-1 and j == 0:
                            rhs_vec_lists[j][i] += dRdw_old_mat_np.T@adjoint_sol_list[i]
                        elif j < i-1 and j == 0:
                            rhs_vec_lists[j][i] += ((-1.)**j)*(2/self.plate_sim.dt)*dRidwi_minus_j_vec
                        else:
                            if np.mod(i, 2) != 0:
                                rhs_vec_lists[j][i] += ((-1.)**j)*(4/self.plate_sim.dt)*dRidwi_minus_j_vec
                            else:
                                rhs_vec_lists[j][i] += ((-1.)**(j+1))*(4/self.plate_sim.dt)*dRidwi_minus_j_vec

                elif self.gradient_mode == 'petsc':
                    # we set up the linear system and store relevant vectors
                    rhs_add_vec_petsc.zeroEntries()
                    if i < self.plate_sim.time_levels - 1:
                        alpha_list = [-1.]*((self.plate_sim.time_levels) - (i+1))
                        vec_list = rhs_vec_lists_petsc[i][i+1:self.plate_sim.time_levels]
                        rhs_add_vec_petsc.maxpy(alpha_list, vec_list)

                    populate_dense_petsc_vec(d_res_cur_petsc, dr_arr[:, i])                    
                    # add rhs_add_vec_petsc to d_res_cur_petsc
                    d_res_cur_petsc.axpy(1., rhs_add_vec_petsc)

                    # calculate the adjoint solution of the current time step
                    if i == 0:
                        adjoint_sol_list_petsc[i].zeroEntries()
                    else:
                        solver.solveTranspose(d_res_cur_petsc, adjoint_sol_list_petsc[i])

                    # calculate the vectors corresponding to dR^{i}dw^{j}
                    # precompute (transpose) matrix-vector product that we can reuse for all j != i-1
                    dRdwdot_old_mat.multTranspose(adjoint_sol_list_petsc[i], dRidwi_minus_j_vec)
                    for j in range(i-1, -1, -1):
                        if j == i-1 and j > 0:
                            dRidwi_minus_1_mat.multTransposeAdd(adjoint_sol_list_petsc[i], rhs_vec_lists_petsc[j][i], rhs_vec_lists_petsc[j][i])
                        elif j == i-1 and j == 0:
                            dRdw_old_mat.multTransposeAdd(adjoint_sol_list_petsc[i], rhs_vec_lists_petsc[j][i], rhs_vec_lists_petsc[j][i])
                        elif j < i-1 and j == 0:
                            rhs_vec_lists_petsc[j][i].axpy(((-1.)**j)*(2/self.plate_sim.dt), dRidwi_minus_j_vec)
                        else:
                            if np.mod(i, 2) != 0:
                                rhs_vec_lists_petsc[j][i].axpy(((-1.)**j)*(4/self.plate_sim.dt), dRidwi_minus_j_vec)
                            else:
                                rhs_vec_lists_petsc[j][i].axpy(((-1.)**(j+1))*(4/self.plate_sim.dt), dRidwi_minus_j_vec)

            # process and stack the solution vectors into a single vector
            if self.gradient_mode == 'petsc':
                adjoint_sol_list = convert_petsc_vec_list_to_np_vec_list(adjoint_sol_list_petsc)

            concat_adj_sol_list = np.concatenate(adjoint_sol_list)
            d_residuals[state_name] = concat_adj_sol_list
            if d_residuals[state_name].shape[0] != concat_adj_sol_list.shape[0]:
                print(d_residuals[state_name].shape)
                print(concat_adj_sol_list.shape)
                raise ValueError("non-matching array shapes")

   
        else:
            raise ValueError("mode must be either 'fwd' or 'rev'.")

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

    filename = "plate_meshes/"+beam[1]
    with dolfinx.io.XDMFFile(comm, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    nn = mesh.topology.index_map(0).size_local
    nel = mesh.topology.index_map(mesh.topology.dim).size_local
    # Define physical parameters
    E = 1e8  # Young's modulus
    nu = 0.3  # Poisson ratio
    h = 0.1  # thickness
    width = 2.  # plate width
    length = 10.  # plate length

    rho = 10. 


    Nsteps  = 200

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
    T = T0+T1+T2 # total time = 0.5s
    # Nsteps  = 1
    dt = T/Nsteps
    def V_g(t):
        V_g = 0.
        if T0 <= t <= T0+T1:
            # V_g = V_p*np.sin(2*np.pi*t/T1)**2
            V_g = V_p*(1-np.cos(2*np.pi*(t-T0)/T1))
        return V_g
        # return 1/2*V_p*(1-np.cos(2*np.pi*t/GGLc))


    # define PlateSim object
    plate_sim = PlateSim(mesh, E, nu, rho, dt, Nsteps)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    thickness = csdl.Variable(value=0.1*np.ones(plate_sim.num_var), name='thickness')


    # # constant loads

    T       = 0.4
    dt = 0.01 #T/Nsteps
    p0 = 5.

    f_0 = Constant(mesh, (0.0,0.0,-p0))
    force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
    force_vector.value[:, 2] = -p0 # body force per node
    force_vector = force_vector.flatten()
    force_history = csdl.expand(force_vector, out_shape=(plate_sim.time_levels,nn*3), action='i->ji')
    force_history.add_name('force_history')

    # # gust loads
    # t = np.linspace(0,T,Nsteps+1)
    # force_history = np.zeros((Nsteps+1, nn*3))
    # for i in range(Nsteps+1):
    #     ti = t[i]
    #     V_gi = V_g(ti)
    #     force_vector = np.zeros((nn, 3))
    #     force_vector[:, 2] = V_gi*0.1
    #     force_vector = force_vector.flatten()
    #     force_history[i,:] = force_vector


    # force_history = csdl.Variable(value=force_history, name='force_history')
    # force_history.add_name('force_history')

    input_vars = csdl.VariableGroup()
    input_vars.thickness = thickness
    # input_vars.force_vector = force_vector
    input_vars.force_history = force_history
    state_operation = StateOperation(plate_sim=plate_sim)
    disp_history = state_operation.evaluate(input_vars)

    # print("disp history norm:", np.linalg.norm(disp_history.value))
    # print("tip disp history:", extractTipDisp(disp_history))
    # print("force history norm:", np.linalg.norm(force_history.value, axis=1))

    import matplotlib.pyplot as plt
    plt.plot(np.linspace(0, T, Nsteps+1), plate_sim.tip_disp_history)
    plt.legend(["dt="+str(dt)])
    plt.show()
    print('check_partials:')
    sim = csdl.experimental.PySimulator(recorder)
    # sim.check_totals([disp_history],[thickness])