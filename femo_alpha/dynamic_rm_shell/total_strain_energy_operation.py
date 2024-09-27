from femo_alpha.dynamic_rm_shell.utils import (create_dense_petsc_vec, 
                                                populate_dense_petsc_vec,
                                                create_dense_np_mat_from_form,                                  
                                                stack_array_into_vector, 
                                                reshape_vector_into_array,
                                                apply_hom_DirichletBCs_to_matrix
                                                    )
from femo_alpha.dynamic_rm_shell.plate_sim import PlateSim
from dolfinx.fem.petsc import assemble_vector
import ufl 
from ufl import dot, grad
from dolfinx.fem import Constant
from dolfinx.fem import form, assemble_scalar
import csdl_alpha as csdl
import numpy as np
import dolfinx
from mpi4py import MPI

class TotalStrainEnergyOperation(csdl.CustomExplicitOperation):
    """
    input: input/state variables
    output: output
    """

    def __init__(self, plate_sim):
        super().__init__()

        # define any checks for the parameters
        csdl.check_parameter(plate_sim, "plate_sim")

        # assign parameters to the class
        self.plate_sim = plate_sim
        self.output_dim = 0 # for scalar outputs

        self.args_dict = ['thickness', 'disp_history']
        self.output_name = 'total_strain_energy'
        
    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        for arg_name in self.args_dict:
            if getattr(inputs, arg_name) is not None:
                self.declare_input(arg_name, getattr(inputs, arg_name))
            else:
                raise ValueError(f"Variable {arg_name} not found in the FEA model.")
            
        # declare output variables
        output = self.create_output(self.output_name, (1,))
        output.add_name(self.output_name)

        # declare any derivative parameters
        self.declare_derivative_parameters(self.output_name, '*', dependent=True)

        return output

    def compute(self, input_vals, output_vals):
        w_array = reshape_vector_into_array(input_vals['disp_history'], self.plate_sim.time_levels)

        self.plate_sim.update_t(input_vals['thickness'])
        self.plate_sim.update_materialmodel()
        self.plate_sim.reset_solution_vectors()

        # loop over time levels to compute total internal energy
        Total_StrainEnergy = 0.
        for i in range(0, self.plate_sim.time_levels):
            self.plate_sim.set_solution_vector(self.plate_sim.w, w_array[:, i])
            total_strain_energy = assemble_scalar(form(self.compute_total_strain_energy()))
            # Total_StrainEnergy += self.plate_sim.assembleStrainEnergy(self.plate_sim.w)
            Total_StrainEnergy += total_strain_energy

        # add regularization term
        regularization = assemble_scalar(form(self.compute_regularization(self.plate_sim.time_levels)))
        #*self.spline_sim.spline_h**2\
        output_vals['total_strain_energy'] = Total_StrainEnergy + regularization

        # print("COMPUTED COMPLIANCE, VALUE: {}".format(Total_StrainEnergy))
        # print("Regularization term: {}".format(regularization))
    

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        """
        We compute the partial derivatives of the total strain energy 
        w.r.t. the thicknesses and displacement solution history
        """
        w_array = reshape_vector_into_array(input_vals['disp_history'], self.plate_sim.time_levels)

        self.plate_sim.update_t(input_vals['thickness'])
        self.plate_sim.update_materialmodel()
        self.plate_sim.reset_solution_vectors()

        # construct numpy arrays to store partial derivatives
        dEdt_arr = np.zeros((input_vals['thickness'].shape[0],))
        dEdw_arr = np.zeros_like(w_array)

        for i in range(0, self.plate_sim.time_levels):
            # set displacement vector
            self.plate_sim.set_solution_vector(self.plate_sim.w, w_array[:, i])
            # construct strain energy function
            # strain_energy_func = self.plate_sim.constructStrainEnergy(self.plate_sim.w)
            total_strain_energy = self.compute_total_strain_energy()
            # compute derivatives w.r.t. thickness and displacement
            dEdt = assemble_vector(form(ufl.derivative(total_strain_energy, self.plate_sim.t)))
            dEdw = assemble_vector(form(ufl.derivative(total_strain_energy, self.plate_sim.w)))

            # add derivatives to storage arrays
            dEdt_arr += dEdt.array
            dEdw_arr[:, i] = dEdw.array



        # add derivative of regularization w.r.t. thickness
        regularization = self.compute_regularization(self.plate_sim.time_levels)
        dRegdt = assemble_vector(form(ufl.derivative(regularization, self.plate_sim.t)))

        # print("dRegdt: {}".format(dRegdt.array))

        dEdt_arr += dRegdt.array

        # reshape dEdw_arr into a vector
        dEdw_arr = stack_array_into_vector(dEdw_arr)

        derivatives[self.output_name, 'thickness'] = dEdt_arr
        derivatives[self.output_name, 'disp_history'] = dEdw_arr

    def compute_total_strain_energy(self):
        total_strain_energy = dot(self.plate_sim.w, self.plate_sim.w)*self.plate_sim.element.dx_inplane
        return total_strain_energy

    def compute_regularization(self, num_timesteps):
        alpha = 1e-2*num_timesteps  # define regularization as constant times number of time steps
        if self.plate_sim.element_wise_thickness:
            regularization = alpha*dot(self.plate_sim.t, self.plate_sim.t)*self.plate_sim.element.dx_inplane
        else:
            regularization = alpha*dot(grad(self.plate_sim.t), grad(self.plate_sim.t))*self.plate_sim.element.dx_inplane
        return regularization


# if __name__ == '__main__':
#     comm = MPI.COMM_WORLD

#     # read in mesh
#     beam = [#### quad mesh ####
#         "plate_2_10_quad_1_5.xdmf",
#         "plate_2_10_quad_2_10.xdmf",
#         "plate_2_10_quad_4_20.xdmf",
#         "plate_2_10_quad_8_40.xdmf",
#         "plate_2_10_quad_10_50.xdmf",
#         "plate_2_10_quad_20_100.xdmf",
#         "plate_2_10_quad_40_200.xdmf",
#         "plate_2_10_quad_80_400.xdmf",]

#     filename = "plate_meshes/"+beam[0]
#     with dolfinx.io.XDMFFile(comm, filename, "r") as xdmf:
#         mesh = xdmf.read_mesh(name="Grid")

#     nn = mesh.topology.index_map(0).size_local
#     nel = mesh.topology.index_map(mesh.topology.dim).size_local

#     # Define physical parameters
#     E = 1e7  # Young's modulus
#     nu = 0.3  # Poisson ratio
#     h = 0.1  # thickness
#     width = 2.  # plate width
#     length = 10.  # plate length (incorrect?)

#     rho = 10. 

#     # Time-stepping parameters
#     T       = 1.0
#     Nsteps  = 5
#     dt = T/Nsteps
#     p0 = 1.

#     f_0 = Constant(mesh, (0.0,0.0,-p0))
#     #f_t = [f_0]*(Nsteps+1)

#     # define PlateSim object
#     plate_sim = PlateSim(mesh, E, nu, rho, dt, Nsteps, f_0)

#     recorder = csdl.Recorder(inline=True)
#     recorder.start()

#     t = csdl.Variable(value=0.1*np.ones(plate_sim.num_var), name='thickness')
#     disp_history = csdl.Variable(value=0.1*np.ones(plate_sim.fe_dofs*plate_sim.time_levels, ), name='disp_history')

#     input_vars = csdl.VariableGroup()
#     input_vars.t = t
#     input_vars.disp_history = disp_history
#     total_strain_energy_operation = TotalStrainEnergyOperation(plate_sim=plate_sim)
#     total_strain_energy = total_strain_energy_operation.evaluate(input_vars)

#     print('check_partials:')
#     sim = csdl.experimental.PySimulator(recorder)
#     sim.check_totals([total_strain_energy],[disp_history, t])