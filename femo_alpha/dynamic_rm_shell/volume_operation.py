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


class VolumeOperation(csdl.CustomExplicitOperation):
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

        self.args_dict = ['thickness']
        self.output_name = 'volume'

        
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
        self.plate_sim.update_t(input_vals['thickness'])

        output_vals['volume'] = assemble_scalar(form(self.compute_volume()))

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        self.plate_sim.update_t(input_vals['thickness'])
        V = self.compute_volume()
        dVdt = assemble_vector(form(ufl.derivative(V, self.plate_sim.t))).getArray()
        derivatives['volume', 'thickness'] = dVdt

    def compute_volume(self):
        V = self.plate_sim.t*self.plate_sim.element.dx_inplane
        return V


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

#     filename = "plate_meshes/"+beam[2]
#     with dolfinx.io.XDMFFile(comm, filename, "r") as xdmf:
#         mesh = xdmf.read_mesh(name="Grid")

#     # Define physical parameters
#     E = 1e9  # Young's modulus
#     nu = 0.3  # Poisson ratio
#     h = 0.1  # thickness
#     width = 2.  # plate width
#     length = 5.  # plate length (incorrect?)

#     rho = 10. 

#     # Time-stepping parameters
#     T       = 10.0
#     Nsteps  = 200
#     dt = T/Nsteps
#     p0 = 1.

#     f_0 = Constant(mesh, (0.0,0.0,-p0))
#     #f_t = [f_0]*(Nsteps+1)

#     # define PlateSim object
#     plate_sim = PlateSim(mesh, E, nu, width, length, rho, dt, Nsteps, f_0)
#     recorder = csdl.Recorder(inline=True)
#     recorder.start()

#     t = csdl.Variable(value=0.1*np.ones(plate_sim.num_var), name='thickness')

#     input_vars = csdl.VariableGroup()
#     input_vars.t = t
#     volume_operation = VolumeOperation(plate_sim=plate_sim)
#     volume = volume_operation.evaluate(input_vars)

#     print('check_partials:')
#     sim = csdl.experimental.PySimulator(recorder)
#     sim.check_totals([volume],[t])