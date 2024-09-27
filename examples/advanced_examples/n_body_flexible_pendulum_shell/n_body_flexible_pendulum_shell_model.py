'''
This is a CSDL model of a two-body pendulum system. The model is constructed using individual residual models for each body
and a time discretization model preppended to each residual model (which computes the derivatives of the states and lagrange multipliers
as a function of the states and lagrange multipliers). The residual models are then combined into a single residual model and a Newton solver
(which is implemented in Python) will converge the residual model at each time step. For this Newton solver:
residual: [df/dx + lambda*dc/dx, c] = 0
Jacobian: KKT matrix which will be assembled with the help of CSDL AD.

In addition to the residual and time discretization models, we also need a constraints model that will compute the constraints
and append the lagrange multiplier term (lambda*dc/dx) to the residuals.

NOTE: In future iteration, decouple time stepping of each residual models by representing states functionally (as a function of time and space)
NOTE: As a first iteration, we take a SAND approach to this inner optimization/solver, where the optimizer picks all of the states
and lagrange multipliers as design variables and the residual model is only used for single evaluation (not driving residual to 0).
NOTE: In the approach from the note above, we do not need to feed the lagrange multipliers into the residual models.

TODO: Figure out how to automate input creation, design variable addition, output registration, and connection.
TODO: Automate the creation of this model altogether.
TODO: Allow use of geometry to specify constraints.
TODO: Think about whether self.bodies should be a dictionary of [body_name, body_residual_model] or perhaps [body_name, body_component/obj].
TODO: As the first note says, decouple time stepping (SIFR)

TODO: Add mapping from structural rotations to geometry
TODO: Figure out how to properly constrain flexible motion to not capture rigid body motion
        -- I think the solution to this is to use lagrange multipliers to constrain that the rigid body states (of center of mass)
           match what the center of mass / orientation of the geometry is. This way, the flexible solver can't change it. This is a nice
           formulation in that it's decoupled from the presence of other flexible solvers.
'''


import csdl_alpha as csdl
import numpy as np
import lsdo_geo
import lsdo_function_spaces as lfs

from mbd_trial.framework_trial.time_discretizations.generalized_alpha import GeneralizedAlphaModel
from mbd_trial.framework_trial.time_discretizations.generalized_alpha import GeneralizedAlphaStateData
from mbd_trial.framework_trial.models.gravitational_potential_energy_model import compute_gravitational_potential_energy
from mbd_trial.framework_trial.models.rigid_body_kinetic_energy_model import compute_rigid_body_kinetic_energy
from mbd_trial.framework_trial.models.pendulum_residual import PendulumResidualModel
from mbd_trial.framework_trial.models.residual_intertial_term import ResidualInertialTerm
from mbd_trial.framework_trial.models.rigid_body_gravity_residual_3d import RigidBodyGravityResidual3D

import dolfinx
from mpi4py import MPI
import csdl_alpha as csdl
import numpy as np

import femo_alpha.rm_shell.rm_shell_model as rmshell
from femo_alpha.fea.utils_dolfinx import createCustomMeasure

from dataclasses import dataclass
from typing import Union

from lsdo_geo.csdl.optimization import Optimization, NewtonOptimizer

from typing import Optional


@dataclass
class Material:
    E: float
    G: float
    nu: float
    density: float
    thickness: float

    name: Optional[str] = None

@dataclass
class NBodyPendulumStateData:
    states:dict[str,csdl.Variable]
    state_derivatives:dict[str,csdl.Variable]


@dataclass
class PendulumBody:
    pendulum_counter = 0
    name : str = None
    geometry : lsdo_geo.Geometry = None
    shell_mesh : lsdo_geo.Mesh = None
    shell_mesh_fea : dolfinx.mesh.Mesh = None
    length : csdl.Variable = csdl.Variable(shape=(1,), value=1.)
    # length : csdl.Variable = 1.
    material : Material = None
    mass : csdl.Variable = None
    # mass : csdl.Variable = 1.
    center_of_mass : csdl.Variable = None
    shell_representation : lsdo_geo.Mesh = None

    # @property
    # def shell_mesh(self) -> csdl.Variable:
    #     # return self.geometry.evaluate_representations(self.shell_representation, plot=False)
    #     return self.design_geometry.evaluate_representations(self.shell_representation, plot=False)

    def __post_init__(self):
        if self.name is None:
            self.name = f'pendulum{PendulumBody.pendulum_counter}'
            PendulumBody.pendulum_counter += 1

        if self.geometry is None:
            # self.geometry = lsdo_geo.import_geometry('mbd_trial/framework_trial/examples/n_body_flexible_pendulum/pendulum_geometry.stp',
            #                                          name=self.name, parallelize=False)
            self.geometry = lsdo_geo.import_geometry('./pendulum_geometry.stp',
                                                     name=self.name, parallelize=False)
            # imported_coefficients = self.geometry.coefficients.value.reshape((-1,3))
            # new_coefficient_values = imported_coefficients.copy()
            # new_coefficient_values[:,1] = -new_coefficient_values[:,1]
            # new_coefficients = csdl.Variable(shape=(len(new_coefficient_values.flatten()),), value=new_coefficient_values.flatten())
            # self.geometry.coefficients = new_coefficients

        # a copy of the undeformed geometry
        self.design_geometry = self.geometry.copy()

        # if self.shell_representation is None:
        #     num_shell_nodes = 5
        #     class PendulumShellMesh(lsdo_geo.Mesh):
        #         def evaluate(self, geometry:lsdo_geo.Geometry, plot:bool=False):
        #             shell_ends = super().evaluate(geometry)
        #             shell_mesh = csdl.linear_combination(shell_ends[0], shell_ends[1], num_shell_nodes)
        #             if plot:
        #                 geometry.plot_meshes([shell_mesh])
        #             return shell_mesh
                    
        #     shell_top = self.geometry.project(np.array([0., 0., 0.5]))
        #     shell_bottom = self.geometry.project(np.array([0., 0., -0.5]))
        #     parametric_mesh = shell_top + shell_bottom
        #     self.shell_representation = PendulumShellMesh(geometry=self.geometry, parametric_coordinates=parametric_mesh)
        # # self.geometry.add_representation(representation=self.shell_representation)
        # self.design_geometry.add_representation(representation=self.shell_representation)

        height = csdl.Variable(value=0.1)
        width = csdl.Variable(value=0.1)

        if self.material is None:
            # E = 79.e9
            # E = 5.e6
            # E = 6.e5    # Young's modulus of soft material
            # E = 1e5
            # E = 3.e4      # Young's modulus of Jello
            # E = 1e3
            # E = 1
            E = 70e9
            nu = 0.3
            G = E/(2*(1+nu))
            # density = 1
            # density = 1e2
            # density = 1080  # Density of soft material
            # density = 1270  # Density of Jello
            density = 2700    # Density of aluminum
            # density = 19.3e3
            # density = 1e5
            thickness = 0.001
            self.material = Material(E=E, G=G, nu=nu, thickness=thickness, density=density)
        if self.mass is None:
            self.mass = (2*(height*self.length) + 2*(height*width) + 2*(self.length*width))*self.material.thickness*self.material.density # hardcoding pendulum dimensions for now to get the idea across

        '''
        1. Define the mesh
        '''

        pendulum = [#### quad mesh ####
                "pendulum_shell_mesh_2_10.xdmf",
                "pendulum_shell_mesh_2_20.xdmf",
                "pendulum_shell_mesh_4_40.xdmf",]

        filename = "./shell_mesh/"+pendulum[0]
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
            shell_mesh_fea = xdmf.read_mesh(name="Grid")
        # Set up mapping from shell displacements to geometry
        # - Fit B-spline to shell mesh
        # TODO: create one bspline surface for each patch
        self.shell_mesh_fea = shell_mesh_fea


        num_shell_nodes = self.shell_mesh_fea.topology.index_map(0).size_local

        self.element_wise_material = True
        self.shell_model = shell_model = rmshell.RMShellModel(shell_mesh_fea, 
                    shell_bc_func = ClampedBoundary,
                    element_wise_material=self.element_wise_material, #node_wise_material=True,
                    record=False,dynamic_only=True)
        num_flexible_states = shell_model.num_states  # 6 dof per node
        self.num_flexible_states = num_flexible_states
        self.num_shell_nodes = num_shell_nodes
        self.disp_dof_coords = shell_model.disp_dof_coords
        self.rot_dof_coords = shell_model.rot_dof_coords
        self.disp_dof_indices = shell_model.disp_dof_indices
        self.rot_dof_indices = shell_model.rot_dof_indices

        self.shell_mesh = csdl.Variable(value=shell_model.disp_dof_coords, name='shell_mesh')
        shell_mesh = self.shell_mesh.value
        
        nn = shell_model.nn
        nel = shell_model.nel

        if self.element_wise_material:
            thickness = csdl.Variable(value=self.material.thickness*np.ones(nel), name='thickness')
            E = csdl.Variable(value=self.material.E*np.ones(nel), name='E')
            nu = csdl.Variable(value=self.material.nu*np.ones(nel), name='nu')
            density = csdl.Variable(value=self.material.density*np.ones(nel), name='density')
        else:
            thickness = csdl.Variable(value=self.material.thickness*np.ones(nn), name='thickness')
            E = csdl.Variable(value=self.material.E*np.ones(nn), name='E')
            nu = csdl.Variable(value=self.material.nu*np.ones(nn), name='nu')
            density = csdl.Variable(value=self.material.density*np.ones(nn), name='density')

        self.shell_thickness = thickness
        self.shell_E = E
        self.shell_nu = nu
        self.shell_density = density


        # # initial solve required
        # node_disp = csdl.Variable(value=np.zeros((nn, 3)), name='node_disp')
        # node_disp.add_name('node_disp')
        # force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
        # force_vector.value[:, 2] = 0.1 # body force per node
        # shell_model.set_up_fea()
        # shell_outputs = shell_model.evaluate(force_vector, thickness, E, nu, density,
        #                         node_disp,
        #                         debug_mode=False,
        #                         is_pressure=True)


        # print(shell_mesh)
        # fishy_plot = self.geometry.plot(show=False, opacity=0.3)
        # import vedo
        # vedo_mesh = vedo.Mesh([shell_mesh, connectivity]).wireframe().color('green').opacity(0.8)
        # plotter = vedo.Plotter()
        # plotter.show([fishy_plot, vedo_mesh], axes=1, viewup='y')

        # # self.geometry.plot_meshes([shell_mesh])
        # self.geometry.plot()
        # b_spline_function_space = lfs.BSplineSpace(num_parametric_dimensions=2, 
        #                             degree=3, 
        #                             coefficients_shape=shell_mesh.shape[0])
        # # shell_b_spline = b_spline_function_space.fit_function(shell_mesh, parametric_coordinates=np.linspace(0, 1, num=shell_mesh.shape[0]))
        
        # # [RX] hardcoded for the pendulum case

        # parametric_coordinates = b_spline_function_space.generate_parametric_grid((3,11))
        # shell_b_spline = b_spline_function_space.fit_function(shell_mesh, 
                                # parametric_coordinates=parametric_coordinates)

        # - Create mapping from shell displacements to grid mesh displacements
        # -- Compute displacement portion of matrix mappings (precomputing matrix for efficiency)

        # - Create mapping from shell displacements to grid mesh displacements
        # -- Compute displacement portion of matrix mappings (precomputing matrix for efficiency)
        self.shell_displacement_to_geometry_displacement_mapping = {}
        self.shell_evaluation_maps = {}
        self.wireframe_coordinates_with_respect_to_shell_coordinates = {}
        self.wireframe_to_geometry_displacement_mapping = {}
        # for function_index, function in self.geometry.functions.items():
        #     # - Create grid mesh over geometry
        #     fitting_resolution = 25
            # parametric_geometry_wireframe = function.space.generate_parametric_grid(grid_resolution=fitting_resolution)
            # geometry_wireframe = function.evaluate(parametric_geometry_wireframe)

            # # - Project grid mesh onto B-spline
            # shell_b_spline.plot()
            # closest_parametric_coordinates = shell_b_spline.project(geometry_wireframe.value, plot=True)
            # shell_to_wireframe_displacement_mapping = b_spline_function_space.compute_basis_matrix(closest_parametric_coordinates)

            # parametric_function_wireframe = function.space.generate_parametric_grid(grid_resolution=fitting_resolution)
            # geometry_to_wireframe_displacement_mapping = function.space.compute_basis_matrix(parametric_function_wireframe)
            # wireframe_to_geometry_displacement_mapping = np.linalg.pinv(geometry_to_wireframe_displacement_mapping.toarray())
            # self.shell_displaccement_to_geometry_displacement_mappings[function_index] = wireframe_to_geometry_displacement_mapping @ shell_to_wireframe_displacement_mapping

        # -- Apply displacements from rotations (using small angle approximation? Not necessary, but more efficient)
        # --- I will use small angle approximation, so the mapping can be pre-computed as a matrix for efficiency?
        # TODO: Come back to this
        # closest_points = shell_b_spline.evaluate(closest_parametric_coordinates)
        # offsets = geometry_wireframe.value - closest_points
        parametric_coordinates = self.geometry.functions[0].project(shell_mesh, plot=False)
        geometry_to_wireframe_displacement_mapping = self.geometry.functions[0].space.compute_basis_matrix(parametric_coordinates)
        wireframe_to_geometry_displacement_mapping = np.linalg.pinv(geometry_to_wireframe_displacement_mapping.toarray())
        self.shell_displacement_to_geometry_displacement_mapping[0] = wireframe_to_geometry_displacement_mapping

        grid_resolution = (6,6)
        parametric_grid = self.design_geometry.generate_parametric_grid(grid_resolution=grid_resolution)
        grid = self.design_geometry.evaluate(parametric_grid)
        grid = grid.reshape((len(self.design_geometry.functions),) + grid_resolution + (3,))
        u_vectors = grid[:,1:] - grid[:,:-1]
        v_vectors = grid[:,:,1:] - grid[:,:,:-1]

        self.design_element_centers = grid[:,:-1,:-1] + u_vectors[:,:,:-1]/2 + v_vectors[:,:-1,:]/2

        _, self.design_center_of_mass, _, _ = self.evaluate_mass_properties(self.design_geometry, properties_to_compute=['center_of_mass'])

        # # if self.center_of_mass is None:
        # #     self.evaluate_center_of_mass()
        # class PendulumCenterOfMassRepresentation(lsdo_geo.Mesh):
        #     def evaluate(self, geometry:lsdo_geo.Geometry, plot:bool=False):
        #         top_and_bottom_ends = super().evaluate(geometry)
        #         center_of_mass = (top_and_bottom_ends[0] + top_and_bottom_ends[1])/2
        #         if plot:
        #             geometry.plot_meshes([center_of_mass])
        #         return center_of_mass
            
        # pendulum_top = self.geometry.project(np.array([0., 0., 0.5]))
        # pendulum_bottom = self.geometry.project(np.array([0., 0., -0.5]))
        # parametric_mesh = pendulum_top + pendulum_bottom
        # self.center_of_mass_representation = PendulumCenterOfMassRepresentation(geometry=self.geometry, parametric_coordinates=parametric_mesh)
        # self.geometry.add_representation(representation=self.center_of_mass_representation)
        # self.parametric_center_of_mass = self.geometry.project(np.array([0., 0., -0.5]), plot=False)


    # def evaluate_center_of_mass(self) -> csdl.Variable:
    #     # num_physical_dimensions = 3
    #     # geometry_num_coefficients = len(self.geometry.coefficients)//num_physical_dimensions
    #     # self.center_of_mass = csdl.sum(self.geometry.coefficients.reshape((geometry_num_coefficients,num_physical_dimensions)), axes=(0,)
    #     #                                 )/geometry_num_coefficients
    #     # self.center_of_mass = self.geometry.evaluate(self.parametric_center_of_mass, plot=False)
    #     self.center_of_mass = self.geometry.evaluate_representations(self.center_of_mass_representation, plot=False)
    #     # self.center_of_mass = self.geometry.evaluate(self.parametric_center_of_mass, plot=False)
    #     return self.center_of_mass
    

    def evaluate_mass_properties(self, geometry:lsdo_geo.Geometry=None, properties_to_compute:list[str]=['all']) -> tuple[csdl.Variable, csdl.Variable, csdl.Variable, csdl.Variable]:
        '''
        Evaluates the mass, center of mass, angle of mass, and moment of inertia of the pendulum body.

        Parameters
        ----------
        geometry : lsdo_geo.Geometry = None
            The geometry of the pendulum body. If None, the geometry of the pendulum body is used. Default is None.
        
        Returns
        -------
        mass : csdl.Variable
            The mass of the pendulum body.
        center_of_mass : csdl.Variable
            The center of mass of the pendulum body.
        moment_of_inertia : csdl.Variable
            The moment of inertia of the pendulum body.
        angular_momentum : csdl.Variable
            The angular momentum of the pendulum body.
        '''
        if geometry is None:
            geometry = self.geometry

        if ['all'] in properties_to_compute:
            properties_to_compute = ['mass', 'center_of_mass', 'moment_of_inertia', 'angular_momentum']
        mass = None
        center_of_mass = None
        moment_of_inertia = None
        angular_momentum = None

        # TODO: Add gauss-quadrature or analytic integration to lsdo_function_spaces so I can properly do center_of_mass = rho*integral(B)*coefficients

        grid_resolution = (6,6)
        parametric_grid = geometry.generate_parametric_grid(grid_resolution=grid_resolution)

        grid = geometry.evaluate(parametric_grid)

        material = self.material
        area_density = material.density*material.thickness

        grid = grid.reshape((len(geometry.functions),) + grid_resolution + (3,))
        u_vectors = grid[:,1:] - grid[:,:-1]
        v_vectors = grid[:,:,1:] - grid[:,:,:-1]

        element_areas_1 = csdl.cross(u_vectors[:,:,:-1], v_vectors[:,:-1,:], axis=3)
        element_areas_2 = csdl.cross(u_vectors[:,:,1:], v_vectors[:,1:,:], axis=3)
        element_areas = (csdl.norm(element_areas_1 + element_areas_2, axes=(3,)))/2

        element_centers = grid[:,:-1,:-1] + u_vectors[:,:,:-1]/2 + v_vectors[:,:-1,:]/2

        element_masses = element_areas*area_density
        element_masses_expanded = csdl.expand(element_masses, element_centers.shape, 'ijk->ijkl')

        if 'mass' or properties_to_compute or 'center_of_mass' in properties_to_compute or 'angular_momentum' in properties_to_compute:
            total_mass = csdl.sum(element_masses, axes=(0,1,2))
            mass = total_mass
            mass.add_name(f'{self.name}_mass')

            first_moment_of_mass = csdl.sum(element_centers*element_masses_expanded, axes=(0,1,2))
            center_of_mass = first_moment_of_mass/total_mass
            center_of_mass.add_name(f'{self.name}_center_of_mass')

        # NOTE: Since this is computed from the center of mass, I think this computation always comes out to 0s
        # total_angles = csdl.Variable(value=np.zeros((3,)))
        # relative_element_centers = element_centers - csdl.expand(center_of_mass, element_centers.shape, 'l->ijkl')
        # angles_x = csdl.arctan(relative_element_centers[:,:,:,2]/relative_element_centers[:,:,:,1])
        # angles_y = csdl.arctan(relative_element_centers[:,:,:,2]/relative_element_centers[:,:,:,0])
        # angles_z = csdl.arctan(relative_element_centers[:,:,:,1]/relative_element_centers[:,:,:,0])
        # total_angles = total_angles.set(csdl.slice[0], total_angles[0] + csdl.sum(angles_x*element_masses, axes=(0,1,2)))
        # total_angles = total_angles.set(csdl.slice[1], total_angles[1] + csdl.sum(angles_y*element_masses, axes=(0,1,2)))
        # total_angles = total_angles.set(csdl.slice[2], total_angles[2] + csdl.sum(angles_z*element_masses, axes=(0,1,2)))
        # angle_of_mass = total_angles/total_mass
        # angle_of_mass.add_name(f'{self.name}_angle_of_mass')

        if 'angular_momentum' in properties_to_compute:
            relative_element_centers = element_centers - csdl.expand(center_of_mass, element_centers.shape, 'l->ijkl')

            element_velocities = element_centers - self.design_element_centers
            element_momentums = element_velocities*csdl.expand(element_masses, element_velocities.shape, 'ijk->ijkl')

            angular_momentums = csdl.cross(self.design_element_centers, element_momentums, axis=3)# / relative_element_centers**2
            angular_momentum = csdl.sum(angular_momentums, axes=(0,1,2))


        if 'moment_of_inertia' in properties_to_compute:
            if 'angular_momentum' not in properties_to_compute:
                relative_element_centers = element_centers - csdl.expand(center_of_mass, element_centers.shape, 'l->ijkl')

            x_component = relative_element_centers[:,:,:,0]
            y_component = relative_element_centers[:,:,:,1]
            z_component = relative_element_centers[:,:,:,2]
            x_component_squared = x_component**2
            y_component_squared = y_component**2
            z_component_squared = z_component**2

            second_moment_of_mass = csdl.Variable(value=np.zeros((3,3)))
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[0,0],
                                                        second_moment_of_mass[0,0] + 
                                                        csdl.sum((y_component_squared + z_component_squared)*element_masses, axes=(0,1,2)))
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[1,1],
                                                        second_moment_of_mass[1,1] + 
                                                        csdl.sum((x_component_squared + z_component_squared)*element_masses, axes=(0,1,2)))
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[2,2],
                                                        second_moment_of_mass[2,2] + 
                                                        csdl.sum((x_component_squared + y_component_squared)*element_masses, axes=(0,1,2)))
            
            xy_term = csdl.sum(x_component*y_component*element_masses, axes=(0,1,2))
            xz_term = csdl.sum(x_component*z_component*element_masses, axes=(0,1,2))
            yz_term = csdl.sum(y_component*z_component*element_masses, axes=(0,1,2))

            term_01 = second_moment_of_mass[0,1] - xy_term
            term_02 = second_moment_of_mass[0,2] - xz_term
            term_12 = second_moment_of_mass[1,2] - yz_term

            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[0,1], term_01)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[1,0], term_01)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[0,2], term_02)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[2,0], term_02)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[1,2], term_12)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[2,1], term_12)

            moment_of_inertia = second_moment_of_mass

        return mass, center_of_mass, moment_of_inertia, angular_momentum
        

    def apply_rigid_body_motion(self, rigid_body_states:csdl.Variable, geometry:lsdo_geo.Geometry=None, function_indices:list=None):
        '''
        Applies rigid body motion to the geometry of the pendulum body.

        Parameters
        ----------
        rigid_body_states : csdl.Variable
            The state of the pendulum body [x, y, z, theta_x, theta_y, theta_z].
        geometry : lsdo_geo.Geometry, optional
            The geometry to apply the rigid body motion to. If None, the geometry of the pendulum body is used. Default is None.
        function_indices : list[int], optional
            The indices of the functions to apply the rigid body motion to. If None, all functions are used. Default is None.
        '''
        # num_physical_dimensions = list(self.geometry.num_physical_dimensions.values())[0]
        # geometry_coefficients_reshaped = \
        #     self.geometry.coefficients.reshape((len(self.geometry.coefficients)//num_physical_dimensions, num_physical_dimensions))

        # current_center_of_mass = self.center_of_mass
        # points_centered_at_origin = geometry_coefficients_reshaped - current_center_of_mass
        # points_rotated = csdl.rotate_using_rotation_matrix(
        #     points_centered_at_origin, angle=rigid_body_states[2], cartesian_axis='z', units='radians')


        if geometry is None:
            geometry = self.geometry

        if function_indices is None:
            function_indices = list(geometry.functions.keys())

        design_center_of_mass = self.design_center_of_mass

        # print('design_center_of_mass', design_center_of_mass.value)
        # print('rigid_body_states', rigid_body_states.value)
        self.center_of_mass = rigid_body_states[:3]
        for function_index in function_indices:
            function = geometry.functions[function_index]
            function.coefficients = function.coefficients - csdl.expand(design_center_of_mass, function.coefficients.shape, 'i->jki')
            function.coefficients = function.coefficients + csdl.expand(self.center_of_mass, function.coefficients.shape, 'i->jki')
            # function.plot()
        geometry.rotate(axis_origin=self.center_of_mass, axis_vector=np.array([0., 0., 1.]), angles=rigid_body_states[5], function_indices=function_indices)
        geometry.rotate(axis_origin=self.center_of_mass, axis_vector=np.array([0., 1., 0.]), angles=rigid_body_states[4], function_indices=function_indices)
        geometry.rotate(axis_origin=self.center_of_mass, axis_vector=np.array([1., 0., 0.]), angles=rigid_body_states[3], function_indices=function_indices)
        # self.geometry.plot()


    def apply_flexible_motion(self, flexible_states:csdl.Variable, rigid_body_states:csdl.Variable=None, 
                              geometry:lsdo_geo.Geometry=None, function_indices:list=None):
        '''
        Applies flexible motion to the geometry of the pendulum body.

        Parameters
        ----------
        flexible_states : csdl.Variable -- shape=(num_nodes,6)
            The state of the pendulum body [x, y, z, theta_x, theta_y, theta_z] for each structural node.
        rigid_body_states : csdl.Variable = None, -- shape=(6,)
            The state of the pendulum body [x, y, z, theta_x, theta_y, theta_z]. This is used to transform flexible states back to global frame.
            If None, it is assumed there are no rigid body rotations.
        geometry : lsdo_geo.Geometry = None
            The geometry to apply the flexible motion to. If None, the geometry of the pendulum body is used. Default is None.
        function_indices : list[int], optional = None
            The indices of the functions to apply the flexible motion to. If None, all functions are used. Default is None.
        '''
        num_physical_dimensions = 3

        # flexible_states = flexible_states.reshape((-1,6))

        # displacements = flexible_states[:,:3]    # first 3 dof are displacements
        # rotations = flexible_states[:,3:]        # last 3 dof are rotations

        displacements = flexible_states[self.disp_dof_indices].reshape((-1,3))

        if geometry is None:
            geometry = self.geometry

        if function_indices is None:
            function_indices = list(geometry.functions.keys())

        # geometry = self.design_geometry.copy()

        # # unvectorized version
        # for function_index in function_indices:
        #     function = geometry.functions[function_index]
        #     function_coefficients_shape = function.coefficients.shape
        #     function_coefficients = function.coefficients
        #     function_coefficients_reshaped = function_coefficients.reshape((function_coefficients.size//num_physical_dimensions, num_physical_dimensions))
        #     local_displacements = csdl.matmat(self.shell_to_geometry_displacement_mapping[function_index], displacements)
        #     temp_displacements = lsdo_geo.rotate(local_displacements, np.array([0., 0., 0.]), axis_vector=np.array([0., 0., 1.]), angles=rigid_body_states[5])
        #     temp_displacements = lsdo_geo.rotate(temp_displacements, np.array([0., 0., 0.]), axis_vector=np.array([0., 1., 0.]), angles=rigid_body_states[4])
        #     global_displacements = lsdo_geo.rotate(temp_displacements, np.array([0., 0., 0.]), axis_vector=np.array([1., 0., 0.]), angles=rigid_body_states[3])
        #     # global_displacements = local_displacements
        #     function_coefficients_reshaped = function_coefficients_reshaped + global_displacements
        #     function.coefficients = function_coefficients_reshaped.reshape(function_coefficients_shape)

        # vectorized version
        geometry_coefficients = []
        # for function_index in function_indices:
        #     function = geometry.functions[function_index]
        #     function_coefficients_shape = function.coefficients.shape
        #     function_coefficients = function.coefficients
        #     function_coefficients_reshaped = function_coefficients.reshape((function_coefficients.size//num_physical_dimensions, num_physical_dimensions))
        #     geometry_coefficients.append(function_coefficients_reshaped)
        function = geometry.functions[0]
        function_coefficients_shape = function.coefficients.shape
        function_coefficients = function.coefficients
        function_coefficients_reshaped = function_coefficients.reshape((function_coefficients.size//num_physical_dimensions, num_physical_dimensions))
        geometry_coefficients.append(function_coefficients_reshaped)
        # geometry_coefficients = csdl.vstack(geometry_coefficients)

        # Geometry displacements from shell displacements
        local_displacements_from_displacements = csdl.matmat(self.shell_displacement_to_geometry_displacement_mapping[0], displacements)

        # Geometry displacements from shell rotations
        # shell_rotations = csdl.matmat(self.shell_evaluation_maps.toarray(), rotations)

        # # Unvectorized : Too much memory for Jax because of feedback stacking
        # rotated_coordinates = self.wireframe_coordinates_with_respect_to_shell_coordinates
        # for i in csdl.frange(shell_rotations.shape[0]):
        #     rotated_point = lsdo_geo.rotate(rotated_coordinates[i], np.array([0., 0., 0.]), axis_vector=np.array([0., 0., 1.]), angles=shell_rotations[i,2])
        #     rotated_coordinates = rotated_coordinates.set(csdl.slice[i], rotated_point.reshape((3,)))
        #     rotated_point = lsdo_geo.rotate(rotated_coordinates[i], np.array([0., 0., 0.]), axis_vector=np.array([0., 1., 0.]), angles=shell_rotations[i,1])
        #     rotated_coordinates = rotated_coordinates.set(csdl.slice[i], rotated_point.reshape((3,)))
        #     rotated_point = lsdo_geo.rotate(rotated_coordinates[i], np.array([0., 0., 0.]), axis_vector=np.array([1., 0., 0.]), angles=shell_rotations[i,0])
        #     rotated_coordinates = rotated_coordinates.set(csdl.slice[i], rotated_point.reshape((3,)))

        # cos_shell_rotations = csdl.cos(shell_rotations)
        # sin_shell_rotations = csdl.sin(shell_rotations)

        # rotation_tensor_z = csdl.Variable(value=np.zeros((shell_rotations.shape[0], 3, 3)))
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,0,0], cos_shell_rotations[:,2])
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,0,1], -sin_shell_rotations[:,2])
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,1,0], sin_shell_rotations[:,2])
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,1,1], cos_shell_rotations[:,2])
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,2,2], 1)

        # rotation_tensor_y = csdl.Variable(value=np.zeros((shell_rotations.shape[0], 3, 3)))
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,0,0], cos_shell_rotations[:,1])
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,0,2], sin_shell_rotations[:,1])
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,2,0], -sin_shell_rotations[:,1])
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,2,2], cos_shell_rotations[:,1])
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,1,1], 1)

        # rotation_tensor_x = csdl.Variable(value=np.zeros((shell_rotations.shape[0], 3, 3)))
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,1,1], cos_shell_rotations[:,0])
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,1,2], -sin_shell_rotations[:,0])
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,2,1], sin_shell_rotations[:,0])
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,2,2], cos_shell_rotations[:,0])
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,0,0], 1)

        # rotated_coordinates = csdl.einsum(rotation_tensor_z, self.wireframe_coordinates_with_respect_to_shell_coordinates, action='ijk,ik->ij')
        # rotated_coordinates = csdl.einsum(rotation_tensor_y, rotated_coordinates, action='ijk,ik->ij')
        # rotated_coordinates = csdl.einsum(rotation_tensor_x, rotated_coordinates, action='ijk,ik->ij')
        # # rotated_coordinates = csdl.einsum(rotation_tensor_x, self.wireframe_coordinates_with_respect_to_shell_coordinates, action='ijk,ik->ij')
         
        # wireframe_displacements = rotated_coordinates - self.wireframe_coordinates_with_respect_to_shell_coordinates

        # wireframe_counter = 0
        # coefficients_counter = 0
        # # rotation_displacements_y = csdl.Variable(value=np.zeros(geometry_coefficients.shape[:-1]))
        # # rotation_displacements_z = csdl.Variable(value=np.zeros(geometry_coefficients.shape[:-1]))
        # rotation_displacements = csdl.Variable(value=np.zeros(geometry_coefficients.shape))
        # for function_index in function_indices:
        #     # function_displacements_y = csdl.matmat(self.wireframe_to_geometry_displacement_mapping[function_index], wireframe_displacements_y[wireframe_counter:wireframe_counter+625])
        #     # function_displacements_z = csdl.matmat(self.wireframe_to_geometry_displacement_mapping[function_index], wireframe_displacements_z[wireframe_counter:wireframe_counter+625])
        #     # function_displacements = csdl.matmat(self.wireframe_to_geometry_displacement_mapping[function_index], wireframe_displacements_from_theta_x[wireframe_counter:wireframe_counter+625])
        #     function_displacements = csdl.matmat(self.wireframe_to_geometry_displacement_mapping[function_index], wireframe_displacements[wireframe_counter:wireframe_counter+625])
        #     num_function_coefficients = geometry.functions[function_index].coefficients.size//num_physical_dimensions

        #     # rotation_displacements_y = rotation_displacements_y.set(csdl.slice[coefficients_counter:coefficients_counter+num_function_coefficients], function_displacements_y)
        #     # rotation_displacements_z = rotation_displacements_z.set(csdl.slice[coefficients_counter:coefficients_counter+num_function_coefficients], function_displacements_z)
        #     rotation_displacements = rotation_displacements.set(csdl.slice[coefficients_counter:coefficients_counter+num_function_coefficients], function_displacements)
        #     wireframe_counter += 625
        #     coefficients_counter += num_function_coefficients

        # local_displacements_from_rotations = rotation_displacements

        # local_displacements = local_displacements_from_displacements + local_displacements_from_rotations
        local_displacements = local_displacements_from_displacements

        if rigid_body_states is not None:
            temp_displacements = lsdo_geo.rotate(local_displacements, np.array([0., 0., 0.]), axis_vector=np.array([0., 0., 1.]), angles=rigid_body_states[5])
            temp_displacements = lsdo_geo.rotate(temp_displacements, np.array([0., 0., 0.]), axis_vector=np.array([0., 1., 0.]), angles=rigid_body_states[4])
            global_displacements = lsdo_geo.rotate(temp_displacements, np.array([0., 0., 0.]), axis_vector=np.array([1., 0., 0.]), angles=rigid_body_states[3])
        else:
            global_displacements = local_displacements
        
        counter = 0
        for function_index in function_indices:
            # function_coefficient_displacements = global_displacements[counter:counter+function.coefficients.size//num_physical_dimensions]

            function = geometry.functions[function_index]
            function_coefficients_shape = function.coefficients.shape
            function_coefficients_reshaped = function.coefficients.reshape((function.coefficients.size//num_physical_dimensions, num_physical_dimensions))
            # function_coefficients_reshaped = function_coefficients_reshaped + function_coefficient_displacements
            function_coefficients_reshaped = function_coefficients_reshaped + global_displacements[counter:counter+function.coefficients.size//num_physical_dimensions]
            function.coefficients = function_coefficients_reshaped.reshape(function_coefficients_shape)

            counter += function.coefficients.size//num_physical_dimensions


    def plot(self, point_types:list=['evaluated_points'], plot_types:list=['function'],
              opacity:float=1., color:lfs.FunctionSet='#00629B', color_map:str='jet', surface_texture:str="",
              line_width:float=3., additional_plotting_elements:list=[], show:bool=True) -> list:
        return self.geometry.plot(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color,
                                  color_map=color_map, surface_texture=surface_texture, line_width=line_width,
                                  additional_plotting_elements=additional_plotting_elements, show=show)

@dataclass
class PendulumSystem:
    pendulums: list[PendulumBody] = None
    constraint_pairs : list[tuple[np.ndarray,np.ndarray]] = None

    def __post_init__(self):
        if self.pendulums is None:
            self.pendulums = []
        if self.constraint_pairs is None:
            self.constraint_pairs = []

    def plot(self, point_types:list=['evaluated_points'], plot_types:list=['function'],
              opacity:float=1., color:lfs.FunctionSet='#00629B', color_map:str='jet', surface_texture:str="",
              line_width:float=3., additional_plotting_elements:list=[], show:bool=True) -> list:
        
        plotting_elements = additional_plotting_elements.copy()
        for pendulum in self.pendulums:
            plotting_elements = pendulum.plot(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color,
                                              color_map=color_map, surface_texture=surface_texture, line_width=line_width,
                                              additional_plotting_elements=plotting_elements, show=False)
        if show:
            lfs.show_plot(plotting_elements=plotting_elements, title='Pendulum System')

# Dummy clamped root boundary condition
DOLFIN_EPS = 3E-16
def ClampedBoundary(x):
    return np.greater(x[2], 0.5-DOLFIN_EPS)

"""
Residuals
"""
class NBodyPendulumModel:
    '''
    The residual model for the n-body pendulum system.
    '''
    def __init__(self, pendulum_system:PendulumSystem, time_step:float=0.0166, rho_infinity:float=0.82, g:float=9.80665) -> None:
        '''
        Creates an N-body pendulum residual model.

        Parameters
        ----------
        time_step : float
            The time step for the time discretization.
        rho_infinity : float
            The rho_infinity parameter for the generalized-alpha time discretization.
        num_bodies : int
            The number of bodies in the system.
        lengths : Union[dict[str,float],float], optional
            The lengths of the pendulums. If a float is provided, it is assumed that all pendulums have the same length.
            If a dictionary is provided, the keys should be the names of the pendulums and the values should be the lengths.
            If None, the default length is 1.
        masses : Union[dict[str,float],float], optional
            The masses of the pendulums. If a float is provided, it is assumed that all pendulums have the same mass.
            If a dictionary is provided, the keys should be the names of the pendulums and the values should be the masses.
            If None, the default mass is 1.
        damping_coefficients : Union[dict[str,float],float], optional
            The damping coefficients of the pendulums. If a float is provided, it is assumed that all pendulums have the same damping coefficient.
            If a dictionary is provided, the keys should be the names of the pendulums and the values should be the damping coefficients.
            If None, the default damping coefficient is 0.5.
        g : float, optional
            The acceleration due to gravity. The default is 9.80665.
        '''
        self.pendulum_system = pendulum_system
        
        self.time_step = time_step
        self.rho_infinity = rho_infinity
        self.num_bodies = len(pendulum_system.pendulums)
        self.g = g

        self.time_stepping_order = 2

        self.body_states_at_n = {}
        self.body_state_derivatives_at_n = {}
        # NOTE: Lagrange multipliers are an automatically generated key in these dictionaries at the same level as body states.
        for body in self.pendulum_system.pendulums:
            num_rigid_body_states = 6 # 3D
            num_flexible_states = body.num_flexible_states
            num_shell_nodes = body.num_shell_nodes

            # self.body_states_at_n[body.name] = csdl.Variable(shape=(num_rigid_body_states,), name=f'{body.name}_states_at_n')
            self.body_states_at_n[body.name] = {}
            self.body_states_at_n[body.name]['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states,), name=f'{body.name}_rigid_body_states_at_n')
            self.body_states_at_n[body.name]['flexible_states'] = csdl.Variable(shape=(num_flexible_states,), name=f'{body.name}_flexible_states_at_n')
            # self.body_state_derivatives_at_n[body.name] = csdl.Variable(shape=(num_rigid_body_states*self.time_stepping_order,), 
            #                                                        name=f'{body.name}_state_derivatives_at_n')
            self.body_state_derivatives_at_n[body.name] = {}
            self.body_state_derivatives_at_n[body.name]['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states*self.time_stepping_order,),
                                                                                              name=f'{body.name}_rigid_body_state_derivatives_at_n')
            self.body_state_derivatives_at_n[body.name]['flexible_states'] = csdl.Variable(shape=(num_flexible_states*self.time_stepping_order,),
                                                                                            name=f'{body.name}_flexible_state_derivatives_at_n')



    def evaluate(self, initial_states:NBodyPendulumStateData,
                 t_initial:float, t_final:float, time_step:float) -> dict[str,csdl.Variable]:
        '''
        Solves the N-body pendulum MBD problem. Returns history of state data across time.
        '''
        # set initial states
        for body in self.pendulum_system.pendulums:
            self.body_states_at_n[body.name]['rigid_body_states'] = initial_states.states[body.name]['rigid_body_states']
            self.body_state_derivatives_at_n[body.name]['rigid_body_states'].set_value(initial_states.state_derivatives[body.name]['rigid_body_states'])
            self.body_states_at_n[body.name]['flexible_states'] = initial_states.states[body.name]['flexible_states']
            self.body_state_derivatives_at_n[body.name]['flexible_states'].set_value(initial_states.state_derivatives[body.name]['flexible_states'])
        # self.lagrange_multipliers_at_n.value = initial_states.states['lagrange_multipliers']
        # self.lagrange_multiplier_derivatives_at_n.value = initial_states.state_derivatives['lagrange_multipliers']

        # Allocate n+1 states. These will be the design variables for the optimizer.
        self.body_states_at_n_plus_1 = {}
        for body in self.pendulum_system.pendulums:
            num_rigid_body_states = 6  # 3D
            num_shell_nodes = body.num_shell_nodes
            num_flexible_states = body.num_flexible_states 

            self.body_states_at_n_plus_1[body.name] = {}
            self.body_states_at_n_plus_1[body.name]['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states,),
                                                                                         name=f'{body.name}_rigid_body_states_at_n_plus_1', value=0.)
            self.body_states_at_n_plus_1[body.name]['flexible_states'] = csdl.Variable(shape=(num_flexible_states,),
                                                                                       name=f'{body.name}_flexible_states_at_n_plus_1', value=0.)
            
        # constraints model        
        # constraints = compute_constraints(states=self.body_states_at_n, lengths=self.lengths)
        # self.num_constraints = len(constraints) # NOTE: This is the number of physical constraints, not the length of the total vector

        # # lagrangian model
        # lagrangian = objective
        # # NOTE: Need to manually compute lagrangian because we need to parameterize lagrange multipliers using generalized alpha model
        self.num_constraints = 5*self.num_bodies

        # Preallocate states so that the variable can be replaced each time step, and the previous time step can be used as the initial value
        self.lagrange_multipliers_at_n = {}
        self.lagrange_multipliers_at_n['physical_constraints'] = []         # Constraints to keep the pendulums from falling apart
        self.lagrange_multipliers_at_n['structural_constraints'] = []   # Constraints to keep the center of mass at the center of the geometry
        self.lagrange_multipliers_at_n_plus_1 = {}
        self.lagrange_multipliers_at_n_plus_1['physical_constraints'] = []         # Constraints to keep the pendulums from falling apart
        self.lagrange_multipliers_at_n_plus_1['structural_constraints'] = []    # Constraints to keep the center of mass at the center of the geometry
        self.lagrange_multiplier_derivatives_at_n = {}
        self.lagrange_multiplier_derivatives_at_n['physical_constraints'] = []
        self.lagrange_multiplier_derivatives_at_n['structural_constraints'] = []
        for i in range(self.num_bodies):
            self.lagrange_multipliers_at_n['physical_constraints'].append(csdl.Variable(shape=(3,), value=0., 
                                                           name='preallocated_physical_translational_consraint_lagrange_multipliers'))
            self.lagrange_multipliers_at_n['physical_constraints'].append(csdl.Variable(shape=(2,), value=0., 
                                                           name='preallocated_physical_rotational_consraint_lagrange_multipliers'))
            self.lagrange_multiplier_derivatives_at_n['physical_constraints'].append(csdl.Variable(shape=(self.time_stepping_order*3,), value=0.,
                                                                            name='preallocated_physical_translational_consraint_lagrange_multiplier_derivatives_at_n'))
            self.lagrange_multiplier_derivatives_at_n['physical_constraints'].append(csdl.Variable(shape=(self.time_stepping_order*2,), value=0.,
                                                                            name='preallocated_physical_rotational_consraint_lagrange_multiplier_derivatives_at_n'))
            
            # This is the design variable so to speak (implicit variable)
            self.lagrange_multipliers_at_n_plus_1['physical_constraints'].append(csdl.Variable(shape=(3,), value=0.,
                                                                        name='preallocated_physical_translational_consraint_lagrange_multiplier_at_n_plus_1'))
            self.lagrange_multipliers_at_n_plus_1['physical_constraints'].append(csdl.Variable(shape=(2,), value=0.,
                                                                        name='preallocated_physical_rotational_consraint_lagrange_multiplier_at_n_plus_1'))
            
            self.lagrange_multipliers_at_n['structural_constraints'].append(csdl.Variable(shape=(3,), value=0.,
                                                              name='preallocated_structural_center_of_mass_lagrange_multipliers'))
            self.lagrange_multipliers_at_n['structural_constraints'].append(csdl.Variable(shape=(3,), value=0.,
                                                              name='preallocated_structural_angle_of_mass_lagrange_multipliers'))
            self.lagrange_multiplier_derivatives_at_n['structural_constraints'].append(csdl.Variable(shape=(self.time_stepping_order*3,), value=0.,
                                                                            name='preallocated_structural_center_of_mass_lagrange_multiplier_derivatives_at_n'))
            self.lagrange_multiplier_derivatives_at_n['structural_constraints'].append(csdl.Variable(shape=(self.time_stepping_order*3,), value=0.,
                                                                            name='preallocated_structural_angle_of_mass_lagrange_multiplier_derivatives_at_n'))
            
            # This is the design variable so to speak (implicit variable)
            self.lagrange_multipliers_at_n_plus_1['structural_constraints'].append(csdl.Variable(shape=(3,), value=0.,
                                                                        name='preallocated_center_of_mass_lagrange_multiplier_at_n_plus_1'))
            self.lagrange_multipliers_at_n_plus_1['structural_constraints'].append(csdl.Variable(shape=(3,), value=0.,
                                                                        name='preallocated_center_of_mass_lagrange_multiplier_at_n_plus_1'))
            
        self.time = np.arange(t_initial, t_final, time_step)
        state_history = {}
        state_derivative_history = {}

        # Allocate state history arrays
        for body in self.pendulum_system.pendulums:
            num_rigid_body_states = 6  # 3D
            num_shell_nodes = body.num_shell_nodes
            num_flexible_states = body.num_flexible_states 

            # state_history[body.name] = csdl.Variable(value=np.zeros((len(self.time)+1, num_rigid_body_states)))
            state_history[body.name] = {}
            state_history[body.name]['rigid_body_states'] = csdl.Variable(value=np.zeros((len(self.time)+1, num_rigid_body_states)))
            state_history[body.name]['flexible_states'] = csdl.Variable(value=np.zeros((len(self.time)+1, num_flexible_states)))

            num_rigid_body_state_derivatives = num_rigid_body_states*self.time_stepping_order
            num_flexible_state_derivatives = num_flexible_states*self.time_stepping_order
            # state_derivative_history[body.name] = csdl.Variable(value=np.zeros((len(self.time)+1, num_state_derivatives)))
            state_derivative_history[body.name] = {}
            state_derivative_history[body.name]['rigid_body_states'] = csdl.Variable(value=np.zeros((len(self.time)+1, num_rigid_body_state_derivatives)))
            state_derivative_history[body.name]['flexible_states'] = csdl.Variable(value=np.zeros((len(self.time)+1, num_flexible_state_derivatives)))

            state_history[body.name]['rigid_body_states'] = state_history[body.name]['rigid_body_states'].set(csdl.slice[0,:], initial_states.states[body.name]['rigid_body_states'])
            state_derivative_history[body.name]['rigid_body_states'] = state_derivative_history[body.name]['rigid_body_states'].set(csdl.slice[0,:], initial_states.state_derivatives[body.name]['rigid_body_states'])
            state_history[body.name]['flexible_states'] = state_history[body.name]['flexible_states'].set(csdl.slice[0,:], initial_states.states[body.name]['flexible_states'])
            state_derivative_history[body.name]['flexible_states'] = state_derivative_history[body.name]['flexible_states'].set(csdl.slice[0,:], initial_states.state_derivatives[body.name]['flexible_states'])

        # Allocate lagrange multipliers
        lagrange_multiplier_history = {}
        lagrange_multiplier_history['physical_constraints'] = []
        lagrange_multiplier_history['structural_constraints'] = []
        for i in range(self.num_bodies):
            lagrange_multiplier_history['physical_constraints'].append(csdl.Variable(name=f'body_{i}_physical_translational_constraint_lagrange_multipliers',
                                                                       value=np.zeros((len(self.time)+1, 3))))
            lagrange_multiplier_history['physical_constraints'].append(csdl.Variable(name=f'body_{i}_physical_alignment_constraint_lagrange_multipliers',
                                                                                     value=np.zeros((len(self.time)+1, 2))))
            lagrange_multiplier_history['structural_constraints'].append(csdl.Variable(name=f'body_{i}_structural_center_of_mass_constraint_lagrange_multipliers',
                                                                                       value=np.zeros((len(self.time)+1, 3))))
            lagrange_multiplier_history['structural_constraints'].append(csdl.Variable(name=f'body_{i}_structural_angle_of_mass_constraint_lagrange_multipliers',
                                                                                       value=np.zeros((len(self.time)+1, 3))))

        # Set initial values
        # for i, lagrange_multipliers in enumerate(self.lagrange_multipliers_at_n):
        #     lagrange_multiplier_history[i][0,:] = initial_states.states['lagrange_multipliers'][i].value

        
        # for i in range(len(self.time)):
        for i in csdl.frange(0, len(self.time)):
            # solver = csdl.nonlinear_solvers.Newton(residual_jac_kwargs={'concatenate_ofs':True, 'loop':False})
            # solver = csdl.nonlinear_solvers.Newton(residual_jac_kwargs={'concatenate_ofs':True, 'loop':True})
            # solver = csdl.nonlinear_solvers.Jacobi(residual_jac_kwargs={'concatenate_ofs':True, 'loop':True}, max_iter=1)
            # recorder = csdl.get_current_recorder()
            # recorder.active_graph.rxgraph.check_cycle = True

            solver = csdl.nonlinear_solvers.Newton()

            # Allocate n+1 states. These will be the design variables for the optimizer.
            for body in self.pendulum_system.pendulums:
                num_rigid_body_states = 6  # 3D
                num_shell_nodes = body.num_shell_nodes
                num_flexible_states = body.num_flexible_states 
                # self.body_states_at_n_plus_1[body.name] = csdl.Variable(shape=(num_states,), name=f'{body.name}_rigid_body_states_at_n_plus_1',
                #                                                 # value=self.body_states_at_n[body.name].value)
                #                                                 value=0.)
                self.body_states_at_n_plus_1[body.name]['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states,),
                                                                                             name=f'{body.name}_rigid_body_states_at_n_plus_1',
                                                                                             value=self.body_states_at_n[body.name]['rigid_body_states'].value)
                                                                                            #  value=0.)
                self.body_states_at_n_plus_1[body.name]['flexible_states'] = csdl.Variable(shape=(num_flexible_states,),
                                                                                           name=f'{body.name}_flexible_states_at_n_plus_1',
                                                                                           value=self.body_states_at_n[body.name]['flexible_states'].value)
                                                                                        #    value=0.)
                

            state_data = {}
            body_residuals = {}
            # Apply time discretization to states
            for body in self.pendulum_system.pendulums:
                state_data[body.name] = {}
                generalized_alpha_model = GeneralizedAlphaModel(rho_infinity=self.rho_infinity, num_states=6,
                                                            time_step=self.time_step)
                body_state_data = generalized_alpha_model.evaluate(states_at_n=self.body_states_at_n[body.name]['rigid_body_states'],
                                    states_at_n_plus_1=self.body_states_at_n_plus_1[body.name]['rigid_body_states'],
                                    state_velocities_at_n=self.body_state_derivatives_at_n[body.name]['rigid_body_states'][0:6],
                                    state_accelerations_at_n=self.body_state_derivatives_at_n[body.name]['rigid_body_states'][6:])
                state_data[body.name]['rigid_body_states'] = body_state_data

                # Apply time discretization to flexible states
                generalized_alpha_model = GeneralizedAlphaModel(rho_infinity=self.rho_infinity, num_states=body.num_flexible_states,
                                                            time_step=self.time_step)
                body_state_data = generalized_alpha_model.evaluate(states_at_n=self.body_states_at_n[body.name]['flexible_states'],
                                    states_at_n_plus_1=self.body_states_at_n_plus_1[body.name]['flexible_states'],
                                    state_velocities_at_n=self.body_state_derivatives_at_n[body.name]['flexible_states'][0:body.num_flexible_states],
                                    state_accelerations_at_n=self.body_state_derivatives_at_n[body.name]['flexible_states'][body.num_flexible_states:])
                state_data[body.name]['flexible_states'] = body_state_data

            
            # Evaluate updated geometry
            for body in self.pendulum_system.pendulums:
                # Start from original geometry each time
                body.geometry = body.design_geometry.copy()

                # Apply rigid body motion
                body.apply_rigid_body_motion(state_data[body.name]['rigid_body_states'].states)


                # Apply flexible motion
                body.apply_flexible_motion(state_data[body.name]['flexible_states'].states,
                                           state_data[body.name]['rigid_body_states'].states)
                # body.geometry.plot()
                # exit()
            # Evaluate rigid body and flexible residuals (Inertial terms and driving gravitational force)
            # Evaluate rigid body and flexible residuals (Inertial terms and driving gravitational force)
            for body in self.pendulum_system.pendulums:
                body_state_data = state_data[body.name]

                # residual
                moment_of_inertia = body.evaluate_mass_properties(properties_to_compute=['moment_of_inertia'])[2]
                mass_matrix = csdl.Variable(value=np.zeros((6,6)))
                mass_matrix = mass_matrix.set(csdl.slice[0,0], body.mass)
                mass_matrix = mass_matrix.set(csdl.slice[1,1], body.mass)
                mass_matrix = mass_matrix.set(csdl.slice[2,2], body.mass)
                mass_matrix = mass_matrix.set(csdl.slice[3:,3:], moment_of_inertia)

                # Evaluate the rigid body dofs portion of the residual
                inertial_term_model = ResidualInertialTerm()
                inertial_term = inertial_term_model.evaluate(state_accelerations=body_state_data['rigid_body_states'].state_accelerations,
                                                             mass_matrix=mass_matrix)

                rigid_body_gravity_residual_3d_model = RigidBodyGravityResidual3D(g=self.g)
                gravity_residual = rigid_body_gravity_residual_3d_model.evaluate(mass=body.mass)

                rigid_body_residual = inertial_term - gravity_residual


                # ############################ RM shell ##########################



                shell_model = body.shell_model
                nn = shell_model.nn
                nel = shell_model.nel

                gravitational_acceleration = csdl.Variable(value=np.array([0, 0, -self.g]))    # gravitational work
                # Defining flexible states/residual in local frame, so rotate gravitational acceleration to local frame
                gravitational_acceleration = lsdo_geo.rotate(gravitational_acceleration, np.array([0., 0., 0.]), axis_vector=np.array([0., 0., 1.]), 
                                                             angles=-state_data[body.name]['rigid_body_states'].states[5]).reshape((3,))
                gravitational_acceleration = lsdo_geo.rotate(gravitational_acceleration, np.array([0., 0., 0.]), axis_vector=np.array([0., 1., 0.]), 
                                                             angles=-state_data[body.name]['rigid_body_states'].states[4]).reshape((3,))
                gravitational_acceleration = lsdo_geo.rotate(gravitational_acceleration, np.array([0., 0., 0.]), axis_vector=np.array([1., 0., 0.]), 
                                                             angles=-state_data[body.name]['rigid_body_states'].states[3]).reshape((3,))

                #TODO: need to implement this for the shell model

                force_vector = csdl.expand(body.material.thickness \
                                            *body.material.density \
                                            *gravitational_acceleration, 
                                            (nn, 3), action='i->ji')
                force_vector.add_name('force_vector')
                
                node_disp = csdl.Variable(value=np.zeros((nn, 3)), name='node_disp')
                node_disp.add_name('node_disp')

                dynamic_shell_outputs = shell_model.evaluate_dynamic_residual(
                            disp_solid=body_state_data['flexible_states'].states, 
                            wdot=body_state_data['flexible_states'].state_velocities, 
                            wddot=body_state_data['flexible_states'].state_accelerations, 
                            force_vector=force_vector, 
                            thickness=body.shell_thickness, 
                            E=body.shell_E, nu=body.shell_nu, density=body.shell_density)


                flexible_residual = dynamic_shell_outputs.dynamic_residual
                # print("flexible_residual", flexible_residual.value)
                # exit()
                # flexible_residual = flexible_residual.set(csdl.slice[:3], flexible_residual[:3] + 10000*self.body_states_at_n_plus_1[body.name]['flexible_states'][:3]) # TODO: This is a hack to get the optimizer to move the states
                # flexible_residual = flexible_residual + 1.e-6*self.body_states_at_n_plus_1[body.name]['flexible_states']

                body_residuals[body.name] = {'rigid_body_residual': rigid_body_residual, 'flexible_residual': flexible_residual}


                # body.geometry.plot()
            # physical constraints model        
            physical_constraints = compute_physical_constraints(pendulum_system=self.pendulum_system)


            # structural constraints
            structural_constraints = compute_structural_constraints(pendulum_system=self.pendulum_system, state_data=state_data)

            lagrange_multiplier_data = {}
            lagrange_multiplier_data['physical_constraints'] = []
            lagrange_multiplier_data['structural_constraints'] = []
            # Compute lagrange multipliers for the phyiscal constraints and add the lagrange multiplier term to the MBD system
            body_residuals, solver = add_constraints_to_system(self, body_residuals, solver,
                                                                        physical_constraints, 'physical_constraints', lagrange_multiplier_data)

            # Compute lagrange multipliers for the structural constraints and add the lagrange multiplier term to the MBD system
            body_residuals, solver = add_constraints_to_system(self, body_residuals, solver, 
                                                                        structural_constraints, 'structural_constraints', lagrange_multiplier_data)

            # Add the state/residual pairs for the body residuals
            for body, states in self.body_states_at_n_plus_1.items():
                solver.add_state(states['rigid_body_states'], body_residuals[body]['rigid_body_residual'], initial_value=self.body_states_at_n[body]['rigid_body_states'])
                solver.add_state(states['flexible_states'], body_residuals[body]['flexible_residual'], initial_value=self.body_states_at_n[body]['flexible_states'])

            # print("rigid body residual", body_residuals[body]['rigid_body_residual'].value)
            # print("flexible residual", body_residuals[body]['flexible_residual'].value)
            # print("physical_constraints "+physical_constraints[0].name, physical_constraints[0].value)
            # print("physical_constraints "+physical_constraints[1].name, physical_constraints[1].value)

            # print("structural_constraints "+structural_constraints[0].name, structural_constraints[0].value)
            # print("structural_constraints "+structural_constraints[1].name, structural_constraints[1].value)

            # print("Lagrange_multiplier "+lagrange_multiplier_data['physical_constraints'][0].states.name, 
            #       lagrange_multiplier_data['physical_constraints'][0].states.value)
            # print("Lagrange_multiplier "+lagrange_multiplier_data['physical_constraints'][1].states.name, 
            #       lagrange_multiplier_data['structural_constraints'][0].states.value)
            

            print("Running Newton solver ...")
            solver.run()

            print("Finished Newton solver ...")
            for body in self.pendulum_system.pendulums:
                state_history[body.name]['rigid_body_states'] = state_history[body.name]['rigid_body_states'].set(csdl.slice[i+1,:], self.body_states_at_n_plus_1[body.name]['rigid_body_states'])
                state_derivative_history[body.name]['rigid_body_states'] = state_derivative_history[body.name]['rigid_body_states'].set(csdl.slice[i,:], self.body_state_derivatives_at_n[body.name]['rigid_body_states'])
                state_history[body.name]['flexible_states'] = state_history[body.name]['flexible_states'].set(csdl.slice[i+1,:], self.body_states_at_n_plus_1[body.name]['flexible_states'])
                state_derivative_history[body.name]['flexible_states'] = state_derivative_history[body.name]['flexible_states'].set(csdl.slice[i,:], self.body_state_derivatives_at_n[body.name]['flexible_states'])

                # Save newly calculated n+1 states and derivatives as n for next time step
                self.body_states_at_n[body.name]['rigid_body_states'] = self.body_states_at_n_plus_1[body.name]['rigid_body_states']
                self.body_state_derivatives_at_n[body.name]['rigid_body_states'] = \
                    self.body_state_derivatives_at_n[body.name]['rigid_body_states'].set(csdl.slice[:6],
                                                                        state_data[body.name]['rigid_body_states'].state_velocities_at_n_plus_1)
                self.body_state_derivatives_at_n[body.name]['rigid_body_states'] = \
                    self.body_state_derivatives_at_n[body.name]['rigid_body_states'].set(csdl.slice[6:],
                                                                            state_data[body.name]['rigid_body_states'].state_accelerations_at_n_plus_1)
                
                self.body_states_at_n[body.name]['flexible_states'] = self.body_states_at_n_plus_1[body.name]['flexible_states']
                self.body_state_derivatives_at_n[body.name]['flexible_states'] = \
                    self.body_state_derivatives_at_n[body.name]['flexible_states'].set(csdl.slice[:6*num_shell_nodes],
                                                                        state_data[body.name]['flexible_states'].state_velocities_at_n_plus_1)
                self.body_state_derivatives_at_n[body.name]['flexible_states'] = \
                    self.body_state_derivatives_at_n[body.name]['flexible_states'].set(csdl.slice[6*num_shell_nodes:],
                                                                            state_data[body.name]['flexible_states'].state_accelerations_at_n_plus_1)
                
            # self.lagrange_multipliers_at_n = self.lagrange_multipliers_at_n_plus_1.copy()
            self.lagrange_multipliers_at_n = {}

            self.lagrange_multipliers_at_n['physical_constraints'] = self.lagrange_multipliers_at_n_plus_1['physical_constraints'].copy()
            for j, constraint in enumerate(physical_constraints):
                # Save newly calculated n+1 lagrange multipliers and derivatives as n for next time step
                self.lagrange_multiplier_derivatives_at_n['physical_constraints'][j] = \
                    self.lagrange_multiplier_derivatives_at_n['physical_constraints'][j].set(csdl.slice[:constraint.size],
                                                                    lagrange_multiplier_data['physical_constraints'][j].state_velocities_at_n_plus_1)
                self.lagrange_multiplier_derivatives_at_n['physical_constraints'][j] = \
                    self.lagrange_multiplier_derivatives_at_n['physical_constraints'][j].set(csdl.slice[constraint.size:],
                                                                    lagrange_multiplier_data['physical_constraints'][j].state_accelerations_at_n_plus_1)

                for j, constraint_lagrange_multiplier_data in enumerate(lagrange_multiplier_data['physical_constraints']):
                    lagrange_multiplier_history['physical_constraints'][j] = lagrange_multiplier_history['physical_constraints'][j].set(
                                                                                    csdl.slice[i+1,:], constraint_lagrange_multiplier_data.states)

            self.lagrange_multipliers_at_n['structural_constraints'] = self.lagrange_multipliers_at_n_plus_1['structural_constraints'].copy()
            for j, constraint in enumerate(structural_constraints):
                # Save newly calculated n+1 lagrange multipliers and derivatives as n for next time step
                self.lagrange_multiplier_derivatives_at_n['structural_constraints'][j] = \
                    self.lagrange_multiplier_derivatives_at_n['structural_constraints'][j].set(csdl.slice[:constraint.size],
                                                                    lagrange_multiplier_data['structural_contraints'][j].state_velocities_at_n_plus_1)
                self.lagrange_multiplier_derivatives_at_n['structural_contraints'][j] = \
                    self.lagrange_multiplier_derivatives_at_n['structural_contraints'][j].set(csdl.slice[constraint.size:],
                                                                    lagrange_multiplier_data['structural_contraints'][j].state_accelerations_at_n_plus_1)

                for j, constraint_lagrange_multiplier_data in enumerate(lagrange_multiplier_data['structural_contraints']):
                    lagrange_multiplier_history['structural_contraints'][j] = lagrange_multiplier_history['structural_contraints'][j].set(
                                                                                    csdl.slice[i+1,:], constraint_lagrange_multiplier_data.states)

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')

        return self.time, state_history, state_derivative_history, lagrange_multiplier_history


def compute_physical_constraints(pendulum_system:PendulumSystem) -> list[csdl.Variable]:
    '''
    Computes the body-to-body physical constraints for the N-body pendulum system.
    '''
    # num_constraints_per_body = 5
    constraints = []
    for i, pendulum_body in enumerate(pendulum_system.pendulums):
        constraint_pair = pendulum_system.constraint_pairs[i]
        # constraint = csdl.Variable(shape=(num_constraints_per_body,), value=0.)
        if i == 0:
            translational_constraint = pendulum_body.geometry.evaluate(constraint_pair[0]) - constraint_pair[1]
            
            alignment_axis = pendulum_body.geometry.evaluate(constraint_pair[0], parametric_derivative_orders=(1,0))
            normalized_alignment_axis = alignment_axis/csdl.norm(alignment_axis)
            # alignment_constraint = normalized_alignment_axis[:2] - np.array([1., 0.])   # NOTE: Hardcoding rotation about x axis
            alignment_constraint = normalized_alignment_axis[1:] - np.array([0., 0.])   # NOTE: Hardcoding rotation about x axis
            # NOTE: For alignment constraint, we only want 2 constraints (6 dof, 5 constraints per body). I think this does it.
        else:
            translational_constraint = pendulum_body.geometry.evaluate(constraint_pair[0]) - \
                                        pendulum_system.pendulums[i-1].geometry.evaluate(constraint_pair[1])
            alignment_axis = pendulum_body.geometry.evaluate(constraint_pair[0], parametric_derivative_orders=(1,0))
            normalized_alignment_axis = alignment_axis/csdl.norm(alignment_axis)
            # alignment_constraint = normalized_alignment_axis[:2] - np.array([1., 0.])   # NOTE: Hardcoding rotation about x axis
            alignment_constraint = normalized_alignment_axis[1:] - np.array([0., 0.])   # NOTE: Hardcoding rotation about x axis
            # NOTE: For alignment constraint, we only want 2 constraints (6 dof, 5 constraints per body). I think this does it.
        
        # constraint = csdl.vstack([translational_constraint, alignment_constraint])
        # constraint = constraint.set(csdl.slice[:3], translational_constraint)
        # constraint = constraint.set(csdl.slice[3:], alignment_constraint)
        # constraints.append(constraint)
        translational_constraint.add_name(f'{pendulum_body.name}_translational_physical_constraint')
        alignment_constraint.add_name(f'{pendulum_body.name}_alignment_physical_constraint')
        constraints.append(translational_constraint)
        constraints.append(alignment_constraint)

    return constraints

def compute_structural_constraints(pendulum_system:PendulumSystem, state_data:GeneralizedAlphaStateData) -> list[csdl.Variable]:
    '''
    Computes the difference between the state center of mass (rigid body state for each body) and the center of mass as
    defined by the geometry.
    '''
    constraints = []

    for pendulum_body in pendulum_system.pendulums:
        # local_pendulum = pendulum_body.copy()
        design_geometry = pendulum_body.design_geometry.copy()

        pendulum_body.apply_flexible_motion(flexible_states=state_data[pendulum_body.name]['flexible_states'].states,
                                             geometry=design_geometry)
        deformed_geometry = design_geometry

        _, center_of_mass, _, angular_momentum = pendulum_body.evaluate_mass_properties(geometry=deformed_geometry,
                                                                                        properties_to_compute=['center_of_mass', 'angular_momentum'])

        constraint = center_of_mass - pendulum_body.design_center_of_mass
        constraint.add_name(f'{pendulum_body.name}_center_of_mass_constraint')
        constraints.append(constraint)

        # angle_of_mass.add_name(f'{pendulum_body.name}_angle_of_mass')
        # constraint = angle_of_mass - pendulum_body.design_angle_of_mass
        # constraint.add_name(f'{pendulum_body.name}_angle_of_mass_constraint')
        # constraints.append(constraint)

        angular_momentum.add_name(f'{pendulum_body.name}_angular_momentum')
        constraint = angular_momentum # Initial "angular momentum" is always 0 because the positions/velocities are relative to the design geometry
        constraint.add_name(f'{pendulum_body.name}_angular_momentum_constraint')
        constraints.append(constraint)

    return constraints


def add_constraints_to_system(model:NBodyPendulumModel, body_residuals:dict[str,dict[str,csdl.Variable]], solver:csdl.nonlinear_solvers.Newton, 
                                       constraints:list[csdl.Variable], constraints_name:str, lagrange_multiplier_data:dict[str,list[GeneralizedAlphaStateData]]) \
                                        -> tuple[dict[str,dict[str,csdl.Variable]], csdl.nonlinear_solvers.Newton]:
    for j, constraint in enumerate(constraints):
        model.lagrange_multipliers_at_n_plus_1[constraints_name][j] = csdl.Variable(shape=(constraint.size,), value=0.,
                                                                name=constraints_name+'_lagrange_multiplier_at_n_plus_1')

        generalized_alpha_lagrange_multipliers = GeneralizedAlphaModel(rho_infinity=model.rho_infinity, num_states=constraint.size,
                                                                        time_step=model.time_step)
        constraint_lagrange_multiplier_data = generalized_alpha_lagrange_multipliers.evaluate(states_at_n=model.lagrange_multipliers_at_n[constraints_name][j],
                                states_at_n_plus_1=model.lagrange_multipliers_at_n_plus_1[constraints_name][j],
                                state_velocities_at_n=model.lagrange_multiplier_derivatives_at_n[constraints_name][j][0:constraint.size],
                                state_accelerations_at_n=model.lagrange_multiplier_derivatives_at_n[constraints_name][j][constraint.size:])
        
        lagrange_multiplier_data[constraints_name].append(constraint_lagrange_multiplier_data)
    
        constraint_lagrange_multipliers = constraint_lagrange_multiplier_data.states

        body_residuals = add_lagrange_multiplier_term_to_residual(body_residuals, constraint, constraint_lagrange_multipliers,
                                                                  model.body_states_at_n_plus_1, 'rigid_body')
        body_residuals = add_lagrange_multiplier_term_to_residual(body_residuals, constraint, constraint_lagrange_multipliers,
                                                                    model.body_states_at_n_plus_1, 'flexible')

        # Add the state/residual pairs for the constraints
        solver.add_state(model.lagrange_multipliers_at_n_plus_1[constraints_name][j], constraint, initial_value=model.lagrange_multipliers_at_n[constraints_name][j])
        # solver.add_state(model.lagrange_multipliers_at_n_plus_1[constraints_name][j], constraint)

    return body_residuals, solver
    

def add_lagrange_multiplier_term_to_residual(body_residuals:dict[str,csdl.Variable], constraint:csdl.Variable, lagrange_multipliers:csdl.Variable,
                                             body_states:dict[str,csdl.Variable], states_name:str) -> dict[str,csdl.Variable]:
    '''
    Adds the dc_dx*lambda term to the body residuals.
    '''
    for body, states in body_states.items():
        current_graph = csdl.get_current_recorder().active_graph
        vecmat = csdl.src.operations.derivatives.reverse.vjp(
            [(constraint,lagrange_multipliers)],
            states[f'{states_name}_states'],
            current_graph,
        )[states[f'{states_name}_states']]
        if vecmat is not None:
            body_residuals[body][f'{states_name}_residual'] = body_residuals[body][f'{states_name}_residual'] + vecmat

    return body_residuals
