import csdl_alpha as csdl
import CADDEE_alpha as cd
from CADDEE_alpha import functions as fs
import numpy as np
import os
import sys
from VortexAD.core.vlm.vlm_solver import vlm_solver
import aframe as af

import matplotlib.pyplot as plt

# lsdo_airfoil must be cloned and installed from https://github.com/LSDOlab/lsdo_airfoil
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker


# Settings
couple = False
optimize = False
check_derivatives = False
inline = True
ML = False
trim = False
element_wise_thickness = True


# Time stepping parameters
Nsteps = 3
dt = 0.01
T = Nsteps*dt

# gust velocity
# [Vx, Vy, Vz]
V_inf = 50.  # freestream velocity magnitude in m/s
# V_p = 50. # peak velocity of the gust
V_p = 5. # peak velocity of the gust
# Time-stepping parameters
# T       = 0.5
l_chord = 1.2 # unit: m, chord length
GGLc = 5 # gust gradient length in chords
T0 = 0.02 # static before the gust
T1 = GGLc*l_chord/V_inf

def V_g(t):
    V_g = 0.
    if T0 <= t <= T0+T1:
        # V_g = V_p*np.sin(2*np.pi*t/T1)**2
        V_g = V_p*(1-np.cos(2*np.pi*(t-T0)/T1))
    return V_g


# The shell solver requires FEniCSx and femo, which require manual installation.
# See https://github.com/LSDOlab/femo_alpha

# from femo_alpha.rm_shell.rm_shell_model import RMShellModel

from femo_alpha.dynamic_rm_shell.plate_sim import PlateSim
from femo_alpha.dynamic_rm_shell.state_operation_dynamic import StateOperation
from femo_alpha.dynamic_rm_shell.total_strain_energy_operation import TotalStrainEnergyOperation
from femo_alpha.dynamic_rm_shell.volume_operation import VolumeOperation
from femo_alpha.fea.utils_dolfinx import readFEAMesh, reconstructFEAMesh

# Quantities
skin_thickness = 0.007
spar_thickness = 0.001
rib_thickness = 0.001

c172_wing = [#### quad mesh ####
        "c172_650",
        "c172_2106",]

file_path = "./cessna_172_shell_meshes/"
mesh_fname = file_path+c172_wing[0]
mass = 1000
stress_bound = 1e8
# strain_energy_bound = 1e8 # TODO: add this constraint by simulation results
# num_ribs = 10
num_ribs = 6
load_factor = 3

# Start recording
rec = csdl.Recorder(inline=True, debug=True)
rec.start()

# Initialize CADDEE and import geometry
caddee = cd.CADDEE()
c172_geom = cd.import_geometry('c172.stp')

def define_base_config(caddee : cd.CADDEE):
    aircraft = cd.aircraft.components.Aircraft(geometry=c172_geom)
    base_config = cd.Configuration(system=aircraft)
    caddee.base_configuration = base_config

    airframe = aircraft.comps["airframe"] = cd.Component()

    # Setup wing component
    wing_geometry = aircraft.create_subgeometry(
        search_names=['MainWing'],
        # The wing coming out of openVSP has some extra surfaces that we don't need
        ignore_names=['0, 8', '0, 9', '0, 14', '0, 15', '1, 16', '1, 17', '1, 22', '1, 23']
    )
    wing = cd.aircraft.components.Wing(AR=1, S_ref=1, geometry=wing_geometry)
    airframe.comps["wing"] = wing
    
    # Generate internal geometry
    wing.construct_ribs_and_spars(
        c172_geom,
        num_ribs=num_ribs,
        LE_TE_interpolation="ellipse",
        # full_length_ribs=True,
        full_length_ribs=False,
        spanwise_multiplicity=10,
        offset=np.array([0.,0.,.15]),
        # export_wing_box=True,
        # export_half_wing=True,
        finite_te=True,
    )
    # wing.plot()
    # exit()
    # extract relevant geometries
    right_wing = wing.create_subgeometry(search_names=[''], ignore_names=[', 1, ', '_r_', '-'])
    right_wing_oml = wing.create_subgeometry(search_names=['MainWing, 0'])
    left_wing_oml = wing.create_subgeometry(search_names=['MainWing, 1'])
    right_wing_spars = wing.create_subgeometry(search_names=['spar'], ignore_names=['_r_', '-'])
    right_wing_ribs = wing.create_subgeometry(search_names=['rib'], ignore_names=['_r_', '-'])
    wing_oml = wing.create_subgeometry(search_names=['MainWing'])
    wing.quantities.right_wing_oml = right_wing_oml
    wing.quantities.oml = wing_oml

    # material
    E = csdl.Variable(value=69E9, name='E')
    G = csdl.Variable(value=26E9, name='G')
    density = csdl.Variable(value=2700, name='density')
    nu = csdl.Variable(value=0.33, name='nu')
    aluminum = cd.materials.IsotropicMaterial(name='aluminum', E=E, G=G, 
                                                density=density, nu=nu)

    # Define thickness functions
    # The ribs and spars have a constant thickness, while the skin has a variable thickness that we will optimize
    thickness_fs = fs.ConstantSpace(2)
    skin_fs = fs.BSplineSpace(2, (2,1), (5,2))
    r_skin_fss = right_wing_oml.create_parallel_space(skin_fs)
    skin_t_coeffs, skin_fn = r_skin_fss.initialize_function(1, value=skin_thickness)
    spar_fn = fs.Function(thickness_fs, spar_thickness)
    rib_fn = fs.Function(thickness_fs, rib_thickness)

    # correlate the left and right wing skin thickness functions - want symmetry
    oml_lr_map = {rind:lind for rind, lind in zip(right_wing_oml.functions, left_wing_oml.functions)}
    wing.quantities.oml_lr_map = oml_lr_map

    # build function set out of the thickness functions
    functions = skin_fn.functions.copy()
    for ind in wing.geometry.functions:
        name = wing.geometry.function_names[ind]
        if "spar" in name:
            functions[ind] = spar_fn
        elif "rib" in name:
            functions[ind] = rib_fn

    for rind, lind in oml_lr_map.items():
        # the v coord is flipped left to right
        functions[lind] = fs.Function(skin_fs, functions[rind].coefficients[:,::-1,:])

    thickness_function_set = fs.FunctionSet(functions)
    wing.quantities.material_properties.set_material(aluminum, thickness_function_set)

    # set skin thickness as a design variable
    skin_t_coeffs.set_as_design_variable(upper=0.05, lower=0.0001, scaler=5e2)
    skin_t_coeffs.add_name('skin_thickness')

    # Spaces for states
    # pressure
    pressure_function_space = fs.IDWFunctionSpace(num_parametric_dimensions=2, order=4, grid_size=(240, 40), conserve=False)
    indexed_pressue_function_space = wing_oml.create_parallel_space(pressure_function_space)
    wing.quantities.pressure_space = indexed_pressue_function_space

    # displacement
    displacement_space = fs.BSplineSpace(2, (1,1), (3,3))
    wing.quantities.displacement_space = wing_geometry.create_parallel_space(
                                                    displacement_space)
    wing.quantities.oml_displacement_space = wing_oml.create_parallel_space(
                                                    displacement_space)

    # meshing
    mesh_container = base_config.mesh_container

    # vlm mesh
    vlm_mesh = cd.mesh.VLMMesh()
    wing_chord_surface = cd.mesh.make_vlm_surface(
        wing, 40, 1, LE_interp="ellipse", TE_interp="ellipse", 
        spacing_spanwise="cosine", ignore_camber=True, plot=False,
    )
    wing_chord_surface.project_airfoil_points(oml_geometry=wing_oml)
    vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface
    mesh_container["vlm_mesh"] = vlm_mesh


    # shell meshing
    # import shell mesh
    wing_shell_mesh = cd.mesh.import_shell_mesh(
        mesh_fname+'.msh', 
        right_wing,
        rescale=[1e-3, 1e-3, 1e-3],
        grid_search_n=5,
        priority_inds=[i for i in right_wing_oml.functions],
        priority_eps=3e-6,
    )
    process_elements(wing_shell_mesh, right_wing_oml, right_wing_ribs, right_wing_spars)

    nodes = wing_shell_mesh.nodal_coordinates
    connectivity = wing_shell_mesh.connectivity
    filename = mesh_fname+"_reconstructed.xdmf"
    if os.path.isfile(filename) and False:
        wing_shell_mesh_fenics = readFEAMesh(filename)
    else:
        # Reconstruct the mesh using the projected nodes and connectivity
        wing_shell_mesh_fenics = reconstructFEAMesh(filename, 
                                                    nodes.value, connectivity)
    # store the xdmf mesh object for shell analysis
    wing_shell_mesh.fea_mesh = wing_shell_mesh_fenics

    wing_shell_mesh_cd = cd.mesh.ShellMesh()
    wing_shell_mesh_cd.discretizations['wing'] = wing_shell_mesh
    mesh_container['shell_mesh'] = wing_shell_mesh_cd
    return skin_t_coeffs

def define_conditions(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    # Cruise
    pitch_angle = csdl.Variable(shape=(1, ), value=np.deg2rad(2.69268269))
    cruise = cd.aircraft.conditions.CruiseCondition(
        altitude=1,
        range=70 * cd.Units.length.kilometer_to_m,
        mach_number=0.18,
        pitch_angle=pitch_angle, # np.linspace(np.deg2rad(-4), np.deg2rad(10), 15),
    )
    cruise.quantities.pitch_angle = pitch_angle
    cruise.configuration = base_config.copy()
    conditions["cruise"] = cruise

def define_analysis(caddee: cd.CADDEE):
    conditions = caddee.conditions
    wing = caddee.base_configuration.system.comps["airframe"].comps["wing"]

    # finalize meshes
    cruise:cd.aircraft.conditions.CruiseCondition = conditions["cruise"]
    cruise.finalize_meshes()
    mesh_container = cruise.configuration.mesh_container

    # run VLM 

    # save the pressure history only for the time-dependent loads
    
    n_steps = int((T1+T0)/dt)
    if n_steps > Nsteps:
        n_steps = Nsteps
    pressure_history, implicit_disp_coeffs = run_vlm_history(mesh_container, cruise, Nsteps=n_steps, dt=dt)

    # Run structural analysis
    plate_sim, shell_outputs = run_dynamic_shell(mesh_container, cruise, pressure_history, 
                                                Nsteps=Nsteps, dt=dt, rec=True)
    # displacement, shell_outputs = run_dynamic_shell(mesh_container, cruise, pressure_history, 
    #                                             Nsteps=1, dt=dt, rec=True)
    # max_stress:csdl.Variable = shell_outputs.aggregated_stress
    total_strain_energy:csdl.Variable = shell_outputs.total_strain_energy
    wing_mass:csdl.Variable = shell_outputs.mass
    displacement = shell_outputs.disp_solid
    

    # [RX] this led to error: IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
    # mirror_function(displacement, wing.quantities.oml_lr_map)

    # max_stress.set_as_constraint(upper=stress_bound, scaler=1e-8)
    # max_stress.add_name('max_stress')

    wing_mass.set_as_constraint(upper=100, scaler=1e-2)
    wing_mass.add_name('wing_mass')

    # total_strain_energy.set_as_objective()
    total_strain_energy.set_as_objective(scaler=1e4)
    total_strain_energy.add_name('total_strain_energy')
    # wing_mass.set_as_objective(scaler=1e-2)
    # wing_mass.add_name('wing_mass')
    # wing.quantities.oml_displacement = displacement
    # wing.quantities.pressure_function = pressure_fn

    # Solver for aerostructural coupling
    # we iterate between the VLM and structural analysis until the displacement converges
    if couple:
        coeffs = displacement.stack_coefficients()
        disp_res = implicit_disp_coeffs[0] - coeffs
        solver = csdl.nonlinear_solvers.Jacobi(max_iter=10, tolerance=1e-6)
        solver.add_state(implicit_disp_coeffs[0], disp_res)
        solver.run()

    return plate_sim, shell_outputs

def run_vlm_history(mesh_container, condition, Nsteps=100, dt=0.01):

    wing = condition.configuration.system.comps["airframe"].comps["wing"]

    # Shell
    pav_shell_mesh = mesh_container["shell_mesh"]
    wing_shell_mesh = pav_shell_mesh.discretizations['wing']
    wing_shell_mesh_fenics = wing_shell_mesh.fea_mesh

    fenics_node_indices = wing_shell_mesh_fenics.geometry.input_global_indices    
    

    nodes = wing_shell_mesh.nodal_coordinates
    oml_node_inds = wing_shell_mesh.oml_node_inds
    oml_nodes_parametric = wing_shell_mesh.oml_nodes_parametric
    
    AoA = 0.  # Angle of Attack in degrees
    AoA_rad = np.deg2rad(AoA)  # Angle of Attack converted to radians

    V_x = V_inf*np.cos(AoA_rad) # chord direction (flight direction)
    V_y = 0. # span direction

    ti = 0.

    pressure_history = csdl.Variable(value=np.zeros((Nsteps+1, nodes.shape[1]*3)))
    for i in range(Nsteps+1):
        ti += dt
        V_z = V_inf*np.sin(AoA_rad)+V_g(ti) # vertical direction (gust direction)
        vlm_outputs, implicit_disp_coeffs = run_vlm([mesh_container], [condition], 
                                                    velocities=[V_x, V_y, V_z])
        forces = vlm_outputs.surface_force[0][0]
        Cp = vlm_outputs.surface_spanwise_Cp[0][0]
        
        # [RX] running into "TypeError: State must not be computed from another operation."
        # because the VLM now takes user-defined velocities directly as input
        # trim the aircraft
        if trim:
            print("Trimming...")
            print(condition.quantities.pitch_angle)
            pitch_angle = condition.quantities.pitch_angle
            z_force = -forces[2]*csdl.cos(pitch_angle) + forces[0]*csdl.sin(pitch_angle)
            residual = z_force - mass*9.81*load_factor
            trim_solver = csdl.nonlinear_solvers.BracketedSearch()
            trim_solver.add_state(pitch_angle, residual, (np.deg2rad(0), np.deg2rad(10)))
            with HiddenPrints():
                # The vlm solver prints stuff every time it's called and it annoys me
                trim_solver.run()

        # fit pressure function to trimmed VLM results
        # we can actually do this before the trim if we wanted, it would be updated automatically

        pressure_fn = fit_pressure_fn(mesh_container, condition, Cp)

        # transfer aero peressures
        pressure_magnitudes = pressure_fn.evaluate(oml_nodes_parametric)
        pressure_normals = wing.geometry.evaluate_normals(oml_nodes_parametric)
        oml_pressures = pressure_normals*csdl.expand(pressure_magnitudes, pressure_normals.shape, 'i->ij')

        shell_pressures = csdl.Variable(value=np.zeros(nodes.shape[1:]))

        shell_pressures = shell_pressures.set(csdl.slice[oml_node_inds], oml_pressures)

        shell_pressures_vector = csdl.reshape(shell_pressures[fenics_node_indices,:], 
                                 shape=(shell_pressures.shape[0]*shell_pressures.shape[1],))
        pressure_history = pressure_history.set(csdl.slice[i,:], shell_pressures_vector)
        print("Step", i, "of", Nsteps, "completed")
        print("V: ", V_x, V_y, V_z)

    return pressure_history, implicit_disp_coeffs

def run_vlm(mesh_containers, conditions, velocities=None):
    # implicit displacement input
    wing = conditions[0].configuration.system.comps["airframe"].comps["wing"]
    displacement_space:fs.FunctionSetSpace = wing.quantities.oml_displacement_space
    implicit_disp_coeffs = []
    implicit_disp_fns = []
    for i in range(len(mesh_containers)):
        coeffs, function = displacement_space.initialize_function(3, implicit=True)
        implicit_disp_coeffs.append(coeffs)
        implicit_disp_fns.append(function)

    # set up VLM analysis
    nodal_coords = []
    nodal_vels = []

    for mesh_container, condition, disp_fn in zip(mesh_containers, conditions, implicit_disp_fns):
        transfer_mesh_para = disp_fn.generate_parametric_grid((5, 5))
        transfer_mesh_phys = wing.geometry.evaluate(transfer_mesh_para)
        transfer_mesh_disp = disp_fn.evaluate(transfer_mesh_para)
        
        wing_lattice = mesh_container["vlm_mesh"].discretizations["wing_chord_surface"]
        wing_lattic_coords = wing_lattice.nodal_coordinates

        map = fs.NodalMap()
        weights = map.evaluate(csdl.reshape(wing_lattic_coords, (np.prod(wing_lattic_coords.shape[0:-1]), 3)), transfer_mesh_phys)
        wing_camber_mesh_displacement = (weights @ transfer_mesh_disp).reshape(wing_lattic_coords.shape)
        
        nodal_coords.append(wing_lattic_coords + wing_camber_mesh_displacement)
        if velocities is not None:
            nodal_velocities = wing_lattice.nodal_velocities
            nodal_velocities = nodal_velocities.set(csdl.slice[0,:,:,0], velocities[0])
            nodal_velocities = nodal_velocities.set(csdl.slice[0,:,:,1], velocities[1])
            nodal_velocities = nodal_velocities.set(csdl.slice[0,:,:,2], velocities[2])
            nodal_vels.append(nodal_velocities)
        else:
            nodal_vels.append(wing_lattice.nodal_velocities) # the velocities should be the same for every node in this case

    if len(nodal_coords) == 1:
        nodal_coordinates = nodal_coords[0]
        nodal_velocities = nodal_vels[0]
    else:
        nodal_coordinates = csdl.vstack(nodal_coords)
        nodal_velocities = csdl.vstack(nodal_vels)


    # Add an airfoil model
    nasa_langley_airfoil_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="ls417",
        aoa_range=np.linspace(-12, 16, 50), 
        reynolds_range=[1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 7e6, 10e6], 
        mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
        num_interp=120,
    )
    Cl_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cl"])
    Cd_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cd"])
    Cp_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cp"])
    alpha_stall_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["alpha_Cl_min_max"])



    vlm_outputs = vlm_solver(
        mesh_list=[nodal_coordinates],
        mesh_velocity_list=[nodal_velocities],
        atmos_states=conditions[0].quantities.atmos_states,
        airfoil_alpha_stall_models=[alpha_stall_model],
        airfoil_Cd_models=[Cd_model],
        airfoil_Cl_models=[Cl_model],
        airfoil_Cp_models=[Cp_model], 
    )

    return vlm_outputs, implicit_disp_coeffs

def fit_pressure_fn(mesh_container, condition, spanwise_Cp):
    wing = condition.configuration.system.comps["airframe"].comps["wing"]
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]
    rho = condition.quantities.atmos_states.density
    v_inf = condition.parameters.speed
    airfoil_upper_nodes = wing_lattice._airfoil_upper_para
    airfoil_lower_nodes = wing_lattice._airfoil_lower_para

    spanwise_p = spanwise_Cp * 0.5 * rho * v_inf**2
    spanwise_p = csdl.blockmat([[spanwise_p[:, 0:120].T()], [spanwise_p[:, 120:].T()]])

    pressure_indexed_space : fs.FunctionSetSpace = wing.quantities.pressure_space
    pressure_function = pressure_indexed_space.fit_function_set(
        values=spanwise_p.reshape((-1, 1)), parametric_coordinates=airfoil_upper_nodes+airfoil_lower_nodes,
        regularization_parameter=1e-4,
    )

    return pressure_function

def run_dynamic_shell(mesh_container, condition:cd.aircraft.conditions.CruiseCondition, pressure_history, Nsteps=100, dt=0.01, rec=False):
    wing = condition.configuration.system.comps["airframe"].comps["wing"]
    
    # Shell
    pav_shell_mesh = mesh_container["shell_mesh"]
    wing_shell_mesh = pav_shell_mesh.discretizations['wing']
    nodes = wing_shell_mesh.nodal_coordinates
    nodes_parametric = wing_shell_mesh.nodes_parametric
    connectivity = wing_shell_mesh.connectivity
    wing_shell_mesh_fenics = wing_shell_mesh.fea_mesh
    element_centers_parametric = wing_shell_mesh.element_centers_parametric
    oml_node_inds = wing_shell_mesh.oml_node_inds
    oml_nodes_parametric = wing_shell_mesh.oml_nodes_parametric
    node_disp = wing.geometry.evaluate(nodes_parametric) - nodes.reshape((-1,3))
    
    f_input_history = pressure_history

    material = wing.quantities.material_properties.material
    element_thicknesses = wing.quantities.material_properties.evaluate_thickness(element_centers_parametric)

    E0, nu0, G0 = material.from_compliance()
    density0 = material.density

    # create node-wise material properties
    nel = connectivity.shape[0]
    E = E0*np.ones(nel)
    E.add_name('E')
    nu = nu0*np.ones(nel)
    nu.add_name('nu')
    density = density0*np.ones(nel)
    density.add_name('density')

    # define boundary conditions
    def clamped_boundary(x):
        eps = 1e-3
        return np.less_equal(x[1], eps)


    plate_sim = PlateSim(wing_shell_mesh_fenics, 
                         custom_bc_func=clamped_boundary,
                         E=E0.value[0], nu=nu0.value[0], rho=density0.value[0], 
                         dt=dt, Nsteps=Nsteps, element_wise_thickness=element_wise_thickness)

    plate_sim.set_up_tip_dofs(x_tip=[-2.67539, 5.59659, -1.13752], cell_tip=386)

    # run solver
    state_operation = StateOperation(plate_sim=plate_sim, record=True, path='./records/')
    input_vars = csdl.VariableGroup()

    #:::::::::::::::::::::: Prepare the inputs :::::::::::::::::::::::::::::
    # sort the material properties based on FEniCS indices
    if element_wise_thickness:
        fenics_mesh_indices = wing_shell_mesh_fenics.topology.original_cell_index.tolist()
    else:
        fenics_mesh_indices = wing_shell_mesh_fenics.geometry.input_global_indices    
    
    input_vars.thickness = element_thicknesses[fenics_mesh_indices]

    input_vars.force_history = f_input_history
    disp_history = state_operation.evaluate(input_vars)
    disp_history.add_name('disp_history')

    total_strain_energy_operation = TotalStrainEnergyOperation(plate_sim=plate_sim)
    input_vars.disp_history = disp_history
    total_strain_energy = total_strain_energy_operation.evaluate(input_vars)
    total_strain_energy.add_name('total_strain_energy')


    volume_operation = VolumeOperation(plate_sim=plate_sim)
    volume = volume_operation.evaluate(input_vars)
    volume.add_name('volume')

    mass = volume*density0
    mass.add_name('mass')

    shell_outputs = csdl.VariableGroup()
    shell_outputs.thickness = input_vars.thickness
    shell_outputs.disp_solid = disp_history[-plate_sim.fe_dofs:]
    shell_outputs.total_strain_energy = total_strain_energy
    shell_outputs.mass = mass

    # # fit displacement function
    # oml_displacement_space:fs.FunctionSetSpace = wing.quantities.oml_displacement_space
    # oml_displacement_function = oml_displacement_space.fit_function_set(disp_history[-plate_sim.fe_dofs:], oml_nodes_parametric)
    
    '''
    4. post-processing
    '''
    if not optimize:
        # disp_solid = shell_outputs.disp_solid
        # # aggregated_stress = shell_outputs.aggregated_stress
        # mass = shell_outputs.mass
        # total_strain_energy = shell_outputs.total_strain_energy
        w = plate_sim.w
        u_mid = w.sub(0).collapse().x.array
        theta = w.sub(1).collapse().x.array

        print("Tip deflection:", max(abs(u_mid)))
        print("Total strain energy:", total_strain_energy.value)
        print("Mass:", mass.value)
        # print("Aggregated stress:", aggregated_stress.value)
        # print("Von Mises stress:", max(von_mises.value))

        print("  Number of elements = "+str(plate_sim.nel))
        print("  Number of vertices = "+str(plate_sim.nn))
        print("  Number of degrees of freedom = "+str(len(u_mid)))

    return plate_sim, shell_outputs

def process_elements(wing_shell_mesh, right_wing_oml, right_wing_ribs, right_wing_spars):
    """
    Process the elements of the shell mesh to determine the type of element (rib, spar, skin)
    """

    nodes = wing_shell_mesh.nodal_coordinates
    connectivity = wing_shell_mesh.connectivity

    # figure out type of surface each element is (rib/spar/skin)
    grid_n = 20
    oml_errors = np.linalg.norm(right_wing_oml.evaluate(right_wing_oml.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)
    rib_errors = np.linalg.norm(right_wing_ribs.evaluate(right_wing_ribs.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)
    spar_errors = np.linalg.norm(right_wing_spars.evaluate(right_wing_spars.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)

    element_centers = np.array([np.mean(nodes.value[connectivity[i].astype(int)], axis=0) for i in range(connectivity.shape[0])])

    rib_correction = 1e-4
    element_centers_parametric = []
    oml_inds = []
    rib_inds = []
    spar_inds = []
    for i in range(connectivity.shape[0]):
        inds = connectivity[i].astype(int)
        
        # rib projection is messed up so we use an alternitive approach - if all the points are in an x-z plane, it's a rib
        if np.all(np.isclose(nodes.value[inds, 1], nodes.value[inds[0], 1], atol=rib_correction)):
            rib_inds.append(i)
            continue

        errors = [np.sum(oml_errors[inds]), np.sum(rib_errors[inds]), np.sum(spar_errors[inds])]
        ind = np.argmin(errors)
        if ind == 0:
            oml_inds.append(i)
        elif ind == 1:
            rib_inds.append(i)
        elif ind == 2:
            spar_inds.append(i)
        else:
            raise ValueError('Error in determining element type')

    oml_centers = right_wing_oml.project(element_centers[oml_inds], grid_search_density_parameter=5, plot=False, force_reprojection=True)
    rib_centers = right_wing_ribs.project(element_centers[rib_inds], grid_search_density_parameter=5, plot=False, force_reprojection=True)
    spar_centers = right_wing_spars.project(element_centers[spar_inds], grid_search_density_parameter=5, plot=False, force_reprojection=True)
    oml_inds_copy = oml_inds.copy()

    for i in range(connectivity.shape[0]):
        if oml_inds and oml_inds[0] == i:
            element_centers_parametric.append(oml_centers.pop(0))
            oml_inds.pop(0)
        elif rib_inds and rib_inds[0] == i:
            element_centers_parametric.append(rib_centers.pop(0))
            rib_inds.pop(0)
        elif spar_inds and spar_inds[0] == i:
            element_centers_parametric.append(spar_centers.pop(0))
            spar_inds.pop(0)
        else:
            raise ValueError('Error in sorting element centers')
        
    wing_shell_mesh.element_centers_parametric = element_centers_parametric
    
    oml_node_inds = []
    for c_ind in oml_inds_copy:
        n_inds = connectivity[c_ind].astype(int)
        for n_ind in n_inds:
            if not n_ind in oml_node_inds:
                oml_node_inds.append(n_ind)

    oml_nodes_parametric = right_wing_oml.project(nodes.value[oml_node_inds], grid_search_density_parameter=5)
    wing_shell_mesh.oml_node_inds = oml_node_inds
    wing_shell_mesh.oml_nodes_parametric = oml_nodes_parametric
    wing_shell_mesh.oml_el_inds = oml_inds_copy

def mirror_function(displacement, oml_lr_map):
    for rind, lind in oml_lr_map.items():
        print(rind, lind)
        print(displacement.functions[rind].coefficients.shape)
        displacement.functions[lind].coefficients = displacement.functions[rind].coefficients[:,::-1,:]

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def load_dv_values(fname, group):
    inputs = csdl.inline_import(fname, group)
    recorder = csdl.get_current_recorder()
    dvs = recorder.design_variables
    for var in dvs:
        var_name = var.name
        var.set_value(inputs[var_name].value)
        scale = 1/np.linalg.norm(inputs[var_name].value)
        dvs[var] = (scale, dvs[var][1], dvs[var][2])

skin_t_coeffs = define_base_config(caddee)
rec.inline = inline
define_conditions(caddee)
plate_sim, shell_outputs = define_analysis(caddee)
csdl.save_optimization_variables()


fname = 'structural_opt_shell_test'

if check_derivatives:
    sim = csdl.experimental.PySimulator(rec)
    sim.check_totals([shell_outputs.total_strain_energy],[skin_t_coeffs])
    # sim.check_totals([aggregated_stress],[force_vector])


if optimize:
    from modopt import CSDLAlphaProblem
    from modopt import PySLSQP

    
    # If you have a GPU, you can set gpu=True - but it may not be faster
    # I think this is because the ml airfoil model can't be run on the GPU when Jax is using it
    # sim = csdl.experimental.JaxSimulator(rec, gpu=False, save_on_update=True, filename=fname)
    
    # If you don't have jax installed, you can use the PySimulator instead (it's slower)
    # To install jax, see https://jax.readthedocs.io/en/latest/installation.html
    sim = csdl.experimental.PySimulator(rec)

    # It's a good idea to check the totals of the simulator before running the optimizer
    # sim.check_totals()
    # exit()

    prob = CSDLAlphaProblem(problem_name=fname, simulator=sim)
    optimizer = PySLSQP(prob, solver_options={'maxiter':200, 'acc':1e-6})

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()
    csdl.inline_export(fname+'_final')


    # # Plotting
    # # load dv values and perform an inline execution to get the final results
    # load_dv_values(fname+'_final.hdf5', 'inline')
    # rec.execute()
    # wing = caddee.base_configuration.system.comps["airframe"].comps["wing"]
    # mesh = wing.quantities.oml.plot_but_good(color=wing.quantities.material_properties.thickness)
    # wing.quantities.oml.plot_but_good(color=wing.quantities.oml_displacement)
    # wing.quantities.oml.plot_but_good(color=wing.quantities.pressure_function)


# run plate_sim with optimized thickness and save dynamic displacements
plate_sim.reset_solution_vectors()
svk_res = plate_sim.SVK_residual()
plate_sim.solve_dynamic_problem(svk_res, saving_outputs=True, PATH='sim_results/')
# z axis is upside down in the aerodynamic model
plt.plot(np.linspace(0, T, Nsteps+1), -plate_sim.tip_disp_history)
plt.legend(["dt="+str(dt)])
plt.show()