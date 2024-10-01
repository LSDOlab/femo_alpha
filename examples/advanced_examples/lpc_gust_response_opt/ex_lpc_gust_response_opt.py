import csdl_alpha as csdl
import CADDEE_alpha as cd
from CADDEE_alpha import functions as fs
import numpy as np
import os
import sys
from VortexAD.core.vlm.vlm_solver import vlm_solver
import aframe as af

import aeroelastic_coupling_utils as acu
import matplotlib.pyplot as plt

# lsdo_airfoil must be cloned and installed from https://github.com/LSDOlab/lsdo_airfoil
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
import lsdo_function_spaces as lfs
from timeit import default_timer as timer

from utils import (
    load_thickness_vars, 
    construct_bay_condition, 
    construct_thickness_function,
    construct_plate_condition, 
    construct_plates, 
    compute_buckling_loads, 
    load_dv_values,
    compute_curved_buckling_loads,
    load_structural_mesh_pre,
    load_structural_mesh_post,
)
# ------------------------------------------------------------------
# ------------------------- LPC parameters -------------------------
# ------------------------------------------------------------------
# Parameters
system_mass = 3754.035713          # kg
total_boom_mass = 140.9903153      # kg
battery_mass = 953.931603          # kg
motor1 = 61.21710382               # kg
motor2 = 35.38429884               # kg
motor3 = 13.00147938               # kg
motor4 = 11.08264674               # kg
boom_masses = [total_boom_mass/8+battery_mass/8+motor1, # f inner
               total_boom_mass/8+battery_mass/8+motor2, # r inner
               total_boom_mass/8+battery_mass/8+motor3, # f outer
               total_boom_mass/8+battery_mass/8+motor4] # r outer
                # kg - just the right wing booms
g = 9.81                           # m/s^2
max_stress = 350E6                 # Pa
max_displacement = 0.55            # m
minimum_thickness = 0.0003         # m
skin_thickness = 0.01              # mm
spar_thickness = 0.001             # mm
rib_thickness = 0.0001             # mm
stress_cf = 1.5 # corrects softmax stress to actual stress



global_start = timer()

# Settings
couple = False
optimize = True
check_derivatives = False
inline = True
ML = False
trim = False
element_wise_thickness = True
add_booms = False
#=============== without booms==================
# Dynamic simulation wall time: 211.1955498870011
# Tip deflection: 0.015926600973292137
# Total strain energy: [12078.485461]
# Mass: [418.59704621]
#   Number of elements = 2468
#   Number of vertices = 2291
#   Number of degrees of freedom = 28503
# Total run time:  289.36618438700134
#=============== with booms==================
# Dynamic simulation wall time: 228.00001642899952
# Tip deflection: 0.01385799468203417
# Total strain energy: [6294.69337779]
# Mass: [418.59704621]
#   Number of elements = 2468
#   Number of vertices = 2291
#   Number of degrees of freedom = 28503
# Total run time:  250.6040047919996

# Optimization settings
max_iter = 100

jax_sim = True
minimize_mass = False
mesh_id = 0
# mass_0 = 298.93714513 # unit: kg
mass_0 = 418.59704621
max_mass = mass_0
strain_energy_scaler = 1e-3
mass_scaler = 1e-2
# ------------------------------------------------------------------
# ---------------------- Time stepping setting ---------------------
# ------------------------------------------------------------------

# Time stepping parameters
dt = 0.005
T = 0.2
Nsteps = int(T/dt) # 40
Nsteps = 40

# ------------------------------------------------------------------
# ------------------------- gust setting ---------------------------
# ------------------------------------------------------------------
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

lpc_wing = [#### quad mesh ####
        "left_wing_c_v2", # 2468 elements
        "left_wing_m_v2", # 9872 elements
        "left_wing_f_v2", # 39488 elements
        ]


file_path = "./data_files/"

x_tip = [3.76643, 7.69544, -2.32919]
if mesh_id == 0:
    cell_tip = 2104 # ndof: 28503
    max_strain_energy = 4752.49776969
elif mesh_id == 1:
    cell_tip = 8416 
    max_strain_energy = 6680.92315078
elif mesh_id == 2:
    cell_tip = 33664

mesh_fname = file_path+lpc_wing[mesh_id]
mass = 1000
stress_bound = 1e8
load_factor = 3

num_ribs = 9
spanwise_multiplicity = 30

# Start recording
rec = csdl.Recorder(inline=inline, debug=True)
rec.start()

# Initialize CADDEE and import geometry
caddee = cd.CADDEE()
lpc_geom = cd.import_geometry("LPC_final_custom_blades.stp", scale=cd.Units.length.foot_to_m)

def define_base_config(caddee : cd.CADDEE):
    aircraft = cd.aircraft.components.Aircraft(geometry=lpc_geom)
    base_config = cd.Configuration(system=aircraft)
    caddee.base_configuration = base_config

    airframe = aircraft.comps["airframe"] = cd.Component()

    # Setup wing component
    top_surface_inds = [75, 79, 83, 87]
    ignore_names = ['72', '73', '90', '91', '92', '93', '110', '111'] # rib-like surfaces
    ignore_names += [str(i-1) for i in top_surface_inds] + [str(i+2) for i in top_surface_inds]
    ignore_names += [str(i-1+20) for i in top_surface_inds] + [str(i+2+20) for i in top_surface_inds]
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing_1"], ignore_names=ignore_names)
    wing = cd.aircraft.components.Wing(AR=12.12, S_ref=19.6, taper_ratio=0.2,
                                       geometry=wing_geometry, 
                                       tight_fit_ffd=True,
                                       compute_surface_area=False
                                    )
    
    # Generate internal geometry
    top_array, bottom_array = wing.construct_ribs_and_spars(
        lpc_geom,
        num_ribs=num_ribs,
        LE_TE_interpolation="ellipse",
        plot_projections=False, 
        export_wing_box=False,  # Set to true to export the wing box as .igs for meshing
        export_half_wing=False,
        full_length_ribs=True,
        spanwise_multiplicity=spanwise_multiplicity,
        num_rib_pts=30,
        offset=np.array([0.,0.,.15]),
        finite_te=True,
        exclute_te=True,
        return_rib_points=True,
        spar_function_space=lfs.BSplineSpace(2, (5, 1), (100, 2)),
    )

    indices = np.array([i for i in range(0, top_array.shape[-1], spanwise_multiplicity)])
    top_array = top_array[:, indices]
    bottom_array = bottom_array[:, indices]

    # wing.plot()
    # exit()


    # material
    E = csdl.Variable(value=69E9, name='E')
    G = csdl.Variable(value=26E9, name='G')
    density = csdl.Variable(value=2700, name='density')
    nu = csdl.Variable(value=0.33, name='nu')
    aluminum = cd.materials.IsotropicMaterial(name='aluminum', E=E, G=G, 
                                                density=density, nu=nu)

    t_vars = construct_thickness_function(wing, num_ribs, top_array, bottom_array, aluminum,
                                          skin_t=skin_thickness, spar_t=spar_thickness, rib_t=rib_thickness, 
                                          minimum_thickness=minimum_thickness)
    # Booms
    booms = cd.Component() # Create a parent component for all the booms
    airframe.comps["booms"] = booms
    boom_points = []
    for i in range(4,8):
        boom_geometry = aircraft.create_subgeometry(search_names=[
            f"Rotor_{i+1}_Support",
        ])
        boom = cd.Component(geometry=boom_geometry)
        boom.quantities.drag_parameters.characteristic_length = 2.4384
        boom.quantities.drag_parameters.form_factor = 1.1
        booms.comps[f"boom_{i+1}"] = boom
        if i%2 == 0:
            points = boom._ffd_block.evaluate(parametric_coordinates=np.array([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.],[0.,1.,1.]]), non_csdl=True)
        else:
            points = boom._ffd_block.evaluate(parametric_coordinates=np.array([[1.,0.,0.],[1.,0.,1.],[1.,1.,0.],[1.,1.,1.]]), non_csdl=True)
        boom_points.append(points)

    boom_y_ranges = []
    for i in range(2):
        points_0 = boom_points[2*i]
        points_1 = boom_points[2*i+1]
        min_y = min(np.min(points_0[:,1]), np.min(points_1[:,1]))
        max_y = max(np.max(points_0[:,1]), np.max(points_1[:,1]))
        boom_y_ranges.append([min_y, max_y])



    # Set material
    wing.quantities.material_properties.set_material(aluminum, None)

    wing_oml = wing.create_subgeometry(search_names=["Wing_1"])
    left_wing_spars = wing.create_subgeometry(search_names=["spar"], ignore_names=['_r_', '-', 'Wing_1, 1'])
    left_wing_ribs = wing.create_subgeometry(search_names=["rib"], ignore_names=['_r_', '-', 'Wing_1, 1'])
    left_wing_oml = wing.create_subgeometry(search_names=["Wing_1"], ignore_names=['_r_', '-', 'Wing_1, 1'])
    wing.quantities.oml_geometry = wing_oml
    left_wing = wing.create_subgeometry(search_names=[""], ignore_names=["Wing_1, 1", '_r_', '-'])
    wing.quantities.left_wing = left_wing
    wing.quantities.left_wing_oml = left_wing_oml
    rear_spar = wing.create_subgeometry(search_names=["spar_2"], ignore_names=['_r_', '-', 'Wing_1, 1'])
    wing.quantities.rear_spar = rear_spar
    wing_bays = construct_plates(num_ribs, top_array, bottom_array, offset=0)



    # shell meshing
    # import shell mesh
    shell_discritizaiton = load_structural_mesh_pre(left_wing, left_wing_oml, left_wing_ribs, left_wing_spars, 
                                                    rear_spar, wing_bays, boom_y_ranges, mesh_fname)
    lpc_shell_mesh = load_structural_mesh_post(shell_discritizaiton, mesh_fname)



    # Spaces for states
    # pressure
    pressure_function_space = lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=4, grid_size=(240, 40), conserve=False, n_neighbors=10)
    # pressure_function_space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 3), coefficients_shape=(2, 10))
    indexed_pressue_function_space = wing.geometry.create_parallel_space(pressure_function_space)
    wing.quantities.pressure_space = indexed_pressue_function_space

    # displacement
    displacement_space = lfs.IDWFunctionSpace(2, 5, grid_size=100, conserve=False)
    wing.quantities.displacement_space = wing_geometry.create_parallel_space(
                                                    displacement_space)

    airframe.comps["wing"] = wing

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
    mesh_container['shell_mesh'] = lpc_shell_mesh
    return t_vars

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
    wing_shell_mesh = caddee.base_configuration.mesh_container["shell_mesh"].discretizations['wing']
    wing_shell_mesh_fenics = wing_shell_mesh.fea_mesh
    nodes = wing_shell_mesh.nodal_coordinates
    # finalize meshes
    cruise:cd.aircraft.conditions.CruiseCondition = conditions["cruise"]
    cruise.finalize_meshes()
    mesh_container = cruise.configuration.mesh_container



    # save the pressure history to a file
    pressure_history_filename = mesh_fname+'_v_p_'+str(V_p)+'_dt_'+str(dt)+'_pressure_history'+'.npy'
    if os.path.isfile(pressure_history_filename):
        print("Loading pressure history from file ...")
        pressure_history_array = np.load(pressure_history_filename)
        pressure_history = csdl.Variable(value=pressure_history_array)

    else:
        # run VLM 
        # save the pressure history only for the time-dependent loads
        n_steps = int((T1+T0)/dt)
        if n_steps > Nsteps:
            n_steps = Nsteps
        pressure_history = run_vlm_history(mesh_container, cruise, Nsteps=n_steps, dt=dt)
        pressure_history_array = pressure_history.value
        np.save(pressure_history_filename, pressure_history_array)

    # Run structural analysis
    plate_sim, shell_outputs = run_dynamic_shell(mesh_container, cruise, pressure_history, 
                                                Nsteps=Nsteps, dt=dt, rec=True, add_booms=add_booms)
    # max_stress:csdl.Variable = shell_outputs.aggregated_stress
    total_strain_energy:csdl.Variable = shell_outputs.total_strain_energy
    wing_mass:csdl.Variable = shell_outputs.mass
    displacement = shell_outputs.disp_solid
    

    if minimize_mass:
        wing_mass.set_as_objective(scaler=mass_scaler)
        wing_mass.add_name('wing_mass')
        total_strain_energy.set_as_constraint(upper=max_strain_energy, scaler=strain_energy_scaler)
        total_strain_energy.add_name('total_strain_energy')
    else: 
        total_strain_energy.set_as_objective(scaler=strain_energy_scaler)
        total_strain_energy.add_name('total_strain_energy')
        wing_mass.set_as_constraint(upper=max_mass, scaler=mass_scaler)
        wing_mass.add_name('wing_mass')

    # wing.quantities.oml_displacement = displacement
    # wing.quantities.pressure_function = pressure_fn

    # # Solver for aerostructural coupling
    # # we iterate between the VLM and structural analysis until the displacement converges
    # if couple:
    #     coeffs = displacement.stack_coefficients()
    #     disp_res = implicit_disp_coeffs[0] - coeffs
    #     solver = csdl.nonlinear_solvers.Jacobi(max_iter=10, tolerance=1e-6)
    #     solver.add_state(implicit_disp_coeffs[0], disp_res)
    #     solver.run()

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
        vlm_outputs = run_vlm([mesh_container], [condition], 
                                                    velocities=[V_x, V_y, V_z])
        forces = vlm_outputs.surface_force[0][0]
        Cp = vlm_outputs.surface_spanwise_Cp[0][0]
        

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
        # print(pressure_history.value[i,:10])
    

    return pressure_history




def run_vlm(mesh_containers, conditions, velocities=None):

    # set up VLM analysis
    nodal_coords = []
    nodal_vels = []

    for mesh_container, condition in zip(mesh_containers, conditions):

        wing_lattice = mesh_container["vlm_mesh"].discretizations["wing_chord_surface"]
        nodal_coords.append(wing_lattice.nodal_coordinates)
        # nodal_vels.append(wing_lattice.nodal_velocities)

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

    return vlm_outputs

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

def run_dynamic_shell(mesh_container, condition:cd.aircraft.conditions.CruiseCondition, 
                    pressure_history, Nsteps=100, dt=0.01, rec=False, add_booms=True, ):
    wing = condition.configuration.system.comps["airframe"].comps["wing"]
    
    # Shell
    pav_shell_mesh = mesh_container["shell_mesh"]
    wing_shell_mesh = pav_shell_mesh.discretizations['wing']
    connectivity = wing_shell_mesh.connectivity
    wing_shell_mesh_fenics = wing_shell_mesh.fea_mesh
    element_centers_parametric = wing_shell_mesh.element_centers_parametric
    nodes = wing_shell_mesh.nodal_coordinates

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

    plate_sim.set_up_tip_dofs(x_tip=x_tip, cell_tip=cell_tip)


    if add_booms:

        # save the pressure history to a file
        pressure_history_filename = mesh_fname+'_v_p_'+str(V_p)+'_dt_'+str(dt)+'_pressure_history_w_booms'+'.npy'
        if os.path.isfile(pressure_history_filename):
            print("Loading pressure history from file with booms...")
            pressure_history_array = np.load(pressure_history_filename)
            pressure_history = csdl.Variable(value=pressure_history_array)

        else:

            # apply weights from booms
            g_factor = 1
            boom_node_inds = wing_shell_mesh.boom_node_inds
            boom_0_force = (boom_masses[0]+boom_masses[1])*9.81*g_factor    # positive is down
            boom_1_force = (boom_masses[2]+boom_masses[3])*9.81*g_factor
            boom_0_node_forces = boom_0_force/len(boom_node_inds[0])
            boom_1_node_forces = boom_1_force/len(boom_node_inds[1])

            force_boom = csdl.Variable(value=np.zeros((nodes.shape[1], 3)))

            force_boom = force_boom.set(csdl.slice[boom_node_inds[0], 2], boom_0_node_forces)
            force_boom = force_boom.set(csdl.slice[boom_node_inds[1], 2], boom_1_node_forces)


            # # verify boom forces
            # print("Verifying boom forces ...")
            # print(np.sum(force_history[0,:,:].value, axis=0))
            # print("Boom 0 force:", boom_0_force)
            # print("Boom 1 force:", boom_1_force)

            # Compute nodal pressures based on forces
            A = plate_sim.construct_force_to_pressure_map()

            fenics_mesh_indices = wing_shell_mesh_fenics.geometry.input_global_indices
            reshaped_force_boom = csdl.reshape(force_boom[fenics_mesh_indices,:], shape=(A.shape[1],))
            pressure_boom = csdl.solve_linear(A.toarray(), reshaped_force_boom)
            # f_func = dolfinx.fem.Function(plate_sim.W_f)
            # f_func.x.array[:] = reshaped_force_boom.value
            # p_func = dolfinx.fem.Function(plate_sim.W_f)
            # p_func.x.array[:] = pressure_boom.value
            # xdmf_file = dolfinx.io.XDMFFile(plate_sim.comm, "sim_c_mesh/boom_forces.xdmf", "w")
            # xdmf_file.write_mesh(plate_sim.mesh)
            # xdmf_file.write_function(f_func)
            # xdmf_file_p = dolfinx.io.XDMFFile(plate_sim.comm, "sim_c_mesh/boom_pressures.xdmf", "w")
            # xdmf_file_p.write_mesh(plate_sim.mesh)
            # xdmf_file_p.write_function(p_func)
            # exit()
            for i in range(pressure_history.shape[0]):
                pressure_history = pressure_history.set(csdl.slice[i,:], pressure_history[i,:]+pressure_boom)


            pressure_history_array = pressure_history.value
            np.save(pressure_history_filename, pressure_history_array)

    f_input_history = pressure_history


    # run solver
    state_operation = StateOperation(plate_sim=plate_sim, gradient_mode='petsc',
                                     record=rec, path='./solutions/')
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


fname = 'gust_structural_opt'

if check_derivatives:
    sim = csdl.experimental.PySimulator(rec)
    sim.check_totals([shell_outputs.total_strain_energy],[skin_t_coeffs])
    # sim.check_totals([aggregated_stress],[force_vector])


if optimize:
    from modopt import CSDLAlphaProblem
    from modopt import PySLSQP, SNOPT

    if jax_sim:
        # If you have a GPU, you can set gpu=True - but it may not be faster
        # I think this is because the ml airfoil model can't be run on the GPU when Jax is using it
        sim = csdl.experimental.JaxSimulator(rec, gpu=False, save_on_update=True, filename=fname)
        
        # If you don't have jax installed, you can use the PySimulator instead (it's slower)
        # To install jax, see https://jax.readthedocs.io/en/latest/installation.html

    else:
        sim = csdl.experimental.PySimulator(rec)
    prob = CSDLAlphaProblem(problem_name=fname, simulator=sim)

    # SLSQP
    optimizer = PySLSQP(prob, solver_options={'maxiter':max_iter, 'acc':1e-3})

    # # SNOPT
    # snopt_options = {
    #             'Major iterations': 1,
    #             'Major optimality': 1e-9,
    #             'Major feasibility': 1e-8
    #             }
    # optimizer = SNOPT(prob, solver_options=snopt_options)

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()
    csdl.inline_export(fname+'_final')


    # # Plotting
    # # load dv values and perform an inline execution to get the final results
    load_dv_values(fname+'_final.hdf5', 'inline')
    # rec.execute()
    # wing = caddee.base_configuration.system.comps["airframe"].comps["wing"]
    # mesh = wing.quantities.oml.plot_but_good(color=wing.quantities.material_properties.thickness)
    # wing.quantities.oml.plot_but_good(color=wing.quantities.oml_displacement)
    # wing.quantities.oml.plot_but_good(color=wing.quantities.pressure_function)


# run plate_sim with optimized thickness and save dynamic displacements
# plate_sim.reset_solution_vectors()
# svk_res = plate_sim.SVK_residual()
# plate_sim.solve_dynamic_problem(svk_res, saving_outputs=True, PATH='sim_results/')


global_end = timer()
print("Total run time: ", global_end-global_start)