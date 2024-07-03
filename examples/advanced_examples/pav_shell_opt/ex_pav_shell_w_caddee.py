'''PAV wing with geometry setup and aeroelastic analysis'''
'''
PAV wing shell optimization setup using the new FEMO-CSDL-CADDEE interface.
This example uses pre-built Reissner-Mindlin shell model in FEMO coupled
    with the VLM aerodynamic solver implemented in CSDL using SIFR.
    It contains material parametrization and shape parametrization 
    for different sets of design variables.

Author: Ru Xiang
Date: 2024-06-20
'''

import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as fs

from VortexAD.core.vlm.vlm_solver import vlm_solver
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from femo_alpha.rm_shell.rm_shell_model import RMShellModel
from femo_alpha.fea.utils_dolfinx import readFEAMesh, reconstructFEAMesh
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
from femo_alpha.rm_shell.rm_shell_model import RMShellModel
import lsdo_function_spaces as lfs
import aeroelastic_coupling_utils as acu
import vedo
import os

fs.num_workers = 1     # uncommont this if projections break
plot=False

recorder = csdl.Recorder(inline=True)
recorder.start()

caddee = cd.CADDEE()

run_optimization = False
run_sweep = False
run_check_derivatives = False

# Import and plot the geometry
pav_geometry = cd.import_geometry('pav.stp')


def define_base_config(caddee: cd.CADDEE):
    aircraft = cd.aircraft.components.Aircraft(geometry=pav_geometry)

    # wing
    # TODO: get rid of non-rib rib stuff in the original geometry maybe
    # wing_geometry = aircraft.create_subgeometry(search_names=["Wing"])
    cap_inds = [172, 173, 190, 191]
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing"],
                                                ignore_names=[str(i) for i in cap_inds]+[str(i+20) for i in cap_inds])
    AR = csdl.Variable(value=5, name='AR')
    S_ref = csdl.Variable(value=15, name='S_ref')
    span = csdl.Variable(value=8.4, name='span')
    taper_ratio = csdl.Variable(value=0.4, name='taper_ratio')
    wing = cd.aircraft.components.Wing(
        AR=AR, S_ref=None, span=span, geometry=wing_geometry, tight_fit_ffd=False,
    )

    rib_locations = np.array([0, 0.1143, 0.2115, 0.4944, 0.7772, 1])
    wing.construct_ribs_and_spars(aircraft.geometry, rib_locations=rib_locations)
    wing_oml = wing.create_subgeometry(search_names=["Wing"], ignore_names=["spar", "rib"])
    wing.quantities.oml_geometry = wing_oml
    # wing.geometry.plot(opacity=0.8)

    # wing material
    in2m = 0.0254
    E = csdl.Variable(value=69E9, name='E')
    G = csdl.Variable(value=26E9, name='G')
    density = csdl.Variable(value=2700, name='density')
    nu = csdl.Variable(value=0.33, name='nu')
    aluminum = cd.materials.IsotropicMaterial(name='aluminum', E=E, G=G, 
                                              density=density, nu=nu)

    # thickness function
    skin_thickness = 0.007 # 7mm
    spar_thickness = 0.015 # 15mm
    rib_thickness = 0.003 # 3mm

    thickness_fs = lfs.ConstantSpace(2)
    skin_fn = lfs.Function(thickness_fs, np.array([skin_thickness]))
    spar_fn = lfs.Function(thickness_fs, np.array([spar_thickness]))
    rib_fn = lfs.Function(thickness_fs, np.array([rib_thickness]))

    functions = {}
    for ind in wing.geometry.functions:
        name = wing.geometry.function_names[ind]
        if "spar" in name:
            functions[ind] = spar_fn
        elif "rib" in name:
            functions[ind] = rib_fn
        else:
            functions[ind] = skin_fn

    thickness_function_set = lfs.FunctionSet(functions)
    wing.quantities.material_properties.set_material(aluminum, thickness_function_set)
    displacement_space = lfs.IDWFunctionSpace(2, 2, grid_size=20, conserve=False)
    # displacement_space = fs.BSplineSpace(2, (1,1), (3,3))
    wing.quantities.displacement_space = wing_geometry.create_parallel_space(
                                                    displacement_space)

    pressure_function_space = lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=4, grid_size=(240, 40), conserve=False)
    # pressure_function_space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 3), coefficients_shape=(2, 10))
    indexed_pressue_function_space = wing.geometry.create_parallel_space(pressure_function_space)
    wing.quantities.pressure_space = indexed_pressue_function_space

    aircraft.comps["wing"] = wing

    wing.quantities.mass_properties.mass = 250
    wing.quantities.mass_properties.cg_vector = np.array([-3, 0., -1])

    # VLM mesh
    pav_vlm_mesh = cd.mesh.VLMMesh()
    # wing_camber_surface = cd.mesh.make_vlm_surface(
    #     wing, 30, 5, grid_search_density=20, ignore_camber=False,
    # )
    wing_chord_surface = cd.mesh.make_vlm_surface(
    wing, 40, 1, LE_interp="ellipse", TE_interp="ellipse", 
    spacing_spanwise="cosine", ignore_camber=True, plot=False,
    )
    wing_chord_surface.project_airfoil_points(oml_geometry=wing_oml)
    pav_vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface

    # pav_geometry.plot_meshes(wing_camber_surface.nodal_coordinates.value)

    meshname = 'pav_wing_6rib_caddee_mesh_2374_quad'
    original_filename = 'pav_wing/'+meshname+'.xdmf'
    wing_shell_discritization = cd.mesh.import_shell_mesh(
        original_filename, 
        wing,
        plot=plot,
        grid_search_n=20,
        rescale=[-1,1,-1]
    )
    nodes = wing_shell_discritization.nodal_coordinates
    connectivity = wing_shell_discritization.connectivity


    reconstructed_mesh_path = "stored_files/meshes/"
    filename = reconstructed_mesh_path+"reconstructed_"+meshname+".xdmf"
    if os.path.isfile(filename):
        wing_shell_mesh_fenics = readFEAMesh(filename)
    else:
        if not os.path.exists(reconstructed_mesh_path):
            os.makedirs(reconstructed_mesh_path)
        # Reconstruct the mesh using the projected nodes and connectivity
        wing_shell_mesh_fenics = reconstructFEAMesh(filename, 
                                                    nodes.value, connectivity)
    # store the xdmf mesh object for shell analysis
    wing_shell_discritization.fea_mesh = wing_shell_mesh_fenics

    pav_shell_mesh = cd.mesh.ShellMesh()
    pav_shell_mesh.discretizations['wing'] = wing_shell_discritization

    # # tail
    tail_geometry = aircraft.create_subgeometry(search_names=["Stabilizer"])
    tail = cd.aircraft.components.Wing(
        AR=5, S_ref=2, geometry=tail_geometry, tight_fit_ffd=False,
    )
    aircraft.comps["tail"] = tail

    # tail_camber_surface = cd.mesh.make_vlm_surface(
    #     tail, 30, 6, grid_search_density=10, ignore_camber=True,
    # )
    tail_surface = cd.mesh.make_vlm_surface(
            tail, 10, 1, ignore_camber=True
        )
    tail.quantities.mass_properties.mass = 250
    tail.quantities.mass_properties.cg_vector = np.array([-3, 0., -0])

    # pav_vlm_mesh.discretizations["wing_camber_surface"] = wing_camber_surface
    # pav_vlm_mesh.discretizations["tail_camber_surface"] = tail_camber_surface
    pav_vlm_mesh.discretizations["tail_chord_surface"] = tail_surface
    base_config = cd.Configuration(aircraft)
    mesh_container = base_config.mesh_container
    mesh_container["vlm_mesh"] = pav_vlm_mesh
    mesh_container["shell_mesh"] = pav_shell_mesh
    caddee.base_configuration = base_config

    # base_config.setup_geometry(plot=plot)



def define_conditions(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    pitch_angle = csdl.ImplicitVariable(shape=(1, ), value=np.deg2rad(2.))
    cruise = cd.aircraft.conditions.CruiseCondition(
        altitude=1e3,
        range=60e3,
        speed=50.,
        pitch_angle=pitch_angle,
    )
    cruise.configuration = base_config
    conditions["cruise"] = cruise
    return pitch_angle

def define_analysis(caddee: cd.CADDEE, pitch_angle=None):
    cruise = caddee.conditions["cruise"]
    cruise_config = cruise.configuration
    mesh_container = cruise_config.mesh_container
    
    aircraft = cruise_config.system
    tail = aircraft.comps["tail"]
    wing = aircraft.comps["wing"]
    elevator = csdl.ImplicitVariable(shape=(1, ), value=0.)
    tail.actuate(elevator)

    cruise.finalize_meshes()

    # prep transfer mesh for wing (SIFR)
    transfer_mesh_para = wing.geometry.generate_parametric_grid((20, 20))
    transfer_mesh_phys = wing.geometry.evaluate(transfer_mesh_para)

    # Set up VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]
    tail_lattice = vlm_mesh.discretizations["tail_chord_surface"]
    
    # Add an airfoil model
    nasa_langley_airfoil_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="ls417",
            aoa_range=np.linspace(-12, 16, 50), 
            reynolds_range=[1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 7e6, 10e6], 
            mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    Cl_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cl"])
    Cd_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cd"])
    Cp_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cp"])
    alpha_stall_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["alpha_Cl_min_max"])

    nodal_coordinates = [wing_lattice.nodal_coordinates, tail_lattice.nodal_coordinates]
    nodal_velocities = [wing_lattice.nodal_velocities, tail_lattice.nodal_velocities]

    vlm_outputs = vlm_solver(
        mesh_list=nodal_coordinates,
        mesh_velocity_list=nodal_velocities,
        atmos_states=cruise.quantities.atmos_states,
        airfoil_alpha_stall_models=[alpha_stall_model, None], #[alpha_stall_model, None],
        airfoil_Cd_models=[Cd_model, None], #[Cd_model, None],
        airfoil_Cl_models=[Cl_model, None],
        airfoil_Cp_models=[Cp_model, None], #[Cp_model, None],
    )

    vlm_forces = vlm_outputs.total_force
    vlm_moments = vlm_outputs.total_moment
    spanwise_Cp = vlm_outputs.surface_spanwise_Cp[0]

    # Compute pressures on the airfoil
    p0 = cruise.quantities.atmos_states.pressure
    rho = cruise.quantities.atmos_states.density
    v_inf = cruise.parameters.speed

    spanwise_p = spanwise_Cp * 0.5 * rho * v_inf**2

    spanwise_p = csdl.blockmat([[spanwise_p[0, :, 0:120].T()], [spanwise_p[0, :, 120:].T()]])

    airfoil_upper_nodes = wing_lattice._airfoil_upper_para
    airfoil_lower_nodes = wing_lattice._airfoil_lower_para



    pressure_indexed_space : lfs.FunctionSetSpace = wing.quantities.pressure_space
    pressure_function = pressure_indexed_space.fit_function_set(
        values=spanwise_p.reshape((-1, 1)), parametric_coordinates=airfoil_upper_nodes+airfoil_lower_nodes,
        regularization_parameter=1e-4,
    )


    # framework to shell (SIFR)
    pav_shell_mesh = mesh_container["shell_mesh"]
    wing_shell_mesh = pav_shell_mesh.discretizations['wing']
    nodes = wing_shell_mesh.nodal_coordinates
    nodes_parametric = wing_shell_mesh.nodes_parametric
    connectivity = wing_shell_mesh.connectivity
    wing_shell_mesh_fenics = wing_shell_mesh.fea_mesh

    new_nodes = wing.geometry.evaluate(nodes_parametric)
    node_disp = new_nodes - nodes.reshape((-1, 3))
    node_disp.add_name('node_disp')

    #####################################################################
    # TODO: make sure the force and pressure projection are correct
    # shell_forces = force_function.evaluate(nodes_parametric)
    shell_pressures = pressure_function.evaluate(nodes_parametric)
    normals = wing.geometry.evaluate_normals(nodes_parametric)
    directed_pressures = normals*csdl.expand(shell_pressures, normals.shape, 'i->ij')


    force_magnitudes, force_para_coords = pressure_function.integrate(wing.geometry, grid_n=50)
    force_magnitudes:csdl.Variable = force_magnitudes.flatten()
    force_coords = wing.geometry.evaluate(force_para_coords)
    force_normals = wing.geometry.evaluate_normals(force_para_coords)
    force_vectors = force_normals*csdl.expand(force_magnitudes, force_normals.shape, 'i->ij')

    mapper = acu.NodalMap()
    force_map = mapper.evaluate(force_coords, nodes.reshape((-1,3)))
    shell_forces = force_map.T() @ force_vectors
    #####################################################################



    # gather material info
    # TODO: make an evaluate that spits out a list of material and a variable 
    #       for thickness (for varying mat props)
    #       This works fine for a single material
    material = wing.quantities.material_properties.material
    thickness = wing.quantities.material_properties.evaluate_thickness(
                                                        nodes_parametric)

    # RM shell analysis

    # Currently only supports a single material
    E0, nu0, G0 = material.from_compliance()
    density0 = material.density

    # create node-wise material properties
    nn = wing_shell_mesh_fenics.geometry.x.shape[0]
    E = csdl.expand(E0, out_shape=(nn,))
    E.add_name('E')
    nu = csdl.expand(nu0, out_shape=(nn,))
    nu.add_name('nu')
    density = csdl.expand(density0, out_shape=(nn,))   
    density.add_name('density')

    # Define structure boundary conditions
    #### Fix all displacements and rotations on the root surface  ####
    DOLFIN_EPS = 3E-16
    y_root = -1E-6 # create margin to make sure the nodes on the boundary are fixed
    def ClampedBoundary(x):
        return np.greater(x[1], y_root+DOLFIN_EPS)
    
    shell_model = RMShellModel(mesh = wing_shell_mesh_fenics, 
                               shell_bc_func=ClampedBoundary,  # set up bc locations
                               record = True) # record flag for saving the structure outputs as # xdmf files
    
    ## Aerodynamic loads as directed nodal pressures
    # shell_outputs = shell_model.evaluate(directed_pressures, # force_vector
    #                         thickness, E, nu, density, # material properties
    #                         node_disp,                 # mesh deformation
    #                         debug_mode=False,
    #                         is_pressure=True)          # debug mode flag
    # Aerodynamic loads as directed nodal forces

    shell_outputs = shell_model.evaluate(shell_forces, # force_vector
                            thickness, E, nu, density, # material properties
                            node_disp,                 # mesh deformation
                            debug_mode=False,
                            is_pressure=False)          # debug mode flag

    # Demostrate all the shell outputs even if they might not be used
    disp_solid = shell_outputs.disp_solid # displacement on the shell mesh
    compliance = shell_outputs.compliance # compliance of the structure
    mass = shell_outputs.mass             # total mass of the structure
    wing_von_Mises_stress = shell_outputs.stress # von Mises stress on the shell
    wing_aggregated_stress = shell_outputs.aggregated_stress # aggregated stress
    disp_extracted = shell_outputs.disp_extracted # extracted displacement 
                                            # for deformation of the OML mesh
    
    # assign color map with specified opacities
    import colorcet
    # https://colorcet.holoviz.org
    mycmap = colorcet.bmy
    alphas = np.linspace(0.8, 0.2, num=len(mycmap))

    scals = np.linalg.norm(disp_extracted.value, axis=1)
    man = vedo.Mesh([nodes.value[0,:,:], connectivity])
    man.cmap(mycmap, scals, alpha=alphas).add_scalarbar()
    plotter = vedo.Plotter()
    plotter.show(man, __doc__, viewup="z", axes=7).close()


    displacement_space:lfs.FunctionSetSpace = wing.quantities.displacement_space
    displacement_function = displacement_space.fit_function_set(disp_extracted, nodes_parametric, 
                                                                regularization_parameter=1e-4)
    wing.geometry.plot_but_good(color=displacement_function)



    w = shell_model.fea.states_dict['disp_solid']['function']
    u_mid = w.sub(0).collapse().x.array
    theta = w.sub(1).collapse().x.array


    print("Wing tip deflection (m):",max(abs(u_mid)))
    print("Wing max rotation (rad):",max(abs(theta)))
    print("Extracted wing tip deflection (m):",max(abs(disp_extracted.value[:,2])))
    print("Wing total mass (kg):", mass.value)
    print("Wing Compliance (N*m):", compliance.value)
    print("Wing aggregated von Mises stress (Pascal):", wing_aggregated_stress.value)
    print("Wing maximum von Mises stress (Pascal):", max(wing_von_Mises_stress.value))
    print("number of elements:", shell_model.fea.nel)   
    print("number of vertices:", shell_model.fea.nn)


    # geo_plot = wing.geometry.plot(opacity=0.5, show=False)
    # geo_plot_2 = wing.geometry.plot(opacity=0.5, plot_types=['point_cloud'], 
    #                                 point_types=['coefficients'], show=False)
    # vedo_mesh_2 = vedo.Mesh([nodes.value[0,:,:], connectivity]).wireframe()
    # vedo_mesh_2.color('blue')
    # vedo_mesh_3 = vedo.Mesh([new_nodes.value, connectivity]).wireframe()
    # vedo_mesh_3.color('red')
    # plotter = vedo.Plotter()
    # plotter.show([geo_plot, geo_plot_2, vedo_mesh_2, vedo_mesh_3], axes=1, viewup='z')


    AR = wing.parameters.AR
    span = wing.parameters.span
    S_ref = wing.parameters.S_ref
    print("Wing AR:", AR.value)
    print("Wing span:", span.value)
    print("Wing S_ref:", S_ref.value)


    sp_vars = [AR, span, S_ref]
    mesh_vars = [new_nodes, node_disp, connectivity]
    shell_vars = [thickness, disp_extracted, mass, wing_aggregated_stress]
    return sp_vars, mesh_vars, shell_vars

define_base_config(caddee)

pitch_angle = define_conditions(caddee)

sp_vars, mesh_vars, shell_vars = define_analysis(caddee, pitch_angle)
AR, span, S_ref = sp_vars
new_nodes, node_disp, connectivity = mesh_vars
thickness, disp_extracted, mass, wing_aggregated_stress = shell_vars

if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import SLSQP
    AR.set_as_design_variable(upper=10, lower=3)
    mass.set_as_constraint(upper=50.)
    wing_aggregated_stress.set_as_objective(scaler=1E-6)
    sim = csdl.experimental.PySimulator(recorder)

    prob = CSDLAlphaProblem(problem_name='PAV_structure', simulator=sim)

    optimizer = SLSQP(prob, ftol=1e-8, maxiter=5, outputs=['AR'])

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()

    print("Wing tip deflection (m):", max(abs(disp_extracted.value[:,2])))
    print("Wing total mass (kg):", mass.value)
    print("Wing aggregated von Mises stress (Pascal):", wing_aggregated_stress.value)

if run_sweep:
    AR_values = np.linspace(3, 10, 2)
    # AR_values = np.array([5.])
    disp_values = np.zeros(len(AR_values))
    mass_values = np.zeros(len(AR_values))
    stress_values = np.zeros(len(AR_values))
    
    new_nodes_value_0 = new_nodes.value
    for i in range(len(AR_values)):
        AR.value = AR_values[i]
        recorder.execute()

        disp_values[i] = max(abs(disp_extracted.value[:,2]))
        mass_values[i] = mass.value
        stress_values[i] = wing_aggregated_stress.value

        print("Wing AR:", AR.value)
        print("Wing span:", span.value)
        print("Wing S_ref:", S_ref.value)

        print("Wing tip deflection (m):", max(abs(disp_extracted.value[:,2])))
        print("Wing total mass (kg):", mass.value)
        print("Wing aggregated von Mises stress (Pascal):", wing_aggregated_stress.value)

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

    axs[0].plot(AR_values, disp_values)
    axs[0].set_ylabel('Wing tip deflection (m)')

    axs[1].plot(AR_values, mass_values)
    axs[1].set_ylabel('Wing total mass (kg)')

    axs[2].plot(AR_values, stress_values)
    axs[2].set_ylabel('Wing aggregated von Mises stress (Pascal)')
    axs[2].set_xlabel('AR')

    plt.show()


if run_check_derivatives:
    from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
    print("Verifying derivatives ...")
    # verify_derivatives_inline([wing_aggregated_stress], 
    #                             [AR], 
    #                             step_size=1e1, raise_on_error=False)
    # verify_derivatives_inline([wing_aggregated_stress], 
    #                             [span], 
    #                             step_size=1e-9, raise_on_error=False)

    verify_derivatives_inline([node_disp], 
                                [AR], 
                                step_size=1e-1, raise_on_error=False)


recorder.stop()