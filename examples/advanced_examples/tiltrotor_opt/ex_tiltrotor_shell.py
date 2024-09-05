'''Example tiltrotor'''
import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs
import aeroelastic_coupling_utils as acu
import pickle

from femo_alpha.rm_shell.rm_shell_model import RMShellModel
import dolfinx
from utils import (
    load_thickness_vars, 
    construct_bay_condition, 
    construct_thickness_function,
    construct_plate_condition, 
    construct_plates, 
    load_structural_mesh_pre,
    load_structural_mesh_post,
)

import lsdo_function_spaces as fs
fs.num_workers = 1     # uncommont this if projections break

# Parameters
system_mass = 3617 # kg
g = 9.81 # m/s^2
max_stress = 350E6 # Pa
max_displacement = 0.55 # m
minimum_thickness = 0.0003 # m
initial_thickness = 0.01
stress_cf = 1.5


recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

caddee = cd.CADDEE()

make_meshes = True
run_ffd = False
run_optimization = False
shutdown_inline = False
do_trim = False

# Import tiltrotor .stp file and convert control points to meters
tiltrotor_geom = cd.import_geometry("nasa_multi_tiltrotor.stp", 
                                    scale=cd.Units.length.foot_to_m)

def define_base_config(caddee : cd.CADDEE):
    """Build the base configuration."""
    
    # system component & airframe
    aircraft = cd.aircraft.components.Aircraft(geometry=tiltrotor_geom, compute_surface_area=False)

    # Make (base) configuration object
    base_config = cd.Configuration(system=aircraft)

    # Airframe container object
    airframe = aircraft.comps["airframe"] = cd.Component()

    # ::::::::::::::::::::::::::: Make components :::::::::::::::::::::::::::
    # ---------- Main wing ----------
    wing_AR = csdl.Variable(name="wing_AR", shape=(1, ), value=8.90189)
    wing_S_ref = csdl.Variable(name="wing_S_ref", shape=(1, ), 
                               value=133.32710*cd.Units.length.foot_to_m) 
    # rib-like surfaces
    ignore_names = ['MainWing, 0, 19, lower, 66', 'MainWing, 0, 22, upper, 67', #left wing tip surfaces
                    'MainWing, 1, 1, lower, 68', 'MainWing, 1, 4, upper, 69', #left wing mid surfaces
                    'MainWing, 1, 19, lower, 74', 'MainWing, 1, 22, upper, 75', #right wing tip surfaces
                    'MainWing, 0, 1, lower, 60', 'MainWing, 0, 4, upper, 61'] #right wing mid surfaces
    wing_geometry = aircraft.create_subgeometry(search_names=["MainWing"], 
                                                ignore_names=ignore_names)
    wing = cd.aircraft.components.Wing(AR=wing_AR, S_ref=wing_S_ref, taper_ratio=1.,
                                       geometry=wing_geometry, tight_fit_ffd=True,
                                       compute_surface_area=False)
    # print(list(wing_geometry.function_names.values()))

    # wing.geometry.plot(opacity=0.3)
    # exit()


    # Make ribs and spars
    num_ribs = 9
    spanwise_multiplicity = 5
    top_array, bottom_array = wing.construct_ribs_and_spars(
        aircraft.geometry, 
        num_ribs=num_ribs, 
        plot_projections=False, 
        export_wing_box=False,  # Set to true to export the wing box as .igs for meshing
        export_half_wing=False,
        spanwise_multiplicity=spanwise_multiplicity, 
        LE_TE_interpolation="ellipse", 
        return_rib_points=True
    )

    # Thickness parametrization
    indices = np.array([i for i in range(0, top_array.shape[-1], spanwise_multiplicity)])
    top_array = top_array[:, indices]
    bottom_array = bottom_array[:, indices]
    
    wing_oml = wing.create_subgeometry(search_names=["MainWing"])
    left_wing_spars = wing.create_subgeometry(search_names=["spar"], ignore_names=['_r_', '-', 'MainWing, 1'])
    left_wing_ribs = wing.create_subgeometry(search_names=["rib"], ignore_names=['_r_', '-', 'MainWing, 1'])
    left_wing_oml = wing.create_subgeometry(search_names=["MainWing"], ignore_names=['_r_', '-', 'MainWing, 1'])
    wing.quantities.oml_geometry = wing_oml
    left_wing = wing.create_subgeometry(search_names=[""], ignore_names=["MainWing, 1", '_r_', '-'])
    wing.quantities.left_wing = left_wing
    wing.quantities.left_wing_oml = left_wing_oml
    front_spar = wing.create_subgeometry(search_names=["spar_0"], ignore_names=['_r_', '-', 'MainWing, 1'])
    rear_spar = wing.create_subgeometry(search_names=["spar_1"], ignore_names=['_r_', '-', 'MainWing, 1'])
    wing.quantities.rear_spar = rear_spar
    wing.quantities.front_spar = front_spar

    # wing_bays = construct_plates(num_ribs, top_array, bottom_array, offset=0)

    # left_wing_spars.plot(opacity=0.3)
    # left_wing_ribs.plot(opacity=0.3)
    # left_wing_oml.plot(opacity=0.3)
    # rear_spar.plot(opacity=0.3)
    # front_spar.plot(opacity=0.3)
    # exit()
    # Wing material
    E = csdl.Variable(value=69E9, name='E')
    G = csdl.Variable(value=26E9, name='G')
    density = csdl.Variable(value=2700, name='density')
    nu = csdl.Variable(value=0.33, name='nu')
    aluminum = cd.materials.IsotropicMaterial(name='aluminum', E=E, G=G, 
                                              density=density, nu=nu)
    skin_thickness = 0.01              # mm
    spar_thickness = 0.001             # mm
    rib_thickness = 0.0001             # mm

    # Construct thickness function
    t_vars = construct_thickness_function(wing, num_ribs, top_array, bottom_array, aluminum,
                                          skin_t=skin_thickness, spar_t=spar_thickness, rib_t=rib_thickness, 
                                          minimum_thickness=minimum_thickness)
    wing.quantities.material_properties.set_material(aluminum, thickness=None)

    # Pressure space
    pressure_function_space = lfs.IDWFunctionSpace(num_parametric_dimensions=2, 
                                                order=6, grid_size=(120, 20), 
                                                conserve=False, n_neighbors=10)
    indexed_pressue_function_space = wing.geometry.create_parallel_space(pressure_function_space)
    wing.quantities.pressure_space = indexed_pressue_function_space

    # Component hierarchy
    airframe.comps["wing"] = wing


    # ---------- Parent component for all rotors ----------
    rotors = cd.Component()
    airframe.comps["rotors"] = rotors
    
    # Pusher prop 
    left_proprotor_geometry = aircraft.create_subgeometry(
        search_names=["Pivot_1", "Hub_1", "Prop_1", "Nacelle_1"] 
    )
    right_proprotor_geometry = aircraft.create_subgeometry(
        search_names=["Pivot_4", "Hub_4", "Prop_4", "Nacelle_4"] 
    )
    proprotor_radius = csdl.Variable(name="proprotor_radius", shape=(1, ), value=9.18690/2*cd.Units.length.foot_to_m)
    left_proprotor = cd.aircraft.components.Rotor(radius=proprotor_radius, geometry=left_proprotor_geometry, compute_surface_area=False, skip_ffd=True)
    right_proprotor = cd.aircraft.components.Rotor(radius=proprotor_radius, geometry=right_proprotor_geometry, compute_surface_area=False, skip_ffd=True)
    
    axis_origin_right = csdl.Variable(value=cd.Units.length.foot_to_m*np.array([3.713,17.225,-2.261]))
    axis_origin_left = csdl.Variable(value=cd.Units.length.foot_to_m*np.array([3.713,-17.225,-2.261]))

    axis_vector = csdl.Variable(value=cd.Units.length.foot_to_m*np.array([3.713,0.,-2.261]))
    # Tilt the rotors to cruise condition
    left_proprotor.actuate(angle=3.1415/8,axis_origin=axis_origin_left, axis_vector=axis_vector)
    right_proprotor.actuate(angle=-3.1415/8,axis_origin=axis_origin_right, axis_vector=axis_vector)

    rotors.comps["left_proprotor"] = left_proprotor
    rotors.comps["right_proprotor"] = right_proprotor

    # ::::::::::::::::::::::::::: Make meshes :::::::::::::::::::::::::::
    if make_meshes:
        # wing shell mesh
        prefix = './data_files/'
        fname = prefix+'tiltrotor_left_wing_shell_mesh'
        shell_discritizaiton = load_structural_mesh_pre(left_wing, left_wing_oml, left_wing_ribs, left_wing_spars, 
                                                        rear_spar, wing_bays=None, boom_y_ranges=None, fname=fname)

        shell_mesh = load_structural_mesh_post(shell_discritizaiton, fname)

        # import vedo
        # nodes = shell_mesh.discretizations['wing'].nodal_coordinates
        # connectivity = shell_mesh.discretizations['wing'].connectivity
        # geo_plot = tiltrotor_geom.plot(opacity=0.2, show=False)
        # vedo_mesh = vedo.Mesh([nodes.value, connectivity]).wireframe()
        # vedo_mesh.color('blue')
        # plotter = vedo.Plotter()
        # plotter.show([geo_plot, vedo_mesh], axes=1, viewup='z')

        # exit()


        # Create panel meshes
        panel_meshes = cd.mesh.PanelMesh()
        # wing panel mesh
        wing_panel_surface = cd.mesh.make_wing_panel_mesh(
            wing, 26, 4, LE_interp="ellipse", TE_interp="ellipse", 
            spacing_spanwise="cosine", spacing_chordwise="cosine",
            ignore_camber=True, plot=False,
        )
        panel_meshes.discretizations["wing_panel_surface"] = wing_panel_surface

        # nacelle panel mesh
        left_nacelle_geometry = aircraft.create_subgeometry(search_names=["Nacelle_1"])
        right_nacelle_geometry = aircraft.create_subgeometry(search_names=["Nacelle_4"])
        left_nacelle_panel_surface = cd.mesh.make_nacelle_panel_mesh(
            left_nacelle_geometry,
            body_id = [1,2,3,4],
            tip_id = [5,6,7,8],
            grid_nr = 5,
            grid_nl_body = 20,
            grid_nl_tip = 3,
            plot=False,
        )
        right_nacelle_panel_surface = cd.mesh.make_nacelle_panel_mesh(
            right_nacelle_geometry,
            body_id = [5,6,7,8],
            tip_id = [1,2,3,4],
            grid_nr = 5,
            grid_nl_body = 20,
            grid_nl_tip = 3,
            plot=False,
        )
        panel_meshes.discretizations["left_nacelle_panel_surface"] = left_nacelle_panel_surface
        panel_meshes.discretizations["right_nacelle_panel_surface"] = right_nacelle_panel_surface


        # Proprotor meshes
        ignore_names_left_prop = ['76', '77', '78', '79', '80', '81', #blade 1
                        '82', '83', '84', '85', '86', '87', #blade 2
                        '88', '89', '90', '91', '92', '93', #blade 3
                        # '94', '95', '96', '97', '98', '99', #blade 4
                        '94', '95', '98', '99', #blade 4 rib like surfaces
                        '100', '101', '102', '103', '104', '105', #blade 5
                        '106', '107', '108', '109', '110', '111', #blade 6
                        '112', '113', '114', '115', '116', '117', #blade 7
                        # '118', '119', '120', '121', '122', '123', #blade 8
                        '118', '119', '122', '123'#blade 8 rib like surfaces
        ]
    
        ignore_names_right_prop = ['132', '133', '134', '135', '136', '137', #blade 1
                                   '138', '139', '140', '141', '142', '143',#blade 2
                                   '144', '145', '146', '147', '148', '149',#blade 3
                                #    '150', '151', '152', '153', '154', '155',#blade 4
                                    '150', '151', '154', '155',#blade 4 rib like surfaces
                                   '156', '157', '158', '159', '160', '161',#blade 5
                                   '162', '163', '164', '165', '166', '167',#blade 6
                                   '168', '169', '170', '171', '172', '173',#blade 7
                                #    '174', '175', '176', '177', '178', '179'#blade 8
                                    '174', '175', '178', '179' #blade 8 rib like surfaces
                                   ]
        left_prop_geometry = aircraft.create_subgeometry(
            search_names=["Prop_1"],
            ignore_names=ignore_names_left_prop
        )
        right_prop_geometry = aircraft.create_subgeometry(
            search_names=["Prop_4"],
            ignore_names=ignore_names_right_prop
        )

        left_prop_surface = cd.mesh.make_rotor_panel_mesh(
            left_prop_geometry,
            blade_keys = [96, 97, 120, 121], # upper and lower surfaces of 2 blades
            grid_nr = 5,
            grid_nl = 10,
            plot=False,
        )
        right_prop_surface = cd.mesh.make_rotor_panel_mesh(
            right_prop_geometry,
            blade_keys = [152, 153, 176, 177], # upper and lower surfaces of 2 blades
            grid_nr = 5,
            grid_nl = 10,
            plot=False,
        )
        panel_meshes.discretizations["left_proprotor_panel_surface"] = left_prop_surface
        panel_meshes.discretizations["right_proprotor_panel_surface"] = right_prop_surface

        # Store meshes
        mesh_container = base_config.mesh_container
        mesh_container["panel_meshes"] = panel_meshes
        mesh_container["shell_mesh"] = shell_mesh

        tiltrotor_geom.plot_meshes([left_nacelle_panel_surface.nodal_coordinates, 
                                    right_nacelle_panel_surface.nodal_coordinates,
                                    left_prop_surface.nodal_coordinates, 
                                    right_prop_surface.nodal_coordinates, 
                                    wing_panel_surface.nodal_coordinates, ])
        exit()

    caddee.base_configuration = base_config
    return


# TODO: 
def run_panel_method():
    '''
    Inputs: mesh_container
    Outputs: directed_pressure
    '''
    pass


def run_shell(mesh_container, condition:cd.aircraft.conditions.CruiseCondition, spanwise_Cp):
    wing = condition.configuration.system.comps["airframe"].comps["wing"]
    
    # set up VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]

    rho = condition.quantities.atmos_states.density
    v_inf = condition.parameters.speed

    spanwise_p = spanwise_Cp * 0.5 * rho * v_inf**2

    spanwise_p = csdl.blockmat([[spanwise_p[:, 0:120].T()], [spanwise_p[:, 120:].T()]])

    airfoil_upper_nodes = wing_lattice._airfoil_upper_para
    airfoil_lower_nodes = wing_lattice._airfoil_lower_para
    pressure_indexed_space : lfs.FunctionSetSpace = wing.quantities.pressure_space
    pressure_function = pressure_indexed_space.fit_function_set(
        values=spanwise_p.reshape((-1, 1)), parametric_coordinates=airfoil_upper_nodes+airfoil_lower_nodes,
        regularization_parameter=1e-4,
    )
    # wing.geometry.plot_but_good(color=pressure_function)

    # Shell
    pav_shell_mesh = mesh_container["shell_mesh"]
    wing_shell_mesh = pav_shell_mesh.discretizations['wing']
    nodes = wing_shell_mesh.nodal_coordinates
    nodes_parametric = wing_shell_mesh.nodes_parametric
    connectivity = wing_shell_mesh.connectivity
    wing_shell_mesh_fenics = wing_shell_mesh.fea_mesh
    element_centers_parametric = wing_shell_mesh.element_centers_parametric
    node_disp = wing.geometry.evaluate(nodes_parametric) - nodes.reshape((-1,3))

    mesh_tags = wing_shell_mesh.meshtags
    association_table = wing_shell_mesh.association_table
    
    shell_pressures = pressure_function.evaluate(nodes_parametric)


    normals = wing.geometry.evaluate_normals(nodes_parametric)
    directed_pressures = normals*csdl.expand(shell_pressures, normals.shape, 'i->ij')


    material = wing.quantities.material_properties.material
    # thickness = wing.quantities.material_properties.evaluate_thickness(nodes_parametric)
    element_thicknesses = wing.quantities.material_properties.evaluate_thickness(element_centers_parametric)

    
    E0, nu0, G0 = material.from_compliance()
    density0 = material.density

    # create node-wise material properties
    nn = wing_shell_mesh_fenics.geometry.x.shape[0]
    nel = connectivity.shape[0]
    E = E0*np.ones(nel)
    E.add_name('E')
    nu = nu0*np.ones(nel)
    nu.add_name('nu')
    density = density0*np.ones(nel)
    density.add_name('density')

    def clamped_boundary(x):
        eps = 1e-3
        return np.less_equal(x[1], eps)

    shell_model = RMShellModel(mesh=wing_shell_mesh_fenics,
                               mesh_tags=mesh_tags, association_table=association_table,
                               shell_bc_func=clamped_boundary,
                               element_wise_material=True,
                               PENALTY_BC = False,
                               record=True) # record=true doesn't work with 2 shell instances

    shell_outputs = shell_model.evaluate(directed_pressures, 
                                         element_thicknesses, E, nu, density,
                                         node_disp,
                                         debug_mode=False)
    disp_extracted = shell_outputs.disp_extracted
    disp_solid = shell_outputs.disp_solid
    stress:csdl.Variable = shell_outputs.stress
    print('max stress np', np.max(stress.value))
    max_stress:csdl.Variable = shell_outputs.aggregated_stress*stress_cf
    print("max stress", max_stress.value)
    print("max displacement", max(disp_solid.value))
    exit()
    print('Subdomain average stresses:')
    for _, subdomain in enumerate(association_table):
        i = association_table[subdomain]
        average_stress_i = getattr(shell_outputs, 'average_stress_'+str(i))
        print(subdomain, average_stress_i.value)

    tip_displacement:csdl.Variable = csdl.maximum(csdl.norm(disp_extracted, axes=(1,))) # assuming the tip displacement is the maximum displacement
    wing_mass:csdl.Variable = shell_outputs.mass

    displacement_space:lfs.FunctionSetSpace = wing.quantities.displacement_space
    displacement_function = displacement_space.fit_function_set(disp_extracted, nodes_parametric, 
                                                                regularization_parameter=1e-4)
    
    # # assign color map with specified opacities
    # import colorcet
    # import vedo
    # # https://colorcet.holoviz.org
    # mycmap = colorcet.bmy
    # alphas = np.linspace(0.8, 0.2, num=len(mycmap))

    # scals = np.linalg.norm(disp_extracted.value, axis=1)
    # man = vedo.Mesh([nodes.value[0,:,:], connectivity])
    # man.cmap(mycmap, scals, alpha=alphas).add_scalarbar()
    # plotter = vedo.Plotter()
    # plotter.show(man, __doc__, viewup="z", axes=7).close()

    # wing.geometry.plot_but_good(color=displacement_function, color_map='coolwarm')
    # wing.geometry.plot_but_good(color=displacement_function, color_map=mycmap)

    return displacement_function, pressure_function, max_stress, tip_displacement, wing_mass





define_base_config(caddee)
