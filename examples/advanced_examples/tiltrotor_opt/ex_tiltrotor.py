'''Example tiltrotor'''
import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs
import aeroelastic_coupling_utils as acu
import pickle

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
do_structural = False
do_trim = False
plot_meshes = True

if do_structural:
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

    if do_structural:
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
        if do_structural:
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
        left_prop_geometry = aircraft.create_subgeometry(
            search_names=["Prop_1"],
        )
        right_prop_geometry = aircraft.create_subgeometry(
            search_names=["Prop_4"],
        )

        left_prop_blade_3_geometry = aircraft.create_subgeometry(
            search_names=["Prop_1, 3, 7, lower, 96", "Prop_1, 3, 10, upper, 97"],
        )
        left_prop_blade_7_geometry = aircraft.create_subgeometry(
            search_names=["Prop_1, 7, 7, lower, 120", "Prop_1, 7, 10, upper, 121"],
        )
        right_prop_blade_3_geometry = aircraft.create_subgeometry(
            search_names=["Prop_4, 3, 7, lower, 152", "Prop_4, 3, 10, upper, 153"],
        )
        right_prop_blade_7_geometry = aircraft.create_subgeometry(
            search_names=["Prop_4, 7, 7, lower, 176", "Prop_4, 7, 10, upper, 177"],
        )

        # TODO: (alternative rotor mesher) fix the spacing of the blade surface mesh to be linspace
        # left_prop_blade_3_surface = cd.mesh.make_blade_panel_mesh(
        #     left_prop_blade_3_geometry, 10, 5, LE_interp="ellipse", TE_interp="ellipse", 
        #     spacing_spanwise="linear", spacing_chordwise="linear",
        #     surface_id = [96, 97],
        #     ignore_camber=True, plot=True,
        # )


        left_prop_blade_3_surface = cd.mesh.make_rotor_panel_mesh(
            left_prop_geometry,
            blade_keys = [96, 97], # upper and lower surfaces
            grid_nr = 5,
            grid_nl = 10,
            plot=False,
        )
        # left_prop_geometry.plot_meshes(left_prop_blade_3_surface.nodal_coordinates)

        left_prop_blade_7_surface = cd.mesh.make_rotor_panel_mesh(
            left_prop_geometry,
            blade_keys = [120, 121], # upper and lower surfaces
            grid_nr = 5,
            grid_nl = 10,
            plot=False,
        )
        right_prop_blade_3_surface = cd.mesh.make_rotor_panel_mesh(
            right_prop_geometry,
            blade_keys = [152, 153], # upper and lower surfaces
            grid_nr = 5,
            grid_nl = 10,
            plot=False,
        )
        right_prop_blade_7_surface = cd.mesh.make_rotor_panel_mesh(
            right_prop_geometry,
            blade_keys = [176, 177], # upper and lower surfaces
            grid_nr = 5,
            grid_nl = 10,
            plot=False,
        )

        # Store meshes

        panel_meshes.discretizations["left_proprotor_blade_3_panel_surface"] = left_prop_blade_3_surface
        panel_meshes.discretizations["right_proprotor_blade_3_panel_surface"] = right_prop_blade_3_surface
        panel_meshes.discretizations["left_proprotor_blade_7_panel_surface"] = left_prop_blade_7_surface
        panel_meshes.discretizations["right_proprotor_blade_7_panel_surface"] = right_prop_blade_7_surface

        mesh_container = base_config.mesh_container
        mesh_container["panel_meshes"] = panel_meshes
        if do_structural:
            mesh_container["shell_mesh"] = shell_mesh

    caddee.base_configuration = base_config


def define_conditions(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    # -1g
    if do_trim:
        minus_1g_pitch = csdl.ImplicitVariable(shape=(1, ), value=-0.14296, name='minus_1g_pitch')
    else:
        minus_1g_pitch = csdl.Variable(shape=(1, ), value=-0.14296, name='minus_1g_pitch')
        # minus_1g_pitch = csdl.Variable(shape=(1, ), value=np.deg2rad(2.69268269))

    minus_1g = cd.aircraft.conditions.CruiseCondition(
        altitude=1,
        pitch_angle=minus_1g_pitch,
        mach_number=0.35,
        range=70*cd.Units.length.kilometer_to_m,
    )
    minus_1g.quantities.pitch = minus_1g_pitch
    minus_1g.configuration = base_config.copy()
    conditions["minus_1g"] = minus_1g


def define_analysis(caddee: cd.CADDEE):
    conditions = caddee.conditions

    # finalize meshes

    # -1g
    minus_1g:cd.aircraft.conditions.CruiseCondition = conditions["minus_1g"]
    # minus_1g.finalize_meshes()
    m1g_mesh_container = minus_1g.configuration.mesh_container

    # run panel method
    panel_outputs = run_panel_method(m1g_mesh_container, minus_1g)


# TODO: 
def run_panel_method(mesh_container, condition):
    '''
    Inputs: mesh_container, condition
    Outputs: ? (VLM outputs: spanwise Cp)
    '''

    panel_meshes = mesh_container['panel_meshes']
    left_prop_blade_3_mesh = panel_meshes.discretizations["left_proprotor_blade_3_panel_surface"]
    left_prop_blade_7_mesh = panel_meshes.discretizations["left_proprotor_blade_7_panel_surface"]
    right_prop_blade_3_mesh = panel_meshes.discretizations["right_proprotor_blade_3_panel_surface"]
    right_prop_blade_7_mesh = panel_meshes.discretizations["right_proprotor_blade_7_panel_surface"]
    left_nacelle_panel_mesh = panel_meshes.discretizations["left_nacelle_panel_surface"]
    right_nacelle_panel_mesh = panel_meshes.discretizations["right_nacelle_panel_surface"]
    wing_panel_mesh = panel_meshes.discretizations["wing_panel_surface"]

    if plot_meshes:
        tiltrotor_geom.plot_meshes([left_prop_blade_3_mesh.nodal_coordinates, 
                                    left_prop_blade_7_mesh.nodal_coordinates,
                                    right_prop_blade_3_mesh.nodal_coordinates, 
                                    right_prop_blade_7_mesh.nodal_coordinates,
                                    left_nacelle_panel_mesh.nodal_coordinates, 
                                    right_nacelle_panel_mesh.nodal_coordinates, 
                                    wing_panel_mesh.nodal_coordinates, ])




define_base_config(caddee)
define_conditions(caddee)

# recorder.inline = not shutdown_inline
define_analysis(caddee)