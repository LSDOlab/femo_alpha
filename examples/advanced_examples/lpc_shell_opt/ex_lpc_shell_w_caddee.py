'''Example lift plus cruise'''
import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
from VortexAD.core.vlm.vlm_solver import vlm_solver
# from ex_utils import plot_vlm
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.utils.var_groups import RotorAnalysisInputs
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
from femo_alpha.rm_shell.rm_shell_model import RMShellModel
from femo_alpha.fea.utils_dolfinx import readFEAMesh, reconstructFEAMesh
import lsdo_function_spaces as lfs
import matplotlib.pyplot as plt
import os
import aeroelastic_coupling_utils as acu
import dolfinx

import lsdo_function_spaces as fs
fs.num_workers = 1     # uncommont this if projections break

# VLM is only forward-coupled to shell model, only in the loop because of trim.

# TODO: set up thickness functions per panel: currently just have skin/spar/rib
# TODO: set up +3g case - done
# TODO: set up -1g case - done
# TODO: set design variables, objectives, and constraints - done

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
run_optimization = True
shutdown_inline = False
do_trim = False

# Import L+C .stp file and convert control points to meters
lpc_geom = cd.import_geometry("LPC_final_custom_blades.stp", scale=cd.Units.length.foot_to_m)

# utility functions
def load_thickness_vars(fname, group):
    inputs = csdl.inline_import(fname, group)
    t_vars = {key:val for key, val in inputs.items() if 'thickness' in key}
    return t_vars

def construct_bay_condition(upper, lower):
    if upper == 1:
        geq = True
    else:
        geq = False
    def condition(parametric_coordinates:np.ndarray):
        if geq:
            out = np.logical_and(np.greater_equal(parametric_coordinates[:,0], lower), np.less_equal(parametric_coordinates[:,0], upper))
        else:
            out = np.logical_and(np.greater(parametric_coordinates[:,0], lower), np.less_equal(parametric_coordinates[:,0], upper))
        if len(out.shape) == 1:
            out.reshape(-1, 1)
        # if len(out.shape) == 1:
        #     out.reshape(1, 1)
        return out.T
    return condition

def construct_thickness_function(wing, num_ribs, top_array, bottom_array, material, t_vars=None):
    bay_eps = 1e-2
    if t_vars is None:
        t_out = {}
    for i in range(num_ribs-1):
        # upper wing bays
        lower = top_array[:, i]
        upper = top_array[:, i+1]
        l_surf_ind = lower[0][0]
        u_surf_ind = upper[0][0]
        l_coord_u = np.mean(np.array([coord[0,0] for (ind, coord) in lower])) - bay_eps
        u_coord_u = np.mean(np.array([coord[0,0] for (ind, coord) in upper])) - bay_eps
        if i == num_ribs-2:
            u_coord_u = 1 + bay_eps
        # l_coord_u = lower[0][1][0,0]
        # u_coord_u = upper[0][1][0,0]
        if l_surf_ind == u_surf_ind:
            condition = construct_bay_condition(u_coord_u, l_coord_u)
            if t_vars is not None:
                thickness = t_vars['upper_wing_thickness_'+str(i)]
            else:
                thickness = csdl.Variable(value=initial_thickness, name='upper_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {l_surf_ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
        else:
            condition1 = construct_bay_condition(1, l_coord_u)
            condition2 = construct_bay_condition(u_coord_u, 0)
            if t_vars is not None:
                thickness = t_vars['upper_wing_thickness_'+str(i)]
            else:
                thickness = csdl.Variable(value=initial_thickness, name='upper_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function1 = lfs.Function(lfs.ConditionalSpace(2, condition1), thickness)
            function2 = lfs.Function(lfs.ConditionalSpace(2, condition2), thickness)
            functions = {l_surf_ind: function1, u_surf_ind: function2}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)

        # lower wing bays
        lower = bottom_array[:, i]
        upper = bottom_array[:, i+1]
        l_surf_ind = lower[0][0]
        u_surf_ind = upper[0][0]
        l_coord_u = np.mean(np.array([coord[0, 0] for (ind, coord) in lower])) - bay_eps
        u_coord_u = np.mean(np.array([coord[0, 0] for (ind, coord) in upper])) - bay_eps
        if i == num_ribs-2:
            u_coord_u = 1 + bay_eps
        # l_coord_u = lower[0][1][0, 0]
        # u_coord_u = upper[0][1][0, 0]
        if l_surf_ind == u_surf_ind:
            condition = construct_bay_condition(u_coord_u, l_coord_u)
            if t_vars is not None:
                thickness = t_vars['lower_wing_thickness_'+str(i)]
            else:
                thickness = csdl.Variable(value=initial_thickness, name='lower_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {l_surf_ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
        else:
            condition1 = construct_bay_condition(1, l_coord_u)
            condition2 = construct_bay_condition(u_coord_u, 0)
            if t_vars is not None:
                thickness = t_vars['lower_wing_thickness_'+str(i)]
            else:
                thickness = csdl.Variable(value=initial_thickness, name='lower_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function1 = lfs.Function(lfs.ConditionalSpace(2, condition1), thickness)
            function2 = lfs.Function(lfs.ConditionalSpace(2, condition2), thickness)
            functions = {l_surf_ind: function1, u_surf_ind: function2}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)


    # ribs
    rib_geometry = wing.create_subgeometry(search_names=["rib"])
    rib_fsp = lfs.ConstantSpace(2)
    for ind in rib_geometry.functions:
        name = rib_geometry.function_names[ind]
        if "-" in name:
            pass
        else:
            if t_vars is not None:
                thickness = t_vars[name+'_thickness']
            else:
                thickness = csdl.Variable(value=initial_thickness, name=name+'_thickness')
                t_out[thickness.name] = thickness
            thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(rib_fsp, thickness)
            functions = {ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)

    # spars
    u_cords = np.linspace(0, 1, num_ribs)
    spar_geometry = wing.create_subgeometry(search_names=["spar"], ignore_names=['_r_'])
    spar_inds = list(spar_geometry.functions)
    for i in range(num_ribs-1):
        lower = u_cords[i] - bay_eps
        upper = u_cords[i+1] - bay_eps
        if i == num_ribs-2:
            upper = 1 + 2*bay_eps
        condition = construct_bay_condition(upper, lower)
        spar_num = 0
        for ind in spar_inds:
            if t_vars is not None:
                thickness = t_vars[f'spar_{spar_num}_thickness_{i}']
            else:
                thickness = csdl.Variable(value=initial_thickness, name=f'spar_{spar_num}_thickness_{i}')
                t_out[thickness.name] = thickness
            thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
            spar_num += 1

def construct_plate_condition(upper, lower, forward, backward, ind):
    if upper == 1:
        geq = True
    else:
        geq = False
    def condition(parametric_coordinate:tuple[int, np.ndarray]):
        index = parametric_coordinate[0]
        if index != ind:
            return False
        coord = parametric_coordinate[1]
        if geq:
            out = np.logical_and(np.greater_equal(coord[:,0], lower), np.less_equal(coord[:,0], upper))
            out = np.logical_and(out, np.logical_and(np.greater_equal(coord[:,1], backward), np.less_equal(coord[:,1], forward)))
        else:
            out = np.logical_and(np.greater(coord[:,0], lower), np.less_equal(coord[:,0], upper))
            out = np.logical_and(out, np.logical_and(np.greater(coord[:,1], backward), np.less_equal(coord[:,1], forward)))
        return out[0]
    return condition

def construct_plates(num_ribs, top_array, bottom_array):
    bays = {}
    bay_eps = 1e-2
    for i in range(num_ribs-2):
        # upper wing bays
        lower = top_array[1:3, i]
        upper = top_array[1:3, i+1]
        l_surf_ind = lower[0][0]
        u_surf_ind = upper[0][0]
        l_coord_u = np.mean(np.array([coord[0,0] for (ind, coord) in lower])) - bay_eps
        u_coord_u = np.mean(np.array([coord[0,0] for (ind, coord) in upper])) + bay_eps
        if i == num_ribs-2:
            u_coord_u = 1 + bay_eps
        # l_coord_u = lower[0][1][0,0]
        # u_coord_u = upper[0][1][0,0]
        if l_surf_ind == u_surf_ind:
            f_coord_v = (lower[0][1][0,1] + upper[0][1][0,1])/2 + bay_eps
            b_coord_v = (lower[1][1][0,1] + upper[1][1][0,1])/2 - bay_eps
            condition = construct_plate_condition(u_coord_u, l_coord_u, f_coord_v, b_coord_v, l_surf_ind)
            bays['upper_wing_bay_'+str(i)] = [condition]
        else:
            condition1 = construct_plate_condition(1, l_coord_u, lower[0][1][0,1]+bay_eps, lower[1][1][0,1]-bay_eps, l_surf_ind)
            condition2 = construct_plate_condition(u_coord_u, 0, upper[0][1][0,1]+bay_eps, upper[1][1][0,1]-bay_eps, u_surf_ind)
            bays['upper_wing_bay_'+str(i)] = [condition1, condition2]

        # lower wing bays
        lower = bottom_array[1:3, i]
        upper = bottom_array[1:3, i+1]
        l_surf_ind = lower[0][0]
        u_surf_ind = upper[0][0]
        l_coord_u = np.mean(np.array([coord[0, 0] for (ind, coord) in lower])) - bay_eps
        u_coord_u = np.mean(np.array([coord[0, 0] for (ind, coord) in upper])) + bay_eps
        if i == num_ribs-2:
            u_coord_u = 1 + bay_eps
        # l_coord_u = lower[0][1][0, 0]
        # u_coord_u = upper[0][1][0, 0]
        if l_surf_ind == u_surf_ind:
            f_coord_v = (lower[0][1][0,1] + upper[0][1][0,1])/2 - bay_eps
            b_coord_v = (lower[1][1][0,1] + upper[1][1][0,1])/2 + bay_eps
            condition = construct_plate_condition(u_coord_u, l_coord_u, b_coord_v, f_coord_v, l_surf_ind)
            bays['lower_wing_bay_'+str(i)] = [condition]
        else:
            condition1 = construct_plate_condition(1, l_coord_u, lower[1][1][0,1]+bay_eps, lower[0][1][0,1]-bay_eps, l_surf_ind)
            condition2 = construct_plate_condition(u_coord_u, 0, upper[1][1][0,1]+bay_eps, upper[0][1][0,1]-bay_eps, u_surf_ind)
            bays['lower_wing_bay_'+str(i)] = [condition1, condition2]
    return bays

def load_dv_values(fname, group):
    inputs = csdl.inline_import(fname, group)
    dvs = recorder.design_variables
    for var in dvs:
        var.set_value(inputs[var.name].value)
        scale = 1/np.linalg.norm(inputs[var.name].value)
        dvs[var] = (scale, dvs[var][1], dvs[var][2])

def define_base_config(caddee : cd.CADDEE):
    """Build the base configuration."""
    
    # system component & airframe
    aircraft = cd.aircraft.components.Aircraft(geometry=lpc_geom)
    airframe = aircraft.comps["airframe"] = cd.Component()

    # ::::::::::::::::::::::::::: Make components :::::::::::::::::::::::::::

    # Main wing
    top_surface_inds = [75, 79, 83, 87]
    ignore_names = ['72', '73', '90', '91', '92', '93', '110', '111'] # rib-like surfaces
    ignore_names += [str(i-1) for i in top_surface_inds] + [str(i+2) for i in top_surface_inds]
    ignore_names += [str(i-1+20) for i in top_surface_inds] + [str(i+2+20) for i in top_surface_inds]
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing_1"], ignore_names=ignore_names)
    wing = cd.aircraft.components.Wing(AR=12.12, S_ref=19.6, taper_ratio=0.2,
                                       geometry=wing_geometry, tight_fit_ffd=True,
                                       compute_surface_area=False)

    num_ribs = 9
    spanwise_multiplicity = 5
    top_array, bottom_array = wing.construct_ribs_and_spars(
        lpc_geom,
        num_ribs=num_ribs,
        LE_TE_interpolation="ellipse",
        plot_projections=False, 
        export_wing_box=False,
        export_half_wing=False,
        full_length_ribs=True,
        spanwise_multiplicity=spanwise_multiplicity,
        num_rib_pts=10,
        offset=np.array([0.,0.,.15]),
        finite_te=True,
        exclute_te=True,
        return_rib_points=True,
    )

    indices = np.array([i for i in range(0, top_array.shape[-1], spanwise_multiplicity)])
    top_array = top_array[:, indices]
    bottom_array = bottom_array[:, indices]

    # material
    E = csdl.Variable(value=69E9, name='E')
    G = csdl.Variable(value=26E9, name='G')
    density = csdl.Variable(value=2700, name='density')
    nu = csdl.Variable(value=0.33, name='nu')
    aluminum = cd.materials.IsotropicMaterial(name='aluminum', E=E, G=G, 
                                              density=density, nu=nu)

    # t_vars = load_thickness_vars('structural_opt_v3_6final.hdf5', 'inline')

    # construct_thickness_function(wing, num_ribs, top_array, bottom_array, aluminum, t_vars)

    construct_thickness_function(wing, num_ribs, top_array, bottom_array, aluminum)


    wing_bays = construct_plates(num_ribs, top_array, bottom_array)

    wing_oml = wing.create_subgeometry(search_names=["Wing_1"])
    left_wing_spars = wing.create_subgeometry(search_names=["spar"], ignore_names=['_r_', '-', 'Wing_1, 1'])
    left_wing_ribs = wing.create_subgeometry(search_names=["rib"], ignore_names=['_r_', '-', 'Wing_1, 1'])
    left_wing_oml = wing.create_subgeometry(search_names=["Wing_1"], ignore_names=['_r_', '-', 'Wing_1, 1'])
    wing.quantities.oml_geometry = wing_oml
    left_wing = wing.create_subgeometry(search_names=[""], ignore_names=["Wing_1, 1", '_r_', '-'])
    wing.quantities.left_wing = left_wing

    # thickness_function_set = lfs.FunctionSet(functions)
    wing.quantities.material_properties.set_material(aluminum, None)

    pressure_function_space = lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=4, grid_size=(240, 40), conserve=False, n_neighbors=10)
    # pressure_function_space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 3), coefficients_shape=(2, 10))
    indexed_pressue_function_space = wing.geometry.create_parallel_space(pressure_function_space)
    wing.quantities.pressure_space = indexed_pressue_function_space

    # displacement_space = lfs.BSplineSpace(2, (1,1), (3,3))
    displacement_space = lfs.IDWFunctionSpace(2, 5, grid_size=100, conserve=False)
    wing.quantities.displacement_space = wing_geometry.create_parallel_space(
                                                    displacement_space)

    airframe.comps["wing"] = wing


    # ::::::::::::::::::::::::::: Make meshes :::::::::::::::::::::::::::
    if make_meshes:
    # wing + tail
        vlm_mesh = cd.mesh.VLMMesh()
        wing_chord_surface = cd.mesh.make_vlm_surface(
            wing, 40, 1, LE_interp="ellipse", TE_interp="ellipse", 
            spacing_spanwise="cosine", ignore_camber=True, plot=False,
        )
        wing_chord_surface.project_airfoil_points(oml_geometry=wing_oml)
        vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface


        # lpc_geom.plot_meshes([wing_chord_surface.nodal_coordinates])

        # shell mesh
        prefix = './data_files/'
        fname = prefix+'left_wing_c'
        wing_shell_discritization = cd.mesh.import_shell_mesh(
            fname+'.msh',
            left_wing,
            plot=False,
            rescale=[1e-3, 1e-3, 1e-3],
            grid_search_n=5,
            priority_inds=[i for i in wing_oml.functions],
            priority_eps=3e-3,
            force_reprojection=False
        )
        connectivity = wing_shell_discritization.connectivity
        nodes_parametric = wing_shell_discritization.nodes_parametric

        # assign bays to nodes
        bay_nodes = {}
        for key, conds in wing_bays.items():
            bay_nodes[key] = []
            for cond in conds:
                for i in range(len(nodes_parametric)):
                    if cond(nodes_parametric[i]):
                        bay_nodes[key].append(i)

        # nodes = wing_shell_discritization.nodal_coordinates
        # for bay_node_list in bay_nodes.values():
        #     wing.geometry.plot_meshes([nodes[bay_node_list]])
        # exit()

        bay_elements = {}
        for key, nodes in bay_nodes.items():
            bay_elements[key] = []
            for i in range(connectivity.shape[0]):
                if np.all(np.isin(connectivity[i], nodes)):
                    bay_elements[key].append(i)

        association_table = {key:i for i, key in enumerate(bay_elements.keys())}
        association_table['none'] = -1


        nodes = wing_shell_discritization.nodal_coordinates

        # find centerpoints for per-element thickness
        # figure out type of surface each element is (rib/spar/skin)
        grid_n = 20
        oml_errors = np.linalg.norm(left_wing_oml.evaluate(left_wing_oml.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)
        rib_errors = np.linalg.norm(left_wing_ribs.evaluate(left_wing_ribs.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)
        spar_errors = np.linalg.norm(left_wing_spars.evaluate(left_wing_spars.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)

        element_centers = np.array([np.mean(nodes.value[connectivity[i].astype(int)], axis=0) for i in range(connectivity.shape[0])])

        rib_correction = 1e-3
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
            
        oml_centers = left_wing_oml.project(element_centers[oml_inds], grid_search_density_parameter=5, plot=False)
        rib_centers = left_wing_ribs.project(element_centers[rib_inds], grid_search_density_parameter=5, plot=False)
        spar_centers = left_wing_spars.project(element_centers[spar_inds], grid_search_density_parameter=5, plot=False)
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
        wing_shell_discritization.element_centers_parametric = element_centers_parametric

        filename = fname+"_reconstructed.xdmf"
        if os.path.isfile(filename) and False:
            wing_shell_mesh_fenics = readFEAMesh(filename)
        else:
            # Reconstruct the mesh using the projected nodes and connectivity
            wing_shell_mesh_fenics = reconstructFEAMesh(filename, 
                                                        nodes.value, connectivity)
        # store the xdmf mesh object for shell analysis
        wing_shell_discritization.fea_mesh = wing_shell_mesh_fenics

        cd2fe_el = np.argsort(wing_shell_mesh_fenics.topology.original_cell_index)

        vals = []
        for i in range(connectivity.shape[0]):
            for key, elements in bay_elements.items():
                if i in elements:
                    vals.append(association_table[key])
                    break
            else:
                vals.append(association_table['none'])

        meshtags = dolfinx.mesh.meshtags(wing_shell_mesh_fenics, 2, cd2fe_el, np.array(vals).astype(np.int32))
        wing_shell_discritization.meshtags = meshtags

        association_table.pop('none')
        wing_shell_discritization.association_table = association_table

        pav_shell_mesh = cd.mesh.ShellMesh()
        pav_shell_mesh.discretizations['wing'] = wing_shell_discritization


    # Make base configuration    
    base_config = cd.Configuration(system=aircraft)
    if run_ffd:
        base_config.setup_geometry()
    caddee.base_configuration = base_config

    # Store meshes
    if make_meshes:
        mesh_container = base_config.mesh_container
        mesh_container["vlm_mesh"] = vlm_mesh
        mesh_container["shell_mesh"] = pav_shell_mesh


def define_conditions(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    # +3g
    if do_trim:
        plus_3g_pitch = csdl.Variable(shape=(1, ), value=np.deg2rad(1), name='plus_3g_pitch')
        plus_3g_pitch.set_as_design_variable(upper=np.deg2rad(10), lower=np.deg2rad(-10), scaler=1)
    else:
        plus_3g_pitch = csdl.Variable(shape=(1, ), value=0.015577, name='plus_3g_pitch')

    plus_3g = cd.aircraft.conditions.CruiseCondition(
        altitude=1,
        pitch_angle=plus_3g_pitch,
        mach_number=0.35,    # gotta go fast or else we can't ge -1g
        range=70*cd.Units.length.kilometer_to_m,
    )
    plus_3g.configuration = base_config.copy()
    conditions["plus_3g"] = plus_3g

    # -1g
    if do_trim:
        minus_1g_pitch = csdl.Variable(shape=(1, ), value=np.deg2rad(-8.2), name='minus_1g_pitch')
        minus_1g_pitch.set_as_design_variable(upper=np.deg2rad(10), lower=np.deg2rad(-10), scaler=1)
    else:
        minus_1g_pitch = csdl.Variable(shape=(1, ), value=-0.14296, name='minus_1g_pitch')

    minus_1g = cd.aircraft.conditions.CruiseCondition(
        altitude=1,
        pitch_angle=minus_1g_pitch,
        mach_number=0.35,
        range=70*cd.Units.length.kilometer_to_m,
    )
    minus_1g.configuration = base_config.copy()
    conditions["minus_1g"] = minus_1g


def define_mass_properties(caddee: cd.CADDEE):
    # Get base config and conditions
    base_config = caddee.base_configuration
    conditions = caddee.conditions

    # directly assign mass based on system optimization
    base_config.system.quantities.mass_properties.mass = system_mass
    base_config.system.quantities.mass_properties.cg_vector = np.array([0., 0., 0.])

    plus_3g = conditions["plus_3g"]
    plus_3g.configuration.system.quantities.mass_properties.mass = system_mass
    plus_3g.configuration.system.quantities.mass_properties.cg_vector = np.array([0., 0., 0.])

    minus_1g = conditions["minus_1g"]
    minus_1g.configuration.system.quantities.mass_properties.mass = system_mass
    minus_1g.configuration.system.quantities.mass_properties.cg_vector = np.array([0., 0., 0.])


def define_analysis(caddee: cd.CADDEE):
    conditions = caddee.conditions

    # finalize meshes

    # 3g
    plus_3g:cd.aircraft.conditions.CruiseCondition = conditions["plus_3g"]
    plus_3g.finalize_meshes()
    p3g_mesh_container = plus_3g.configuration.mesh_container
    p3g_pitch_angle = plus_3g.parameters.pitch_angle

    # -1g
    minus_1g:cd.aircraft.conditions.CruiseCondition = conditions["minus_1g"]
    minus_1g.finalize_meshes()
    m1g_mesh_container = minus_1g.configuration.mesh_container
    m1g_pitch_angle = minus_1g.parameters.pitch_angle

    # run VLM (vectorized)
    vlm_outputs = run_vlm([p3g_mesh_container, m1g_mesh_container], [plus_3g, minus_1g])
    p3g_forces = vlm_outputs.surface_force[0][0]
    m1g_forces = vlm_outputs.surface_force[0][1]
    p3g_Cp = vlm_outputs.surface_spanwise_Cp[0][0]
    m1g_Cp = vlm_outputs.surface_spanwise_Cp[0][1]

    # 3g - run beam and set up constraints
    p3g_displacement, p3g_pressure, p3g_max_stress, p3g_tip_displacement, wing_mass = run_shell(p3g_mesh_container, plus_3g, p3g_Cp)

    p3g_max_stress.add_name('p3g_max_stress')
    p3g_tip_displacement.add_name('p3g_tip_displacement')
    p3g_forces.add_name('p3g_forces')
    p3g_max_stress.set_as_constraint(upper=max_stress, scaler=1e-8)
    p3g_tip_displacement.set_as_constraint(upper=max_displacement, lower=-max_displacement, scaler=1e2)
    p3g_wing_z_force = -p3g_forces[2]*csdl.cos(p3g_pitch_angle) + p3g_forces[0]*csdl.sin(p3g_pitch_angle)
    p3g_wing_z_force.add_name('p3g_wing_z_force')
    if do_trim:
        p3g_wing_z_force.set_as_constraint(upper=3*system_mass*g, lower=3*system_mass*g, scaler=1/(3*system_mass*g))


    # -1g - run beam and set up constraints
    m1g_displacement, m1g_pressure, m1g_max_stress, m1g_tip_displacement, _ = run_shell(m1g_mesh_container, minus_1g, m1g_Cp)

    m1g_max_stress.add_name('m1g_max_stress')
    m1g_tip_displacement.add_name('m1g_tip_displacement')
    m1g_forces.add_name('m1g_forces')
    m1g_max_stress.set_as_constraint(upper=max_stress, scaler=1e-8)
    m1g_tip_displacement.set_as_constraint(upper=max_displacement, lower=-max_displacement, scaler=1e2)
    m1g_wing_z_force = -m1g_forces[2]*csdl.cos(m1g_pitch_angle) + m1g_forces[0]*csdl.sin(m1g_pitch_angle)
    m1g_wing_z_force.add_name('m1g_wing_z_force')
    if do_trim:
        m1g_wing_z_force.set_as_constraint(upper=-system_mass*g, lower=-system_mass*g, scaler=-1/(system_mass*g))

    # set up objective
    wing_mass.add_name('wing_mass')
    wing_mass.set_as_objective(scaler=1e-2)


def run_vlm(mesh_containers, conditions):
    # set up VLM analysis
    nodal_coords = []
    nodal_vels = []

    for mesh_container, condition in zip(mesh_containers, conditions):
        wing_lattice = mesh_container["vlm_mesh"].discretizations["wing_chord_surface"]
        nodal_coords.append(wing_lattice.nodal_coordinates)
        nodal_vels.append(wing_lattice.nodal_velocities)

    nodal_coordinates = csdl.vstack(nodal_coords)
    nodal_velocities = csdl.vstack(nodal_vels)


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

    # shell_model = RMShellModel(wing_shell_mesh_fenics,
    #                            clamped_boundary,
    #                            record=True, # record=true doesn't work with 2 shell instances
    #                            rho=10)
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

define_conditions(caddee)

define_mass_properties(caddee)

recorder.inline = not shutdown_inline

define_analysis(caddee)

csdl.save_optimization_variables()
# exit()

load_dv_values('structural_opt_v3_6final.hdf5', 'inline')

if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import PySLSQP, SNOPT, IPOPT
    import time

    fname = 'structural_opt_v3_8'
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False, save_on_update=True, filename=fname)

    prob = CSDLAlphaProblem(problem_name=fname, simulator=sim)

    iter = 0

    def save_vars(x):
        iter += 1
        sim.save_external(fname, 'iteration_'+str(iter))


    optimizer = PySLSQP(prob, solver_options={'maxiter':100, 'acc':1e-3})
    # snoptions = {
    # 'Major iterations': 100, 
    # 'Major optimality': 1e-3, 
    # 'Major feasibility': 1e-3
    # }
    # optimizer = SNOPT(prob, solver_options=snoptions)
    # optimizer = IPOPT(prob, solver_options={'max_iter':100, 'tol':1e-3})

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()
    csdl.inline_export(fname+'_final')

# print(accel_cruise.accel_norm.value)
# print(accel_hover.accel_norm.value)
# print(total_forces_cruise.value)
# print(total_forces_hover.value)
    
