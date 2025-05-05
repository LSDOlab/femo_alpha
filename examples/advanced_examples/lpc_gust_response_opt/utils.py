import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs
import CADDEE_alpha as cd
import os

def load_structural_mesh_pre(left_wing, left_wing_oml, left_wing_ribs, left_wing_spars, rear_spar, wing_bays, boom_y_ranges, fname):
    
    wing_shell_discritization = cd.mesh.import_shell_mesh(
        fname+'.msh',
        left_wing,
        plot=False,
        rescale=[1e-3, 1e-3, 1e-3],
        grid_search_n=5,
        priority_inds=[i for i in left_wing_oml.functions],
        priority_eps=3e-6,
        force_reprojection=False
    )
    connectivity = wing_shell_discritization.connectivity
    nodes_parametric = wing_shell_discritization.nodes_parametric
    node_vals = wing_shell_discritization.nodal_coordinates.value

    # find centerpoints for per-element thickness
    # figure out type of surface each element is (rib/spar/skin)
    grid_n = 20
    oml_errors = np.linalg.norm(left_wing_oml.evaluate(left_wing_oml.project(node_vals, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - node_vals, axis=1)
    rib_errors = np.linalg.norm(left_wing_ribs.evaluate(left_wing_ribs.project(node_vals, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - node_vals, axis=1)
    spar_errors = np.linalg.norm(left_wing_spars.evaluate(left_wing_spars.project(node_vals, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - node_vals, axis=1)
    rear_spar_errors = np.linalg.norm(rear_spar.evaluate(rear_spar.project(node_vals, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - node_vals, axis=1)

    element_centers = np.array([np.mean(node_vals[connectivity[i].astype(int)], axis=0) for i in range(connectivity.shape[0])])

    rib_correction = 1e-3
    element_centers_parametric = []
    oml_inds = []
    rib_inds = []
    spar_inds = []
    rear_spar_inds = []
    for i in range(connectivity.shape[0]):
        inds = connectivity[i].astype(int)
        
        # rib projection is messed up so we use an alternitive approach - if all the points are in an x-z plane, it's a rib
        if np.all(np.isclose(node_vals[inds, 1], node_vals[inds[0], 1], atol=rib_correction)):
            rib_inds.append(i)
            continue

        errors = [np.sum(oml_errors[inds]), np.sum(rib_errors[inds]), np.sum(spar_errors[inds])]
        rear_spar_error = np.sum(rear_spar_errors[inds])
        ind = np.argmin(errors)
        if ind == 0:
            oml_inds.append(i)
        elif ind == 1:
            rib_inds.append(i)
        elif ind == 2:
            spar_inds.append(i)
            if rear_spar_error < 1e-3+errors[2]:
                rear_spar_inds.append(i)
        else:
            raise ValueError('Error in determining element type')
        
    oml_centers = left_wing_oml.project(element_centers[oml_inds], grid_search_density_parameter=5, plot=False)
    rib_centers = left_wing_ribs.project(element_centers[rib_inds], grid_search_density_parameter=5, plot=False)
    spar_centers = left_wing_spars.project(element_centers[spar_inds], grid_search_density_parameter=5, plot=False)
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
    wing_shell_discritization.element_centers_parametric = element_centers_parametric

    oml_node_inds = []
    for c_ind in oml_inds_copy:
        n_inds = connectivity[c_ind].astype(int)
        for n_ind in n_inds:
            if not n_ind in oml_node_inds:
                oml_node_inds.append(n_ind)
    rear_spar_node_inds = []
    for c_ind in rear_spar_inds:
        n_inds = connectivity[c_ind].astype(int)
        for n_ind in n_inds:
            if not n_ind in rear_spar_node_inds:
                rear_spar_node_inds.append(n_ind)

    oml_nodes_parametric = left_wing_oml.project(node_vals[oml_node_inds], grid_search_density_parameter=5)
    wing_shell_discritization.oml_node_inds = oml_node_inds
    wing_shell_discritization.oml_nodes_parametric = oml_nodes_parametric
    wing_shell_discritization.oml_el_inds = oml_inds_copy
    wing_shell_discritization.rear_spar_node_inds = rear_spar_node_inds


    # assign bays to nodes
    bay_nodes = {}
    for key, conds in wing_bays.items():
        bay_nodes[key] = []
        for cond in conds:
            for i in range(len(oml_nodes_parametric)):
                if cond(oml_nodes_parametric[i]):
                    bay_nodes[key].append(oml_node_inds[i])

    # nodes = wing_shell_discritization.nodal_coordinates
    # for bay_node_list in bay_nodes.values():
    #     left_wing.plot_meshes([nodes[bay_node_list]])
    # exit()

    bay_elements = {}
    for key, nodes in bay_nodes.items():
        bay_elements[key] = []
        for i in range(connectivity.shape[0]):
            if np.all(np.isin(connectivity[i], nodes)):
                bay_elements[key].append(i)

    association_table = {key:i for i, key in enumerate(bay_elements.keys())}
    association_table['none'] = -1
    wing_shell_discritization.association_table = association_table
    wing_shell_discritization.bay_elements = bay_elements

    # boom nodes
    boom_node_inds = [[],[]]
    for i, node in enumerate(node_vals):
        for j in range(2):
            if boom_y_ranges[j][0] <= node[1] <= boom_y_ranges[j][1]:
                boom_node_inds[j].append(i)
    wing_shell_discritization.boom_node_inds = boom_node_inds

    return wing_shell_discritization


def load_structural_mesh_post(wing_shell_discritization, fname):
    import dolfinx
    from femo_alpha.fea.utils_dolfinx import readFEAMesh, reconstructFEAMesh

    nodes = wing_shell_discritization.nodal_coordinates
    connectivity = wing_shell_discritization.connectivity
    association_table = wing_shell_discritization.association_table
    bay_elements = wing_shell_discritization.bay_elements

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
    
    pav_shell_mesh = cd.mesh.ShellMesh()
    pav_shell_mesh.discretizations['wing'] = wing_shell_discritization
    
    return pav_shell_mesh

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

def construct_thickness_function(wing, num_ribs, top_array, bottom_array, material, t_vars=None, 
                                 skin_t=0.01, spar_t=0.01, rib_t=0.01, minimum_thickness=0.0003,
                                 top_dv=True, bottom_dv=True, rib_dv=True, spar_dv=True):
    bay_eps = 1e-5
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
                thickness = csdl.Variable(value=skin_t, name='upper_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            if top_dv:
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
                thickness = csdl.Variable(value=skin_t, name='upper_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            if top_dv:
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
                thickness = csdl.Variable(value=skin_t, name='lower_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            if bottom_dv:
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
                thickness = csdl.Variable(value=skin_t, name='lower_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            if bottom_dv:
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
                thickness = csdl.Variable(value=rib_t, name=name+'_thickness')
                t_out[thickness.name] = thickness
            if rib_dv:
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
                thickness = csdl.Variable(value=spar_t, name=f'spar_{spar_num}_thickness_{i}')
                t_out[thickness.name] = thickness
            if spar_dv:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
            spar_num += 1
    if t_vars is None:
        return t_out
    

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

def construct_plates(num_ribs, top_array, bottom_array, offset=1):
    bays = {}
    bay_eps = 1e-2
    for i in range(num_ribs-(1+offset)):
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

def compute_buckling_loads(wing, material, point_array, t_vars):
    compression_k_lookup = {0.2:22.2, 0.3:10.9, 0.4:6.92, 0.6:4.23, 0.8:3.45, 
                        1.0:3.29, 1.2:3.40, 1.4:3.68, 1.6:3.45, 1.8:3.32, 
                        2.0:3.29, 2.2:3.32, 2.4:3.40, 2.7:3.32, 3.0:3.29}
    shear_k_lookup = {1.0:7.75, 1.2:6.58, 1.4:6.00, 1.5:5.84, 1.6:5.76, 
                    1.8:5.59, 2.0:5.43, 2.5:5.18, 3.0:5.02}
    E, nu, G = material.get_constants()
    sigma_cr = []
    tau_cr = []
    for i in range(point_array.shape[1]-1):
        # get relevant parametric points
        lower = point_array[:, i].tolist()   # [s1, s2]
        upper = point_array[:, i+1].tolist() # [s1, s2]

        # get thickness
        t = t_vars['upper_wing_thickness_'+str(i)]

        # approximate the bay as a rectangle between ribs (lower and upper) and the spars (s1 and s2)
        # compute the side lengths of the rectangle
        corner_points_parametric = lower + upper
        corner_points = wing.geometry.evaluate(corner_points_parametric, non_csdl=True)
        b1 = np.linalg.norm(corner_points[0] - corner_points[1])
        b2 = np.linalg.norm(corner_points[2] - corner_points[3])
        a1 = np.linalg.norm(corner_points[0] - corner_points[2])
        a2 = np.linalg.norm(corner_points[1] - corner_points[3])
        a = (a1 + a2)/2
        b = (b1 + b2)/2
        aspect_ratio = a/b
        compression_k = compression_k_lookup[min(compression_k_lookup, key=lambda x:abs(x-aspect_ratio))]
        if aspect_ratio < 1:
            aspect_ratio = 1/aspect_ratio
        shear_k = shear_k_lookup[min(shear_k_lookup, key=lambda x:abs(x-aspect_ratio))]

        sigma_cr.append(compression_k*E/(1-nu**2)*(t/b)**2)
        tau_cr.append(shear_k*E/(1-nu**2)*(t/b)**2)
    return sigma_cr, tau_cr

def compute_curved_buckling_loads(wing, material, point_array, t_vars, key_prefix):
    E, nu, G = material.get_constants()
    if isinstance(E, csdl.Variable):
        E = E.value
        nu = nu.value
        G = G.value
    sigma_cr = []
    for i in range(point_array.shape[1]-1):
        # get relevant parametric points
        lower = point_array[:, i].tolist()
        upper = point_array[:, i+1].tolist()
        lower_middle = (lower[0][0], (lower[0][1]+lower[1][1])/2)
        upper_middle = (upper[0][0], (upper[0][1]+upper[1][1])/2)

        # get thickness
        t = t_vars[key_prefix+str(i)]

        # approximate the bay as a rectangle between ribs (lower and upper) and the spars (s1 and s2)
        # compute the side lengths of the rectangle
        corner_points_parametric = lower + upper
        corner_points = wing.geometry.evaluate(corner_points_parametric, non_csdl=True)
        b1 = np.linalg.norm(corner_points[0] - corner_points[1])
        b2 = np.linalg.norm(corner_points[2] - corner_points[3])
        a1 = np.linalg.norm(corner_points[0] - corner_points[2])
        a2 = np.linalg.norm(corner_points[1] - corner_points[3])
        a = (a1 + a2)/2
        b = (b1 + b2)/2
        
        # compute average radius of curvature
        r = np.sum(roc(wing, lower+upper+[lower_middle, upper_middle]))/6

        sigma_cr.append(1/6*E/(1-nu**2)*((12*(1-nu**2)*(t/r)**2+(np.pi*t/b)**4)**(1/2)+(np.pi*t/b)**2))
    return sigma_cr

def roc(wing:cd.Component, point):
    u_prime = wing.geometry.evaluate(point, parametric_derivative_orders=(0,1), non_csdl=True)[:,0]
    u_double_prime = wing.geometry.evaluate(point, parametric_derivative_orders=(0,2), non_csdl=True)[:,0]
    return np.abs(1+np.abs(u_prime)**(1/2)/u_double_prime)


def load_dv_values(fname, group, rib_mult=1):
    inputs = csdl.inline_import(fname, group)
    recorder = csdl.get_current_recorder()
    dvs = recorder.design_variables
    for var in dvs:
        if 'rib' in var.name:
            # divide the number in the name string by rib_div
            num = int(var.name.split('_')[2])
            num = int(num/rib_mult)
            # remake the string
            var_name = f'Wing_rib_{num}_thickness'
            print(f'Changing {var.name} to {var_name}')
        else:
            var_name = var.name

        var.set_value(inputs[var_name].value)
        scale = 1/np.linalg.norm(inputs[var_name].value)
        dvs[var] = (scale, dvs[var][1], dvs[var][2])
