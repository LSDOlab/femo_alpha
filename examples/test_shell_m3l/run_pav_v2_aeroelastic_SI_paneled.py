"""
Structural analysis of the PAV wing
with the Reissner--Mindlin shell model

-----------------------------------------------------------
Test the integration of m3l and shell model
-----------------------------------------------------------
"""
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from VAST.core.generate_mappings_m3l import VASTNodalForces
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
import caddee.core.primitives.bsplines.bspline_functions as bsf
from caddee.core.caddee_core.system_representation.system_primitive.system_primitive import SystemPrimitive
from caddee.core.caddee_core.system_representation.spatial_representation import SpatialRepresentation
from caddee.core.caddee_core.system_representation.utils.mesh_utils import import_mesh as caddee_import_mesh

from caddee import GEOMETRY_FILES_FOLDER

import numpy as np
from mpi4py import MPI
import caddee.api as cd
import csdl
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import m3l
import lsdo_geo as lg
import array_mapper as am
import pickle
import pathlib

from lsdo_rotor.core.BEM_caddee.BEM_caddee import BEM, BEMMesh
from caddee.utils.aircraft_models.pav.pav_weight import PavMassProperties
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem

import meshio
fenics = True
if fenics:
    import dolfinx
    from femo.fea.utils_dolfinx import *
    import shell_module as rmshell
    from shell_pde import ShellPDE


in2m = 0.0254
ft2m = 0.3048
lbs2kg = 0.453592
psf2pa = 50

debug_geom_flag = False
visualize_flag = False
do_plots = False
force_reprojection = False

# CADDEE geometry initialization
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation


## Generate geometry

# import initial geomrty
# file_name = '/pav_wing/pav.stp'
file_name = '/pav_wing/pav_SI.stp'
cfile = str(pathlib.Path(__file__).parent.resolve())
spatial_rep.import_file(file_name=cfile+file_name)
spatial_rep.refit_geometry(file_name=cfile+file_name)

# fix naming
primitives_new = {}
indicies_new = {}
for key, item in spatial_rep.primitives.items():
    item.name = item.name.replace(' ','_').replace(',','')
    primitives_new[key.replace(' ','_').replace(',','')] = item

for key, item in spatial_rep.primitive_indices.items():
    indicies_new[key.replace(' ','_').replace(',','')] = item

spatial_rep.primitives = primitives_new
spatial_rep.primitive_indices = indicies_new

wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())

# Manual surface identification
if False:
    for key in wing_primitive_names:
        surfaces = wing_primitive_names
        surfaces.remove(key)
        print(key)
        spatial_rep.plot(primitives=surfaces)

# make wing components
left_wing_names = []
left_wing_top_names = []
left_wing_bottom_names = []
left_wing_te_top_names = []
left_wing_te_bottom_names = []
for i in range(22+168,37+168):
    surf_name = 'Wing_1_' + str(i)
    left_wing_names.append(surf_name)
    if i%4 == 2:
        left_wing_te_bottom_names.append(surf_name)
    elif i%4 == 3:
        left_wing_bottom_names.append(surf_name)
    elif i%4 == 0:
        left_wing_top_names.append(surf_name)
    else:
        left_wing_te_top_names.append(surf_name)
#wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=list(spatial_rep.primitives.keys()))
wing_left = LiftingSurface(name='wing_left', spatial_representation=spatial_rep, primitive_names=left_wing_names)
wing_left_top = LiftingSurface(name='wing_left_top', spatial_representation=spatial_rep, primitive_names=left_wing_top_names)
wing_left_bottom = LiftingSurface(name='wing_left_bottom', spatial_representation=spatial_rep, primitive_names=left_wing_bottom_names)

# structural_left_wing_names = left_wing_names.copy()
structural_left_wing_names = []

# projections for internal structure
num_pts = 10
spar_rib_spacing_ratio = 3
num_rib_pts = 20

# Important points from openVSP
root_te = np.array([15.170, 0., 1.961]) * ft2m
root_le = np.array([8.800, 0, 1.989]) * ft2m
tip_te = np.array([11.300, -14.000, 1.978]) * ft2m
tip_le = np.array([8.796, -14.000, 1.989]) * ft2m



root_25 = (3*root_le+root_te)/4
root_75 = (root_le+3*root_te)/4
tip_25 = (3*tip_le+tip_te)/4
tip_75 = (tip_le+3*tip_te)/4

avg_spar_spacing = (np.linalg.norm(root_25-root_75)+np.linalg.norm(tip_25-tip_75))/2
half_span = root_le[1] - tip_le[1]
num_ribs = int(spar_rib_spacing_ratio*half_span/avg_spar_spacing)+1

f_spar_projection_points = np.linspace(root_25, tip_25, num_ribs)
r_spar_projection_points = np.linspace(root_75, tip_75, num_ribs)

rib_projection_points = np.linspace(f_spar_projection_points, r_spar_projection_points, num_rib_pts)

f_spar_top = wing_left_top.project(f_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)
f_spar_bottom = wing_left_bottom.project(f_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)

r_spar_top = wing_left_top.project(r_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)
r_spar_bottom = wing_left_bottom.project(r_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)

ribs_top = wing_left_top.project(rib_projection_points, direction=[0.,0.,1.], plot=do_plots, grid_search_n=100, force_reprojection=force_reprojection)
ribs_bottom = wing_left_bottom.project(rib_projection_points, direction=[0.,0.,1.], plot=do_plots, grid_search_n=100, force_reprojection=force_reprojection)


# make monolithic spars - makes gaps in internal structure, not good
if False:
    n_cp_spar = (num_ribs,2)
    order_spar = (2,)

    f_spar_points = np.zeros((num_ribs, 2, 3))
    f_spar_points[:,0,:] = f_spar_top.value
    f_spar_points[:,1,:] = f_spar_bottom.value
    f_spar_bspline = bsf.fit_bspline(f_spar_points, num_control_points=n_cp_spar, order = order_spar)
    f_spar = SystemPrimitive('front_spar', f_spar_bspline)
    spatial_rep.primitives[f_spar.name] = f_spar

    r_spar_points = np.zeros((num_ribs, 2, 3))
    r_spar_points[:,0,:] = r_spar_top.value
    r_spar_points[:,1,:] = r_spar_bottom.value
    r_spar_bspline = bsf.fit_bspline(r_spar_points, num_control_points=n_cp_spar, order = order_spar)
    r_spar = SystemPrimitive('rear_spar', r_spar_bspline)
    spatial_rep.primitives[r_spar.name] = r_spar

# make multi-patch spars - for coherence
n_cp = (2,2)
order = (2,)

for i in range(num_ribs-1):
    f_spar_points = np.zeros((2,2,3))
    f_spar_points[0,:,:] = ribs_top.value[0,(i,i+1),:]
    f_spar_points[1,:,:] = ribs_bottom.value[0,(i,i+1),:]
    f_spar_bspline = bsf.fit_bspline(f_spar_points, num_control_points=n_cp, order=order)
    f_spar = SystemPrimitive('f_spar_' + str(i), f_spar_bspline)
    spatial_rep.primitives[f_spar.name] = f_spar
    structural_left_wing_names.append(f_spar.name)

    r_spar_points = np.zeros((2,2,3))
    r_spar_points[0,:,:] = ribs_top.value[-1,(i,i+1),:]
    r_spar_points[1,:,:] = ribs_bottom.value[-1,(i,i+1),:]
    r_spar_bspline = bsf.fit_bspline(r_spar_points, num_control_points=n_cp, order=order)
    r_spar = SystemPrimitive('r_spar_' + str(i), r_spar_bspline)
    spatial_rep.primitives[r_spar.name] = r_spar
    structural_left_wing_names.append(r_spar.name)


# make ribs
n_cp_rib = (num_rib_pts,2)
order_rib = (2,)

for i in range(num_ribs):
    rib_points = np.zeros((num_rib_pts, 2, 3))
    rib_points[:,0,:] = ribs_top.value[:,i,:]
    rib_points[:,1,:] = ribs_bottom.value[:,i,:]
    rib_bspline = bsf.fit_bspline(rib_points, num_control_points=n_cp_rib, order = order_rib)
    rib = SystemPrimitive('rib_' + str(i), rib_bspline)
    spatial_rep.primitives[rib.name] = rib
    structural_left_wing_names.append(rib.name)


# region Generate structure mesh
#
# ## make additional geometry for meshing
# write_geometry = False
# save_file = '/pav_wing/pav_wing_v2_structue.iges'
#
# mesh_spatial_rep = SpatialRepresentation()
# # mesh_spatial_rep.primitives[f_spar.name] = f_spar
# # mesh_spatial_rep.primitives[r_spar.name] = r_spar
#
# # add ribs
# for i in range(num_ribs):
#     name = 'rib_' + str(i)
#     mesh_spatial_rep.primitives[name] = spatial_rep.primitives[name]
#
# # add spars
# for i in range(num_ribs-1):
#     name = 'f_spar_' + str(i)
#     mesh_spatial_rep.primitives[name] = spatial_rep.primitives[name]
#     name = 'r_spar_' + str(i)
#     mesh_spatial_rep.primitives[name] = spatial_rep.primitives[name]
#
# # make surface panels
# n_cp = (num_rib_pts,2)
# order = (2,)
#
# for i in range(num_ribs-1):
#     t_panel_points = ribs_top.value[:,(i,i+1),:]
#     t_panel_bspline = bsf.fit_bspline(t_panel_points, num_control_points=n_cp, order=order)
#     t_panel = SystemPrimitive('t_panel_' + str(i), t_panel_bspline)
#     mesh_spatial_rep.primitives[t_panel.name] = t_panel
#
#     b_panel_points = ribs_bottom.value[:,(i,i+1),:]
#     b_panel_bspline = bsf.fit_bspline(b_panel_points, num_control_points=n_cp, order=order)
#     b_panel = SystemPrimitive('b_panel_' + str(i), b_panel_bspline)
#     mesh_spatial_rep.primitives[b_panel.name] = b_panel
#
# mesh_spatial_rep.assemble()
# if do_plots:
#     mesh_spatial_rep.plot(plot_types=['wireframe'])
#
# if write_geometry:
#     mesh_spatial_rep.write_iges(cfile + save_file)
# ## At this point, use gmsh to generate a mesh using ^this^ .iges file
# endregion


# generate between-rib panels for structural optimization
# mesh_spatial_rep.primitives[f_spar.name] = f_spar
# mesh_spatial_rep.primitives[r_spar.name] = r_spar

# make surface panels
n_cp = (num_rib_pts,2)
order = (2,)

surface_dict = {}
for i in range(num_ribs-1):
    t_panel_points = ribs_top.value[:,(i,i+1),:]
    t_panel_bspline = bsf.fit_bspline(t_panel_points, num_control_points=n_cp, order=order)
    t_panel = SystemPrimitive('t_panel_' + str(i), t_panel_bspline)
    surface_dict[t_panel.name] = t_panel
    structural_left_wing_names.append(t_panel.name)

    b_panel_points = ribs_bottom.value[:,(i,i+1),:]
    b_panel_bspline = bsf.fit_bspline(b_panel_points, num_control_points=n_cp, order=order)
    b_panel = SystemPrimitive('b_panel_' + str(i), b_panel_bspline)
    surface_dict[b_panel.name] = b_panel
    structural_left_wing_names.append(b_panel.name)

surface_dict.update(spatial_rep.primitives)
spatial_rep.primitives = surface_dict

spatial_rep.assemble()

# Wing
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
wing = LiftingSurface(name='Wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
if debug_geom_flag:
    wing.plot()
sys_rep.add_component(wing)


if False:
    spatial_rep.plot(plot_types=['wireframe'])


# structural wing component:
wing_left_structural = LiftingSurface(name='wing_left_structural',
                                      spatial_representation=spatial_rep,
                                      primitive_names = structural_left_wing_names)
sys_rep.add_component(wing_left_structural)

### Non-wing stuff

# Horizontal tail
tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Stabilizer']).keys())
htail = cd.LiftingSurface(name='HTail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)
if debug_geom_flag:
    htail.plot()
sys_rep.add_component(htail)


# region Rotors
# Pusher prop
pp_disk_prim_names = list(spatial_rep.get_primitives(search_names=['PropPusher']).keys())
pp_disk = cd.Rotor(name='pp_disk', spatial_representation=spatial_rep, primitive_names=pp_disk_prim_names)
if debug_geom_flag:
    pp_disk.plot()
sys_rep.add_component(pp_disk)
# endregion


# region Actuations
# Tail FFD
htail_geometry_primitives = htail.get_geometry_primitives()
htail_ffd_bspline_volume = cd.create_cartesian_enclosure_volume(
    htail_geometry_primitives,
    num_control_points=(11, 2, 2), order=(4,2,2),
    xyz_to_uvw_indices=(1,0,2)
)
htail_ffd_block = cd.SRBGFFDBlock(name='htail_ffd_block',
                                  primitive=htail_ffd_bspline_volume,
                                  embedded_entities=htail_geometry_primitives)
htail_ffd_block.add_scale_v(name='htail_linear_taper',
                            order=2, num_dof=3, value=np.array([0., 0., 0.]),
                            cost_factor=1.)
htail_ffd_block.add_rotation_u(name='htail_twist_distribution',
                               connection_name='h_tail_act', order=1,
                               num_dof=1, value=np.array([np.deg2rad(1.75)]))
ffd_set = cd.SRBGFFDSet(
    name='ffd_set',
    ffd_blocks={htail_ffd_block.name : htail_ffd_block}
)
sys_param.add_geometry_parameterization(ffd_set)
sys_param.setup()
# endregion

# region Meshes
plots_flag = False

structure = True


# region Wing
num_wing_vlm = 41
num_chordwise_vlm = 5
point00 = np.array([8.796,  14.000,  1.989]) * ft2m # Right tip leading edge
point01 = np.array([11.300, 14.000,  1.989]) * ft2m # Right tip trailing edge
point10 = np.array([8.800,  0.000,   1.989]) * ft2m # Center Leading Edge
point11 = np.array([15.170, 0.000,   1.989]) * ft2m # Center Trailing edge
point20 = np.array([8.796,  -14.000, 1.989]) * ft2m # Left tip leading edge
point21 = np.array([11.300, -14.000, 1.989]) * ft2m # Left tip

leading_edge_points = np.concatenate(
    (np.linspace(point00, point10, int(num_wing_vlm/2+1))[0:-1,:],
     np.linspace(point10, point20, int(num_wing_vlm/2+1))),
    axis=0)
trailing_edge_points = np.concatenate(
    (np.linspace(point01, point11, int(num_wing_vlm/2+1))[0:-1,:],
     np.linspace(point11, point21, int(num_wing_vlm/2+1))),
    axis=0)

leading_edge = wing.project(leading_edge_points, direction=np.array([-1., 0., 0.]), plot=debug_geom_flag, force_reprojection=force_reprojection)
trailing_edge = wing.project(trailing_edge_points, direction=np.array([1., 0., 0.]), plot=debug_geom_flag, force_reprojection=force_reprojection)

# Chord Surface
wing_chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
if debug_geom_flag:
    spatial_rep.plot_meshes([wing_chord_surface])

# Upper and lower surface
wing_upper_surface_wireframe = wing.project(wing_chord_surface.value + np.array([0., 0., 0.5*ft2m]),
                                            direction=np.array([0., 0., -1.]), grid_search_n=25,
                                            plot=debug_geom_flag, max_iterations=200, force_reprojection=force_reprojection)
wing_lower_surface_wireframe = wing.project(wing_chord_surface.value - np.array([0., 0., 0.5*ft2m]),
                                            direction=np.array([0., 0., 1.]), grid_search_n=25,
                                            plot=debug_geom_flag, max_iterations=200, force_reprojection=force_reprojection)

# Chamber surface
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)
if debug_geom_flag:
    spatial_rep.plot_meshes([wing_camber_surface])

# OML mesh
wing_oml_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
wing_oml_mesh_name = 'wing_oml_mesh'
sys_rep.add_output(wing_oml_mesh_name, wing_oml_mesh)
if debug_geom_flag:
    spatial_rep.plot_meshes([wing_oml_mesh])
# endregion

# left wing only
num_wing_vlm = 21
num_chordwise_vlm = 5
point10 = root_le
point11 = root_te
point20 = tip_le
point21 = tip_te

leading_edge_points = np.linspace(point10, point20, num_wing_vlm)
trailing_edge_points = np.linspace(point11, point21, num_wing_vlm)

leading_edge = wing.project(leading_edge_points, direction=np.array([-1., 0., 0.]), plot=plots_flag, force_reprojection=force_reprojection)
trailing_edge = wing.project(trailing_edge_points, direction=np.array([-1., 0., 0.]), plot=plots_flag, force_reprojection=force_reprojection)

# Chord Surface
wing_chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
if plots_flag:
    spatial_rep.plot_meshes([wing_chord_surface])

# Upper and lower surface
wing_upper_surface_wireframe = wing.project(wing_chord_surface.value + np.array([0., 0., 0.5*ft2m]),
                                            direction=np.array([0., 0., -1.]), grid_search_n=25,
                                            plot=plots_flag, max_iterations=200, force_reprojection=force_reprojection)
wing_lower_surface_wireframe = wing.project(wing_chord_surface.value - np.array([0., 0., 0.5*ft2m]),
                                            direction=np.array([0., 0., 1.]), grid_search_n=25,
                                            plot=plots_flag, max_iterations=200, force_reprojection=force_reprojection)

# Chamber surface
left_wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
left_wing_vlm_mesh_name = 'left_wing_vlm_mesh'
sys_rep.add_output(left_wing_vlm_mesh_name, left_wing_camber_surface)

# OML mesh
left_wing_oml_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
left_wing_oml_mesh_name = 'left_wing_oml_mesh'
sys_rep.add_output(left_wing_oml_mesh_name, left_wing_oml_mesh)

sys_rep.add_output(name='left_wing_chord_distribution',
                                    quantity=am.norm(leading_edge-trailing_edge))
# endregion



# region Tail

num_htail_vlm = 13
num_chordwise_vlm = 5
point00 = np.array([20.713-4., 8.474+1.5, 0.825+1.5]) * ft2m # Right tip leading edge
point01 = np.array([22.916, 8.474, 0.825]) * ft2m # Right tip trailing edge
point10 = np.array([18.085, 0.000, 0.825]) * ft2m # Center Leading Edge
point11 = np.array([23.232, 0.000, 0.825]) * ft2m # Center Trailing edge
point20 = np.array([20.713-4., -8.474-1.5, 0.825+1.5]) * ft2m # Left tip leading edge
point21 = np.array([22.916, -8.474, 0.825]) * ft2m # Left tip trailing edge

leading_edge_points = np.linspace(point00, point20, num_htail_vlm)
trailing_edge_points = np.linspace(point01, point21, num_htail_vlm)

leading_edge = htail.project(leading_edge_points, direction=np.array([0., 0., -1.]), plot=debug_geom_flag, force_reprojection=force_reprojection)
trailing_edge = htail.project(trailing_edge_points, direction=np.array([1., 0., 0.]), plot=debug_geom_flag, force_reprojection=force_reprojection)

# Chord Surface
htail_chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
if debug_geom_flag:
    spatial_rep.plot_meshes([htail_chord_surface])

# Upper and lower surface
htail_upper_surface_wireframe = htail.project(htail_chord_surface.value + np.array([0., 0., 0.5 * ft2m]),
                                              direction=np.array([0., 0., -1.]), grid_search_n=25,
                                              plot=debug_geom_flag, max_iterations=200, force_reprojection=force_reprojection)
htail_lower_surface_wireframe = htail.project(htail_chord_surface.value - np.array([0., 0., 0.5 * ft2m]),
                                              direction=np.array([0., 0., 1.]), grid_search_n=25,
                                              plot=debug_geom_flag, max_iterations=200, force_reprojection=force_reprojection)

# Chamber surface
htail_camber_surface = am.linspace(htail_upper_surface_wireframe, htail_lower_surface_wireframe, 1)
htail_vlm_mesh_name = 'htail_vlm_mesh'
sys_rep.add_output(htail_vlm_mesh_name, htail_camber_surface)
if debug_geom_flag:
    spatial_rep.plot_meshes([htail_camber_surface])

# OML mesh
htail_oml_mesh = am.vstack((htail_upper_surface_wireframe, htail_lower_surface_wireframe))
htail_oml_mesh_name = 'htail_oml_mesh'
sys_rep.add_output(htail_oml_mesh_name, htail_oml_mesh)
if debug_geom_flag:
    spatial_rep.plot_meshes([htail_oml_mesh])
# endregion

if visualize_flag:
    spatial_rep.plot_meshes([left_wing_camber_surface, htail_camber_surface])


# region Sizing
pav_wt = PavMassProperties()
mass, cg, I = pav_wt.evaluate()

total_mass_properties = cd.TotalMassPropertiesM3L()
total_mass, total_cg, total_inertia = total_mass_properties.evaluate(mass, cg, I)
# endregion


process_gmsh = False


#############################################
if process_gmsh:
    file = '/pav_wing/pav_v2_gmsh_6492.msh'
    nodes, connectivity = caddee_import_mesh(cfile + file,
                                    spatial_rep,
                                    component = wing_left_structural,
                                    targets = list(wing_left_structural.get_primitives().values()),
                                    rescale=1e-3,
                                    remove_dupes=True,
                                    optimize_projection=False,
                                    tol=1e-8,
                                    plot=do_plots,
                                    grid_search_n=100)


    cells = [("quad",connectivity)]
    mesh = meshio.Mesh(nodes.value, cells)
    meshio.write(cfile + '/pav_wing/pav_wing_v2_caddee_mesh_' + str(nodes.shape[0]) + '_quad.msh', mesh, file_format='gmsh')
# exit()

#############################################

# filename = "./pav_wing/pav_wing_v2_caddee_mesh_SI_6307_quad.xdmf"
filename = "./pav_wing/pav_wing_v2_caddee_mesh_SI_2303_quad.xdmf"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    fenics_mesh = xdmf.read_mesh(name="Grid")
nel = fenics_mesh.topology.index_map(fenics_mesh.topology.dim).size_local
nn = fenics_mesh.topology.index_map(0).size_local

nodes = fenics_mesh.geometry.x


## make thickness function and function space
wing_spaces = {}
wing_t_spaces = {}
coefficients = {}
thickness_coefficients = {}
t = 0.005 # starting wing thickness control point values
for name in structural_left_wing_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
    space = lg.BSplineSpace(name=name,
                            order=(primitive.order_u, primitive.order_v),
                            control_points_shape=primitive.shape,
                            knots=(primitive.knots_u, primitive.knots_v))
    # print((primitive.control_points.shape[0], primitive.control_points.shape[1], 1))
    # print(name + '_t_coefficients')
    space_t = lg.BSplineSpace(name=name,
                            order=(primitive.order_u, primitive.order_v),
                            control_points_shape=(primitive.control_points.shape[0], primitive.control_points.shape[1], 1),
                            knots=(primitive.knots_u, primitive.knots_v))
    wing_spaces[name] = space
    wing_t_spaces[name] = space_t
    coefficients[name] = m3l.Variable(name = name + '_geo_coefficients', shape = primitive.control_points.shape, value = primitive.control_points)
    thickness_coefficients[name] = m3l.Variable(name = name + '_t_coefficients', shape = (primitive.control_points.shape[0], primitive.control_points.shape[1], 1), value = t * np.ones((primitive.control_points.shape[0], primitive.control_points.shape[1], 1)))

wing_space_m3l = m3l.IndexedFunctionSpace(name='wing_space', spaces=wing_spaces)
wing_t_space_m3l = m3l.IndexedFunctionSpace(name='wing_space', spaces=wing_t_spaces)
wing_geo = m3l.IndexedFunction('wing_geo', space=wing_space_m3l, coefficients=coefficients)
wing_thickness = m3l.IndexedFunction('wing_thickness', space = wing_t_space_m3l, coefficients=thickness_coefficients)


# make thickness mesh as m3l variable
# this will take ~4 minutes

if force_reprojection:
    file_name = '/pav_wing/pav_wing_v2_paneled_mesh_data_'+str(nodes.shape[0])+'.pickle'

    nodes_parametric = []

    targets = spatial_rep.get_primitives(structural_left_wing_names)
    num_targets = len(targets.keys())

    projected_points_on_each_target = []
    target_names = []
    # Project all points onto each target
    for target_name in targets.keys():
        target = targets[target_name]
        target_projected_points = target.project(points=nodes, properties=['geometry', 'parametric_coordinates'])
                # properties are not passed in here because we NEED geometry
        projected_points_on_each_target.append(target_projected_points)
        target_names.append(target_name)
    num_targets = len(target_names)
    distances = np.zeros(tuple((num_targets,)) + (nodes.shape[0],))
    for i in range(num_targets):
            distances[i,:] = np.linalg.norm(projected_points_on_each_target[i]['geometry'].value - nodes, axis=-1)
            # distances[i,:] = np.linalg.norm(projected_points_on_each_target[i]['geometry'].value - nodes.value, axis=-1)
    tol = 1e-2
    distances[abs(distances) < tol] = 0
    closest_surfaces_indices = np.argmin(distances, axis=0) # Take argmin across surfaces
    flattened_surface_indices = closest_surfaces_indices.flatten()
    for i in range(nodes.shape[0]):
        target_index = flattened_surface_indices[i]
        receiving_target_name = target_names[target_index]
        receiving_target = targets[receiving_target_name]
        u_coord = projected_points_on_each_target[target_index]['parametric_coordinates'][0][i]
        v_coord = projected_points_on_each_target[target_index]['parametric_coordinates'][1][i]
        node_parametric_coordinates = np.array([u_coord, v_coord])
        nodes_parametric.append((receiving_target_name, node_parametric_coordinates))
    with open(cfile + file_name, 'wb') as f:
        pickle.dump(nodes_parametric, f)
# exit()
with open(cfile + '/pav_wing/pav_wing_v2_paneled_mesh_data_'+str(nodes.shape[0])+'.pickle', 'rb') as f:
    nodes_parametric = pickle.load(f)

for i in range(len(nodes_parametric)):
    # print(nodes_parametric[i][0].replace(' ', '_').replace(',',''))
    nodes_parametric[i] = (nodes_parametric[i][0].replace(' ', '_').replace(',',''), np.array([nodes_parametric[i][1]]))


thickness_nodes = wing_thickness.evaluate(nodes_parametric)

shell_pde = ShellPDE(fenics_mesh)

#
# nu = 0.327
# # Aluminum 7050
# E = 6.9E10 # unit: Pa (N/m^2)
# h = 3E-3 # overall thickness (unit: m)
# rho = 2700
# f_d = -rho*h*9.81


# Unstiffened Aluminum 2024
# reference: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4
E = 73.1E9 # unit: Pa
nu = 0.33
h = 0.02*in2m # unit: m
rho = 2780 # unit: kg/m^3
f_d = -rho*h*9.81 # self-weight unit: N
tensile_yield_strength = 324E6 # unit: Pa
safety_factor = 1.5

y_bc = -1e-6
semispan = tip_te[1]

G = E/2/(1+nu)

#### Getting facets of the LEFT and the RIGHT edge  ####
DOLFIN_EPS = 3E-16
def ClampedBoundary(x):
    return np.greater(x[1], y_bc)
def RightChar(x):
    return np.less(x[1], semispan)
fdim = fenics_mesh.topology.dim - 1

ds_1 = createCustomMeasure(fenics_mesh, fdim, ClampedBoundary, measure='ds', tag=100)
dS_1 = createCustomMeasure(fenics_mesh, fdim, ClampedBoundary, measure='dS', tag=100)
dx_2 = createCustomMeasure(fenics_mesh, fdim+1, RightChar, measure='dx', tag=10)

g = Function(shell_pde.W)
with g.vector.localForm() as uloc:
     uloc.set(0.)

###################  m3l ########################

# create the shell dictionaries:
shells = {}
shells['wing_shell'] = {'E': E, 'nu': nu, 'rho': rho,# material properties
                        'dss': ds_1(100), # custom integrator: ds measure
                        'dSS': dS_1(100), # custom integrator: dS measure
                        'dxx': dx_2(10),  # custom integrator: dx measure
                        'g': g}
# Meshes definitions
# Wing VLM Mesh

plots_flag = False
################# PAV  Wing #################

# Wing shell Mesh
z_offset = 0.0
wing_shell_mesh = am.MappedArray(input=fenics_mesh.geometry.x).reshape((-1,3))
shell_mesh = rmshell.LinearShellMesh(
                    meshes=dict(
                    wing_shell_mesh=wing_shell_mesh,
                    ))

# # design scenario
design_scenario = cd.DesignScenario(name='aircraft_trim')

# region Cruise condition
cruise_model = m3l.Model()
cruise_condition = cd.CruiseCondition(name="cruise_1")
cruise_condition.atmosphere_model = cd.SimpleAtmosphereModel()
cruise_condition.set_module_input(name='altitude', val=600*ft2m)
cruise_condition.set_module_input(name='mach_number', val=0.145972)  # 112 mph = 0.145972 Mach = 50m/s
cruise_condition.set_module_input(name='range', val=80467.2)  # 50 miles = 80467.2 m
cruise_condition.set_module_input(name='pitch_angle', val=np.deg2rad(6), dv_flag=True, lower=np.deg2rad(-10), upper=np.deg2rad(10))
cruise_condition.set_module_input(name='flight_path_angle', val=0)
cruise_condition.set_module_input(name='roll_angle', val=0)
cruise_condition.set_module_input(name='yaw_angle', val=0)
cruise_condition.set_module_input(name='wind_angle', val=0)
cruise_condition.set_module_input(name='observer_location', val=np.array([0, 0, 600*ft2m]))

cruise_ac_states = cruise_condition.evaluate_ac_states()
cruise_model.register_output(cruise_ac_states)
# endregion

# region Propulsion
pusher_bem_mesh = BEMMesh(
    airfoil='NACA_4412',
    num_blades=5,
    num_radial=25,
    use_airfoil_ml=False,
    use_rotor_geometry=False,
    mesh_units='m',
    chord_b_spline_rep=True,
    twist_b_spline_rep=True
)
bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
bem_model.set_module_input('rpm', val=4000)
bem_model.set_module_input('propeller_radius', val=3.97727/2*ft2m)
bem_model.set_module_input('thrust_vector', val=np.array([1., 0., 0.]))
bem_model.set_module_input('thrust_origin', val=np.array([19.700, 0., 2.625]))
bem_model.set_module_input('chord_cp', val=np.linspace(0.2, 0.05, 4),
                           dv_flag=True,
                           upper=np.array([0.25, 0.25, 0.25, 0.25]), lower=np.array([0.05, 0.05, 0.05, 0.05]), scaler=1
                           )
bem_model.set_module_input('twist_cp', val=np.deg2rad(np.linspace(65, 15, 4)),
                           dv_flag=True,
                           lower=np.deg2rad(5), upper=np.deg2rad(85), scaler=1
                           )
bem_forces, bem_moments, _, _, _, _ = bem_model.evaluate(ac_states=cruise_ac_states)
cruise_model.register_output(bem_forces)
cruise_model.register_output(bem_moments)
# endregion

# region Inertial loads
inertial_loads_model = cd.InertialLoadsM3L(load_factor=1.)
inertial_forces, inertial_moments = inertial_loads_model.evaluate(total_cg_vector=total_cg, totoal_mass=total_mass, ac_states=cruise_ac_states)
cruise_model.register_output(inertial_forces)
cruise_model.register_output(inertial_moments)
# endregion

# region Aerodynamics (left wing only)
if structure:

    left_wing_vlm_model = VASTFluidSover(
        surface_names=[
            left_wing_vlm_mesh_name,
        ],
        surface_shapes=[
            (1, ) + left_wing_camber_surface.evaluate().shape[1:],
            ],
        fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
        mesh_unit='m',
        # cl0=[0.0, 0.0] # need to tune the coefficient
        cl0=[0.3475, 0.0] # need to tune the coefficient
    )
    left_wing_vlm_panel_forces, left_wing_vlm_forces, left_wing_vlm_moments  = left_wing_vlm_model.evaluate(ac_states=cruise_ac_states)
    cruise_model.register_output(left_wing_vlm_forces)
    cruise_model.register_output(left_wing_vlm_moments)
# endregion


# wing and h tail
vlm_model = VASTFluidSover(
    surface_names=[
        wing_vlm_mesh_name,
        htail_vlm_mesh_name,
    ],
    surface_shapes=[
        (1, ) + wing_camber_surface.evaluate().shape[1:],
        (1, ) + htail_camber_surface.evaluate().shape[1:],
        ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='m',
    cl0=[0.55, 0.0]
)
vlm_panel_forces, vlm_forces, vlm_moments  = vlm_model.evaluate(ac_states=cruise_ac_states)
cruise_model.register_output(vlm_forces)
cruise_model.register_output(vlm_moments)

# endregion


# Total loads
total_forces_moments_model = cd.TotalForcesMomentsM3L()
total_forces, total_moments = total_forces_moments_model.evaluate(
    inertial_forces, inertial_moments,
    vlm_forces, vlm_moments,
    bem_forces, bem_moments
)
cruise_model.register_output(total_forces)
cruise_model.register_output(total_moments)

# Equations of motions
eom_m3l_model = cd.EoMM3LEuler6DOF()
trim_residual = eom_m3l_model.evaluate(
    total_mass=total_mass,
    total_cg_vector=total_cg,
    total_inertia_tensor=total_inertia,
    total_forces=total_forces,
    total_moments=total_moments,
    ac_states=cruise_ac_states
)

cruise_model.register_output(trim_residual)


if structure:
    vlm_force_mapping_model = VASTNodalForces(
        surface_names=[
            left_wing_vlm_mesh_name,
        ],
        surface_shapes=[
            (1, ) + left_wing_camber_surface.evaluate().shape[1:],
        ],
        initial_meshes=[
            left_wing_camber_surface,
            ]
    )

    oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=left_wing_vlm_panel_forces, nodal_force_meshes=[left_wing_oml_mesh, left_wing_oml_mesh])
    wing_forces = oml_forces[0]

    shell_force_map_model = rmshell.RMShellForces(component=wing,
                                                    mesh=shell_mesh,
                                                    pde=shell_pde,
                                                    shells=shells)
    cruise_structural_wing_mesh_forces = shell_force_map_model.evaluate(
                            nodal_forces=wing_forces,
                            nodal_forces_mesh=left_wing_oml_mesh)

    shell_displacements_model = rmshell.RMShell(component=wing,
                                                mesh=shell_mesh,
                                                pde=shell_pde,
                                                shells=shells)

    cruise_structural_wing_mesh_displacements, cruise_structural_wing_mesh_rotations, wing_mass = \
                                    shell_displacements_model.evaluate(
                                        forces=cruise_structural_wing_mesh_forces,
                                        thicknesses=thickness_nodes)
    cruise_model.register_output(cruise_structural_wing_mesh_displacements)
    cruise_model.register_output(wing_mass)

# Add cruise m3l model to cruise condition
cruise_condition.add_m3l_model('cruise_model', cruise_model)
# Add design condition to design scenario
design_scenario.add_design_condition(cruise_condition)

system_model.add_design_scenario(design_scenario=design_scenario)

caddee_csdl_model = caddee.assemble_csdl()

system_model_name = 'system_model.aircraft_trim.cruise_1.cruise_1.'

caddee_csdl_model.create_input(name='h_tail_act', val=np.deg2rad(0.))
caddee_csdl_model.add_design_variable(dv_name='h_tail_act', lower=np.deg2rad(-10), upper=np.deg2rad(10), scaler=1.)

caddee_csdl_model.create_input(name='RPM', val=4000.)
caddee_csdl_model.add_design_variable(dv_name='RPM', lower=2000., upper=6000, scaler=1E-3)
caddee_csdl_model.connect('RPM',system_model_name+'pp_disk_bem_model.rpm')


if not structure:
    #### Trim cruise only
    caddee_csdl_model.add_objective(system_model_name+'euler_eom_gen_ref_pt.trim_residual')
    caddee_csdl_model.add_constraint(
        name=system_model_name+'pp_disk_bem_model.induced_velocity_model.eta',
        equals=0.8)
    caddee_csdl_model.add_constraint(
        name=system_model_name+'wing_vlm_meshhtail_vlm_mesh_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.L_over_D',
        equals=8.,
        scaler=1e-1
    )
else:
    #### structural sizing
    caddee_csdl_model.add_constraint(system_model_name+'euler_eom_gen_ref_pt.trim_residual', equals=0.)
    caddee_csdl_model.add_constraint(system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress',
                                    upper=tensile_yield_strength/safety_factor,scaler=1E-8)
    caddee_csdl_model.add_objective(system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass', scaler=1e-1)

    # Minimum thickness: 0.02 inch
    h_min = h
    h_spar = caddee_csdl_model.create_input('h_spar', val=h_min)
    caddee_csdl_model.add_design_variable('h_spar',
                                          lower=0.01 * in2m,
                                          upper=0.1 * in2m,
                                          scaler=1000,
                                          )
    h_skin = caddee_csdl_model.create_input('h_skin', val=h_min)
    caddee_csdl_model.add_design_variable('h_skin',
                                          lower=0.01 * in2m,
                                          upper=0.1* in2m,
                                          scaler=1000,
                                          )
    h_rib = caddee_csdl_model.create_input('h_rib', val=h_min)
    caddee_csdl_model.add_design_variable('h_rib',
                                          lower=0.01 * in2m,
                                          upper=0.1 * in2m,
                                          scaler=1000,
                                          )
    # h_spar, h_skin, h_rib = 0.003, 0.001, 0.002
    i = 0
    # skin_shape = (625, 1)
    spar_shape = (4, 1)
    rib_shape = (40, 1)
    skin_shape = rib_shape
    shape = (4, 1)
    # valid_wing_surf = [23, 24, 27, 28, 31, 32, 35, 36]
    # valid_structural_left_wing_names = []
    # for name in structural_left_wing_names:
    #     if "spar" in name:
    #         valid_structural_left_wing_names.append(name)
    #     elif "rib" in name:
    #         valid_structural_left_wing_names.append(name)
    #     elif "Wing" in name:
    #         for id in valid_wing_surf:
    #             if str(id+168) in name:
    #                 valid_structural_left_wing_names.append(name)

    valid_structural_left_wing_names = structural_left_wing_names
    # print("Full list of surface names for left wing:", structural_left_wing_names)
    # print("Valid list of surface names for left wing:", svalid_structural_left_wing_names)
    for name in valid_structural_left_wing_names:
        primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
        name = name
        surface_id = i
        if "spar" in name:
            shape = spar_shape
            h_init = h_spar
        elif "panel" in name:
            shape = skin_shape
            h_init = h_skin
        elif "rib" in name:
            shape = rib_shape
            h_init = h_rib

        # h_i = caddee_csdl_model.create_input('wing_thickness_'+name, val=h_init)
        # caddee_csdl_model.register_output('wing_thickness_surface_'+name, csdl.expand(h_i, shape))

        caddee_csdl_model.register_output('wing_thickness_surface_'+name, csdl.expand(h_init, shape))
        caddee_csdl_model.connect('wing_thickness_surface_'+name,
                                    system_model_name+'wing_thickness_evaluation.'+\
                                    name+'_t_coefficients')
        i += 1
#################### end of m3l ########################

# region Optimization Setup


sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()


print('C_L (left wing): ', sim['system_model.aircraft_trim.cruise_1.cruise_1.left_wing_vlm_mesh_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.left_wing_vlm_mesh_C_L'])
print('Total lift (left wing): ', sim['system_model.aircraft_trim.cruise_1.cruise_1.left_wing_vlm_mesh_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.total_lift'])
print('Total lift (whole wing + Tail): ', sim['system_model.aircraft_trim.cruise_1.cruise_1.wing_vlm_meshhtail_vlm_mesh_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.total_lift'])
print('Total forces: ', sim[system_model_name+'euler_eom_gen_ref_pt.total_forces'])
print('Total moments:', sim[system_model_name+'euler_eom_gen_ref_pt.total_moments'])
print('Total mass: ', sim[system_model_name+'total_constant_mass_properties.total_mass'])
# sim.check_totals(of=[system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress'],
#                                     wrt=['h_spar', 'h_skin', 'h_rib'])

# sim.check_totals(of=[system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass'],
#                                     wrt=['h_spar', 'h_skin', 'h_rib'])
########################### Run optimization ##################################
# prob = CSDLProblem(problem_name='pav', simulator=sim)
#
# optimizer = SLSQP(prob, maxiter=50, ftol=1E-5)
# # from modopt.snopt_library import SNOPT
# # optimizer = SNOPT(prob,
# #                   Major_iterations = 10,
# #                   Major_optimality = 1e-5,
# #                   append2file=False)
#
# optimizer.solve()
# optimizer.print_results()


print('Trim residual: ', sim[system_model_name+'euler_eom_gen_ref_pt.trim_residual'])
print('Trim forces: ', sim[system_model_name+'euler_eom_gen_ref_pt.total_forces'])
print('Trim moments:', sim[system_model_name+'euler_eom_gen_ref_pt.total_moments'])
print('Pitch: ', np.rad2deg(
    sim[system_model_name+'cruise_1_ac_states_operation.cruise_1_pitch_angle']))
print('RPM: ', sim[system_model_name+'pp_disk_bem_model.rpm'])
print('Horizontal tail actuation: ',
      np.rad2deg(sim['system_parameterization.ffd_set.rotational_section_properties_model.h_tail_act']))

if structure:
    ####### Structural output
    print("="*20+'structue outputs'+"="*20)
    # Comparing the solution to the Kirchhoff analytical solution
    f_shell = sim[system_model_name+'Wing_rm_shell_force_mapping.wing_shell_forces']
    f_vlm = sim[system_model_name+'left_wing_vlm_mesh_vlm_force_mapping_model.left_wing_vlm_mesh_oml_forces'].reshape((-1,3))
    u_shell = sim[system_model_name+'Wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
    # u_nodal = sim['Wing_rm_shell_displacement_map.wing_shell_nodal_displacement']
    uZ = u_shell[:,2]
    # uZ_nodal = u_nodal[:,2]


    wing_tip_compliance = sim[system_model_name+'Wing_rm_shell_model.rm_shell.compliance_model.compliance']
    wing_mass = sim[system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass']
    wing_elastic_energy = sim[system_model_name+'Wing_rm_shell_model.rm_shell.elastic_energy_model.elastic_energy']
    wing_aggregated_stress = sim[system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress']
    wing_von_Mises_stress = sim[system_model_name+'Wing_rm_shell_model.rm_shell.von_Mises_stress_model.von_Mises_stress']
    ########## Output: ##########
    print("Spar, rib, skin thicknesses:", sim['h_spar'], sim['h_rib'], sim['h_skin'])

    fz_func = Function(shell_pde.VT)
    fz_func.x.array[:] = f_shell[:,-1]

    fx_func = Function(shell_pde.VT)
    fx_func.x.array[:] = f_shell[:,0]

    fy_func = Function(shell_pde.VT)
    fy_func.x.array[:] = f_shell[:,1]

    dummy_func = Function(shell_pde.VT)
    dummy_func.x.array[:] = 1.0
    print("vlm forces:", sum(f_vlm[:,0]),sum(f_vlm[:,1]),sum(f_vlm[:,2]))
    print("shell forces:", dolfinx.fem.assemble_scalar(form(fx_func*ufl.dx)),
                            dolfinx.fem.assemble_scalar(form(fy_func*ufl.dx)),
                            dolfinx.fem.assemble_scalar(form(fz_func*ufl.dx)))

    print("Wing surface area:", dolfinx.fem.assemble_scalar(form(dummy_func*ufl.dx)))
    print("Wing tip deflection (m):",max(abs(uZ)))
    print("Wing tip compliance (= tip deflection^3/2 m^3):",wing_tip_compliance)
    print("Wing total mass (kg):", wing_mass)
    print("Wing aggregated von Mises stress (Pascal):", wing_aggregated_stress)
    print("Wing maximum von Mises stress (Pascal):", max(wing_von_Mises_stress))
    print("  Number of elements = "+str(nel))
    print("  Number of vertices = "+str(nn))


# ######## Visualization: ##############
import vedo
# #
# plotter = vedo.Plotter()
# wing_shell_mesh_plot = vedo.Points(wing_shell_mesh.value.reshape((-1,3)))
# wing_oml_plot = vedo.Points(wing_oml_mesh.value.reshape((-1,3)))
# plotter.show([wing_shell_mesh_plot, wing_oml_plot], interactive=True, axes=1)    # Plotting point cloud
#
# plotter = vedo.Plotter()
# wing_oml_plot = vedo.Points(wing_oml_mesh.value.reshape((-1,3)))
# plotter.show([wing_oml_plot], interactive=True, axes=1)    # Plotting point cloud

# plotter = vedo.Plotter()
# wing_shell_mesh_plot = vedo.Points(wing_shell_mesh.value.reshape((-1,3)))
# wing_oml_plot = vedo.Points(left_wing_oml_mesh.value.reshape((-1,3)))
# plotter.show([wing_shell_mesh_plot, wing_oml_plot], interactive=True, axes=1)    # Plotting point cloud
