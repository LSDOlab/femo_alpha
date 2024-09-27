import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update(plt.rcParamsDefault)
import lsdo_geo
import csdl_alpha as csdl
import pickle
import lsdo_function_spaces as lfs

recorder = csdl.Recorder(inline=True, debug=False)
recorder.start()

# from mbd_trial.framework_trial.examples.n_body_flexible_pendulum.n_body_flexible_pendulum_model import NBodyPendulumModel, NBodyPendulumStateData
# from mbd_trial.framework_trial.examples.n_body_flexible_pendulum.n_body_flexible_pendulum_model import PendulumBody
# from mbd_trial.framework_trial.examples.n_body_flexible_pendulum.n_body_flexible_pendulum_model import PendulumSystem

from mbd_trial.framework_trial.examples.n_body_flexible_pendulum_shell.n_body_flexible_pendulum_shell_model import NBodyPendulumModel, NBodyPendulumStateData
from mbd_trial.framework_trial.examples.n_body_flexible_pendulum_shell.n_body_flexible_pendulum_shell_model import PendulumBody
from mbd_trial.framework_trial.examples.n_body_flexible_pendulum_shell.n_body_flexible_pendulum_shell_model import PendulumSystem




# geo = lsdo_geo.import_geometry('mbd_trial/framework_trial/examples/n_body_pendulum_with_geometry/pendulum_geometry.stp')
# geo.plot()

num_coefficients_v = 21
num_coefficients_u = 3
space_of_linear_25_cp_b_spline_surfaces = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2,2), 
                                                        coefficients_shape=(num_coefficients_u,num_coefficients_v))

coefficients_line_v = np.linspace(-0.5, 0.5, num_coefficients_v)
coefficients_line_u = np.linspace(-0.05, 0.05, num_coefficients_u)
coefficients_z, coefficients_x = np.meshgrid(coefficients_line_v,coefficients_line_u)
coefficients_y = np.zeros((num_coefficients_u,num_coefficients_v))
coefficients = np.stack((coefficients_x, np.zeros((num_coefficients_u,num_coefficients_v)), coefficients_z), axis=-1)

coefficients = csdl.Variable(value=coefficients.reshape((num_coefficients_u,num_coefficients_v,3)))
b_spline = lfs.Function(space=space_of_linear_25_cp_b_spline_surfaces, coefficients=coefficients)
pendulum_geometry = lsdo_geo.Geometry(functions={0: b_spline})
# pendulum_geometry.plot()
# b_spline.plot()
# exit()
plot = True
make_video = True
save_stuff = True
run_stuff = True

pendulum_system = PendulumSystem()

num_bodies = 1

# Make bodies
for i in range(num_bodies):
    pendulum = PendulumBody(geometry=pendulum_geometry, 
                            mass=csdl.Variable(value=np.array([1.])), 
                            length=csdl.Variable(value=np.array([1.])))
    pendulum_system.pendulums.append(pendulum)

    # Add constraints
    if i == 0:
        # pendulum.geometry.plot()
        point_on_top = pendulum.geometry.project(np.array([0., 0., 0.5]))
        constraint_pair = (point_on_top, np.array([0., 0., 0.5]))
    else:
        point_on_top = pendulum.geometry.project(np.array([0., 0., 0.5]), plot=False)
        point_on_bottom_of_previous_body = pendulum_system.pendulums[i-1].geometry.project(np.array([0., 0., -0.5]), plot=False)
        constraint_pair = (point_on_top, point_on_bottom_of_previous_body)

    pendulum_system.constraint_pairs.append(constraint_pair)

    # pendulum_geometry.plot()
    # exit()

'''
# recommended rho_infinity values:
# 0.82 for 1,2 bodies
# 0.7 for 3,4 bodies
# 0.6 for 5,6 bodies
# Not sure after that
'''

# Create a two-body pendulum model=
rho_infinity = 0.5
# time_step = 1.e-2
time_step = 1.e-3
model = NBodyPendulumModel(pendulum_system=pendulum_system, time_step=time_step, rho_infinity=rho_infinity)


def generate_initial_states(thetas):
    '''
    This function helps generates intial states for the n-body pendulum that satisfy the constraints.
    '''
    num_rigid_body_states = 6
    nel_x = 2
    nel_z = 10
    num_shell_nodes = (nel_x+1)*(nel_z+1)
    num_flexible_states = (2*nel_x+1)*(2*nel_z+1)*3 + num_shell_nodes*3
    initial_states = {}
    initial_state_derivatives = {}
    for i, pendulum in enumerate(pendulum_system.pendulums):
        delta_y = pendulum.length*np.sin(thetas[i])
        delta_z = -pendulum.length*np.cos(thetas[i])
        if i == 0:
            # y = delta_y
            y = delta_y*0.5
            # z = 0.5 + delta_z
            z = 0.5 + delta_z*0.5
            # z = delta_z*0.5
        else:
            # y = initial_states[f'pendulum{i-1}']['rigid_body_states'][1] + delta_y
            # z = initial_states[f'pendulum{i-1}']['rigid_body_states'][2] + delta_z
            y = initial_states[f'pendulum{i-1}']['rigid_body_states'][1] + pendulum.length*np.sin(thetas[i-1])*0.5 + delta_y*0.5
            z = initial_states[f'pendulum{i-1}']['rigid_body_states'][2] - pendulum.length*np.cos(thetas[i-1])*0.5 + delta_z*0.5
        
        initial_states[f'pendulum{i}'] = {}
        initial_states[f'pendulum{i}']['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states,), value=0.)
        initial_states[f'pendulum{i}']['rigid_body_states'] = initial_states[f'pendulum{i}']['rigid_body_states'].set(csdl.slice[1], y)
        initial_states[f'pendulum{i}']['rigid_body_states'] = initial_states[f'pendulum{i}']['rigid_body_states'].set(csdl.slice[2], z)
        initial_states[f'pendulum{i}']['rigid_body_states'] = initial_states[f'pendulum{i}']['rigid_body_states'].set(csdl.slice[3], thetas[i])

        initial_state_derivatives[f'pendulum{i}'] = {}
        initial_state_derivatives[f'pendulum{i}']['rigid_body_states'] = np.zeros(2*num_rigid_body_states)

        # initial_flexible_states = np.loadtxt('./disp_solid.txt')
        # initial_states[f'pendulum{i}']['flexible_states'] = csdl.Variable(shape=(num_flexible_states,), value=initial_flexible_states)
        initial_states[f'pendulum{i}']['flexible_states'] = csdl.Variable(shape=(num_flexible_states,), value=0.)
        initial_state_derivatives[f'pendulum{i}']['flexible_states'] = np.zeros(2*num_flexible_states)


    # initial_states['lagrange_multipliers'] = np.zeros(2*num_bodies)
    # initial_state_derivatives['lagrange_multipliers'] = np.zeros(2*num_bodies)

    initial_state_data = NBodyPendulumStateData(states=initial_states, state_derivatives=initial_state_derivatives)
    return initial_state_data
        

thetas = np.zeros(num_bodies)
# thetas[0] = 0.
thetas[0] = np.pi/2
# thetas[0] = 3*np.pi/4
# thetas[1] = 3*np.pi/4
# thetas[2] = 3*np.pi/4
initial_state_data = generate_initial_states(thetas)

t_initial = 0.
# t_final = 5.
t_final = 0.05
# t_final = 0.1
# t_final = 1.
tolerance = 1.e-8
max_iterations = 10

recorder.inline = True

from time import time as timer
t1 = timer()
t, states, state_derivatives, lagrange_multipliers = model.evaluate(initial_state_data, t_initial, t_final, time_step)
t2 = timer()
jax_inputs = []
jax_inputs.append(pendulum_system.pendulums[0].mass)
jax_inputs.append(pendulum_system.pendulums[0].length)

additional_outputs = []
for body in model.pendulum_system.pendulums:
    additional_outputs.append(states[body.name]['rigid_body_states'])
    additional_outputs.append(state_derivatives[body.name]['rigid_body_states'])
    additional_outputs.append(states[body.name]['flexible_states'])
    additional_outputs.append(state_derivatives[body.name]['flexible_states'])
additional_outputs = additional_outputs + lagrange_multipliers['physical_constraints']
additional_outputs = additional_outputs + lagrange_multipliers['structural_constraints']

jax_sim = csdl.experimental.JaxSimulator(
    recorder = recorder,
    # additional_inputs = list(initial_state_data.states.values()),
    additional_inputs=jax_inputs,
    # additional_outputs = list(states.values()['rigid_body_states']) + list(state_derivatives.values()['rigid_body_states']) \
    #                     + list(states.values()['flexible_states']) + list(state_derivatives.values()['flexible_states']) \
    #                     + lagrange_multipliers,
    additional_outputs=additional_outputs,
    gpu=False
)
# recorder.print_largest_variables(n = 50)
# exit()

if run_stuff:

    t3 = timer()
    jax_sim.run()
    t4 = timer()
    jax_sim.run()
    t5 = timer()

    print(f'Python time: {t2-t1}')
    print(f'JAX 1st time: {t4-t3}')
    print(f'JAX Nth time: {t5-t4}')
recorder.inline = True
# exit()


if save_stuff and run_stuff:
    for body in model.pendulum_system.pendulums:
        states[body.name]['rigid_body_states'] = states[body.name]['rigid_body_states'].value
        state_derivatives[body.name]['rigid_body_states'] = state_derivatives[body.name]['rigid_body_states'].value
        states[body.name]['flexible_states'] = states[body.name]['flexible_states'].value
        state_derivatives[body.name]['flexible_states'] = state_derivatives[body.name]['flexible_states'].value

    for key, lagrange_multiplier_list in lagrange_multipliers.items():
        for i, lagrange_multiplier in enumerate(lagrange_multiplier_list):
            lagrange_multipliers[key][i] = lagrange_multiplier.value


    # file_path = 'mbd_trial/framework_trial/examples/n_body_flexible_pendulum/'
    file_path = './pickle_files/'
    file_name = file_path + "t.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = file_path + "states.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = file_path + "state_derivatives.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(state_derivatives, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = file_path + "lagrange_multipliers.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(lagrange_multipliers, handle, protocol=pickle.HIGHEST_PROTOCOL)

load_stuff = True
if load_stuff:
    # file_path = 'mbd_trial/framework_trial/examples/n_body_flexible_pendulum/'
    file_path = './pickle_files/'
    file_name = file_path + "t.pickle"
    with open(file_name, 'rb') as handle:
        t = pickle.load(handle)
    file_name = file_path + "states.pickle"
    with open(file_name, 'rb') as handle:
        states = pickle.load(handle)
    file_name = file_path + "state_derivatives.pickle"
    with open(file_name, 'rb') as handle:
        state_derivatives = pickle.load(handle)
    file_name = file_path + "lagrange_multipliers.pickle"
    with open(file_name, 'rb') as handle:
        lagrange_multipliers = pickle.load(handle)

if save_stuff or load_stuff:
    # Convert the states and state derivatives to csdl.Variables
    for body in model.pendulum_system.pendulums:
        states[body.name]['rigid_body_states'] = csdl.Variable(value=states[body.name]['rigid_body_states'])
        state_derivatives[body.name]['rigid_body_states'] = csdl.Variable(value=state_derivatives[body.name]['rigid_body_states'])
        states[body.name]['flexible_states'] = csdl.Variable(value=states[body.name]['flexible_states'])
        state_derivatives[body.name]['flexible_states'] = csdl.Variable(value=state_derivatives[body.name]['flexible_states'])

    # for key, lagrange_multiplier_list in lagrange_multipliers.items():
        # for i, lagrange_multiplier in enumerate(lagrange_multiplier_list):
        #     lagrange_multipliers[key][i] = csdl.Variable(value=lagrange_multiplier)
    lagrange_multipliers['physical_constraints'] = [csdl.Variable(value=lagrange_multiplier) for lagrange_multiplier in lagrange_multipliers['physical_constraints']]

# r0 = f(x0)
# r1 = f(x1)

# X0[0] = x0
# X01[1] = x1
# R01 = F(X01)
# r1 = F01[1]
# r0 = F01[0]



state_velocities = {}
state_accelerations = {}
for body in model.pendulum_system.pendulums:
    state_velocities[body.name] = state_derivatives[body.name]['rigid_body_states'].value[:,:6]
    state_accelerations[body.name] = state_derivatives[body.name]['rigid_body_states'].value[:,6:]

if plot:

    # calculate energy through the simulation
    energy = np.zeros(len(t))
    for body in model.pendulum_system.pendulums:
        kinetic_energy = 0.5*body.mass.value*np.linalg.norm(state_velocities[body.name][:,:3], axis=1)**2
        potential_energy = body.mass.value*model.g*states[body.name]['rigid_body_states'].value[:,2]
        energy += kinetic_energy[:-1] + potential_energy[:-1]

    # plot the results
    plt.figure()
    for i, body in enumerate(model.pendulum_system.pendulums):
        plt.plot(t,states[body.name]['rigid_body_states'].value[:-1,3],linewidth=2, label=f'Angular Displacement {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('$\\theta_{1}$,$\\theta_{2}$')
    plt.legend()
    plt.title('Angular Displacements')
    plt.grid(True)

    plt.figure()
    for i, body in enumerate(model.pendulum_system.pendulums):
        plt.plot(t,state_velocities[body.name][:-1,3],linewidth=2, label=f'Angular Velocity {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('$\dot{\\theta_{1}}$,$\dot{\\theta_{2}}$')
    plt.legend()
    plt.title('Angular Velocities')
    plt.grid(True)

    plt.figure()
    for i, body in enumerate(model.pendulum_system.pendulums):
        plt.plot(t,state_accelerations[body.name][:-1,3],linewidth=2, label=f'Angular Acceleration {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('$\ddot{\\theta_{1}}$,$\ddot{\\theta_{2}}$')
    plt.legend()
    plt.title('Angular Accelerations')
    plt.grid(True)

    plt.figure()
    for i, body in enumerate(model.pendulum_system.pendulums):
        plt.plot(states[body.name]['rigid_body_states'].value[:-1,1],states[body.name]['rigid_body_states'].value[:-1,2],linewidth=2, label=f'Mass {i+1}')
    plt.xlabel('$x_{1},x_{2}$')
    plt.ylabel('$y_{1},y_{2}$')
    plt.legend()
    plt.grid(True)
    plt.title('Mass Trajectories')

    # for i, body in enumerate(model.pendulum_system.pendulums):  # whatever, number of bodies = number of physical connections
    #     plt.figure()
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,0],linewidth=2, label=f'Lagrange Multiplier {2*i+1}')
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,1],linewidth=2, label=f'Lagrange Multiplier {2*i+2}')
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,2],linewidth=2, label=f'Lagrange Multiplier {2*i+3}')
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,3],linewidth=2, label=f'Lagrange Multiplier {2*i+4}')
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,4],linewidth=2, label=f'Lagrange Multiplier {2*i+5}')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('$\lambda_{1}$,$\lambda_{2}$')
    #     plt.legend()
    #     plt.title('Lagrange Multipliers For Body ' + str(i+1))
    #     plt.grid(True)

    plt.figure()
    plt.plot(t,energy,'k-',linewidth=2, label='Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Energy')
    plt.grid(True)

    plt.show()

if make_video:

    import vedo
    # folder_path = 'mbd_trial/framework_trial/examples/n_body_flexible_pendulum/videos/'
    folder_path = './videos/'
    file_name = f'{num_bodies}_body_pendulum.mp4'
    video = vedo.Video(folder_path + file_name, duration=t_final, backend="cv")
    # pendulum_colors = np.random.rand(num_bodies, 3)*255
    # dot_colors = 0.5*pendulum_colors[:-1,:] + 0.5*pendulum_colors[1:,:]
    # dot_colors = np.insert(dot_colors, 0, np.zeros(3), axis=0)

    total_length = 0
    for body in pendulum_system.pendulums:
        total_length += body.length.value
    bounding_size = total_length
    # ax = vedo.Axes(xrange=(-bounding_size,bounding_size), yrange=(-bounding_size,lengths['pendulum1']*0.8), htitle=__doc__)
    # ax = vedo.Axes(yrange=(-bounding_size,bounding_size), zrange=(-bounding_size-bounding_size/2,bounding_size-bounding_size/2), htitle=__doc__)

    camera = {
        # 'pos': (4*num_bodies + 1, 0, -1*num_bodies/2),
        'pos': (5*num_bodies + 1, 0, -1*num_bodies/2 + 1/4),
        # 'pos': (5*num_bodies + 1, 0, 0),
        'focalPoint': (0, 0, -1*num_bodies/2 + 1/4),
        # 'focalPoint': (0, 0, 0),
        'viewup': (0, 0, 1),
        'distance': 3,
    }

    for body in model.pendulum_system.pendulums:
        for function in body.geometry.functions.values():
            function.coefficients = function.coefficients.value

    # for i in range(len(t)):
    skipping_rate = 2
    for i in np.arange(0, len(t), skipping_rate):
        print(f'{i}/{len(t)}')

        plotting_elements = []
        for body in model.pendulum_system.pendulums:
            body.geometry = body.design_geometry.copy()
            body.apply_rigid_body_motion(states[body.name]['rigid_body_states'].value[i,:])
            shell_mesh = body.geometry.evaluate_representations(body.shell_representation, plot=False)
            body.apply_flexible_motion(states[body.name]['flexible_states'].value[i,:],
                                       states[body.name]['rigid_body_states'].value[i,:])
            print('rigid body states:', states[body.name]['rigid_body_states'].value[i,:])
            # print('flexible states: \n', states[body.name]['flexible_states'].value[i,:].reshape((-1,6)))
            # plotting_elements = body.geometry.plot(additional_plotting_elements=plotting_elements, show=False)
            plotting_elements = body.geometry.plot_meshes(meshes=[shell_mesh], function_opacity=0.5, additional_plotting_elements=plotting_elements, show=False)

            # pendulum_tops[j,0] = states[model.bodies[j]].value[i,0] - lengths[f'pendulum{j+1}'].value*np.sin(states[model.bodies[j]].value[i,2])
            # pendulum_tops[j,1] = states[model.bodies[j]].value[i,1] + lengths[f'pendulum{j+1}'].value*np.cos(states[model.bodies[j]].value[i,2])
            # pendulum_bottoms[j,0] = states[model.bodies[j]].value[i,0]
            # pendulum_bottoms[j,1] = states[model.bodies[j]].value[i,1]

        # plotting_elements = []
        # for j in range(num_bodies):
        #     # pendulum = vedo.Cylinder(pos=(pendulum_tops[j,0], pendulum_tops[j,1], 0), r=0.05, height=lengths[f'pendulum{j+1}'], c=(0,0,1))
        #     pendulum_line = vedo.Line(pendulum_tops[j], pendulum_bottoms[j]).linewidth(5).color(pendulum_colors[j])
        #     pendulum_top_and_bot = vedo.Points([pendulum_tops[j], pendulum_bottoms[j]]).c(dot_colors[j]).point_size(10)
        #     # pendulum.rotate_z(states[bodies[j]][i,2], rad=True)
        #     plotting_elements.append(pendulum_line)
        #     plotting_elements.append(pendulum_top_and_bot)

        # plotter = vedo.Plotter(size=(3200,2000),offscreen=True)
        plotter = vedo.Plotter(offscreen=True)
        # plotter = vedo.Plotter(offscreen=True)
        plotter.show(plotting_elements, camera=camera, axes=1)
        # plotter.show(plotting_elements, axes=0)
        # plotter.show(plotting_elements, axes=1, viewup='z')
        # plotter.show(plotting_elements, ax, viewup='z')

        video.add_frame()
    video.close()
