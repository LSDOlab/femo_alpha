import dolfinx
from dolfinx.fem import Function
import csdl_alpha as csdl
import ufl
import numpy as np
from mpi4py import MPI

from femo_alpha.fea.fea_dolfinx import FEA
from femo_alpha.fea.utils_dolfinx import createCustomMeasure, convertToDense
from femo_alpha.rm_shell.rm_shell_pde import RMShellPDE
from femo_alpha.csdl_alpha_opt.fea_model import FEAModel

class RMShellModel:
    '''
    Class for the RM shell model for aircraft optimization
    ------------------------------------------------------
    Args:
    mesh: dolfinx.mesh object for the shell mesh
    shell_bc_func: callable for shell Dirichlet BC locations - returns True if 
                    it is the boundary location, otherwise returns False
    record: boolean to record the FEA model variables in xdmf format
    '''
    def __init__(self, mesh: dolfinx.mesh, 
                            shell_bc_func: callable=None, 
                            record=True):
        self.mesh = mesh
        self.shell_bc_func = shell_bc_func # shell bc information
        self.record = record
        self.m, self.rho = 1e-6, 100

        if shell_bc_func is not None:
            self.set_up_bcs(shell_bc_func)
        else:
            raise ValueError('Please provide the shell bc location function.\n \
                             Example:\n \
                             def ClampedBoundary(x):\n \
                                return np.less(x[1], 0.0)')
        self.set_up_fea()

    def set_up_bcs(self, bc_locs_func): 
        '''
        Set up the boundary conditions for the shell model and the tip displacement
        ** helper function for aircraft optimization with clamped root bc **
        '''
        mesh = self.mesh

        fdim = mesh.topology.dim - 1

        ds_1 = createCustomMeasure(mesh, fdim, bc_locs_func, measure='ds', tag=100)
        dS_1 = createCustomMeasure(mesh, fdim, bc_locs_func, measure='dS', tag=100)

        self.dss = ds_1(100) # custom ds measure for the Dirichlet BC
        self.dSS = dS_1(100) # custom ds measure for the Dirichlet BC

    def set_up_fea(self):
        '''
        Set up the FEMO FEA model for RM shell analysis
        '''
        print('-'*40)
        print('Setting up the FEA model for RM shell analysis ...')
        mesh = self.mesh
        shell_pde = self.shell_pde = RMShellPDE(mesh)
        dss = self.dss
        dSS = self.dSS

        PENALTY_BC = True

        fea = FEA(mesh)
        fea.PDE_SOLVER = 'Newton'
        fea.REPORT = False
        fea.record = self.record
        fea.linear_problem = True
        # Add input to the PDE problem:
        h = Function(shell_pde.VT)
        f = Function(shell_pde.VF)
        E = Function(shell_pde.VT)
        nu = Function(shell_pde.VT)
        density = Function(shell_pde.VT)
        uhat = Function(shell_pde.VF)

        # Add state to the PDE problem:
        w_space = shell_pde.W
        w = Function(w_space)

        # Simple isotropic material
        g = Function(shell_pde.W)
        with g.vector.localForm() as uloc:
            uloc.set(0.)
        residual_form = shell_pde.pdeRes(h=h, # thickness
                                         w=w, # displacement
                                         uhat=uhat, # mesh displacement
                                         f=f, # force
                                         E=E, # Young's modulus
                                         nu=nu, # Poisson ratio
                                         penalty=PENALTY_BC, 
                                         dss=dss, dSS=dSS, g=g)

        # Add output to the PDE problem:
        u_mid, theta = ufl.split(w)
        compliance_form = shell_pde.compliance(u_mid,uhat,h,f)
        mass_form = shell_pde.mass(uhat, h, density)
        elastic_energy_form = shell_pde.elastic_energy(w,uhat,h,E)
        dx_reduced = ufl.Measure('dx', domain=mesh, 
                                 metadata={'quadrature_degree':4})
        pnorm_stress_form = shell_pde.pnorm_stress(
                        w,uhat,h,E,nu,
                        dx_reduced,m=self.m,rho=self.rho,
                        alpha=None,regularization=False)
        stress_form = shell_pde.von_Mises_stress(
                        w,uhat,h,E,nu,surface='Top')

        fea.add_input('thickness', h, init_val=0.001, record=self.record)
        fea.add_input('F_solid', f, init_val=1., record=self.record)
        fea.add_input('E', E, init_val=1., record=self.record)
        fea.add_input('nu', nu, init_val=1., record=self.record)
        fea.add_input('density', density, init_val=1., record=self.record)
        fea.add_input('uhat', uhat, init_val=0., record=self.record)

        fea.add_state(name='disp_solid',
                        function=w,
                        residual_form=residual_form,
                        arguments=['thickness','F_solid',
                                    'E','nu','uhat'])
        fea.add_output(name='compliance',
                        form=compliance_form,
                        arguments=['disp_solid','F_solid','thickness','uhat'])
        fea.add_output(name='mass',
                        form=mass_form,
                        arguments=['thickness','density','uhat'])
        fea.add_output(name='elastic_energy',
                        form=elastic_energy_form,
                        arguments=['thickness','disp_solid', 'E','uhat'])
        fea.add_output(name='pnorm_stress',
                        form=pnorm_stress_form,
                        arguments=['thickness','disp_solid','E', 'nu','uhat'])
        fea.add_field_output(name='stress',
                        form=stress_form,
                        arguments=['thickness','disp_solid','E', 'nu','uhat'],
                        record=self.record)
        
        self.fea = fea
        
    def evaluate(self, force_vector: csdl.Variable, 
                thickness: csdl.Variable,
                E: csdl.Variable, 
                nu: csdl.Variable, 
                density: csdl.Variable,
                node_disp: csdl.Variable=None,
                debug_mode=False,
                is_pressure=True) -> csdl.VariableGroup:
        '''
        Parameters:
        -----------
        Vector csdl.Variable:
            > force_vector: the force vector applied on the shell mesh nodes
            > thickness: the thickness on the shell mesh nodes
            > E: the Young's modulus on the shell mesh nodes
            > nu: the Poisson's ratio on the shell mesh nodes
            > density: the density on the shell mesh nodes

        Returns:
        --------
        Vector csdl.Variable:
            > disp_solid: the displacements (3 translational dofs, 3 rotation dofs)
                            on the shell mesh nodes
            > stress: the von Mises stress on the shell mesh elements
        Scalar csdl.Variable:
            > aggregated_stress: the aggregated stress of the shell model
            > compliance: the compliance of the shell model
            > tip_disp: the tip displacement of the shell model
            > mass: the mass of the shell model
        '''
        shell_inputs = csdl.VariableGroup()

        #:::::::::::::::::::::: Prepare the inputs :::::::::::::::::::::::::::::
        # sort the material properties based on FEniCS indices
        fenics_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices    
        shell_inputs.thickness = thickness[fenics_mesh_indices]
        shell_inputs.E = E[fenics_mesh_indices]
        shell_inputs.nu = nu[fenics_mesh_indices]
        shell_inputs.density = density[fenics_mesh_indices]

        # reshape the force matrix to vector and sort indices
        force_reshaping_model = ForceReshapingModel(shell_pde=self.shell_pde)
        reshaped_force = force_reshaping_model.evaluate(force_vector)
        reshaped_force.add_name('F_solid')

        if is_pressure:
            shell_inputs.F_solid = reshaped_force
        else:
            # Compute nodal pressures based on forces
            print('Converting forces to pressures ...')
            A = self.shell_pde.construct_force_to_pressure_map()
            pressure = csdl.solve_linear(convertToDense(A), reshaped_force)
            shell_inputs.F_solid = pressure

        # sort the nodal mesh deformation based on FEniCS indices
        if node_disp is None:
            node_disp = csdl.Variable(value=0.0, shape=force_vector.shape, 
                                      name='node_disp')
        reshaped_node_disp = force_reshaping_model.evaluate(node_disp)
        reshaped_node_disp.add_name('uhat')
        shell_inputs.uhat = reshaped_node_disp

        #:::::::::::::::::::::: Evaluate the model :::::::::::::::::::::::::::::
        # Evaluate the shell model
        print('Evaluating the RM shell model ...')
        solid_model = FEAModel(fea=[self.fea], fea_name='rm_shell')
        shell_outputs = solid_model.evaluate(shell_inputs, debug_mode=debug_mode)

        #:::::::::::::::::::::: Postprocess the outputs ::::::::::::::::::::::::
        disp_extraction_model = DisplacementExtractionModel(shell_pde=self.shell_pde)
        disp_extracted = disp_extraction_model.evaluate(shell_outputs.disp_solid)
        disp_extracted.add_name('disp_extracted')
        shell_outputs.disp_extracted = disp_extracted
        
        aggregated_stress_model = AggregatedStressModel(m=self.m, rho=self.rho)
        aggregated_stress = aggregated_stress_model.evaluate(shell_outputs.pnorm_stress)
        aggregated_stress.add_name('aggregated_stress')
        shell_outputs.aggregated_stress = aggregated_stress

        print('RM shell model evaluation completed.')
        print('-'*40)
        return shell_outputs

class AggregatedStressModel:
    '''
    Compute the aggregated stress
    '''
    def __init__(self, m: float, rho: int):
        self.m = m
        self.rho = rho

    def evaluate(self, pnorm_stress: csdl.Variable):
        aggregated_stress = 1/self.m*pnorm_stress**(1/self.rho)
        return aggregated_stress

class DisplacementExtractionModel:
    '''
    Extract and reshape displacement vector into matrix
    '''
    def __init__(self, shell_pde: RMShellPDE):
        self.shell_pde = shell_pde

    def evaluate(self, disp_vec: csdl.Variable):
        shell_pde = self.shell_pde

        disp_extraction_mats = shell_pde.construct_nodal_disp_map()
        # Both vector or tensors need to be numpy arrays
        shape = shell_pde.mesh.geometry.x.shape
        # contains nodal displacements only (CG1)
        nodal_disp_vec = csdl.sparse.matvec(disp_extraction_mats, disp_vec)
        nodal_disp_mat = csdl.transpose(csdl.reshape(nodal_disp_vec, shape=(shape[1],shape[0])))

        # reorder the matrix to match the importing mesh node indices
        # FEniCS --> CADDEE
        fenics_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices
        reverse_fenics_mesh_indices = np.argsort(fenics_mesh_indices).tolist()
        reordered_nodal_disp_mat = nodal_disp_mat[reverse_fenics_mesh_indices,:]
        return reordered_nodal_disp_mat

class ForceReshapingModel:
    '''
    Reshape force matrix to vector
    '''
    def __init__(self, shell_pde: RMShellPDE):
        self.shell_pde = shell_pde

    def evaluate(self, nodal_force_mat: csdl.Variable):
        shell_pde = self.shell_pde
        dummy_func = Function(shell_pde.VF)
        size = len(dummy_func.x.array)
        # reorder the matrix to match the FEniCS mesh node indices
        # CADDEE --> FEniCS
        fenics_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices    
        output = csdl.reshape(nodal_force_mat[fenics_mesh_indices,:], shape=(size,))
        return output
