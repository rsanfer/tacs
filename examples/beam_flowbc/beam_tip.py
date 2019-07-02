# A test of FLOWBC boundary condition in TACS
# Works only in single core
# Ruben Sanchez
from __future__ import print_function   

# Import necessary libraries
import numpy as np
from mpi4py import MPI
from tacs import TACS, elements, constitutive, functions

# Load structural mesh from BDF file
tacs_comm = MPI.COMM_WORLD
struct_mesh = TACS.MeshLoader(tacs_comm)
struct_mesh.scanBDFFile("beam_tip.bdf")

# Set constitutive properties
rho = 2500.0 # density, kg/m^3
E = 70e9 # elastic modulus, Pa
nu = 0.3 # poisson's ratio
kcorr = 5.0 / 6.0 # shear correction factor
ys = 350e6 # yield stress, Pa
min_thickness = 0.002
max_thickness = 0.20
thickness = 0.01

# Loop over components, creating stiffness and element object for each
num_components = struct_mesh.getNumComponents()
a = struct_mesh.getBCs()
b = struct_mesh.getConnectivity()

for i in range(num_components):
    descriptor = struct_mesh.getElementDescript(i)
    stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i,
                                 min_thickness, max_thickness)                                
    element = None
    if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
        element = elements.MITCShell(2, stiff, component_num=i)    
    struct_mesh.setElement(i, element)
    
# Create tacs assembler object from mesh loader
tacs = struct_mesh.createTACS(6)

# Create the KS Function
ksWeight = 100.0
funcs = [functions.KSFailure(tacs, ksWeight)]

# Get the design variable values
x = np.zeros(num_components, TACS.dtype)
tacs.getDesignVars(x)

# Get the node locations
X = tacs.createNodeVec()
tacs.getNodes(X)
nNodes = tacs.getNumNodes()
X_array = X.getArray()
tacs.setNodes(X)

# Get the Flow BC node indices
nflowbc, flow_nodes = tacs.getBCNodeArray()
print("---------- Beam tip coordinates and local numbering ----------")
ipoint=0
for inode in flow_nodes:
  print(ipoint,inode, X_array[3*inode], X_array[3*inode+1], X_array[3*inode+2])
  ipoint+=1

# Create the forces
forces = tacs.createVec()
force_array = forces.getArray()
for inode in flow_nodes:
  force_array[6*inode] += 100.0   # point load at the flow BC nodes
tacs.applyBCs(forces)

# Set up and solve the analysis problem
res = tacs.createVec()
ans = tacs.createVec()
u = tacs.createVec()
mat = tacs.createFEMat()
pc = TACS.Pc(mat)
subspace = 100
restarts = 2
gmres = TACS.KSM(mat, pc, subspace, restarts)

# Assemble the Jacobian and factor
alpha = 1.0
beta = 0.0
gamma = 0.0
tacs.zeroVariables()
tacs.assembleJacobian(alpha, beta, gamma, res, mat)
pc.factor()

# Solve the linear system
gmres.solve(forces, ans)
tacs.setVariables(ans)

disp_array = ans.getArray()

# Output for visualization 
flag = (TACS.ToFH5.NODES |
        TACS.ToFH5.DISPLACEMENTS |
        TACS.ToFH5.STRAINS |
        TACS.ToFH5.EXTRAS)
f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
f5.writeToFile('beam_tip.f5')
