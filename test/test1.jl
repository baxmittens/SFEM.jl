include("../src/SFEM.jl")

using .SFEM
using .SFEM.Elements: Tri3Ref, Tri6Ref, evaluateShapeFuns, gaussSimplex, dim, nnodes, flatten_tuple, shape_functions
using .SFEM.MeshReader: GmshMesh

using StaticArrays
import DensePolynomials
import DensePolynomials: DensePoly, coordinates, evaluate, fast_binomial, diff

tri3 = Tri3Ref()
tri6 = Tri6Ref()
shape_functions(tri6,Val{2})
#evaluatedShapeFuns,evaluatedShapeFunDerivs = evaluateShapeFuns(tri6, gaussSimplex, 6)
#meshfilepath = "../models/2d/patchtest.msh"
#mesh = GmshMesh(meshfilepath)

