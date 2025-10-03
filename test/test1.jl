include("../src/SFEM.jl")

using .SFEM
using .SFEM.Elements: Tri3Ref, Tri6Ref, evaluateShapeFuns, gaussSimplex, dim, nnodes, flatten_tuple, shape_functions, monomials
using .SFEM.MeshReader: GmshMesh

using StaticArrays
import DensePolynomials
import DensePolynomials: DensePoly, coordinates, evaluate, fast_binomial, diff


#nodes = SMatrix{2,6,Float64,12}(0.,0.,1.,0.,0.,1.,.5,0.,.5,.5,.0,.5)
#shapeFuns = shape_functions(nodes, Val{2})
tri3 = Tri3Ref()
tri6 = Tri6Ref()
#N=P=K=2
#shape_functions(tri6,Val{2})
#N = K = 2
#exps = monomial_exponents(Val{N}, Val{K})
#evaluatedShapeFuns,evaluatedShapeFunDerivs = evaluateShapeFuns(tri6, gaussSimplex, 6)
#meshfilepath = "../models/2d/patchtest.msh"
#mesh = GmshMesh(meshfilepath)

#fbnk = fast_binomial(N+K,K)
#ntuple(i->begin; dp=DensePoly{N, Float64}(fbnk); dp.c[i]=1.0;dp end, fbnk)