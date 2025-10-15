ENV["OMP_NUM_THREADS"] = Base.Threads.nthreads()
ENV["MKL_NUM_THREADS"] = Base.Threads.nthreads()
ENV["OPENBLAS_NUM_THREADS"] = Base.Threads.nthreads()

#using Logging
#global_logger(ConsoleLogger(stdout))

import SFEM
import SFEM: LinearElasticity, HeatConduction, J2Plasticity, NoMaterial
import SFEM.Elements: Tri3, Tri6, Tri, ElementStateVars2D, MatPars
import SFEM.MeshReader: GmshMesh
import SFEM.Domains: ProcessDomain, Domain, solve!, setBCandUCMaps!, init_loadstep!, tsolve!

using StaticArrays
using LinearAlgebra
#using ProfileView

meshfilepath = "../models/2d/beam.msh"
mesh = GmshMesh(meshfilepath);
nips = 7


ls = [vcat(zeros(6),collect(0.0:-0.0005:-0.002)),vcat(zeros(1),ones(Float64,10)*-100)]
ts = collect(0.0:1.0:10.0)
nts = length(ts)

states = [ElementStateVars2D(Val{nips},Val{nts}) for elinds in mesh.connectivity];
matpars = MatPars(7000.0, 450.0, 1e-5, 1e-5, 0.0, 50.0, 50.0, 0.0, 2.1e11, 0.3, 200.0)
els1 = Tri{2,3,nips,6}[Tri3(J2Plasticity, SMatrix{2,3,Float64,6}(mesh.nodes[elinds,1:2]'), SVector{3,Int}(elinds), state, matpars, Val{nips}) for (elinds,state) in zip(mesh.connectivity, states)];
els2 = Tri{2,3,nips,6}[Tri3(NoMaterial, SMatrix{2,3,Float64,6}(mesh.nodes[elinds,1:2]'), SVector{3,Int}(elinds), state, matpars, Val{nips}) for (elinds,state) in zip(mesh.connectivity, states)];

ndofs1 = size(mesh.nodes,1)*2
dofmap1 = convert(Matrix{Int}, reshape(1:ndofs1,2,:))
ndofs2 = size(mesh.nodes,1)*1
dofmap2 = convert(Matrix{Int}, reshape(ndofs1+1:ndofs1+ndofs2,1,:))
linelasticity = ProcessDomain(LinearElasticity, mesh.nodes, mesh.connectivity, els1, dofmap1, nips, nts, Val{2})
heatconduction = ProcessDomain(HeatConduction, mesh.nodes, mesh.connectivity, els2, dofmap2, nips, nts, Val{1})
dom = Domain((linelasticity,heatconduction),ls,ts)
tsolve!(dom)