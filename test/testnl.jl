ENV["OMP_NUM_THREADS"] = Base.Threads.nthreads()
ENV["MKL_NUM_THREADS"] = Base.Threads.nthreads()
ENV["OPENBLAS_NUM_THREADS"] = Base.Threads.nthreads()

using StaticArrays
using SFEM
using SFEM.Elements: Tri3, Tri6, Tri
using SFEM.MeshReader: GmshMesh
using SFEM.Domains: Domain, solve!, setBCandUCMaps!, init_loadstep!, tsolve!
using LinearAlgebra

#using ProfileView

#meshfilepath = "../models/2d/beam.msh"
meshfilepath = "../models/2d/beam.msh"
mesh = GmshMesh(meshfilepath)
nips = 3
ts = collect(0.0:-0.01:-0.05)
nts = length(ts)
els = Tri{2,3,nips,6}[Tri3(SMatrix{2,3,Float64,6}(mesh.nodes[elinds,1:2]'), SVector{3,Int}(elinds), Val{nips}, Val{nts}) for elinds in mesh.connectivity]
dom = Domain(mesh,els,nips,ts)
tsolve!(dom)
plotting = true
if plotting
	include("../src/Plotting.jl")
end

