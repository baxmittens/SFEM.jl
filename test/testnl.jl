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
nips = 4
ts = collect(0.0:-0.005:-0.05)
nts = length(ts)
els = Tri{2,3,nips,6}[Tri3(SMatrix{2,3,Float64,6}(mesh.nodes[elinds,1:2]'), SVector{3,Int}(elinds), Val{nips}, Val{nts}) for elinds in mesh.connectivity]
dom = Domain(mesh,els,nips,ts)
t1 = time()
tsolve!(dom)
t2 = time()
println("Gesamtzeit = $(round(t2-t1,digits=2))")
plotting = false
if plotting
	include("../src/Plotting.jl")
end

