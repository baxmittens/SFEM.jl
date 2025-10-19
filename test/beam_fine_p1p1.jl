ENV["OMP_NUM_THREADS"] = Base.Threads.nthreads()
ENV["MKL_NUM_THREADS"] = Base.Threads.nthreads()
ENV["OPENBLAS_NUM_THREADS"] = Base.Threads.nthreads()

import SFEM
import SFEM: LinearElasticity, HeatConduction
import SFEM.Elements: Tri3, Tri6, Tri10, Tri, ElementStateVars2D, MatPars, Line, globalToLocal
import SFEM.MeshReader: GmshMesh
import SFEM.Domains: ProcessDomain, Domain, solve!, setBCandUCMaps!, init_loadstep!, tsolve!

using StaticArrays
using LinearAlgebra

plotting=false

### Load mesh
meshfilepath = "../models/2d/beam_fine_tri3.msh"
mesh = GmshMesh(meshfilepath);
nnodes_element = 3
nnodes_neumann = 2


# connectivity is missing. Load that too
filestream = open(meshfilepath)
lines = readlines(filestream)
close(filestream)
indline1 = findfirst(x->occursin("\$Elements", x), lines)
indline2 = findfirst(x->occursin("\$EndElements", x), lines)
nodes = mesh.nodes
matids = mesh.elemPhysNums
connectivity = filter(x->length(x)==nnodes_element, map(x->map(y->parse(Int,y), split(x)[2:end]),lines[indline1+3:indline2-1]))
filter!(x->!(x[2]==2&&(x[1]==1||x[1]==2||x[1]==3)), connectivity)
###

### Define Elements
# Number integration points
nips = 4
nips_neumann = 2
# Define Eltype
ElType = Tri{2,nnodes_element,nips,2*nnodes_element}
ElLineType = Line{2,nnodes_neumann,nips_neumann,2*nnodes_neumann}
# Time stepping
ts = collect(0.0:1e5:3e5)
nts = length(ts)
# init state variables per nip and timestep
states = [ElementStateVars2D(Val{nips},Val{nts}) for elinds in connectivity];
## Material 
matpars = MatPars(7000.0, 450.0, 1e-5, 1e-5, 0.0, 50.0, 50.0, 0.0, 2.1e11, 0.3, 200.0, 0)
## Create elements T->Tri3, M->Tri3
els1 = ElType[Tri3(SMatrix{2,nnodes_element,Float64,2*nnodes_element}(nodes[elinds,1:2]'), SVector{nnodes_element,Int}(elinds), state, matpars, Val{nips}) for (i,(elinds,state)) in enumerate(zip(connectivity, states))];
els2 = ElType[Tri3(SMatrix{2,nnodes_element,Float64,2*nnodes_element}(nodes[elinds,1:2]'), SVector{nnodes_element,Int}(elinds), state, matpars, Val{nips}) for (i,(elinds,state)) in enumerate(zip(connectivity, states))];
###

### Boundary Conditions
# Boundary Conditions are handled by the user fun(ΔU, U, nodes, dofmap, actt)->Vector{Int}
## Dirichlet
# Mechanics
function dirichletM(ΔU, U, nodes, dofmap, actt)
	ls = range(0,-0.1,nts)
	Uval = ls[actt]
	inds_left = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(nodes))
	inds_right = findall(x->isapprox(x[1],10.0,atol=1e-9), eachrow(nodes))
	left_bc_x = dofmap[1,inds_left]
	left_bc_y = dofmap[2,inds_left]
	right_bc_x = dofmap[1,inds_right]
	right_bc_y = dofmap[2,inds_right]
	ΔU[right_bc_y] .= (Uval .- U[right_bc_y])
	return vcat(left_bc_x, left_bc_y, right_bc_x, right_bc_y)
	#return vcat(bottom_bc_y, left_bc_x)
end
# Thermo
function dirichletT(ΔU, U, nodes, dofmap, actt)
	inds_left = findall(x->isapprox(x[2],0.0,atol=1e-9), eachrow(nodes))
	ind_right = findall(x->isapprox(x[2],10.0,atol=1e-9), eachrow(nodes))
	left_bc = dofmap[1,inds_left]
	right_bc = dofmap[1,ind_right]
	#return vcat(left_bc,right_bc)
	return Int[]
end

## Neumann
# Compute Line elements
neumann_inds = findall(x->isapprox(x[1],10.0,atol=1e-5), eachrow(nodes))
lines = [SVector{2,Int}(1,2), SVector{2,Int}(2,3), SVector{2,Int}(3,1)]

neumann_els_M = ElLineType[]
neumann_els_T = ElLineType[]
for el in els1
	for line in lines
		inds = el.inds[line]
		if all(map(x->x ∈ neumann_inds, inds))
			push!(neumann_els_M, Line(SMatrix{2,nnodes_neumann,Float64,2*nnodes_neumann}(nodes[inds,1:2]'), inds, Val{nips_neumann}))
		end
	end
end
# Define boundary functions
# Mechanics
fun_neumann_M(x, actt, ts=ts) = SVector{2,Float64}(0.0,-14000000.0)
# Thermo
#fun_neumann_T(x, actt, ts=ts) = x[2]*10.0
fun_neumann_T(x, actt, ts=ts) = 10.0
###

### Create Dofmaps
ndofs1 = size(nodes,1)*2
dofmap1 = convert(Matrix{Int}, reshape(1:ndofs1,2,:))
ndofs2 = size(nodes,1)*1
dofmap2 = convert(Matrix{Int}, reshape(ndofs1+1:ndofs1+ndofs2,1,:))
###

### Create ProcessDomains
linelasticity = ProcessDomain(LinearElasticity, nodes, connectivity, els1, dofmap1, nts, Val{2}, ElLineType, els_neumann=neumann_els_M, fun_neumann=fun_neumann_M)
heatconduction = ProcessDomain(HeatConduction, nodes, connectivity, els2, dofmap2, nts, Val{1}, ElLineType, els_neumann=neumann_els_M, fun_neumann=fun_neumann_T)
###
### Create Domain
dom = Domain((linelasticity,), ts, dirichletM=dirichletM, dirichletT=dirichletT)
###
### Solve
@time tsolve!(dom)
###

### Plotting

if plotting
using GLMakie
import GeometryBasics
using LinearAlgebra
if !isdefined(Main,:sampleResult!)
	include("../src/Plotting.jl")
end

# Sample over line xStart ... xEnd
xStart = SVector{2,Float64}(1.0,0.0)
xEnd = SVector{2,Float64}(1.0,1.0)
nsamplepoints = 40
valkeys_line = [:U_1, :U_2, :σ_1, :σ_2, :σ_3, :ΔT_1, :εpl_1, :εpl_2, :εpl_3]
valkeys_dom = [[:U, :σ, :εpl],[:ΔT, :q]]

f = Figure(size=(1600,900));
controlview = f[1,1] = GridLayout()
controlsubview = controlview[2,1] = GridLayout()
fieldmenu = Menu(controlsubview[1,1], options=["_1/xx", "_2/yy", "_3/xy", "norm"])
Label(controlsubview[1,2], text="show mesh:")
togglemesh = Toggle(controlsubview[1,3], active = false)
plotview = f[2,1] = GridLayout()
lineplotview = plotview[1,2] = GridLayout()
domplotview = plotview[1,1] = GridLayout()
colsize!(plotview, 1, Relative(4/5))
timeslider = Slider(controlview[1,1], range = 1:length(ts), startvalue=length(ts), update_while_dragging=false)
dispmult = Slider(controlsubview[1,4], range = [1,10,100,1000,2000,5000,10000,20000], startvalue=1, update_while_dragging=false)
timetext = map!(Observable{Any}(), timeslider.value) do i
	return string(round(ts[i]/60/60/24/365.25,digits=2))
end
dispmultslidertext = map!(Observable{Any}(), dispmult.value) do val
	return "disp. mult.=$val"
end
Label(controlview[1,2], text=timetext)
Label(controlsubview[1,5], text=dispmultslidertext)
plotLine!(lineplotview, valkeys_line, timeslider, dom, xStart, xEnd, nsamplepoints, true)
plotconn = plotConnectivity(dom.processes[1])
points = getPoints2f(dom.processes[1], timeslider, dispmult)
axhandles = Dict{Symbol, Any}()
for (i,plotrow) in enumerate(valkeys_dom)
	if length(dom.processes) >= i
		for (j,valk) in enumerate(plotrow)
			ax_handle = plotField!(domplotview[j,i], dom.processes[i], valk, points, plotconn, timeslider, fieldmenu)
			axhandles[valk] = ax_handle
		end
	end
end
faces = [GeometryBasics.TriangleFace(dom.processes[1].connectivity[j][1], dom.processes[1].connectivity[j][2], dom.processes[1].connectivity[j][3]) for j = 1:length(dom.processes[1].connectivity)]
meshobs = map!(Observable{Any}(), points) do p
	GeometryBasics.Mesh(p, faces)
end
wireframe!(axhandles[:U], meshobs, color = (:black, 0.75), linewidth = 0.5, transparency = true, visible=togglemesh.active)
display(f)
end
###









