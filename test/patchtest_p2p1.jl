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

# Define nnodes per element
nnodes_element1 = 6
nnodes_element2 = 3
nnodes_neumann1 = 3
nnodes_neumann2 = 2


### Load mesh tri 10
meshfilepath = "../models/2d/patchtest_tri6.msh"
mesh = GmshMesh(meshfilepath);
# connectivity is missing. Load that too
filestream = open(meshfilepath)
lines = readlines(filestream)
close(filestream)
indline1 = findfirst(x->occursin("\$Elements", x), lines)
indline2 = findfirst(x->occursin("\$EndElements", x), lines)
nodes1 = mesh.nodes
matids1 = mesh.elemPhysNums
connectivity1 = filter(x->length(x)==nnodes_element1, map(x->map(y->parse(Int,y), split(x)[2:end]),lines[indline1+3:indline2-1]))
###

meshfilepath = "../models/2d/patchtest_tri3.msh"
mesh = GmshMesh(meshfilepath);
# connectivity is missing. Load that too
filestream = open(meshfilepath)
lines = readlines(filestream)
close(filestream)
indline1 = findfirst(x->occursin("\$Elements", x), lines)
indline2 = findfirst(x->occursin("\$EndElements", x), lines)
nodes2 = mesh.nodes
matids2 = mesh.elemPhysNums
connectivity2 = filter(x->length(x)==nnodes_element2, map(x->map(y->parse(Int,y), split(x)[2:end]),lines[indline1+3:indline2-1]))
filter!(x->!(x[2]==2&&(x[1]==1||x[1]==2||x[1]==3)), connectivity2) # filter material group declaration
###

# Meshes must be nested! Partially checked here.
@assert length(connectivity1) == length(connectivity2)
@assert length(matids1) == length(matids2)

### Define Elements
# Number integration points
nips = 7
nips_neumann = 4
# Define Eltype
ElType1 = Tri{2,nnodes_element1,nips,2*nnodes_element1}
ElType2 = Tri{2,nnodes_element2,nips,2*nnodes_element2}
ElLineType1 = Line{2,nnodes_neumann1,nips_neumann,2*nnodes_neumann1}
ElLineType2 = Line{2,nnodes_neumann2,nips_neumann,2*nnodes_neumann2}
# Time stepping
ts = collect(0.0:1e5:1e5)
nts = length(ts)
# init state variables per nip and timestep
states = [ElementStateVars2D(Val{nips},Val{nts}) for elinds in connectivity1];
## define body forces as function
import SFEM.Elements: bodyforceM, bodyforceT 
# Thermo
bodyforceT(x, matpars, actt, ts=ts) = actt > 1 ? 100.0 : 0.0
# Mechanic no body force
## Material 
matpars = MatPars(7000.0, 450.0, 1e-5, 1e-5, 0.0, 50.0, 50.0, 0.0, 2.1e11, 0.3, Inf, 0)
## Create elements T->Tri3, M->Tri3
els1 = ElType1[Tri6(SMatrix{2,nnodes_element1,Float64,2*nnodes_element1}(nodes1[elinds,1:2]'), SVector{nnodes_element1,Int}(elinds), state, matpars, Val{nips}) for (i,(elinds,state)) in enumerate(zip(connectivity1, states))];
els2 = ElType2[Tri3(SMatrix{2,nnodes_element2,Float64,2*nnodes_element2}(nodes2[elinds,1:2]'), SVector{nnodes_element2,Int}(elinds), state, matpars, Val{nips}) for (i,(elinds,state)) in enumerate(zip(connectivity2, states))];
###

### Boundary Conditions
# Boundary Conditions are handled by the user fun(ΔU, U, nodes, dofmap, actt)->Vector{Int}
## Dirichlet
# Mechanics
function dirichletM(ΔU, U, nodes, dofmap, actt)
	inds_left = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(nodes))
	inds_bottom = findall(x->isapprox(x[2],0.0,atol=1e-9), eachrow(nodes))
	left_bc_x = dofmap[1,inds_left]
	bottom_bc_y = dofmap[2,inds_bottom]
	return vcat(left_bc_x, bottom_bc_y)
end
# Thermo
function dirichletT(ΔU, U, nodes, dofmap, actt)
	inds_left = findall(x->isapprox(x[2],0.0,atol=1e-9), eachrow(nodes))
	ind_right = findall(x->isapprox(x[2],1.0,atol=1e-9), eachrow(nodes))
	left_bc = dofmap[1,inds_left]
	right_bc = dofmap[1,ind_right]
	return Int[]
end

## Neumann
# Compute Line elements
neumann_inds = findall(x->isapprox(x[1],1.0,atol=1e-5), eachrow(nodes1))
lines = [SVector{3,Int}(1,4,2), SVector{3,Int}(2,5,3), SVector{3,Int}(3,6,1)]

neumann_els_M = ElLineType1[]
neumann_els_T = ElLineType2[]
for el in els1
	for line in lines
		inds = el.inds[line]
		if all(map(x->x ∈ neumann_inds, inds))
			push!(neumann_els_M, Line(SMatrix{2,nnodes_neumann1,Float64,2*nnodes_neumann1}(nodes1[inds,1:2]'), inds, Val{nips_neumann}))
		end
	end
end
# Define boundary functions
# Mechanics
fun_neumann_M(x, actt, ts=ts) = actt > 1 ? SVector{2,Float64}(1000.0,0.0) : SVector{2,Float64}(0.0,0.0)
###

### Create Dofmaps
ndofs1 = size(nodes1,1)*2
dofmap1 = convert(Matrix{Int}, reshape(1:ndofs1,2,:))
ndofs2 = size(nodes2,1)*1
dofmap2 = convert(Matrix{Int}, reshape(ndofs1+1:ndofs1+ndofs2,1,:))
###

### Create ProcessDomains
linelasticity = ProcessDomain(LinearElasticity, nodes1, connectivity1, els1, dofmap1, nts, Val{2}, ElLineType1, els_neumann=neumann_els_M, fun_neumann=fun_neumann_M)
heatconduction = ProcessDomain(HeatConduction, nodes2, connectivity2, els2, dofmap2, nts, Val{1}, Nothing)
###
### Create Domain
dom = Domain((linelasticity,heatconduction), ts, dirichletM=dirichletM, dirichletT=dirichletT)
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
xStart = SVector{2,Float64}(0.5,0.0)
xEnd = SVector{2,Float64}(0.5,1.0)
nsamplepoints = 40
valkeys_line = [:U_1, :U_2, :σ_1, :σ_2, :σ_3, :ΔT_1, :εpl_1, :εpl_2, :εpl_3]
valkeys_dom = [[:U, :σ, :εpl],[:ΔT, :q]]

f = Figure(size=(1600,900));
controlview = f[1,1] = GridLayout()
controlsubview = controlview[2,1] = GridLayout()
fieldmenu = Menu(controlsubview[1,1], options=["_1/xx", "_2/yy", "_3/xy", "norm"])
Label(controlsubview[1,2], text="global colormap:")
toggleclrmp = Toggle(controlsubview[1,3], active = true)
Label(controlsubview[1,4], text="show mesh:")
togglemesh = Toggle(controlsubview[1,5], active = false)
plotview = f[2,1] = GridLayout()
lineplotview = plotview[1,2] = GridLayout()
domplotview = plotview[1,1] = GridLayout()
colsize!(plotview, 1, Relative(4/5))
timeslider = Slider(controlview[1,1], range = 1:length(ts), startvalue=length(ts), update_while_dragging=false)
dispmult = Slider(controlsubview[1,6], range = [1,10,100,1000,2000,5000,10000,20000], startvalue=1, update_while_dragging=false)
timetext = map!(Observable{Any}(), timeslider.value) do i
	return string(round(ts[i]/60/60/24/365.25,digits=2))
end
dispmultslidertext = map!(Observable{Any}(), dispmult.value) do val
	return "disp. mult.=$val"
end
Label(controlview[1,2], text=timetext)
Label(controlsubview[1,7], text=dispmultslidertext)
plotLine!(lineplotview, valkeys_line, timeslider, dom, xStart, xEnd, nsamplepoints, true)
plotconns = ntuple(i->plotConnectivity(dom.processes[i]), length(dom.processes))
points = ntuple(i->getPoints2f(dom.processes[i], timeslider, dispmult), length(dom.processes))
axhandles = Dict{Symbol, Any}()
for (i,plotrow) in enumerate(valkeys_dom)
	if length(dom.processes) >= i
		for (j,valk) in enumerate(plotrow)
			ax_handle = plotField!(domplotview[j,i], dom.processes[i], valk, points[i], plotconns[i], timeslider, fieldmenu, toggleclrmp.active)
			axhandles[valk] = ax_handle
		end
	end
end
faces = [GeometryBasics.TriangleFace(dom.processes[1].connectivity[j][1], dom.processes[1].connectivity[j][2], dom.processes[1].connectivity[j][3]) for j = 1:length(dom.processes[1].connectivity)]
meshobs = map!(Observable{Any}(), points[1]) do p
	GeometryBasics.Mesh(p, faces)
end
if haskey(axhandles, :U)
	wireframe!(axhandles[:U], meshobs, color = (:black, 0.75), linewidth = 0.5, transparency = true, visible=togglemesh.active)
else
	wireframe!(axhandles[:ΔT], meshobs, color = (:black, 0.75), linewidth = 0.5, transparency = true, visible=togglemesh.active)
end
display(f)
end
###









