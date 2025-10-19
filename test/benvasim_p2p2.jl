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
using VTUFileHandler

function lin_func(x,xmin,ymin,xmax,ymax)
	a = (ymax-ymin)/(xmax-xmin)
	b = ymax-a*xmax
	return a*x+b
end

### Load mesh
# Canister -> ID 0
# Backfill -> ID 1
# Domain -> ID 2
meshfilepath = "../models/2d/benvasim_tri6.msh"
mesh = GmshMesh(meshfilepath);
nnodes_element = 6
nnodes_neumann = 3


# connectivity is missing. Load that too
filestream = open(meshfilepath)
lines = readlines(filestream)
close(filestream)
indline1 = findfirst(x->occursin("\$Elements", x), lines)
indline2 = findfirst(x->occursin("\$EndElements", x), lines)
nodes = mesh.nodes
matids = mesh.elemPhysNums
connectivity = filter(x->length(x)==nnodes_element, map(x->map(y->parse(Int,y), split(x)[2:end]),lines[indline1+3:indline2-1]))
###

### Define Elements
# Number integration points
nips = 7
nips_neumann = 3
# Define Eltype
ElType = Tri{2,nnodes_element,nips,2*nnodes_element}
ElLineType = Line{2,nnodes_neumann,nips_neumann,2*nnodes_neumann}
# Time stepping
ts = vcat(0.0,collect(range(1, 63115200000.0, 10)))
nts = length(ts)
# init state variables per nip and timestep
states = [ElementStateVars2D(Val{nips},Val{nts}) for elinds in connectivity];
## define body forces as function
import SFEM.Elements: bodyforceM, bodyforceT 
# Thermo ts 0...31557600000 qs 51...0
bodyforceT(x, matpars, actt, ts=ts) = matpars.materialID==0 && actt > 2 && actt < 27 ? lin_func(ts[actt],0.0,51.0,31557600000.0,0) : 0.0
# Mechanic no body force
bodyforceM(x, matpars, actt, ts=ts) = actt > 1 ? SVector{2,Float64}(0.0,0.0) : SVector{2,Float64}(0.0,0.0)
## Material 
# Canister
matpars0 = MatPars(6700.0, 500.0, 1.7e-05, 1.7e-05, 0.0, 16.0, 16.0, 0.0, 195_000_000_000.0, 0.3, Inf, 0)
# Backfill
matpars1 = MatPars(1575.0, 1090.0,  2.5e-5, 2.5e-5, 0.0, 1.17, 1.17, 0.0, 100_000_000.0, 0.1  , Inf, 1)
# Domain
matpars2 = MatPars(2495.0, 1060.0,  2e-5, 2e-5, 0.0, 1.84, 1.84, 0.0, 5_000_000_000.0, 0.3, Inf, 2)
matparsdict = Dict(1=>matpars0, 2=>matpars1, 3=>matpars2)
## Create elements T->Tri10, M->Tri10
els1 = ElType[Tri6(SMatrix{2,nnodes_element,Float64,2*nnodes_element}(nodes[elinds,1:2]'), SVector{nnodes_element,Int}(elinds), state, matparsdict[matids[i]], Val{nips}) for (i,(elinds,state)) in enumerate(zip(connectivity, states))];
els2 = ElType[Tri6(SMatrix{2,nnodes_element,Float64,2*nnodes_element}(nodes[elinds,1:2]'), SVector{nnodes_element,Int}(elinds), state, matparsdict[matids[i]], Val{nips}) for (i,(elinds,state)) in enumerate(zip(connectivity, states))];
###

### Boundary Conditions
# Boundary Conditions are handled by the user fun(ΔU, U, nodes, dofmap, actt)->Vector{Int}
## Dirichlet
# Mechanics
function dirichletM(ΔU, U, nodes, dofmap, actt)
	inds_left = findall(x->isapprox(x[1],-25.0,atol=1e-9), eachrow(nodes))
	inds_right = findall(x->isapprox(x[1],25.0,atol=1e-9), eachrow(nodes))
	inds_bottom = findall(x->isapprox(x[2],-100.0,atol=1e-9), eachrow(nodes))
	inds_top = findall(x->isapprox(x[2],-0.0,atol=1e-9), eachrow(nodes))
	left_bc_x = dofmap[1,inds_left]
	right_bc_x = dofmap[1,inds_right]
	bottom_bc_y = dofmap[2,inds_bottom]
	return vcat(bottom_bc_y, left_bc_x, right_bc_x)
	#return vcat(bottom_bc_y, left_bc_x)
end
# Thermo
function dirichletT(ΔU, U, nodes, dofmap, actt)
	inds_bottom = findall(x->isapprox(x[2],-100.0,atol=1e-9), eachrow(nodes))
	inds_top = findall(x->isapprox(x[2],100.0,atol=1e-9), eachrow(nodes))
	bottom_bc = dofmap[1,inds_bottom]
	top_bc = dofmap[1,inds_top]
	return vcat(bottom_bc,top_bc)
	#return Int[]
end

## Neumann
# Compute Line elements
neumann_inds = findall(x->isapprox(x[2],100.0,atol=1e-5), eachrow(nodes))
lines = [SVector{3,Int}(1,4,2), SVector{3,Int}(2,5,3), SVector{3,Int}(3,6,1)]

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
#fun_neumann_T(x, actt, ts=ts) = 0.0
###

### Create Dofmaps
ndofs1 = size(nodes,1)*2
dofmap1 = convert(Matrix{Int}, reshape(1:ndofs1,2,:))
ndofs2 = size(nodes,1)*1
dofmap2 = convert(Matrix{Int}, reshape(ndofs1+1:ndofs1+ndofs2,1,:))
###

### Create ProcessDomains
linelasticity = ProcessDomain(LinearElasticity, nodes, connectivity, els1, dofmap1, nts, Val{2}, ElLineType, els_neumann=neumann_els_M, fun_neumann=fun_neumann_M)
heatconduction = ProcessDomain(HeatConduction, nodes, connectivity, els2, dofmap2, nts, Val{1}, Nothing)
###
### Create Domain
dom = Domain((linelasticity,heatconduction), ts, dirichletM=dirichletM, dirichletT=dirichletT)
###
### Solve
@time tsolve!(dom)
###

### Plotting
plotting=true
if plotting
using GLMakie
import GeometryBasics
using LinearAlgebra
if !isdefined(Main,:sampleResult!)
	include("../src/Plotting.jl")
end

# Sample over line xStart ... xEnd
xStart = SVector{2,Float64}(0.0,0.0)
xEnd = SVector{2,Float64}(25.0,12.5)
nsamplepoints = 200
valkeys_line = [:U_1, :U_2, :σ_1, :σ_2, :σ_3, :ΔT_1]
valkeys_dom = [[:U, :σ,],[:ΔT, :q]]

f = Figure(size=(1600,900));
controlview = f[1,1] = GridLayout()
controlsubview = controlview[2,1] = GridLayout()
fieldmenu = Menu(controlsubview[1,1], options=["_1/xx", "_2/yy", "_3/xy", "norm"])
Label(controlsubview[1,2], text="show mesh:")
togglemesh = Toggle(controlsubview[1,3], active = false)
plotview = f[2,1] = GridLayout()
lineplotview = plotview[1,2] = GridLayout()
domplotview = plotview[1,1] = GridLayout()
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
plotLine!(lineplotview, valkeys_line,timeslider, dom, xStart, xEnd, nsamplepoints)
plotconn = plotConnectivity(dom.processes[1])
points = getPoints2f(dom.processes[1], timeslider, dispmult)
axhandles = Dict{Symbol, Any}()
for (i,plotrow) in enumerate(valkeys_dom)
	if length(dom.processes) >= i
		for (j,valk) in enumerate(plotrow)
			ax_handle = plotField!(domplotview[i,j], dom.processes[i], valk, points, plotconn, timeslider, fieldmenu)
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






