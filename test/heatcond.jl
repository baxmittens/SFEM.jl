
ENV["OMP_NUM_THREADS"] = Base.Threads.nthreads()
ENV["MKL_NUM_THREADS"] = Base.Threads.nthreads()
ENV["OPENBLAS_NUM_THREADS"] = Base.Threads.nthreads()

#using Logging
#global_logger(ConsoleLogger(stdout))

import SFEM
import SFEM: HeatConduction
import SFEM.Elements: Tri3, Tri6, Tri, ElementStateVars2D
import SFEM.MeshReader: GmshMesh
import SFEM.Domains: ProcessDomain, Domain, solve!, setBCandUCMaps!, init_loadstep!, tsolve!

using StaticArrays
using LinearAlgebra
using ProfileView

meshfilepath = "../models/2d/beam_grob.msh"
mesh = GmshMesh(meshfilepath);
nips = 4
ls = vcat(0.0,ones(5)*-100)
ts = collect(0.0:10000000.0:50000000.0)
nts = length(ls)
states = [ElementStateVars2D(Val{nips},Val{nts}) for elinds in mesh.connectivity];
els = Tri{2,3,nips,6}[Tri3(SMatrix{2,3,Float64,6}(mesh.nodes[elinds,1:2]'), SVector{3,Int}(elinds), state, Val{nips}) for (elinds,state) in zip(mesh.connectivity, states)];

ndofs = size(mesh.nodes,1)
dofmap = convert(Matrix{Int}, reshape(1:ndofs,1,:))


heatconduction = ProcessDomain(HeatConduction, mesh.nodes, mesh.connectivity, els, dofmap, nips, nts, Val{1})
dom = Domain((heatconduction,),[ls],ts)
tsolve!(dom)

test = false
if test
import SFEM.Elements: smallDet, Blin0, grad, MaterialStiffness

pdom = dom.processes[1];
els = pdom.els
shapeFuns = pdom.shapeFuns;
dofmap = pdom.dofmap;

d𝐍s = shapeFuns.d𝐍s
𝐍s = shapeFuns.𝐍s

wips = shapeFuns.wips
el = els[1]
elT0 = el.nodes
eldofs = dofmap[1,el.inds][:]
U = dom.mma.U

nodalT = U[eldofs]
NIPs = 7
Js = ntuple(ip->elT0*d𝐍s[ip], NIPs)
detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
@assert all(detJs .> 0) "error: det(JM) < 0"
invJs = ntuple(ip->inv(Js[ip]), NIPs)
grad𝐍s = ntuple(ip->d𝐍s[ip]*invJs[ip], NIPs)

𝐍=𝐍s[1]
d𝐍=d𝐍s[1]
grad𝐍_temp = grad𝐍s[1]

Nm = SVector{6,Float64}(𝐍[1],𝐍[1],𝐍[2],𝐍[2],𝐍[3],𝐍[3])
transpose(𝐁)*ones(SMatrix{3,4,Float64,12})*vcat(𝐁,Nm')
end

plotting=true
if plotting
using GLMakie
using GeometryBasics
using LinearAlgebra

f = Figure(size=(1000,600));
mainview = f[1,1] = GridLayout()
controlview = f[2,1] = GridLayout()
ax = Axis(f[1,1], autolimitaspect = 1)
timeslider = Slider(controlview[1,2], range = 1:length(ls), startvalue = length(ls), update_while_dragging=false)
timeslidertext = map!(Observable{Any}(), timeslider.value) do val
	return "t=$val"
end
Label(controlview[1,1], text=timeslidertext)
fieldview = controlview[2,2] = GridLayout()
itemmenu = Menu(fieldview[1,1], options=["ΔT"])
fieldmenu = Menu(fieldview[1,2], options=["1"])
Label(fieldview[1,3], text="show mesh:")
togglemesh = Toggle(fieldview[1,4], active = true)


_conn = dom.processes[1].connectivity

if length(_conn[1])>3
	conn = Vector{Vector{Int64}}(undef, 4*length(_conn))
	for i = 1:length(_conn)
		ii = 4*(i-1)+1
		conn[ii] = _conn[i][[1,4,6]]
		conn[ii+1] = _conn[i][[4,2,5]]
		conn[ii+2] = _conn[i][[6,5,3]]
		conn[ii+3] = _conn[i][[4,5,6]]
	end
else
	conn = _conn
end
pdom = dom.processes[1]
X = pdom.nodes[:,1]
Y = pdom.nodes[:,2]

postData = map!(Observable{Any}(), itemmenu.selection, fieldmenu.selection, timeslider.value) do item,field,val
	return pdom.postdata.timesteps[val].pdat[:ΔT][:,1]
end

postData_limits = map!(Observable{Any}(), postData) do u
	lims = minimum(u),maximum(u)
	if abs(lims[2]-lims[1]) < 1e-6
		return lims[1]-0.000001,lims[1]+0.0000001
	else
		return lims
	end
end


tricontourf!(ax, X, Y, postData, triangulation = hcat(conn...)',levels=16)
Colorbar(mainview[1,2], limits=postData_limits)

#faces = [GeometryBasics.TriangleFace(conn[j][1], conn[j][2], conn[j][3]) for j = 1:length(conn)]
#mesh = map!(Observable{Any}(), points) do p
#	GeometryBasics.Mesh(p, faces)
#end
#wireframe!(ax, mesh, color = (:black, 0.75), linewidth = 0.5, transparency = true, visible=togglemesh.active)

display(f)

end


