ENV["OMP_NUM_THREADS"] = Base.Threads.nthreads()
ENV["MKL_NUM_THREADS"] = Base.Threads.nthreads()
ENV["OPENBLAS_NUM_THREADS"] = Base.Threads.nthreads()

#using Logging
#global_logger(ConsoleLogger(stdout))

import SFEM
import SFEM: LinearElasticity, HeatConduction
import SFEM.Elements: Tri3, Tri6, Tri, ElementStateVars2D, MatPars
import SFEM.MeshReader: GmshMesh
import SFEM.Domains: ProcessDomain, Domain, solve!, setBCandUCMaps!, init_loadstep!, tsolve!

using StaticArrays
using LinearAlgebra
#using ProfileView

meshfilepath = "../models/2d/beam_grob_tri6.msh"
mesh = GmshMesh(meshfilepath);
nips = 7


ls = [vcat(zeros(6),collect(0.0:-0.0005:-0.002)),vcat(zeros(1),ones(Float64,10)*-100)]
ts = collect(0.0:100000.0:1000000.0)
nts = length(ts)

states = [ElementStateVars2D(Val{nips},Val{nts}) for elinds in mesh.connectivity];

matpars = MatPars(7000.0, 450.0, 1e-5, 1e-5, 0.0, 50.0, 50.0, 0.0, 2.1e11, 0.3, 200.0)
els1 = Tri{2,6,nips,12}[Tri6(SMatrix{2,6,Float64,12}(mesh.nodes[elinds,1:2]'), SVector{6,Int}(elinds), state, matpars, Val{nips}) for (elinds,state) in zip(mesh.connectivity, states)];
els2 = Tri{2,6,nips,12}[Tri6(SMatrix{2,6,Float64,12}(mesh.nodes[elinds,1:2]'), SVector{6,Int}(elinds), state, matpars, Val{nips}) for (elinds,state) in zip(mesh.connectivity, states)];


ndofs1 = size(mesh.nodes,1)*2
dofmap1 = convert(Matrix{Int}, reshape(1:ndofs1,2,:))
ndofs2 = size(mesh.nodes,1)*1
dofmap2 = convert(Matrix{Int}, reshape(ndofs1+1:ndofs1+ndofs2,1,:))
linelasticity = ProcessDomain(LinearElasticity, mesh.nodes, mesh.connectivity, els1, dofmap1, nips, nts, Val{2})
heatconduction = ProcessDomain(HeatConduction, mesh.nodes, mesh.connectivity, els2, dofmap2, nips, nts, Val{1})
dom = Domain((linelasticity,heatconduction),ls,ts)

tsolve!(dom)



plotting=true
if plotting
using GLMakie
using GeometryBasics
using LinearAlgebra

f = Figure(size=(1000,600));
mainview = f[1,1] = GridLayout()
controlview = f[2,1] = GridLayout()
ax = Axis(f[1,1], autolimitaspect = 1)
timeslider = Slider(controlview[1,2], range = ts, startvalue=ts[end], update_while_dragging=false)
dispmult = Slider(controlview[3,2], range = [1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000,10000000000], startvalue=1, update_while_dragging=false)
timeslidertext = map!(Observable{Any}(), timeslider.value) do val
	return "t=$val"
end
dispmultslidertext = map!(Observable{Any}(), dispmult.value) do val
	return "t=$val"
end
Label(controlview[1,1], text=timeslidertext)
Label(controlview[3,1], text=dispmultslidertext)
fieldview = controlview[2,2] = GridLayout()
itemmenu = Menu(fieldview[1,1], options=["U", "σ", "εpl", "ΔT","q"])
fieldmenu = Menu(fieldview[1,2], options=["xx", "yy", "xy", "eq"])
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

Ux = map!(Observable{Any}(), timeslider.value, dispmult.value) do val,val2
	ti = findfirst(x->x==val, dom.timesteps)
	pdom.postdata.timesteps[ti].pdat[:U][:,1].*val2
	#pdom.postdata.timesteps[1].pdat[:U][:,1]
end
Uy = map!(Observable{Any}(), timeslider.value, dispmult.value) do val,val2
	ti = findfirst(x->x==val, dom.timesteps)
	pdom.postdata.timesteps[ti].pdat[:U][:,2].*val2
	#pdom.postdata.timesteps[1].pdat[:U][:,2]
end
Xd = map!(Observable{Any}(), Ux) do _Ux
	X .+ _Ux
end
Yd = map!(Observable{Any}(), Uy) do _Uy
	Y .+ _Uy
end
points = map!(Observable{Any}(), Xd, Yd) do _Xd,_Yd
	points = Point2f.(_Xd, _Yd)
end

on(points) do points
	xx = map(x->x[1], points)
	yy = map(x->x[2], points)
	minx,maxx = minimum(xx), maximum(xx)
	miny,maxy = minimum(yy), maximum(yy)
	distx = (maxx-minx)/10.0
	disty = (maxy-miny)/10.0
	xlims!(ax, [minx-distx,maxx+distx])
	ylims!(ax, [miny-disty,maxy+disty])
end

postData = map!(Observable{Any}(), itemmenu.selection, fieldmenu.selection, timeslider.value) do item,field,val
	#println(val)
	#return pdom.postdata.timesteps[2].pdat[:U][:,1]
	ti = findfirst(x->x==val, dom.timesteps)
	if item == "U"
		if field == "xx"
			pdom.postdata.timesteps[ti].pdat[:U][:,1]
		elseif field == "yy"
			pdom.postdata.timesteps[ti].pdat[:U][:,2]
		else
			zeros(pdom.postdata.timesteps[ti].pdat[:U][:,1])
		end
	elseif item == "σ"
		if field == "xx"
			pdom.postdata.timesteps[ti].pdat[:σ][:,1]
		elseif field == "yy"
			pdom.postdata.timesteps[ti].pdat[:σ][:,2]
		elseif field == "xy"
			pdom.postdata.timesteps[ti].pdat[:σ][:,3]
		else
			zeros(pdom.postdata.timesteps[ti].pdat[:U][:,1])
		end
	elseif item == "εpl"
		if field == "xx"
			abs.(pdom.postdata.timesteps[ti].pdat[:εpl][:,1])
		elseif field == "yy"
			abs.(pdom.postdata.timesteps[ti].pdat[:εpl][:,2])
		elseif field == "xy"
			abs.(pdom.postdata.timesteps[ti].pdat[:εpl][:,3])
		else
			zeros(pdom.postdata.timesteps[ti].pdat[:U][:,1])
		end
	elseif item == "ΔT"
		dom.processes[2].postdata.timesteps[ti].pdat[:ΔT][:,1]
	else
		if field == "xx"
			abs.(dom.processes[2].postdata.timesteps[ti].pdat[:q][:,1])
		else
			abs.(dom.processes[2].postdata.timesteps[ti].pdat[:q][:,2])
		end
	end
	
end

postData_limits = map!(Observable{Any}(), postData) do u
	lims = minimum(u),maximum(u)
	if abs(lims[2]-lims[1]) < 1e-6
		return lims[2]-0.00001,lims[2]+0.00001
	else
		return lims
	end
end

tricontourf!(ax, Xd, Yd, postData, triangulation = hcat(conn...)',levels=80, colormap=:prism)
faces = [GeometryBasics.TriangleFace(conn[j][1], conn[j][2], conn[j][3]) for j = 1:length(conn)]
mesh = map!(Observable{Any}(), points) do p
	GeometryBasics.Mesh(p, faces)
end
wireframe!(ax, mesh, color = (:black, 0.75), linewidth = 0.5, transparency = true, visible=togglemesh.active)
Colorbar(mainview[1,2], limits=postData_limits, colormap=:prism)
f
end
