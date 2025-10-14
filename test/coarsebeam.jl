
ENV["OMP_NUM_THREADS"] = Base.Threads.nthreads()
ENV["MKL_NUM_THREADS"] = Base.Threads.nthreads()
ENV["OPENBLAS_NUM_THREADS"] = Base.Threads.nthreads()

#using Logging
#global_logger(ConsoleLogger(stdout))

import SFEM
import SFEM: LinearElasticity
import SFEM.Elements: Tri3, Tri6, Tri, ElementStateVars2D
import SFEM.MeshReader: GmshMesh
import SFEM.Domains: ProcessDomain, Domain, solve!, setBCandUCMaps!, init_loadstep!, tsolve!

using StaticArrays
using LinearAlgebra
#using ProfileView

meshfilepath = "../models/2d/beam.msh"
mesh = GmshMesh(meshfilepath);
nips = 4
ts = collect(0.0:-0.1:-0.5)
nts = length(ts)
states = [ElementStateVars2D(Val{nips},Val{nts}) for elinds in mesh.connectivity];
els = Tri{2,3,nips,6}[Tri3(SMatrix{2,3,Float64,6}(mesh.nodes[elinds,1:2]'), SVector{3,Int}(elinds), state, Val{nips}) for (elinds,state) in zip(mesh.connectivity, states)];
ndofs = size(mesh.nodes,1)*2
dofmap = convert(Matrix{Int}, reshape(1:ndofs,2,:))
linela = ProcessDomain(LinearElasticity, mesh.nodes, mesh.connectivity, els, dofmap, nips, nts, Val{2})
dom = Domain((linela,),[ts],ts)

t1 = time()
tsolve!(dom)
#@profview tsolve!(dom)
t2 = time()
println("Gesamtzeit = $(round(t2-t1,digits=2))")

#plotting = true
#if plotting
#	include("../src/Plotting.jl")
#end
plotting=true
if plotting
using GLMakie
using GeometryBasics
using LinearAlgebra

f = Figure(size=(1000,600));
mainview = f[1,1] = GridLayout()
controlview = f[2,1] = GridLayout()
ax = Axis(f[1,1], autolimitaspect = 1)
timeslider = Slider(controlview[1,2], range = ts, startvalue = last(ts), update_while_dragging=false)
timeslidertext = map!(Observable{Any}(), timeslider.value) do val
	return "t=$val"
end
Label(controlview[1,1], text=timeslidertext)
fieldview = controlview[2,2] = GridLayout()
itemmenu = Menu(fieldview[1,1], options=["U", "σ", "εpl"])
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

Ux = map!(Observable{Any}(), timeslider.value) do val
	ti = findfirst(x->x==val, dom.timesteps)
	pdom.postdata.timesteps[ti].pdat[:U][:,1]
end
Uy = map!(Observable{Any}(), timeslider.value) do val
	ti = findfirst(x->x==val, dom.timesteps)
	pdom.postdata.timesteps[ti].pdat[:U][:,2]
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
	else
		if field == "xx"
			abs.(pdom.postdata.timesteps[ti].pdat[:εpl][:,1])
		elseif field == "yy"
			abs.(pdom.postdata.timesteps[ti].pdat[:εpl][:,2])
		elseif field == "xy"
			abs.(pdom.postdata.timesteps[ti].pdat[:εpl][:,3])
		else
			zeros(pdom.postdata.timesteps[ti].pdat[:U][:,1])
		end
	end
	
end

postData_limits = map!(Observable{Any}(), postData) do u
	lims = minimum(u),maximum(u)
	if abs(lims[2]-lims[1]) < 1e-6
		return -1.0,1.0
	else
		return lims
	end
end



tricontourf!(ax, Xd, Yd, postData, triangulation = hcat(conn...)',levels=17)

faces = [GeometryBasics.TriangleFace(conn[j][1], conn[j][2], conn[j][3]) for j = 1:length(conn)]
#mesh = map!(Observable{Any}(), points) do p
#	GeometryBasics.Mesh(p, faces)
#end
#wireframe!(ax, mesh, color = (:black, 0.75), linewidth = 0.5, transparency = true, visible=togglemesh.active)
Colorbar(mainview[1,2], limits=postData_limits)
f

#function facecolor(vertices,faces,facecolors)
#	v = zeros(size(faces,1)*3,2)
#	f = zeros(Int, size(faces))
#	fc = zeros(size(v,1))
#	for i in 1:size(faces,1)
#		face = faces[i,:]
#		verts = vertices[face,:]
#		j = 3*(i-1)+1
#		f[i,:] = [j,j+1,j+2]
#		v[j:j+2,:] .= verts
#		fc[j:j+2] .= facecolors[i]
#	end
#	return v,f,fc
#end
#vertices = pdom.mesh.nodes[:,1:2]
#faces = hcat(_conn...)'[:,1:3]
#facecolors = pdom.postdata.postdata[end].σ_avg
#v,fa,fc = facecolor(vertices,faces,facecolors)
#cm_repo_2d = mesh!(ax, v, fa, color=fc, shading=NoShading)
end