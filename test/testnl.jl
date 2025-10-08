ENV["OMP_NUM_THREADS"] = "6"
ENV["MKL_NUM_THREADS"] = "6"
ENV["OPENBLAS_NUM_THREADS"] = "6"

using StaticArrays
using SFEM
using SFEM.Elements: Tri3Ref, Tri6Ref, Tri3, Tri6
using SFEM.MeshReader: GmshMesh
using SFEM.Domains: Domain, solve!, setBCandUCMaps!, init_loadstep!, tsolve!
#using ProfileView

#meshfilepath = "../models/2d/beam.msh"
meshfilepath = "../models/2d/beam.msh"
mesh = GmshMesh(meshfilepath)
nips = 3
ts = collect(0.0:-0.01:-0.05)
nts = length(ts)
els = Tri3[Tri3(SMatrix{2,3,Float64,6}(mesh.nodes[elinds,1:2]'), SVector{3,Int}(elinds), nips, nts) for elinds in mesh.connectivity]
dom = Domain(mesh,els,Tri3Ref,nips,ts)
tsolve!(dom)
plotting = true


if plotting
using GLMakie
using GeometryBasics

f = Figure(size=(1000,600));
mainview = f[1,1] = GridLayout()
controlview = f[2,1] = GridLayout()
ax = Axis(f[1,1], autolimitaspect = 1)
timeslider = Slider(controlview[1,2], range = ts, startvalue = first(ts), update_while_dragging=false)
timeslidertext = map!(Observable{Any}(), timeslider.value) do val
	return "t=$val"
end
Label(controlview[1,1], text=timeslidertext)
fieldview = controlview[2,2] = GridLayout()
itemmenu = Menu(fieldview[1,1], options=["U", "σ", "εpl"])
fieldmenu = Menu(fieldview[1,2], options=["xx", "yy", "xy"])
Label(fieldview[1,3], text="show mesh:")
togglemesh = Toggle(fieldview[1,4], active = true)


conn = dom.mesh.connectivity
X = dom.mesh.nodes[:,1]
Y = dom.mesh.nodes[:,2]

oU = map!(Observable{Any}(), timeslider.value) do val
	ti = findfirst(x->x==val, dom.ts)
	return dom.postdata.postdata[ti].U
end
Xd = map!(Observable{Any}(), oU) do U
	Ux = U[dom.dofmap[1,:]]
	X .+ Ux
end
Yd = map!(Observable{Any}(), oU) do U
	Uy = U[dom.dofmap[2,:]]
	Y .+ Uy
end
points = map!(Observable{Any}(), oU) do U
	Ux = U[dom.dofmap[1,:]]
	Uy = U[dom.dofmap[2,:]]
	Xd = X .+ Ux
	Yd = Y .+ Uy
	points = Point2f.(Xd, Yd)
end
Ux = map!(Observable{Any}(), oU) do U
	U[dom.dofmap[1,:]]
end
Uy = map!(Observable{Any}(), oU) do U
	U[dom.dofmap[2,:]]
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
	ti = findfirst(x->x==val, dom.ts)
	if item == "U"
		if field == "xx"
			dom.postdata.postdata[ti].U[dom.dofmap[1,:]].+eps()
		elseif field == "yy"
			dom.postdata.postdata[ti].U[dom.dofmap[2,:]].+eps()
		else
			zeros(length(dom.postdata.postdata[ti].U[dom.dofmap[2,:]])).+eps()
		end
	elseif item == "σ"
		if field == "xx"
			dom.postdata.postdata[ti].σ[:,1].+eps()
		elseif field == "yy"
			dom.postdata.postdata[ti].σ[:,2].+eps()
		else
			dom.postdata.postdata[ti].σ[:,3].+eps()
		end
	else
		if field == "xx"
			abs.(dom.postdata.postdata[ti].εpl[:,1]).+eps()
		elseif field == "yy"
			abs.(dom.postdata.postdata[ti].εpl[:,2]).+eps()
		else
			abs.(dom.postdata.postdata[ti].εpl[:,3]).+eps()
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



tricontourf!(ax, Xd, Yd, postData, triangulation = hcat(conn...)')
faces = [GeometryBasics.TriangleFace(conn[j][1], conn[j][2], conn[j][3]) for j = 1:length(conn)]
mesh = map!(Observable{Any}(), points) do p
	GeometryBasics.Mesh(p, faces)
end
wireframe!(ax, mesh, color = (:black, 0.75), linewidth = 0.5, transparency = true, visible=togglemesh.active)
Colorbar(mainview[1,2], limits=postData_limits)
display(f)
end