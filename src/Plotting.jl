using GLMakie
using GeometryBasics
using LinearAlgebra

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
fieldmenu = Menu(fieldview[1,2], options=["xx", "yy", "xy", "eq"])
Label(fieldview[1,3], text="show mesh:")
togglemesh = Toggle(fieldview[1,4], active = true)


_conn = dom.mesh.connectivity

if length(_conn[1])>3
	conn = Vector{Vector{Int64}}(undef, 4*length(_conn))
	for i = 1:length(_conn)
		ii = 4*(i-1)+1
		conn[ii] = _conn[i][[1,2,4]]
		conn[ii+1] = _conn[i][[4,2,5]]
		conn[ii+2] = _conn[i][[6,5,3]]
		conn[ii+3] = _conn[i][[4,5,6]]
	end
else
	conn = _conn
end

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
		elseif field == "xy"
			dom.postdata.postdata[ti].σ[:,3].+eps()
		else
			norm.(vcat(dom.postdata.postdata[ti].σ[:,1], dom.postdata.postdata[ti].σ[:,3], dom.postdata.postdata[ti].σ[:,2], dom.postdata.postdata[ti].σ[:,3])).+eps()
		end
	else
		if field == "xx"
			abs.(dom.postdata.postdata[ti].εpl[:,1]).+eps()
		elseif field == "yy"
			abs.(dom.postdata.postdata[ti].εpl[:,2]).+eps()
		elseif field == "xy"
			abs.(dom.postdata.postdata[ti].εpl[:,3]).+eps()
		else
			norm.(vcat(dom.postdata.postdata[ti].εpl[:,1], dom.postdata.postdata[ti].εpl[:,3]./2.0, dom.postdata.postdata[ti].εpl[:,2], dom.postdata.postdata[ti].εpl[:,3]./2.0)).+eps()
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



tricontourf!(ax, Xd, Yd, postData, triangulation = hcat(conn...)',levels=40)

faces = [GeometryBasics.TriangleFace(conn[j][1], conn[j][2], conn[j][3]) for j = 1:length(conn)]
mesh = map!(Observable{Any}(), points) do p
	GeometryBasics.Mesh(p, faces)
end
wireframe!(ax, mesh, color = (:black, 0.75), linewidth = 0.5, transparency = true, visible=togglemesh.active)
Colorbar(mainview[1,2], limits=postData_limits)
display(f)
function facecolor(vertices,faces,facecolors)
	v = zeros(size(faces,1)*3,2)
	f = zeros(Int, size(faces))
	fc = zeros(size(v,1))
	for i in 1:size(faces,1)
		face = faces[i,:]
		verts = vertices[face,:]
		j = 3*(i-1)+1
		f[i,:] = [j,j+1,j+2]
		v[j:j+2,:] .= verts
		fc[j:j+2] .= facecolors[i]
	end
	return v,f,fc
end
vertices = dom.mesh.nodes[:,1:2]
faces = hcat(conn...)'
facecolors = dom.postdata.postdata[end].σ_avg
v,fa,fc = facecolor(vertices,faces,facecolors)
cm_repo_2d = mesh!(ax, v, fa, color=fc, shading=NoShading)
