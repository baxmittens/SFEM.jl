ENV["OMP_NUM_THREADS"] = Base.Threads.nthreads()
ENV["MKL_NUM_THREADS"] = Base.Threads.nthreads()
ENV["OPENBLAS_NUM_THREADS"] = Base.Threads.nthreads()

#using Logging
#global_logger(ConsoleLogger(stdout))

import SFEM
import SFEM: LinearElasticity, HeatConduction
import SFEM.Elements: Tri3, Tri6, Tri, ElementStateVars2D, MatPars, Line
import SFEM.MeshReader: GmshMesh
import SFEM.Domains: ProcessDomain, Domain, solve!, setBCandUCMaps!, init_loadstep!, tsolve!

using StaticArrays
using LinearAlgebra
using VTUFileHandler

#meshfilepath = "../models/2d/beam_medium_tri6.msh"
#mesh = GmshMesh(meshfilepath);
#mesh.nodes nnodes×3 Matrix{Float64}
#connectivity Vector{Vector{Int64}

meshfilepath = "../models/2d/model1.vtu"
mesh = VTUFile(meshfilepath);
matids = mesh["MaterialIDs"]
_connectivity = reshape(mesh["connectivity"],6,:).+1
nodes = collect(reshape(mesh["Points"],3,:)')
connectivity = Vector{Vector{Int}}()
#tria=[1 2 4 5 9 8]
#trib=[2 3 4 6 7 9]
for (i,conn) in enumerate(eachcol(_connectivity))
	push!(connectivity, _connectivity[:,i])
#	push!(connectivity, vec(map(i->conn[i], trib)))
#	push!(matids, _matids[i])
#	push!(matids, _matids[i])
end

nips = 7
ts = collect(range(0,63115200000.0,50))
nts = length(ts)

states = [ElementStateVars2D(Val{nips},Val{nts}) for elinds in connectivity];

function lin_func(x,xmin,ymin,xmax,ymax)
	a = (ymax-ymin)/(xmax-xmin)
	b = ymax-a*xmax
	return a*x+b
end

#funT(x, matpars, actt, ts=ts) = matpars.materialID==0 && actt > 1 && actt < 26 ? lin_func(ts[actt],0.0,51.0,31557600000.0,0) : 0.0
funT(x, matpars, actt, ts=ts) = 0.0
funM(x, matpars, actt, ts=ts) = actt > 1 ? SVector{2,Float64}(0.0,0.0) : SVector{2,Float64}(0.0,0.0)
matpars = MatPars(7000.0, 450.0, 1e-5, 1e-5, 0.0, 50.0, 50.0, 0.0, 2.1e11, 0.3, 200.0, funM, funT, 1)
matpars0 = MatPars(6700.0, 500.0, 1.7e-05, 1.7e-05, 0.0, 16.0, 16.0, 0.0, 195000000000.0, 0.3, Inf, funM, funT, 0)
matpars1 = MatPars(1575.0, 1090.0,  2.5e-5, 2.5e-5, 0.0, 1.17, 1.17, 0.0, 100000000.0, 0.1, Inf, funM, funT, 1)
matpars2 = MatPars(2495.0, 1060.0,  2e-5, 2e-5, 0.0, 1.84, 1.84, 0.0, 5000000000.0, 0.3, Inf, funM, funT, 2)
matparsdict = Dict(0=>matpars0, 1=>matpars1, 2=>matpars2)
els1 = Tri{2,6,nips,12}[Tri6(SMatrix{2,6,Float64,12}(nodes[elinds,1:2]'), SVector{6,Int}(elinds), state, matparsdict[matids[i]], Val{nips}) for (i,(elinds,state)) in enumerate(zip(connectivity, states))];
els2 = Tri{2,6,nips,12}[Tri6(SMatrix{2,6,Float64,12}(nodes[elinds,1:2]'), SVector{6,Int}(elinds), state, matparsdict[matids[i]], Val{nips}) for (i,(elinds,state)) in enumerate(zip(connectivity, states))];

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

function dirichletT(ΔU, U, nodes, dofmap, actt)
	inds_bottom = findall(x->isapprox(x[2],-100.0,atol=1e-9), eachrow(nodes))
	inds_top = findall(x->isapprox(x[2],100.0,atol=1e-9), eachrow(nodes))
	bottom_bc = dofmap[1,inds_bottom]
	top_bc = dofmap[1,inds_top]
	return vcat(bottom_bc,top_bc)
	#return Int[]

end

ndofs1 = size(nodes,1)*2
dofmap1 = convert(Matrix{Int}, reshape(1:ndofs1,2,:))
ndofs2 = size(nodes,1)*1
dofmap2 = convert(Matrix{Int}, reshape(ndofs1+1:ndofs1+ndofs2,1,:))

neumann_inds = findall(x->isapprox(x[2],100.0,atol=1e-5), eachrow(nodes))
lines = [SVector{3,Int}(1,4,2), SVector{3,Int}(2,5,3), SVector{3,Int}(3,6,1)]
nips_neumann = 3
neumann_els_M = Line{2,3,nips_neumann,6}[]
neumann_els_T = Line{2,3,nips_neumann,6}[]
for el in els1
	for line in lines
		inds = el.inds[line]
		if all(map(x->x ∈ neumann_inds, inds))
			push!(neumann_els_M, Line(SMatrix{2,3,Float64,6}(nodes[inds,1:2]'), inds, Val{nips_neumann}))
		end
	end
end
#fun_neumann_M(x, actt, ts=ts) = actt > 1 ? SVector{2,Float64}(0.0,0.0) : SVector{2,Float64}(0.0,0.0)
fun_neumann_M(x, actt, ts=ts) = SVector{2,Float64}(0.0,-14000000.0)
#fun_neumann_M(x, actt, ts=ts) = SVector{2,Float64}(0.0,0.0)

linelasticity = ProcessDomain(LinearElasticity, nodes, connectivity, els1, dofmap1, nts, Val{2}, Line{2,3,nips_neumann,6}, els_neumann=neumann_els_M, fun_neumann=fun_neumann_M)
heatconduction = ProcessDomain(HeatConduction, nodes, connectivity, els2, dofmap2, nts, Val{1}, Nothing)
dom = Domain((linelasticity,heatconduction), ts, dirichletM=dirichletM, dirichletT=dirichletT)
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
timeslider = Slider(controlview[1,2], range = ts[2:end], startvalue=ts[end], update_while_dragging=false)
dispmult = Slider(controlview[3,2], range = [1,10,100,1000,2000,5000,10000,20000], startvalue=1, update_while_dragging=false)
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
togglemesh = Toggle(fieldview[1,4], active = false)


_conn = dom.processes[1].connectivity;

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
pdom = dom.processes[1];
X = pdom.nodes[:,1];
Y = pdom.nodes[:,2];

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
#	if abs(lims[2]-lims[1]) < 1e-6
#		return lims
#	else
#		return lims
#	end
	return lims
end

#tricontourf!(ax, Xd, Yd, postData, triangulation = hcat(conn...)',levels=80, colormap=:prism)
#tch = tricontourf!(ax, Xd, Yd, postData, triangulation = hcat(conn...)', levels=31, colormap=:RdBu)
cmap = cgrad(:RdBu)
cmap_reversed = reverse(cmap)
tch = mesh!(points, hcat(conn...)', color=postData,colormap=cmap_reversed, shading=false)
faces = [GeometryBasics.TriangleFace(conn[j][1], conn[j][2], conn[j][3]) for j = 1:length(conn)]
mesh = map!(Observable{Any}(), points) do p
	GeometryBasics.Mesh(p, faces)
end
wireframe!(ax, mesh, color = (:black, 0.75), linewidth = 0.5, transparency = true, visible=togglemesh.active)
#Colorbar(mainview[1,2], limits=postData_limits, colormap=:RdBu) #colormap=:prism)
Colorbar(mainview[1,2], tch) #colormap=:prism)
#Colorbar(mainview[1,2], limits=postData_limits)
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
#vertices = pdom.nodes[:,1:2]
#faces = hcat(_conn...)'[:,1:3]
#facecolors = pdom.postdata.timesteps[end].pdat[:σ_avg]
#v,fa,fc = facecolor(vertices,faces,facecolors)
#cm_repo_2d = mesh!(ax, v, fa, color=fc, shading=NoShading)

end


#f = Figure(size=(1000,600));
#ax = Axis(f[1,1])
#_pdat = dom.processes[1].postdata.timesteps[1].pdat[:σ][:,1]
#pdat = zeros(Float64, length(_pdat))
#tch = mesh!(Point2f.(X, Y), hcat(conn...)', color=pdat,colormap=:RdBu)
#f









