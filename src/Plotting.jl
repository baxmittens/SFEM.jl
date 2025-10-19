using GLMakie
using LinearAlgebra
using Printf

using NearestNeighbors
function elementCoords(xPt::SVector{N,Float64}, pdom, kdtree_faces) where {N}
	ind_face,dist_face = knn(kdtree_faces, xPt, 10)
	shapeFuns,dshapeFuns = pdom.refel.shapeFuns,pdom.refel.dshapeFuns
	for ind in ind_face
		success, ξ = globalToLocal(xPt, pdom.els[ind].nodes, shapeFuns, dshapeFuns)
		if success
			return success, ind, ξ 
		end
	end
	return false, -1, -ones(SVector{N,Float64})
end

function sampleLine2D(xStart, xEnd, nsamplepoints)
	xlength = norm(xEnd-xStart)
	xnormal = (xEnd-xStart)/xlength
	xdist = xlength/(nsamplepoints-1)
	xPts = Vector{SVector{2,Float64}}(undef, nsamplepoints)
	pts = Vector{Float64}(undef, nsamplepoints)
	xPts[1] = xStart
	pts[1] =  0.0
	for i = 2:nsamplepoints
		xPts[i] = xStart + (i-1)*xnormal*xdist
		pts[i] = (i-1)*xdist
	end
	return xPts, pts
end

import DensePolynomials: evaluate
function sampleResult!(valDict, pdom::ProcessDomain{P, Tri{N,M,NIPs,NM}}, xStart, xEnd, nsamplepoints) where {P,N,M,NIPs,NM}
	connectivity = pdom.connectivity
	ntimessteps = length(pdom.postdata.timesteps)
	nodes = pdom.nodes
	shapeFuns,dshapeFuns = pdom.refel.shapeFuns,pdom.refel.dshapeFuns
	kdtree_faces = KDTree(hcat(vcat(map(face->sum(map(faceind->nodes[faceind,:], face[1:3]),dims=1)./3, connectivity)...)...))
	xPts, pts = sampleLine2D(xStart, xEnd, nsamplepoints)
	elCoords = Vector{Tuple{Int, SVector{2,Float64}}}(undef, nsamplepoints)
	for (i,xPt) in enumerate(xPts)
		success, elind, ξ = elementCoords(xPt, pdom, kdtree_faces)
		@assert success "$xPt not found in processdomain $(typeof(pdom))"
		elCoords[i] = elind,ξ
	end
	pdatkeys = collect(keys(pdom.postdata.timesteps[1].pdat))
	pdatkeysizes = map(k->size(pdom.postdata.timesteps[1].pdat[k],2), pdatkeys)
	for (k,sizek) in zip(pdatkeys,pdatkeysizes)
		for r in 1:sizek
			valvec = Matrix{Float64}(undef, ntimessteps,nsamplepoints)
			for (i,(elind,ξ)) in enumerate(elCoords)
				for ti in 1:ntimessteps
					if occursin("_avg", string(k))
						elVals = pdom.postdata.timesteps[ti].pdat[k][elind,r]
						valvec[ti,i] = elVals
					else
						elVals = pdom.postdata.timesteps[ti].pdat[k][pdom.els[elind].inds,r]
						Ne = SVector{M,Float64}(ntuple(i->evaluate(shapeFuns[i], ξ), M))
						valvec[ti,i] = transpose(elVals)*Ne
					end
				end
			end
			valDict[Symbol(string(k)*"_$r")] = valvec
		end
	end
	return xPts, pts
end

function plotLine!(gridlayout, valkeys, timeslider, dom, xStart, xEnd, nsamplepoints=50, flipaxes=false)
	valDict = Dict{Symbol, Matrix{Float64}}()
	xPts, pts = sampleResult!(valDict, dom.processes[1], xStart, xEnd, nsamplepoints)
	for pdom in dom.processes[2:end]
		xPts, pts = sampleResult!(valDict, pdom, xStart, xEnd, nsamplepoints)
	end
	for (i,valkey) in enumerate(sort(collect(keys(valDict))∩valkeys))
		ax = Axis(gridlayout[i,1], ylabel=string(valkey))
		vals = map!(Observable{Any}(), timeslider.value) do ts
			vals = valDict[valkey][ts,:]
			minval = minimum(valDict[valkey])
			maxval = maximum(valDict[valkey])
			diffval = maxval-minval
			if diffval > 1e-6
				if flipaxes
					xlims!(ax, (minval-0.05*diffval, maxval+0.05*diffval))
				else
					ylims!(ax, (minval-0.05*diffval, maxval+0.05*diffval))
				end
			end
			return vals
		end
		if flipaxes
			lines!(ax, vals, pts)
		else
			lines!(ax, pts, vals)
		end
	end
	return nothing
end

function getXY(pdom,  timeslider, dispslider)
	X = map!(Observable{Any}(), timeslider.value, dispslider.value) do val,val2
		X = pdom.nodes[:,1];
		U = pdom.postdata.timesteps[val].pdat[:U][:,1].*val2
		return X .+ U
	end
	Y = map!(Observable{Any}(), timeslider.value, dispslider.value) do val,val2
		Y = pdom.nodes[:,2];
		U = pdom.postdata.timesteps[val].pdat[:U][:,1].*val2
		return Y .+ U
	end
	return X,Y
end

function getPoints2f(pdom,  timeslider, dispslider)
	points = map!(Observable{Any}(), timeslider.value, dispslider.value) do val,val2
		X = pdom.nodes[:,1];
		Y = pdom.nodes[:,2];
		U1 = similar(X)
		U2 = similar(Y)
		if haskey(pdom.postdata.timesteps[val].pdat, :U)
			U1 = pdom.postdata.timesteps[val].pdat[:U][:,1].*val2
			U2 = pdom.postdata.timesteps[val].pdat[:U][:,2].*val2
		else
			fill!(U1, 0.0)
			fill!(U2, 0.0)
		end
		return GeometryBasics.Point2f.(X .+ U1, Y .+ U2)	
	end
	return points
end

function plotConnectivity(pdom)
	_conn = pdom.connectivity;
	if length(_conn[1])==6
		conn = Vector{Vector{Int64}}(undef, 4*length(_conn))
		for i = 1:length(_conn)
			ii = 4*(i-1)+1
			conn[ii] = _conn[i][[1,4,6]]
			conn[ii+1] = _conn[i][[4,2,5]]
			conn[ii+2] = _conn[i][[6,5,3]]
			conn[ii+3] = _conn[i][[4,5,6]]
		end
	elseif length(_conn[1])==10
		conn = Vector{Vector{Int64}}(undef, 9*length(_conn))
		for i = 1:length(_conn)
			ii = 9*(i-1)+1
			conn[ii] = _conn[i][[1,4,9]]
			conn[ii+1] = _conn[i][[4,5,10]]
			conn[ii+2] = _conn[i][[5,2,6]]
			conn[ii+3] = _conn[i][[4,10,9]]
			conn[ii+4] = _conn[i][[5,6,10]]
			conn[ii+5] = _conn[i][[9,10,8]]
			conn[ii+6] = _conn[i][[10,6,7]]
			conn[ii+7] = _conn[i][[10,7,8]]
			conn[ii+8] = _conn[i][[8,7,3]]
		end
	else
		conn = _conn
	end
	return conn
end

function _getfield(x,ind)
	if size(x,2) >= ind
		return x[:,ind]
	else
		tmp = similar(x[:,1])
		fill!(tmp,0.0)
		return tmp
	end
end

function plotField!(gridlayout, pdom, fieldname::Symbol, points, conn, timeslider, fieldmenu, global_colormap, colormap=reverse(cgrad(:RdBu)))
	compdict = Dict("_1/xx"=>x->_getfield(x,1), "_2/yy"=>x->_getfield(x,2), "_3/xy"=>x->_getfield(x,3), "norm"=>x->map(norm,eachrow(x)))
	globalcolormapdict = Dict{String,Any}()
	for (k,fun) in compdict
		maxval = maximum(map(ti->maximum(fun(pdom.postdata.timesteps[ti].pdat[fieldname])), 1:length(pdom.postdata.timesteps)))
		minval = minimum(map(ti->minimum(fun(pdom.postdata.timesteps[ti].pdat[fieldname])), 1:length(pdom.postdata.timesteps)))
		if isapprox(minval,maxval, atol=1e-8)
			globalcolormapdict[k] = (minval-1,maxval+1)
		else
			globalcolormapdict[k] = (minval,maxval)
		end
	end
	postData = map!(Observable{Any}(), fieldmenu.selection, timeslider.value) do field,ti
		return compdict[field](pdom.postdata.timesteps[ti].pdat[fieldname])
	end
	ax = Axis(gridlayout[1,1], autolimitaspect = 1, title=string(fieldname))
	cr = map!(Observable{Any}(), fieldmenu.selection, global_colormap) do field, gclrm
		if gclrm
			return globalcolormapdict[field]
		else
			Makie.automatic
		end
	end
	tch = mesh!(ax, points, hcat(conn...)', color=postData, colormap=colormap, shading=false, colorrange=cr)
	Colorbar(gridlayout[1,2], tch, tickformat=vals->[@sprintf("%.2e", val) for val in vals])
	return ax
end


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
#vertices = pdom.nodes[:,1:2]
#faces = hcat(_conn...)'[:,1:3]
#facecolors = pdom.postdata.timesteps[end].pdat[:σ_avg][:,1]
#v,fa,fc = facecolor(vertices,faces,facecolors)
