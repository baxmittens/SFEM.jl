module Domains

using StaticArrays
using LinearAlgebra
using Printf
using MUMPS, MPI
import ..MeshReader: GmshMesh
import ..Elements: GenericRefElement, GenericElement, EvaluatedShapeFunctions, dim, elStiffness, saveHistory!, nips, Tri3, Tri6, elMass, elPost
import ..IntegrationRules: gaussSimplex


include("./Domains/assembler.jl")
include("./Domains/malloc.jl")

mutable struct Domain
	mma::Malloc
	mesh::GmshMesh
	refel::GenericRefElement
	els::Vector{Tri3}
	nnodes::Int
	ndofs::Int
	nels::Int
	ndofs_el::Int
	dofmap::Matrix{Int}
	cmap::Vector{Int}
	ucmap::Vector{Int}
	shapeFuns::EvaluatedShapeFunctions
	ts::Vector{Float64}
	actt::Int
	MMat::SparseMatrixCSC{Float64, Int64}
	postdata::PostData
	function Domain(mesh, els::Vector{Tri3}, RefEl::Type{T}, nips, ts) where {T<:GenericRefElement}
		refel = RefEl()
		nels = length(els)
		nnodes = size(mesh.nodes,1)
		ndofs = size(mesh.nodes,1)*dim(refel)
		ndofs_el = length(els[1].inds)*dim(els[1])
		dofmap = convert(Matrix{Int}, reshape(1:ndofs,2,:))
		mma = Malloc(nels,ndofs,ndofs_el, length(els[1].inds), nnodes)
		cmap = Vector{Int}()
		ucmap = Vector{Int}()
		shapeFuns = EvaluatedShapeFunctions(refel, gaussSimplex, nips)
		elMMats = SMatrix{3, 3, Float64, 9}[elMass(el, dofmap, shapeFuns) for el in els];
		MMat = assembleMass!(mma.Im, mma.Jm, mma.Vm, dofmap, els, elMMats, nnodes, length(els[1].inds))
		return new(mma, mesh, refel, els, nnodes, ndofs, nels, ndofs_el, dofmap, cmap, ucmap, shapeFuns, ts, 0, MMat, PostData(ndofs, nnodes, length(ts)))
	end
end

function setBC!(dom::Domain, Uval)
	mesh = dom.mesh
	U = dom.mma.U
	dofmap = dom.dofmap
	inds_xc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(mesh.nodes))
	inds_yc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9) && isapprox(x[2],0.5,atol=1e-9), eachrow(mesh.nodes))
	inds_xc_1 = findall(x->isapprox(x[1],10.0,atol=1e-9), eachrow(mesh.nodes))
	uc_y_0 = dofmap[2,inds_yc_0]
	uc_x_0 = dofmap[1,inds_xc_0]
	uc_y_1 = dofmap[2,inds_xc_1]
	U[uc_y_1] .= -Uval
	return nothing
end

function setBCandUCMaps!(dom::Domain, Uval)
	mesh = dom.mesh
	U = dom.mma.U
	dofmap = dom.dofmap
	inds_xc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(mesh.nodes))
	inds_yc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9) && isapprox(x[2],0.5,atol=1e-9), eachrow(mesh.nodes))
	inds_xc_1 = findall(x->isapprox(x[1],10.0,atol=1e-9), eachrow(mesh.nodes))
	uc_y_0 = dofmap[2,inds_yc_0]
	uc_x_0 = dofmap[1,inds_xc_0]
	uc_y_1 = dofmap[2,inds_xc_1]
	U[uc_y_1] .= -Uval
	dom.cmap = vcat(uc_y_0, uc_x_0, uc_y_1)
	dom.ucmap = setdiff(1:dom.ndofs, dom.cmap)
	return nothing
end

function solveMUMPS!(ΔU, A,rhs)
	MPI.Initialized() ? nothing : MPI.Init()
	icntl = default_icntl[:]
	icntl[1] = 0
	icntl[2] = 0
	icntl[3] = 0
	icntl[4] = 0
	mumps1_unsafe = Mumps{Float64}(mumps_definite, icntl, default_cntl32);
	associate_matrix!(mumps1_unsafe, A; unsafe = true)
	factorize!(mumps1_unsafe);
	associate_rhs!(mumps1_unsafe, rhs; unsafe = true)
	MUMPS.solve!(mumps1_unsafe)
	MUMPS.get_sol!(ΔU, mumps1_unsafe)
	finalize(mumps1_unsafe)
	return nothing
end

function solve!(dom::Domain)
	I,J,V,U,ΔU,F,els,ndofs = dom.mma.I,dom.mma.J,dom.mma.V,dom.mma.U,dom.mma.ΔU,dom.mma.F,dom.els,dom.ndofs
	dofmap,ucmap,cmap,shapeFuns,ndofs_el,actt = dom.dofmap, dom.ucmap, dom.cmap, dom.shapeFuns,dom.ndofs_el, dom.actt
	#@info "Element Stiffness"
	elMats = Tuple{SMatrix{6, 6, Float64, 36}, SVector{6, Float64}}[elStiffness(el, dofmap, U, ΔU, shapeFuns, actt) for el in els];
	#@info "Assemble"
	Kglob = assemble!(I, J, V, F, dofmap, els, elMats, ndofs, ndofs_el)
	#display(Kglob)
	#@info "Solve"
	ΔU[ucmap] = Kglob[ucmap, ucmap] \ F[ucmap] #( F[ucmap] - Kglob[ucmap, cmap] * U[cmap])
	#@time solveMUMPS!(ΔU[ucmap], Kglob[ucmap, ucmap], F[ucmap])
	#F[cmap] =  Kglob[cmap, ucmap] * U[ucmap] + Kglob[cmap, cmap] * U[cmap]
	U .+= ΔU
	return nothing
end

function postSolve!(dom::Domain)
	@time elPosts = Tuple{SMatrix{3, 3, Float64, 9},SMatrix{3, 3, Float64, 9}}[elPost(el, dom.dofmap, dom.shapeFuns, dom.actt) for el in dom.els];
	assemblePost!(dom.mma.σ, dom.mma.εpl, dom.dofmap, dom.els, elPosts, dom.nnodes, length(dom.els[1].inds))
	dom.postdata.postdata[dom.actt].U .= dom.mma.U
	dom.postdata.postdata[dom.actt].σ .= dom.MMat \ dom.mma.σ
	dom.postdata.postdata[dom.actt].εpl .= dom.MMat \ dom.mma.εpl
	return nothing
end

function saveHistory!(dom)
	foreach(el->saveHistory!(el,dom.actt), dom.els) 
end

function init_loadstep!(dom, loadstep)	
	fill!(dom.mma.ΔU, 0.0)
	fill!(dom.mma.F, 0.0)
	setBC!(dom, loadstep)
	return nothing
end

function newtonraphson!(dom)
	init_loadstep!(dom, dom.ts[dom.actt])
	normdU = Inf
	println("converg. history")
	numit = 0
	while normdU>1e-7 && numit < 10
		solve!(dom)
		normdU = norm(dom.mma.ΔU)
		strnormdU = @sprintf("%.4e", normdU) 
		println("normdU = $strnormdU")
		numit += 1
	end
end

function tsolve!(dom::Domain)
	setBCandUCMaps!(dom, 0.0)
	for t in dom.ts
		@info "newtonraphson solve t=$t"
		dom.actt += 1
		@time newtonraphson!(dom)
		saveHistory!(dom)
		postSolve!(dom)
	end
end

end #module Domains