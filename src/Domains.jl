module Domains

using Base.Threads
using SparseArrays
using StaticArrays
using LinearAlgebra
using Printf
#using MUMPS, MPI
import ..MeshReader: GmshMesh
import ..Elements: GenericRefElement, GenericElement, EvaluatedShapeFunctions, dim, elStiffness, saveHistory!, 
	Tri, Tri3, Tri6, elMass, elPost, updateTrialStates!, σ_avg, RefEl, flatten_tuple
import ..IntegrationRules: gaussSimplex
import Pardiso


include("./Domains/malloc.jl")
include("./Domains/assembler.jl")

abstract type LinearSolver; end
abstract type PardisoSolver <: LinearSolver; end
abstract type UMPFPackSolver <: LinearSolver; end
abstract type MUMPSSolver <: LinearSolver; end

mutable struct Domain{T}
	mma::Malloc
	mesh::GmshMesh
	refel::GenericRefElement
	els::Vector{T}
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
	SOLVER::DataType
	function Domain(mesh, els::Vector{T}, nips, ts) where {T}
		refel = RefEl(T)
		nels = length(els)
		nnodes = size(mesh.nodes,1)
		ndofs = size(mesh.nodes,1)*dim(refel)
		ndofs_el = length(els[1].inds)*dim(els[1])
		dofmap = convert(Matrix{Int}, reshape(1:ndofs,2,:))
		mma = Malloc(nels,ndofs, ndofs_el, Val{length(els[1].inds)}, nnodes)
		cmap = Vector{Int}()
		ucmap = Vector{Int}()
		shapeFuns = EvaluatedShapeFunctions(refel, gaussSimplex, nips)
		elMMats = [elMass(el, dofmap, shapeFuns) for el in els];
		MMat = assembleMass!(mma.Im, mma.Jm, mma.Vm, dofmap, els, elMMats, nnodes)
		#MMat = spzeros(10,10)
		if haskey(ENV, "MKLROOT")
			SOLVER = PardisoSolver
		else
			SOLVER = UMPFPackSolver
		end
		return new{T}(mma, mesh, refel, els, nnodes, ndofs, nels, ndofs_el, dofmap, cmap, ucmap, shapeFuns, ts, 0, MMat, PostData(ndofs, nnodes, length(ts), nels), SOLVER)
	end
end

function setBC!(dom::Domain, Uval)
	mesh = dom.mesh
	U = dom.mma.U
	ΔU = dom.mma.ΔU
	dofmap = dom.dofmap
	inds_xc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(mesh.nodes))
	#inds_yc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9) && isapprox(x[2],0.5,atol=1e-9), eachrow(mesh.nodes))
	inds_yc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(mesh.nodes))
	inds_xc_1 = findall(x->isapprox(x[1],10.0,atol=1e-9), eachrow(mesh.nodes))
	uc_y_0 = dofmap[2,inds_yc_0]
	uc_x_0 = dofmap[1,inds_xc_0]
	uc_y_1 = dofmap[2,inds_xc_1]
	ΔU[uc_y_1] .= (Uval .- U[uc_y_1])
	return nothing
end

function setBCandUCMaps!(dom::Domain, Uval)
	mesh = dom.mesh
	U = dom.mma.U
	ΔU = dom.mma.ΔU
	dofmap = dom.dofmap
	inds_xc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(mesh.nodes))
	#inds_yc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9) && isapprox(x[2],0.5,atol=1e-9), eachrow(mesh.nodes))
	inds_yc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(mesh.nodes))
	inds_xc_1 = findall(x->isapprox(x[1],10.0,atol=1e-9), eachrow(mesh.nodes))
	uc_y_0 = dofmap[2,inds_yc_0]
	uc_x_0 = dofmap[1,inds_xc_0]
	uc_y_1 = dofmap[2,inds_xc_1]
	#U[uc_y_1] .= Uval
	ΔU[uc_y_1] .= (Uval .- U[uc_y_1])
	dom.cmap = vcat(uc_y_0, uc_x_0, uc_y_1)
	dom.ucmap = setdiff(1:dom.ndofs, dom.cmap)
	return nothing
end


#function solve!(::Type{MUMPSSolver}, x, A, rhs::AbstractVector{Float64})
#
#    if !MPI.Initialized()
#        MPI.Init()
#    end
#    comm = MPI.COMM_WORLD
# 
#    rhs_work = copy(rhs)
#
#    icntl = default_icntl[:]
#    #icntl[1:4] .= 0
#    m = MUMPS.Mumps{Float64}(mumps_unsymmetric, icntl, default_cntl32);
#    
#    MUMPS.associate_matrix!(m, A; unsafe = false)
#    MUMPS.factorize!(m)
#    MUMPS.associate_rhs!(m, copy(rhs); unsafe = false)
#    MUMPS.mumps_solve!(x, m)
#    MUMPS.finalize!(m)
#    MPI.Barrier(comm)
#    return x
#end

function solve!(::Type{PardisoSolver}, x, A, rhs::AbstractVector{Float64})
	ps = Pardiso.MKLPardisoSolver()
	Pardiso.set_nprocs!(ps, Base.Threads.nthreads())
	Pardiso.set_matrixtype!(ps, 11)
	#Pardiso.solve!(ps, x, A, rhs)
	Pardiso.pardiso(ps, x, A, rhs)
	return nothing
end

function solve!(::Type{UMPFPackSolver}, x, A, rhs::AbstractVector{Float64})
	copy!(x, A \ rhs)
	return nothing
end

function solve!(dom::Domain)
	I,J,V,U,ΔU,F,els,ndofs,elMats = dom.mma.I,dom.mma.J,dom.mma.V,dom.mma.U,dom.mma.ΔU,dom.mma.F,dom.els,dom.ndofs,dom.mma.elMats
	dofmap,ucmap,cmap,shapeFuns,ndofs_el,actt = dom.dofmap, dom.ucmap, dom.cmap, dom.shapeFuns,dom.ndofs_el, dom.actt
	#@time elMats = Tuple{SMatrix{6, 6, Float64, 36}, SVector{6, Float64}}[elStiffness(el, dofmap, U, ΔU, shapeFuns, actt) for el in els];
	t1 = time()
	@threads for i in eachindex(els)
    	el = els[i]
    	elMats[i] = elStiffness(el, dofmap, U, ΔU, shapeFuns, actt)
	end
	t2 = time()
	println("Integrating element matrices took $(round(t2-t1,digits=2)) seconds")
	Kglob = assemble!(dom.mma, F, dofmap, els, elMats, ndofs)
	t3 = time()
	println("Assembling blfs and lfs took $(round(t3-t2,digits=2)) seconds")
	rhs = F[ucmap] -  Kglob[ucmap, cmap] * ΔU[cmap]
	Klgobuc = Kglob[ucmap, ucmap]
	x = zeros(Float64, length(ΔU[ucmap]))
	solve!(dom.SOLVER, x, Klgobuc, rhs)
	ΔU[ucmap] .= x
	t4 = time()
	println("Solving the linear system took $(round(t4-t3,digits=2)) seconds")
	percsolver =  @sprintf("%.2f", (t4-t1)/(t4-t3)*100)
	@info "Solver time: $percsolver%"
	U .+= ΔU
	return nothing
end

function updateTrialStates!(dom)
	for el in dom.els
		updateTrialStates!(el , dom.dofmap, dom.mma.U, dom.shapeFuns, dom.actt)
	end
end


function postSolve!(dom::Domain)
	@time elPosts = [elPost(el, dom.dofmap, dom.shapeFuns, dom.actt) for el in dom.els];
	assemblePost!(dom.mma.σ, dom.mma.εpl, dom.dofmap, dom.els, elPosts, dom.nnodes)
	dom.postdata.postdata[dom.actt].U .= dom.mma.U
	dom.postdata.postdata[dom.actt].σ .= dom.MMat \ dom.mma.σ
	dom.postdata.postdata[dom.actt].εpl .= dom.MMat \ dom.mma.εpl
	@threads for i in 1:dom.nels
		el = dom.els[i]
		dom.postdata.postdata[dom.actt].σ_avg[i] = σ_avg(el, dom.actt)
	end
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
	numit = 0
	str = "\nconvergence history"
	@info "Start Newton-Raphson iteration"
	while normdU>1e-7 && numit < 10
		@info "Newton-Raphson iteration $(numit+1)"
		solve!(dom)
		updateTrialStates!(dom)
		normdU = norm(dom.mma.ΔU)
		fill!(dom.mma.ΔU,0.0)
		strnormdU = @sprintf("%.4e", normdU) 
		str *= "\nstep $(numit+1): normdU = $strnormdU"
		numit += 1
	end
	println(str)
	println()
end

function tsolve!(dom::Domain)
	dom.actt = 0
	setBCandUCMaps!(dom, 0.0)
	for t in dom.ts
		@info "newtonraphson solve t=$t"
		dom.actt += 1
		t1 = time()
		newtonraphson!(dom)
		@info "Save history variables"
		@time saveHistory!(dom)
		@info "Postprocessing"
		@time postSolve!(dom)
		t2 = time()
		timestr = @sprintf("%.2f", t2-t1)
		@info "Analysis time $timestr seconds"
		println()
	end
end

end #module Domains