module Domains

using Base.Threads
using SparseArrays
using StaticArrays
using LinearAlgebra
using Printf
#using MUMPS, MPI
import ..MeshReader: GmshMesh
import ..Elements: GenericRefElement, GenericElement, EvaluatedShapeFunctions, dim, elStiffness, saveHistory!, 
	Tri, Tri3, Tri6, elMass, elPost, updateTrialStates!, initStates!, σ_avg, RefEl, flatten_tuple
import ..IntegrationRules: gaussSimplex
import Pardiso
import ...SFEM: Process, LinearElasticity, HeatConduction

include("./Domains/malloc.jl")

abstract type LinearSolver; end
abstract type PardisoSolver <: LinearSolver; end
abstract type UMPFPackSolver <: LinearSolver; end
abstract type MUMPSSolver <: LinearSolver; end

mutable struct ProcessDomain{P,T,ESF}
	mma::ProcessDomainMalloc
	nodes::Matrix{Float64}
	connectivity::Vector{Vector{Int64}}
	refel::GenericRefElement
	els::Vector{T}
	nnodes::Int
	nels::Int
	ndofs_el::Int
	dofmap::Matrix{Int}
	shapeFuns::ESF
	MMat::SparseMatrixCSC{Float64, Int64}
	postdata::PostData
	function ProcessDomain(process::Type{P}, nodes, connectivity, els::Vector{T}, dofmap, nips, nts) where {P<:Process,T}
		refel = RefEl(T)
		nels = length(els)
		nnodes = size(nodes,1)
		ndofs_el = length(els[1].inds)*dim(els[1])
		mma = ProcessDomainMalloc(nels, ndofs_el, Val{length(els[1].inds)}, nnodes) 
		shapeFuns = EvaluatedShapeFunctions(refel, gaussSimplex, nips)
		ESF = typeof(shapeFuns)
		elMMats = [elMass(el, shapeFuns) for el in els];
		MMat = assembleMass!(mma.Im, mma.Jm, mma.Vm, els, elMMats, nnodes)
		return new{P,T,ESF}(mma, nodes, connectivity, refel, els, nnodes, nels, ndofs_el, dofmap, shapeFuns, MMat, PostData(P, nnodes, nts, nels))
	end
end

mutable struct Domain{T}
	processes::T
	mma::DomainMalloc
	ndofs::Int
	nels::Int
	cmap::Vector{Int}
	ucmap::Vector{Int}
	loadsteps::Vector{Float64}
	timesteps::Vector{Float64}
	actt::Int
	SOLVER::DataType
	function Domain(processes::T,loadsteps::AbstractVector{Float64}, timesteps::AbstractVector{Float64}) where {T}
		
		@assert length(loadsteps) == length(timesteps)
		@assert all(map(x->length(x.els), processes) .== length(processes[1].els))
		
		ndofs_el = sum(map(p->p.ndofs_el, processes))
		nels = processes[1].nels
		ndofs = sum(map(p->size(p.nodes,1) * 2, processes))
		mma = DomainMalloc(nels,ndofs,ndofs_el)
		cmap = Vector{Int}()
		ucmap = Vector{Int}()

		if haskey(ENV, "MKLROOT")
			SOLVER = PardisoSolver
		else
			SOLVER = UMPFPackSolver
		end
		return new{T}(processes, mma, ndofs, nels, cmap, ucmap, loadsteps, timesteps, 0, SOLVER)
	end
end

include("./Domains/assembler.jl")

function setBC!(dom::Domain, eladom::ProcessDomain{LinearElasticity,T}, Uval) where {T}
	nodes = eladom.nodes
	U = dom.mma.U
	ΔU = dom.mma.ΔU
	dofmap = eladom.dofmap
	inds_xc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(nodes))
	#inds_yc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9) && isapprox(x[2],0.5,atol=1e-9), eachrow(mesh.nodes))
	inds_yc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(nodes))
	inds_xc_1 = findall(x->isapprox(x[1],10.0,atol=1e-9), eachrow(nodes))
	uc_y_0 = dofmap[2,inds_yc_0]
	uc_x_0 = dofmap[1,inds_xc_0]
	uc_y_1 = dofmap[2,inds_xc_1]
	ΔU[uc_y_1] .= (Uval .- U[uc_y_1])
	return vcat(uc_y_0, uc_x_0, uc_y_1)
end

function setBCandCMap!(dom::Domain, eladom::ProcessDomain{LinearElasticity,T}, Uval) where {T}
	inds = setBC!(dom, eladom, Uval)
	append!(dom.cmap, inds)
	return nothing
end

function setBC!(dom::Domain, Uval)
	foreach(proc->setBC!(dom, proc, Uval), dom.processes)
	return nothing
end

function setBCandUCMaps!(dom::Domain, Uval)
	empty!(dom.ucmap)
	empty!(dom.cmap)
	foreach(pdom->setBCandCMap!(dom, pdom, Uval), dom.processes)
	append!(dom.ucmap, setdiff(1:dom.ndofs, dom.cmap))
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

function integrate!(::Type{LinearElasticity}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{ENNODES,ENNODES,Float64,ENNODESSQ}, SVector{ENNODES,Float64}}}, dofmap, shapeFuns, U, ΔU, actt) where {T, ENNODES, ENNODESSQ}
	@threads for i in eachindex(els)
    	el = els[i]
    	elMats[i] = elStiffness(el, dofmap, U, ΔU, shapeFuns, actt)
	end
	return nothing
end

function integrate!(pdom::ProcessDomain{P,T}, U, ΔU, actt) where {P,T}
	els = pdom.els
	elMats = pdom.mma.elMats
	integrate!(P, els, elMats, pdom.dofmap, pdom.shapeFuns, U, ΔU, actt)
	return nothing		
end

function integrate!(::Type{Val{N}}, pdoms::Tuple{ProcessDomain{P1,T1},ProcessDomain{P2,T2}}, U, ΔU, actt) where {N, P1,P2,T1,T2}
	els1 = pdoms[N].els
	els2 = pdoms[rem(N,2)+1].els
	elMats = pdom.mma.elMats
	@threads for i in eachindex(pdom.els)
    	el1 = els1[i]
    	el2 = els2[i]
    	elMats[i] = elStiffness(el, pdom.dofmap, U, ΔU, pdom.shapeFuns, actt)
	end
end

function integrate!(dom::Domain{Tuple{PD}}) where {PD}
	integrate!(dom.processes[1], dom.mma.U, dom.mma.ΔU, dom.actt)
	return nothing
end

function solve!(dom::Domain)
	ucmap,cmap = dom.ucmap, dom.cmap
	t1 = time()
	integrate!(dom)
	t2 = time()
	println("Integrating element matrices took $(round(t2-t1,digits=8)) seconds")
	Kglob = assemble!(dom)
	#Kglob = SFEM.Domains.assemble!(dom)
	t3 = time()
	println("Assembling blfs and lfs took $(round(t3-t2,digits=8)) seconds")
	F,ΔU,U = dom.mma.F,dom.mma.ΔU,dom.mma.U
	rhs = F[ucmap] -  Kglob[ucmap, cmap] * ΔU[cmap]
	Klgobuc = Kglob[ucmap, ucmap]
	x = zeros(Float64, length(ΔU[ucmap])) #???
	solve!(dom.SOLVER, x, Klgobuc, rhs)
	ΔU[ucmap] .= x
	t4 = time()
	println("Solving the linear system took $(round(t4-t3,digits=8)) seconds")
	percsolver =  @sprintf("%.2f", (t4-t3)/(t4-t1)*100)
	@info "Solver time: $percsolver%"
	U .+= ΔU
	return nothing
end

function updateTrialStates!(::Type{P}, els::Vector{T}, dofmap, U, shapeFuns::ESF, actt) where {P,T,ESF}
	for el in els
		updateTrialStates!(P, el, dofmap, U, shapeFuns, actt)
	end
	return nothing
end
function updateTrialStates!(pdom::ProcessDomain{P,T,ESF}, U, actt) where {P,T,ESF}
	updateTrialStates!(P, pdom.els, pdom.dofmap, U, pdom.shapeFuns, actt)
	return nothing
end
function updateTrialStates!(dom::Domain{Tuple{PD}}) where {PD}
	updateTrialStates!(dom.processes[1], dom.mma.U, dom.actt)
	return nothing
end

function initStates!(pdom::ProcessDomain{P,T}) where {P,T}
	for el in pdom.els
		initStates!(P, el)
	end
	return nothing
end

function initStates!(dom::Domain{T}) where {T}
	for pdom in dom.processes
		initStates!(pdom)
	end
	return nothing
end

function postSolve!(pdom::ProcessDomain{LinearElasticity, T}, U, actt) where {T}
	@time elPosts = [elPost(el, pdom.shapeFuns, actt) for el in pdom.els];
	assemblePost!(pdom.mma.σ, pdom.mma.εpl, pdom.els, elPosts, pdom.nnodes)
	pdom.postdata.timesteps[actt].pdat[:U] .= transpose(U[pdom.dofmap])
	pdom.postdata.timesteps[actt].pdat[:σ] .= pdom.MMat \ pdom.mma.σ
	pdom.postdata.timesteps[actt].pdat[:εpl] .= pdom.MMat \ pdom.mma.εpl
	σ_avg_mat = pdom.postdata.timesteps[actt].pdat[:σ_avg]
	@threads for i in 1:pdom.nels
		el = pdom.els[i]
		σ_avg_mat[i,:] .= σ_avg(el, actt)
	end
	return nothing
end

function postSolve!(dom::Domain{T}) where {T}
	foreach(pdom->postSolve!(pdom, dom.mma.U, dom.actt), dom.processes)
	return nothing
end

function saveHistory!(dom::Domain)
	foreach(el->saveHistory!(el, dom.actt), dom.processes[1].els) 
end

function init_loadstep!(dom, loadstep)	
	fill!(dom.mma.ΔU, 0.0)
	fill!(dom.mma.F, 0.0)
	setBC!(dom, loadstep)
	return nothing
end

function newtonraphson!(dom::Domain)
	init_loadstep!(dom, dom.loadsteps[dom.actt])
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
	fill!(dom.mma.U, 0.0)
	initStates!(dom)
	setBCandUCMaps!(dom, 0.0)
	for t in dom.loadsteps
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