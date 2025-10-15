module Domains

using Base.Threads
using SparseArrays
using StaticArrays
using LinearAlgebra
using Printf
#using MUMPS, MPI
import ..MeshReader: GmshMesh
import ..Elements: GenericRefElement, GenericElement, EvaluatedShapeFunctions, dim, elStiffness, elStiffnessTM, elStiffnessT, saveHistory!, 
	Tri, Tri3, Tri6, elMass, elPost, elPostT, updateTrialStates!, initStates!, σ_avg, RefEl, flatten_tuple
import ..IntegrationRules: gaussSimplex
import Pardiso
import ...SFEM: Process, LinearElasticity, HeatConduction

include("./Domains/malloc.jl")

abstract type LinearSolver; end
abstract type PardisoSolver <: LinearSolver; end
abstract type UMPFPackSolver <: LinearSolver; end
abstract type MUMPSSolver <: LinearSolver; end

mutable struct ProcessDomain{P,T,ESF,DMD1}
	mma::ProcessDomainMalloc
	nodes::Matrix{Float64}
	connectivity::Vector{Vector{Int64}}
	refel::GenericRefElement
	els::Vector{T}
	nnodes::Int
	nels::Int
	dofmap::Matrix{Int}
	shapeFuns::ESF
	MMat::SparseArrays.UMFPACK.UmfpackLU{Float64, Int64}
	postdata::PostData
	function ProcessDomain(::Type{P}, nodes, connectivity, els::Vector{T}, dofmap, nips, nts, ::Type{Val{DOFMAPDIM1}}) where {P<:Process,T,DOFMAPDIM1}
		refel = RefEl(T)
		nels = length(els)
		nnodes = size(nodes,1)
		mma = ProcessDomainMalloc(nels, Val{length(els[1].inds)}, nnodes) 
		shapeFuns = EvaluatedShapeFunctions(refel, gaussSimplex, nips)
		ESF = typeof(shapeFuns)
		@threads for i in 1:nels
			el = els[i]
			mma.elMMats[i] = elMass(el, shapeFuns)
		end
		MMat = lu(assembleMass!(mma.Im, mma.Jm, mma.Vm, els, mma.elMMats, nnodes))
		return new{P,T,ESF,DOFMAPDIM1}(mma, nodes, connectivity, refel, els, nnodes, nels, dofmap, shapeFuns, MMat, PostData(P, nnodes, nts, nels))
	end
end

mutable struct Domain{T}
	processes::T
	mma::DomainMalloc
	ndofs::Int
	nels::Int
	cmap::Vector{Int}
	ucmap::Vector{Int}
	loadsteps::Vector{Vector{Float64}}
	timesteps::Vector{Float64}
	actt::Int
	SOLVER::DataType
	function Domain(processes::T, loadsteps::Vector{Vector{Float64}}, timesteps::Vector{Float64}) where {T}
		
		@assert all(map(x->length(x.els), processes) .== length(processes[1].els))
		@assert length(processes) == length(loadsteps)
		@assert all(map(length, loadsteps) .== length(timesteps))
		
		ndofs_el = sum(map(p->size(p.dofmap,1)*length(p.els[1].inds), processes))
		nels = processes[1].nels
		ndofs = sum(map(p->size(p.nodes,1) * size(p.dofmap,1), processes))
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

function setBC!(dom::Domain, eladom::ProcessDomain{LinearElasticity,T}, ls::Vector{Float64}) where {T}
	Uval = dom.actt>0 ? ls[dom.actt] : 0.0
	nodes = eladom.nodes
	U = dom.mma.U
	ΔU = dom.mma.ΔU
	dofmap = eladom.dofmap
	
	inds_left = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(nodes))
	inds_right = findall(x->isapprox(x[1],10.0,atol=1e-9), eachrow(nodes))

	inds_left_middle = findall(x->isapprox(x[1],0.0,atol=1e-9)&&isapprox(x[2],0.0,atol=1e-9), eachrow(nodes))
	inds_right_middle = findall(x->isapprox(x[1],10.0,atol=1e-9)&&isapprox(x[2],0.0,atol=1e-9), eachrow(nodes))
	
	left_bc_x = dofmap[1,inds_left]
	left_bc_y = dofmap[2,inds_left]
	right_bc_x = dofmap[1,inds_right]
	#right_bc_y = dofmap[2,inds_right_middle]
	right_bc_y = dofmap[2,inds_right]
	
	ΔU[right_bc_y] .= (Uval .- U[right_bc_y])
	return vcat(left_bc_x, left_bc_y, right_bc_x, right_bc_y)
	#return vcat(left_bc_x, left_bc_y, right_bc_y)
	
end

function setBC!(dom::Domain, eladom::ProcessDomain{HeatConduction,T}, ls::Vector{Float64}) where {T}
	Uval = dom.actt>0 ? ls[dom.actt] : 0.0
	nodes = eladom.nodes
	U = dom.mma.U
	ΔU = dom.mma.ΔU
	dofmap = eladom.dofmap
	inds1 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(nodes))
	inds2 = findall(x->isapprox(x[1],10.0,atol=1e-9), eachrow(nodes))
	#uc_y_0 = dofmap[2,inds_yc_0]
	uc_x_0 = dofmap[1,inds1]
	uc_x_10 = dofmap[1,inds2]
	ΔU[uc_x_0] .= (Uval .- U[uc_x_0])
	ΔU[uc_x_10] .= (Uval .- U[uc_x_10])
	return vcat(uc_x_0, uc_x_10)
	#return uc_y_1
end

function setBCandCMap!(dom::Domain, eladom::ProcessDomain{P,T}, ls::Vector{Float64}) where {P,T}
	inds = setBC!(dom, eladom, ls)
	append!(dom.cmap, inds)
	return nothing
end

function setBC!(dom::Domain)
	foreach((pdom,ls)->setBC!(dom, pdom, ls), dom.processes, dom.loadsteps)
	return nothing
end

function setBCandUCMaps!(dom::Domain)
	empty!(dom.ucmap)
	empty!(dom.cmap)
	foreach((pdom,ls)->setBCandCMap!(dom, pdom, ls), dom.processes, dom.loadsteps)
	append!(dom.ucmap, setdiff(1:dom.ndofs, dom.cmap))
	resize!(dom.mma.Ftmp, length(dom.ucmap))
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

function integrateTM!(elMats, els1, els2, dofmap1, dofmap2, shapeFuns1, shapeFuns2, U, Uprev, actt, Δt)
	@threads for i in eachindex(els1)
		el1 = els1[i]
		el2 = els2[i]
    	elMats[i] = elStiffnessTM(el1, el2, dofmap1, dofmap2, U, Uprev, shapeFuns1, shapeFuns2, actt, Δt)
	end
end
function integrateTM!(elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SVector{N, Float64}}}, pdoms::Tuple{ProcessDomain{LinearElasticity,T1},ProcessDomain{HeatConduction,T2}}, U, Uprev, actt, Δt) where {N,NN,T1,T2}
	pdom1,pdom2 = pdoms
	els1,els2 = pdom1.els,pdom2.els
	integrateTM!(elMats, els1, els2, pdom1.dofmap, pdom2.dofmap, pdom1.shapeFuns, pdom2.shapeFuns, U, Uprev, actt, Δt)
	return nothing		
end

function integrate!(dom::Domain{Tuple{ProcessDomain{LinearElasticity, T1, ESF1, D1}, ProcessDomain{HeatConduction, T2, ESF2, D2}}}) where {T1,T2,ESF1,ESF2,D1,D2}
	Δt = dom.actt > 1 ? dom.timesteps[dom.actt]-dom.timesteps[dom.actt-1] : 1.0
	integrateTM!(dom.mma.elMats, dom.processes, dom.mma.U, dom.mma.Uprev, dom.actt, Δt)
	return nothing
end

function integrate!(::Type{LinearElasticity}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{ENNODES,ENNODES,Float64,ENNODESSQ}, SVector{ENNODES,Float64}}}, dofmap, shapeFuns, U, Uprev, actt) where {T, ENNODES, ENNODESSQ}
	@threads for i in eachindex(els)
    	el = els[i]
    	elMats[i] = elStiffness(el, dofmap, U, shapeFuns, actt)
	end
	return nothing
end
function integrate!(::Type{HeatConduction}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{ENNODES,ENNODES,Float64,ENNODESSQ}, SVector{ENNODES,Float64}}}, dofmap, shapeFuns, U, Uprev, actt) where {T, ENNODES, ENNODESSQ}
	@threads for i in eachindex(els)
    	el = els[i]
    	elMats[i] = elStiffnessT(el, dofmap, U, Uprev, shapeFuns, actt)
	end
	return nothing
end
function integrate!(elMats::Vector{Tuple{SMatrix{ENNODES,ENNODES,Float64,ENNODESSQ}, SVector{ENNODES,Float64}}}, pdom::ProcessDomain{P,T}, U, Uprev, actt) where {ENNODES,ENNODESSQ,P,T}
	els = pdom.els
	integrate!(P, els, elMats, pdom.dofmap, pdom.shapeFuns, U, Uprev, actt)
	return nothing		
end
function integrate!(dom::Domain{Tuple{PD}}) where {PD}
	integrate!(dom.mma.elMats, dom.processes[1], dom.mma.U, dom.mma.Uprev, dom.actt)
	return nothing
end

using IterativeSolvers
using IncompleteLU

function solve!(dom::Domain)
	ucmap,cmap = dom.ucmap, dom.cmap
	t1 = time()
	integrate!(dom)
	t2 = time()
	println("Integrating element matrices took $(round(t2-t1,digits=8)) seconds")
	Kglob = assemble!(dom)
	t3 = time()
	println("Assembling blfs and lfs took $(round(t3-t2,digits=8)) seconds")

	F,ΔU,U,Ftmp = dom.mma.F,dom.mma.ΔU,dom.mma.U,dom.mma.Ftmp
	mul!(Ftmp, Kglob[ucmap, cmap], ΔU[cmap])
	Ftmp .= F[ucmap] .- Ftmp
	Klgobuc = Kglob[ucmap, ucmap]
	
	if isempty(dom.mma.luKglob)
		luKglob = lu(Klgobuc)
		push!(dom.mma.luKglob, luKglob)
	else
		luKglob = first(dom.mma.luKglob)
		lu!(luKglob, Klgobuc)
	end
	ΔU[ucmap] .= luKglob \ Ftmp
	
	t4 = time()
	println("Solving the linear system took $(round(t4-t3,digits=8)) seconds")
	percsolver =  @sprintf("%.2f", (t4-t3)/(t4-t1)*100)
	@info "Solver time: $percsolver%"
	U .+= ΔU
	return nothing
end

function updateTrialStates!(pdom::ProcessDomain{LinearElasticity, T, E, D}, dom::Domain) where {T,E,D}
	@threads for el in pdom.els
		updateTrialStates!(LinearElasticity, el, pdom.dofmap, dom.mma.U, pdom.shapeFuns, dom.actt)
	end
	return nothing
end

function updateTrialStates!(pdom::ProcessDomain{HeatConduction, T, E, D}, dom::Domain) where {T,E,D}
	@threads for el in pdom.els
		updateTrialStates!(HeatConduction, el, pdom.dofmap, dom.mma.U, pdom.shapeFuns, dom.actt)
	end
	return nothing
end
function updateTrialStates!(pdom1::ProcessDomain{LinearElasticity, T1, E1, D1}, pdom2::ProcessDomain{HeatConduction, T2, E2, D2}, dom::Domain) where {T1,E1,D1,T2,E2,D2}
	@threads for i in 1:pdom1.nels
		el1 = pdom1.els[i]
		el2 = pdom2.els[i]
		updateTrialStates!(LinearElasticity, HeatConduction, el1, el2, pdom1.dofmap, pdom2.dofmap, dom.mma.U, pdom1.shapeFuns, pdom2.shapeFuns, dom.actt)
	end
	return nothing
end

function updateTrialStates!(dom::Domain{Tuple{PD1,PD2}}) where {PD1<:ProcessDomain,PD2<:ProcessDomain}
	updateTrialStates!(dom.processes[1], dom.processes[2], dom)
	return nothing
end
function updateTrialStates!(dom::Domain{Tuple{PD}}) where {PD<:ProcessDomain}
	updateTrialStates!(first(dom.processes), dom)
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
	elPosts = [elPost(el, pdom.shapeFuns, actt) for el in pdom.els];
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
function postSolve!(pdom::ProcessDomain{HeatConduction, T}, U, actt) where {T}
	elPosts = [elPostT(el, pdom.shapeFuns, actt) for el in pdom.els];
	assemblePostT!(pdom.mma.q, pdom.els, elPosts, pdom.nnodes)
	pdom.postdata.timesteps[actt].pdat[:ΔT] .= transpose(U[pdom.dofmap])
	pdom.postdata.timesteps[actt].pdat[:q] .= pdom.MMat \ pdom.mma.q
	return nothing
end

function postSolve!(dom::Domain{Tuple{PD}}) where {PD}
	postSolve!(dom.processes[1], dom.mma.U, dom.actt)
	return nothing
end
function postSolve!(dom::Domain{Tuple{PD1,PD2}}) where {PD1,PD2}
	postSolve!(dom.processes[1], dom.mma.U, dom.actt)
	postSolve!(dom.processes[2], dom.mma.U, dom.actt)
	return nothing
end

function saveHistory!(dom::Domain)
	updateTrialStates!(dom)
	foreach(el->saveHistory!(el, dom.actt), dom.processes[1].els) 
	return nothing
end

function init_loadstep!(dom::Domain)
	fill!(dom.mma.ΔU, 0.0)
	fill!(dom.mma.F, 0.0)
	setBC!(dom)
	return nothing
end

function newtonraphson!(dom::Domain)
	init_loadstep!(dom)
	normdU = Inf
	numit = 0
	str = "\nconvergence history"
	@info "Start Newton-Raphson iteration"
	while normdU>1e-7 && numit < 10
		@info "Newton-Raphson iteration $(numit+1)"
		solve!(dom)
		normdU = norm(dom.mma.ΔU)
		fill!(dom.mma.ΔU,0.0)
		strnormdU = @sprintf("%.4e", normdU) 
		str *= "\nstep $(numit+1): normdU = $strnormdU"
		numit += 1
	end
	println(str)
	println()
	return nothing
end

function tsolve!(dom::Domain)
	dom.actt = 0
	fill!(dom.mma.U, 0.0)
	initStates!(dom)
	setBCandUCMaps!(dom)
	for t in dom.timesteps
		@info "newtonraphson solve t=$t"
		dom.actt += 1
		t1 = time()
		newtonraphson!(dom)
		dom.mma.Uprev .= dom.mma.U
		@info "Save history variables"
		@time saveHistory!(dom)
		@info "Postprocessing"
		@time postSolve!(dom)
		t2 = time()
		timestr = @sprintf("%.2f", t2-t1)
		@info "Analysis time $timestr seconds"
		println()
	end
	pop!(dom.mma.Kglob)
	pop!(dom.mma.luKglob)
	return nothing
end

end #module Domains