module Elements

using StaticArrays
import LinearAlgebra
import ...SFEM: LinearElasticity

abstract type GenericRefElement
end
abstract type ContinuumRefElement <: GenericRefElement
end
abstract type TriRefElement <: ContinuumRefElement
end

include("./Elements/ShapeFunctions.jl")

struct Tri3Ref <: TriRefElement
	nodes::SMatrix{2,3,Float64,6}
	shapeFuns::Tuple
	pOrder::Int
	function Tri3Ref()
		pOrder = 1
		nodes = SMatrix{2,3,Float64,6}(0.,0.,1.,0.,0.,1.)
		shapeFuns = shape_functions(TriRefElement, nodes, Val{pOrder})
		return new(nodes, shapeFuns, pOrder)
	end
end

struct Tri6Ref <: TriRefElement
	nodes::SMatrix{2,6,Float64,12}
	shapeFuns::Tuple
	pOrder::Int
	function Tri6Ref()
		pOrder = 2
		nodes = SMatrix{2,6,Float64,12}(0.,0.,1.,0.,0.,1.,.5,0.,.5,.5,.0,.5)
		shapeFuns = shape_functions(TriRefElement, nodes, Val{pOrder})
		return new(nodes, shapeFuns, pOrder)
	end
end
dim(el::C) where {C<:GenericRefElement} = size(el.nodes,1)
nnodes(el::C) where {C<:GenericRefElement} = size(el.nodes,2)


abstract type GenericElement{DIM, NNODES, NIPs, DIMtimesNNodes}
end
abstract type ContinuumElement{DIM, NNODES, NIPs, DIMtimesNNodes} <: GenericElement{DIM, NNODES, NIPs, DIMtimesNNodes}
end

mutable struct IPStateVars2D
	σ::Vector{SVector{3,Float64}}
	εpl::Vector{SVector{3,Float64}}
	σtr::SVector{3,Float64}
	εpltr::SVector{3,Float64}
	function IPStateVars2D(::Type{Val{NTs}}) where {NTs}
		σ = SVector{3,Float64}[SVector{3,Float64}(0.,0.,0.) for i in 1:NTs]
		εpl = SVector{3,Float64}[SVector{3,Float64}(0.,0.,0.) for i in 1:NTs]
		return new(σ, εpl, SVector{3,Float64}(0.,0.,0.),SVector{3,Float64}(0.,0.,0.))
	end
end
function saveHistory!(ipstate::IPStateVars2D, actt)
	ipstate.σ[actt] = ipstate.σtr
	ipstate.εpl[actt] = ipstate.εpltr
	return nothing
end

mutable struct ElementStateVars2D{NIPs}
	state::NTuple{NIPs, IPStateVars2D}
	function ElementStateVars2D(::Type{Val{NIPs}},::Type{Val{NTs}}) where {NIPs, NTs}
		return new{NIPs}(ntuple(i->IPStateVars2D(Val{NTs}), NIPs))
	end
end
σ_avg(state::ElementStateVars2D{NIPs}, actt::Int) where {NIPs} = ntuple(i->sum(ntuple(ip->state.state[ip].σ[actt][i], NIPs))/NIPs, 3)

struct Tri{DIM, NNODES, NIPs, DIMtimesNNodes} <: ContinuumElement{DIM, NNODES, NIPs, DIMtimesNNodes}
	nodes::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}
	inds::SVector{NNODES,Int}
	state::ElementStateVars2D{NIPs}
end
#struct Tri6{NIPs} <: TriElement
#	nodes::SMatrix{2,6,Float64,12}
#	inds::SVector{6,Int}
#	state::ElementStateVars2D{NIPs}
#	Tri6(nodes,inds,nips,nts) = new{nips}(nodes, inds, ElementStateVars2D(nips,nts))
#end
dim(el::Tri) = 2
nnodes(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}) where {DIM, NNODES, NIPs, DIMtimesNNodes} = NNODES
#nips(state::ElementStateVars2D{NIPs}) where {NIPs} = Val{NIPs}()
#nips(el::Tri{NNODES, NIPs, DIMtimesNNodes})  where {NNODES, NIPs, DIMtimesNNodes} = Val{NIPs}
σ_avg(el::C, actt::Int) where {C<:GenericElement} = σ_avg(el.state, actt)
RefEl(::Type{Tri{DIM, 3, NIPs, DIMtimesNNodes}}) where {DIM, NIPs, DIMtimesNNodes} = Tri3Ref()
RefEl(::Type{Tri{DIM, 6, NIPs, DIMtimesNNodes}}) where {DIM, NIPs, DIMtimesNNodes} = Tri6Ref()

function Tri3(nodes,inds, state::ElementStateVars2D, ::Type{Val{NIPs}}) where {NIPs} 
	return Tri{2,3,NIPs,6}(nodes, inds, state)
end

function Tri6(nodes,inds,state::ElementStateVars2D, ::Type{Val{NIPs}}) where {NIPs}
	return Tri{2,6,NIPs,12}(nodes, inds, state)
end

function saveHistory!(el::C, actt) where {C<:GenericElement}
	foreach(ipstate->saveHistory!(ipstate,actt), el.state.state)
	return nothing
end

include("./Elements/elementstiffness.jl")

function updateTrialStates!(::Type{LinearElasticity}, state::IPStateVars2D, 𝐁, nodalU, actt)
	εtr = 𝐁*nodalU
	εpl = state.εpl[actt]
	state.σtr,state.εpltr = response(εtr, εpl)
	return nothing
end

function updateTrialStates!(::Type{LinearElasticity}, el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, dofmap, U, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	d𝐍s = shapeFuns.d𝐍s
	wips = shapeFuns.wips
	elX0 = el.nodes
	eldofs = dofmap[SVector{2,Int}(1,2),el.inds][:]
	nodalU = U[eldofs]
	Js = ntuple(ip->elX0*d𝐍s[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	invJs = ntuple(ip->inv(Js[ip]), NIPs)
	grad𝐍s = ntuple(ip->d𝐍s[ip]*invJs[ip], NIPs)
	𝐁s = ntuple(ip->Blin0(Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, grad𝐍s[ip]), NIPs)
	foreach((ipstate,𝐁)->updateTrialStates!(LinearElasticity, ipstate, 𝐁, nodalU, actt), el.state.state, 𝐁s)
	return nothing
end

end #module Elements

