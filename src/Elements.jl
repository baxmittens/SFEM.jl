module Elements

using StaticArrays
import LinearAlgebra
import ...SFEM: LinearElasticity, HeatConduction

abstract type GenericRefElement
end
abstract type ContinuumRefElement <: GenericRefElement
end
abstract type TriRefElement <: ContinuumRefElement
end
abstract type LineRefElement <: GenericRefElement
end

include("./Elements/ShapeFunctions.jl")

struct Line2Ref <: LineRefElement
	nodes::SMatrix{1,2,Float64,2}
	shapeFuns::Tuple
	pOrder::Int
	function Line2Ref()
		pOrder = 1
		nodes = SMatrix{1,2,Float64,2}(-1.0, 1.0)
		shapeFuns = shape_functions(TriRefElement, nodes, Val{pOrder})
		return new(nodes, shapeFuns, pOrder)
	end
end

struct Line3Ref <: LineRefElement
	nodes::SMatrix{1,3,Float64,3}
	shapeFuns::Tuple
	pOrder::Int
	function Line3Ref()
		pOrder = 2
		nodes = SMatrix{1,3,Float64,3}(-1.0, 0.0, 1.0)
		shapeFuns = shape_functions(TriRefElement, nodes, Val{pOrder})
		return new(nodes, shapeFuns, pOrder)
	end
end

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
	q::Vector{SVector{2,Float64}}
	σtr::SVector{3,Float64}
	εpltr::SVector{3,Float64}
	qtr::SVector{2,Float64}
	function IPStateVars2D(::Type{Val{NTs}}) where {NTs}
		σ = SVector{3,Float64}[SVector{3,Float64}(0.,0.,0.) for i in 1:NTs]
		εpl = SVector{3,Float64}[SVector{3,Float64}(0.,0.,0.) for i in 1:NTs]
		q = SVector{2,Float64}[SVector{2,Float64}(0.,0.) for i in 1:NTs]
		return new(σ, εpl, q, SVector{3,Float64}(0.,0.,0.),SVector{3,Float64}(0.,0.,0.),SVector{2,Float64}(0.,0.))
	end
end
function saveHistory!(ipstate::IPStateVars2D, actt)
	ipstate.σ[actt] = ipstate.σtr
	ipstate.εpl[actt] = ipstate.εpltr
	ipstate.q[actt] = ipstate.qtr
	ipstate.σtr,ipstate.εpltr,ipstate.qtr = zeros(SVector{3,Float64}),zeros(SVector{3,Float64}),zeros(SVector{2,Float64})
	return nothing
end

mutable struct ElementStateVars2D{NIPs}
	state::NTuple{NIPs, IPStateVars2D}
	function ElementStateVars2D(::Type{Val{NIPs}},::Type{Val{NTs}}) where {NIPs, NTs}
		return new{NIPs}(ntuple(i->IPStateVars2D(Val{NTs}), NIPs))
	end
end
σ_avg(state::ElementStateVars2D{NIPs}, actt::Int) where {NIPs} = ntuple(i->sum(ntuple(ip->state.state[ip].σ[actt][i], NIPs))/NIPs, 3)

struct MatPars
	ϱ::Float64
	c_p::Float64
	α_Tx::Float64
	α_Ty::Float64
	α_Tz::Float64
	k_x::Float64
	k_y::Float64
	k_z::Float64
	E::Float64
	ν::Float64
	σy::Float64
	T0::Float64
	bodyforceM::Function
	bodyforceT::Function
	materialID::Int
end

struct Line{DIM, NNODES, NIPs, DIMtimesNNodes} <: GenericElement{DIM, NNODES, NIPs, DIMtimesNNodes}
	nodes::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}
	inds::SVector{NNODES,Int}
	#state::ElementStateVars2D{NIPs}
	#matpars::MatPars
end
struct Tri{DIM, NNODES, NIPs, DIMtimesNNodes} <: ContinuumElement{DIM, NNODES, NIPs, DIMtimesNNodes}
	nodes::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}
	inds::SVector{NNODES,Int}
	state::ElementStateVars2D{NIPs}
	matpars::MatPars
end

_NIPs(el::Line{DIM, NNODES, NIPs, DIMtimesNNodes}) where {DIM, NNODES, NIPs, DIMtimesNNodes} = NIPs
_NIPs(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}) where {DIM, NNODES, NIPs, DIMtimesNNodes} = NIPs
dim(el::Tri) = 2
nnodes(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}) where {DIM, NNODES, NIPs, DIMtimesNNodes} = NNODES
σ_avg(el::C, actt::Int) where {C<:GenericElement} = σ_avg(el.state, actt)
RefEl(::Type{Tri{DIM, 3, NIPs, DIMtimesNNodes}}) where {DIM, NIPs, DIMtimesNNodes} = Tri3Ref()
RefEl(::Type{Tri{DIM, 6, NIPs, DIMtimesNNodes}}) where {DIM, NIPs, DIMtimesNNodes} = Tri6Ref()
RefEl(::Type{Line{DIM, 2, NIPs, DIMtimesNNodes}}) where {DIM, NIPs, DIMtimesNNodes} = Line2Ref()
RefEl(::Type{Line{DIM, 3, NIPs, DIMtimesNNodes}}) where {DIM, NIPs, DIMtimesNNodes} = Line3Ref()
RefEl(::Type{Nothing}) = Line2Ref()

function Line(nodes::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}, inds, ::Type{Val{NIPs}}) where {DIM,NNODES,DIMtimesNNodes,NIPs}
	return Line{DIM,NNODES,NIPs,DIMtimesNNodes}(nodes, inds)
end

function Tri3(nodes, inds, state::ElementStateVars2D, matpars, ::Type{Val{NIPs}}) where {NIPs} 
	return Tri{2,3,NIPs,6}(nodes, inds, state, matpars)
end

function Tri6(nodes, inds, state::ElementStateVars2D, matpars, ::Type{Val{NIPs}}) where {NIPs}
	return Tri{2,6,NIPs,12}(nodes, inds, state, matpars)
end

function saveHistory!(el::C, actt) where {C<:GenericElement}
	foreach(ipstate->saveHistory!(ipstate,actt), el.state.state)
	return nothing
end

include("./Elements/matrices.jl")
include("./Elements/elementstiffness.jl")
include("./Elements/elementstiffnessT.jl")
include("./Elements/elementstiffnessTM.jl")

function initStates!(::Type{LinearElasticity}, state::IPStateVars2D)
	fill!(state.εpl,zeros(SVector{3,Float64}))
	fill!(state.σ,zeros(SVector{3,Float64}))
	fill!(state.q,zeros(SVector{2,Float64}))
	state.σtr,state.εpltr,state.qtr = zeros(SVector{3,Float64}),zeros(SVector{3,Float64}),zeros(SVector{2,Float64})
	return nothing
end

function initStates!(::Type{LinearElasticity}, el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	foreach(ipstate->initStates!(LinearElasticity, ipstate), el.state.state)
	return nothing
end

function initStates!(::Type{HeatConduction}, state::IPStateVars2D)
	return nothing
end

function initStates!(::Type{HeatConduction}, el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	foreach(ipstate->initStates!(LinearElasticity, ipstate), el.state.state)
	return nothing
end

end #module Elements

