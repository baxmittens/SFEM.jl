module Elements

using StaticArrays
import LinearAlgebra

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


abstract type GenericElement
end
abstract type ContinuumElement <: GenericElement
end
abstract type TriElement <: ContinuumElement
end

mutable struct IPStateVars2D
	σ::Vector{SVector{3,Float64}}
	εpl::Vector{SVector{3,Float64}}
	σtr::SVector{3,Float64}
	εpltr::SVector{3,Float64}
	function IPStateVars2D(nts)
		σ = SVector{3,Float64}[SVector{3,Float64}(0.,0.,0.) for i in 1:nts]
		εpl = SVector{3,Float64}[SVector{3,Float64}(0.,0.,0.) for i in 1:nts]
		return new(σ, εpl, SVector{3,Float64}(0.,0.,0.))
	end
end
function saveHistory!(ipstate::IPStateVars2D, actt)
	ipstate.σ[actt] = ipstate.σtr
	ipstate.εpl[actt] = ipstate.εpltr
	return nothing
end

mutable struct ElementStateVars2D{NIPs}
	state::NTuple{NIPs, IPStateVars2D}
	function ElementStateVars2D(nips,nts)
		return new{nips}(ntuple(i->IPStateVars2D(nts), nips))
	end
end

struct Tri3{NIPs} <: TriElement
	nodes::SMatrix{2,3,Float64,6}
	inds::SVector{3,Int}
	state::ElementStateVars2D{NIPs}
	Tri3(nodes,inds,nips,nts) = new{nips}(nodes, inds, ElementStateVars2D(nips,nts))
end
struct Tri6{NIPs} <: TriElement
	nodes::SMatrix{2,6,Float64,12}
	inds::SVector{6,Int}
	state::ElementStateVars2D{NIPs}
	Tri6(nodes,inds,nips,nts) = new{nips}(nodes, inds, ElementStateVars2D(nips,nts))
end
dim(el::C) where {C<:GenericElement} = size(el.nodes,1)
nnodes(el::C) where {C<:GenericElement} = size(el.nodes,2)
nips(state::ElementStateVars2D{NIPs}) where {NIPs} = Val{NIPs}()
nips(el::C) where {C<:GenericElement} = nips(el.state)

function saveHistory!(el::C, actt) where {C<:GenericElement}
	foreach(ipstate->saveHistory!(ipstate,actt), el.state.state)
	return nothing
end

include("./Elements/elementstiffness.jl")

end #module Elements
