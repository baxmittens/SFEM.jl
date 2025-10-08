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
	function ElementStateVars2D(nips,nts)
		return new{nips}(ntuple(i->IPStateVars2D(nts), nips))
	end
end
σ_avg(state::ElementStateVars2D{NIPs}, actt::Int) where {NIPs} = sum(ntuple(ip->state.state[ip].σ[actt][1], NIPs))/NIPs

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
σ_avg(el::C, actt::Int) where {C<:GenericElement} = σ_avg(el.state, actt)

function saveHistory!(el::C, actt) where {C<:GenericElement}
	foreach(ipstate->saveHistory!(ipstate,actt), el.state.state)
	return nothing
end

include("./Elements/elementstiffness.jl")

function updateTrialStates!(state::IPStateVars2D, 𝐁, nodalU, actt)
	εtr = 𝐁*nodalU
	εpl = state.εpl[actt]
	state.σtr,state.εpltr = response(εtr, εpl)
	return nothing
end

function updateTrialStates!(el::Tri3{NIPs}, dofmap, U, shapeFuns, actt) where {NIPs}
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
	𝐁s = ntuple(ip->Blin0(Tri3, grad𝐍s[ip]), NIPs)
	foreach((ipstate,𝐁)->updateTrialStates!(ipstate, 𝐁, nodalU, actt), el.state.state, 𝐁s)
	return nothing
end

end #module Elements

