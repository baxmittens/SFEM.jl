module Elements

using StaticArrays
using StaticPolynomials
import DensePolynomials
import DensePolynomials: DensePoly, coordinates, evaluate, fast_binomial

#@inline function unsafe_mono_next_grlex(x::NTuple{M,Int}, m::Int) where {M}
#    i = 0
#    @inbounds for j = m:-1:1
#        if x[j] > 0
#            i = j
#            break
#        end
#    end
#
#    if i == 0
#        return Base.setindex(x, 1, m)
#    elseif i == 1
#        t   = x[1] + 1
#        im1 = m
#    else
#        t   = x[i]
#        im1 = i - 1
#    end
#
#    x = Base.setindex(x, 0, i)
#    x = Base.setindex(x, x[im1] + 1, im1)
#    x = Base.setindex(x, x[m] + t - 1, m)
#    return x
#end
#
#@generated function monomial_exponents(::Type{Val{N}}, ::Type{Val{K}}) where {N,K}
#    len = binomial(N+K, K)
#    function build_indices(N,K)
#        
#        buf = Vector{NTuple{N,Int}}()
#        f = ntuple(_->0, N)
#        push!(buf, f)
#        for _ in 2:len
#            f = unsafe_mono_next_grlex(f, N)
#            push!(buf, f)
#        end
#        return Tuple(buf)
#    end
#    indices = build_indices(N,K)
#    return :( ($(indices...),) )
#end

@generated function monomials(::Type{Val{N}}, ::Type{Val{K}}) where {N,K}
    #exps = monomial_exponents(Val{N}, Val{K})
    fbnk = binomial(N+K,K)
    tup = ntuple(i->begin; dp=DensePoly{N, Float64}(fbnk); dp.c[i]=1.0;dp end, fbnk)
    return :( $tup )
end

function flatten_tuple(t::NTuple{M,NTuple{N,T}}) where {M,N,T}
    return ntuple(k -> begin
        j = fld(k-1, N) + 1
        i = (k-1) % N + 1
        t[j][i]
    end, M * N)
end

@generated function shape_functions(nodes::SMatrix{N,M,Float64,NM}, ::Type{Val{P}}) where {N,M,NM,P}
    monos = monomials(Val{N}, Val{P})
    nmonos = length(monos)
    MN = M * nmonos
    evals = [:(evaluate($(monos[i]), nodes[:, $j])) for j in 1:M, i in 1:nmonos]
    shape_exprs = Vector{Any}()
    for i in 1:M
        coeffs = [:(Vinv[$k, $i]) for k in 1:nmonos]
        push!(shape_exprs, :(DensePoly{N,Float64}($P, Float64[$(coeffs...)])))
    end
    return quote
        V = SMatrix{$M,$nmonos,Float64,$MN}($(evals...))
        Vinv = inv(V)
        $(Expr(:tuple, shape_exprs...))
    end
end

abstract type GenericRefElement
end
abstract type ContinuumRefElement <: GenericRefElement
end
abstract type ContactRefElement <: GenericRefElement
end

struct Tri3Ref <: ContinuumRefElement
	nodes::SMatrix{2,3,Float64,6}
	shapeFuns::Tuple
	pOrder::Int
	function Tri3Ref()
		dim = 2
		pOrder = 1
		nodes = SMatrix{dim,3,Float64,6}(0.,0.,1.,0.,0.,1.)
		shapeFuns = shape_functions(nodes, Val{pOrder})
		return new(nodes, shapeFuns, pOrder)
	end
end

struct Tri6Ref <: ContinuumRefElement
	nodes::SMatrix{2,6,Float64,12}
	shapeFuns::Tuple
	pOrder::Int
	function Tri6Ref()
		dim = 2
		pOrder = 2
		nodes = SMatrix{dim,6,Float64,12}(0.,0.,1.,0.,0.,1.,.5,0.,.5,.5,.0,.5)
		shapeFuns = shape_functions(nodes, Val{pOrder})
		return new(nodes, shapeFuns, pOrder)
	end
end
dim(el::C) where {C<:GenericRefElement} = size(el.nodes,1)
nnodes(el::C) where {C<:GenericRefElement} = size(el.nodes,2)

include("./IntegrationRules.jl")

import .IntegrationRules.gaussSimplex

function evaluateShapeFuns(refel::C, intrulefun::F, nips::Int) where {C<:ContinuumRefElement, F<:Function}
	N = dim(refel)
	((ξs,ηs),w) = intrulefun(N, nips)
	evaledshapefuns = ntuple(
		j->SVector{nnodes(refel),Float64}(
			ntuple(
				i->evaluate(refel.shapeFuns[i], SVector{N,Float64}(ξs[j], ηs[j])), 
			nnodes(refel))), 
		length(ξs)
	)
	evaledshapefunderivs = ntuple(nip->
		SMatrix{nnodes(refel),N,Float64,nnodes(refel)*N}(flatten_tuple(ntuple(dir->
			ntuple(nnode->
				evaluate(DensePolynomials.diff(refel.shapeFuns[nnode],dir), SVector{N,Float64}(ξs[nip], ηs[nip])), 
			nnodes(refel)), 
		N))), 
	length(ξs))
	return evaledshapefuns, evaledshapefunderivs
end



end #module Elements
