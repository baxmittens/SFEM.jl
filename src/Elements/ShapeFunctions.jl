import DensePolynomials
import DensePolynomials: DensePoly, coordinates, evaluate, fast_binomial

@generated function monomials(::Type{Val{N}}, ::Type{Val{K}}) where {N,K}
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

@generated function shape_functions(::Type{E}, nodes::SMatrix{N,M,Float64,NM}, ::Type{Val{P}}) where {E<:GenericRefElement,N,M,NM,P}
    monos = monomials(Val{N}, Val{P})
    nmonos = length(monos)
    MN = M * nmonos
    evals = [:(evaluate($(monos[i]), nodes[:, $j])) for j in 1:M, i in 1:nmonos]
    shape_exprs = Vector{Expr}()
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
struct EvaluatedShapeFunctions{DIM, NNodes, NIPs, NNodestimesDIM}
    𝐍s::NTuple{NIPs, SVector{NNodes, Float64}}
    d𝐍s::NTuple{NIPs, SMatrix{NNodes, DIM, Float64, NNodestimesDIM}}
    wips::SVector{NIPs, Float64}
end

function EvaluatedShapeFunctions(refel::C, intrulefun::F, nips::Int) where {C<:TriRefElement, F<:Function}
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
                evaluate(DensePolynomials.diff(refel.shapeFuns[nnode], dir), SVector{N,Float64}(ξs[nip], ηs[nip])), 
            nnodes(refel)), 
        N))), 
    length(ξs))
    return EvaluatedShapeFunctions(evaledshapefuns, evaledshapefunderivs, w)
end

function EvaluatedShapeFunctions(refel::C, intrulefun::F, nips::Int) where {C<:LineRefElement, F<:Function}
    N = dim(refel)
    ((ξs,),w) = intrulefun(N, nips)
    evaledshapefuns = ntuple(
        j->SVector{nnodes(refel),Float64}(
            ntuple(
                i->evaluate(refel.shapeFuns[i], SVector{N,Float64}(ξs[j])), 
            nnodes(refel))), 
        length(ξs)
    )
    evaledshapefunderivs = ntuple(nip->
        SMatrix{nnodes(refel),N,Float64,nnodes(refel)*N}(flatten_tuple(ntuple(dir->
            ntuple(nnode->
                evaluate(DensePolynomials.diff(refel.shapeFuns[nnode], dir), SVector{N,Float64}(ξs[nip])), 
            nnodes(refel)), 
        N))), 
    length(ξs))
    return EvaluatedShapeFunctions(evaledshapefuns, evaledshapefunderivs, w)
end