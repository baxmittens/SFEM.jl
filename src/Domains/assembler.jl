using SparseArrays

@generated function _dofmap(::Type{Val{DIM}}, ::Type{Val{NNODES2}}, dofmap, elinds) where {DIM,NNODES2}
    @assert isapprox(round(NNODES2/DIM), NNODES2/DIM)
    NNODES = round(Int, NNODES2/DIM)
    tup = flatten_tuple(ntuple(node->ntuple(dim->(dim,node),DIM),NNODES))
    exprs = Vector{Expr}()
    for t in tup
        push!(exprs, :(dofmap[$(t[1]),elinds[$(t[2])]]))
    end 
    return quote
        c = $(Expr(:tuple, exprs...))
        return SVector{$NNODES2,Int}(c)
    end
end

function assemble!(mma::DomainMalloc, dofmap::Matrix{Int}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SVector{N, Float64}}}, offset) where {N,NN,T<:Tri}
    I, J, V, F = mma.I, mma.J, mma.V, mma.F
    nels = length(els)
    fill!(F,0.0)
    k = 0
    for i in 1:nels        
        @inbounds el = els[i]
        @inbounds Ke = elMats[i][1] 
        @inbounds Rint = elMats[i][2]
        eldofs = _dofmap(Val{2}, Val{N}, dofmap, el.inds)            
        k = (i-1)*NN + 1
        @simd for a in 1:N
            @inbounds for b in 1:N
                I[k+offset] = eldofs[a]
                J[k+offset] = eldofs[b]
                V[k+offset] = Ke[a, b]
                k += 1
            end
        end
        @inbounds F[eldofs] .-= Rint
    end
    return k+offset-1
end

function assemble!(dom::Domain{N}) where {N}
    offset = 0
    for pdom in dom.processes
        offset = assemble!(dom.mma, pdom.dofmap, pdom.els, pdom.mma.elMats, offset)
    end
    SparseArrays.sparse!(dom.mma.I, dom.mma.J, dom.mma.V, dom.ndofs, dom.ndofs, +, dom.mma.klasttouch, dom.mma.csrrowptr, 
        dom.mma.csrcolval, dom.mma.csrnzval, dom.mma.csccolptr, dom.mma.Iptr, dom.mma.Vptr)
end

function assembleMass!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64}, els::Vector{T}, elMats::Vector{SMatrix{N, N, Float64, NN}}, ndofs) where {N,NN,T<:Tri}
    #k = 1
    nels = length(els)
    @threads for i in 1:nels
        @inbounds el = els[i]
        @inbounds  Me  = elMats[i]
        k = (i-1)*NN + 1 
        @simd for a in 1:N
            @inbounds for b in 1:N
                I[k] = el.inds[a]
                J[k] = el.inds[b]
                V[k] = Me[a, b]
                k += 1
            end
        end
    end
    return sparse(I, J, V, ndofs, ndofs)
end

function assemblePost!(σ::Matrix{Float64}, εpl::Matrix{Float64}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{N, M, Float64, NM}, SMatrix{N, M, Float64, NM}}}, ndofs) where {N,M,NM,T<:Tri}
    k = 1
    nels = length(els)
    fill!(σ,0.0)
    fill!(εpl,0.0)
    @inbounds for (i,el) in enumerate(els)
        σ[el.inds,:] .+= elMats[i][1]
        εpl[el.inds,:] .+= elMats[i][2]
    end
    return nothing
end