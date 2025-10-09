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

function _assemble!(mma::Malloc, F::Vector{Float64}, dofmap::Matrix{Int}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SVector{N, Float64}}}, ndofs) where {N,NN,T<:Tri}
    I, J, V, It, Jt, Vt, thrranges, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr = mma.I, mma.J, mma.V, mma.It, mma.Jt, mma.Vt, mma.thrranges, mma.klasttouch, mma.csrrowptr, mma.csrcolval, mma.csrnzval, mma.csccolptr
    nels = length(els)
    nt = length(It)
    @threads for tid in 1:nt
    #@threads for i in 1:nels
    #for i in 1:nels
        II = It[tid]
        JJ = Jt[tid]
        VV = Vt[tid]
        k = 1
        for i in div(thrranges[tid].start-1, NN)+1:div(thrranges[tid].stop, NN)
            @inbounds el = els[i]
            @inbounds Ke = elMats[i][1] 
            eldofs = _dofmap(Val{2}, Val{N}, dofmap, el.inds)            
            @simd for a in 1:N
                @inbounds for b in 1:N
                    II[k] = eldofs[a]
                    JJ[k] = eldofs[b]
                    VV[k] = Ke[a, b]
                    k += 1
                end
            end
        end
    end

    @inbounds for (i,thrrange) in enumerate(thrranges)
        I[thrrange] .= It[i]
        J[thrrange] .= Jt[i]
        V[thrrange] .= Vt[i]
    end

    fill!(F,0.0)
    @inbounds @simd for i in 1:nels
        el = els[i]
        Rint  = elMats[i][2]
        eldofs = _dofmap(Val{2}, Val{N}, dofmap, el.inds)
        F[eldofs] .-= Rint
    end

    return SparseArrays.sparse!(I, J, V, ndofs, ndofs, +, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr)
    #return SparseMatrixCSR(I, J, V, ndofs, ndofs)
    #return SparseArrays.sparse!(I, J, V, ndofs, ndofs)
end

function assemble!(mma::Malloc, F::Vector{Float64}, dofmap::Matrix{Int}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SVector{N, Float64}}}, ndofs) where {N,NN,T<:Tri}
    I, J, V, It, Jt, Vt, thrranges, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr = mma.I, mma.J, mma.V, mma.It, mma.Jt, mma.Vt, mma.thrranges, mma.klasttouch, mma.csrrowptr, mma.csrcolval, mma.csrnzval, mma.csccolptr
    Iptr,Vptr = mma.Iptr,mma.Vptr
    nels = length(els)
    nt = length(It)    
    fill!(F,0.0)
    for i in 1:nels        
        @inbounds el = els[i]
        @inbounds Ke = elMats[i][1] 
        @inbounds Rint = elMats[i][2]
        eldofs = _dofmap(Val{2}, Val{N}, dofmap, el.inds)            
        k = (i-1)*NN + 1
        @simd for a in 1:N
            @inbounds for b in 1:N
                I[k] = eldofs[a]
                J[k] = eldofs[b]
                V[k] = Ke[a, b]
                k += 1
            end
        end
        @inbounds F[eldofs] .-= Rint
    end
    @time spm = SparseArrays.sparse!(I, J, V, ndofs, ndofs, +, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, Iptr, Vptr)
    return spm
end

function assembleMass!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64}, dofmap::Matrix{Int}, els::Vector{T}, elMats::Vector{SMatrix{N, N, Float64, NN}}, ndofs) where {N,NN,T<:Tri}
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

function assemblePost!(σ::Matrix{Float64}, εpl::Matrix{Float64}, dofmap::Matrix{Int}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{N, M, Float64, NM}, SMatrix{N, M, Float64, NM}}}, ndofs) where {N,M,NM,T<:Tri}
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