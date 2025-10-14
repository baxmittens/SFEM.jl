using SparseArrays

function build_index_mapping!(dom::Domain)
    I, J, A = dom.mma.I, dom.mma.J, first(dom.mma.Kglob)
    nnz_total = length(I)
    # baue Hilfsstruktur für schnellen Lookup: Dict[(row,col)] -> nzval_index
    lookup = Dict{Tuple{Int,Int},Int}()
    for col in 1:size(A,2)
        for p in A.colptr[col]:(A.colptr[col+1]-1)
            lookup[(A.rowval[p], col)] = p
        end
    end
    # dann Mapping aufbauen
    for k in 1:nnz_total
        dom.mma.idxmap[k] = lookup[(I[k], J[k])]
    end
    return nothing
end

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

function assemble!(mma::DomainMalloc, dofmap::Matrix{Int}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SVector{N, Float64}}}, ::Type{Val{DOFMAPDIM1}}) where {N,NN,T<:Tri,DOFMAPDIM1}
    I, J, V, F = mma.I, mma.J, mma.V, mma.F
    nels = length(els)
    fill!(F,0.0)
    k = 0
    for i in 1:nels        
        @inbounds el = els[i]
        @inbounds Ke = elMats[i][1] 
        @inbounds Rint = elMats[i][2]
        eldofs = _dofmap(Val{DOFMAPDIM1}, Val{N}, dofmap, el.inds)
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
    return nothing
end

function assemble!(Kglob::SparseMatrixCSC,mma::DomainMalloc, dofmap::Matrix{Int}, els::Vector{T}, elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SVector{N, Float64}}}, ::Type{Val{DOFMAPDIM1}}) where {N,NN,T<:Tri,DOFMAPDIM1}
    Kglob.nzval .= 0.0
    I, J, V, F = mma.I, mma.J, mma.V, mma.F
    nels = length(els)
    fill!(F,0.0)
    k = 0
    for i in 1:nels        
        @inbounds el = els[i]
        @inbounds Ke = elMats[i][1] 
        @inbounds Rint = elMats[i][2]
        eldofs = _dofmap(Val{DOFMAPDIM1}, Val{N}, dofmap, el.inds)
        k = (i-1)*NN + 1
        @simd for a in 1:N
            @inbounds for b in 1:N
                #I[k] = eldofs[a]
                #J[k] = eldofs[b]
                #V[k] = Ke[a, b]
                Kglob.nzval[mma.idxmap[k]] += Ke[a, b]
                k += 1
            end
        end
        @inbounds F[eldofs] .-= Rint
    end
    return nothing
end

function assemble!(dom::Domain{Tuple{ProcessDomain{P,T,E,DMD1}}}) where {P,T,E,DMD1}
    pdom = first(dom.processes)
    if isempty(dom.mma.Kglob)
        assemble!(dom.mma, pdom.dofmap, pdom.els, dom.mma.elMats, Val{DMD1})
        push!(dom.mma.Kglob,SparseArrays.sparse!(dom.mma.I, dom.mma.J, dom.mma.V, dom.ndofs, dom.ndofs, +, dom.mma.klasttouch, dom.mma.csrrowptr, dom.mma.csrcolval, dom.mma.csrnzval, dom.mma.csccolptr, dom.mma.Iptr, dom.mma.Vptr))
        build_index_mapping!(dom)
    else
        assemble!(first(dom.mma.Kglob), dom.mma, pdom.dofmap, pdom.els, dom.mma.elMats, Val{DMD1})
    end
    return first(dom.mma.Kglob)
end

@generated function _dofmap(::Type{Val{DIM}}, ::Type{Val{NNODES2}}, dofmap1, dofmap2, elinds1::SVector{N,Int}, elinds2::SVector{M,Int}) where {DIM,NNODES2,N,M}
    @assert isapprox(round(NNODES2/(DIM+1)), NNODES2/(DIM+1))
    NNODES = round(Int, NNODES2/(DIM+1))
    exprs = Vector{Expr}()
    for i in 1:N
        for j = 1:2
            push!(exprs, :(dofmap1[$j,elinds1[$i]]))
        end
    end
    for i in 1:M
        push!(exprs, :(dofmap2[1,elinds2[$i]]))
    end 
    return quote
        c = $(Expr(:tuple, exprs...))
        return SVector{$NNODES2,Int}(c)
    end
end

function assemble!(mma::DomainMalloc, dofmap1::Matrix{Int}, dofmap2::Matrix{Int}, els1::Vector{T}, els2::Vector{T}, elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SVector{N, Float64}}}) where {N,NN,T<:Tri}
    I, J, V, F = mma.I, mma.J, mma.V, mma.F
    nels = length(els1)
    fill!(F,0.0)
    k = 0
    for i in 1:nels        
        @inbounds el1 = els1[i]
        @inbounds el2 = els2[i]
        @inbounds Ke = elMats[i][1] 
        @inbounds Rint = elMats[i][2]
        eldofs = _dofmap(Val{2}, Val{N}, dofmap1, dofmap2, el1.inds, el2.inds)
        #println(eldofs)
        #error()     
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
    return nothing
end

function assemble!(Kglob::SparseMatrixCSC, mma::DomainMalloc, dofmap1::Matrix{Int}, dofmap2::Matrix{Int}, els1::Vector{T}, els2::Vector{T}, elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SVector{N, Float64}}}) where {N,NN,T<:Tri}
    Kglob.nzval .= 0.0
    I, J, V, F = mma.I, mma.J, mma.V, mma.F
    nels = length(els1)
    fill!(F,0.0)
    k = 0
    for i in 1:nels        
        @inbounds el1 = els1[i]
        @inbounds el2 = els2[i]
        @inbounds Ke = elMats[i][1] 
        @inbounds Rint = elMats[i][2]
        eldofs = _dofmap(Val{2}, Val{N}, dofmap1, dofmap2, el1.inds, el2.inds)
        #println(eldofs)
        #error()     
        k = (i-1)*NN + 1
        @simd for a in 1:N
            @inbounds for b in 1:N
                #I[k] = eldofs[a]
                #J[k] = eldofs[b]
                #V[k] = Ke[a, b]
                Kglob.nzval[mma.idxmap[k]] += Ke[a, b]
                k += 1
            end
        end
        @inbounds F[eldofs] .-= Rint
    end
    return nothing
end

function assemble!(dom::Domain{Tuple{PD1,PD2}}) where {PD1,PD2}
    pdom1,pdom2 = dom.processes
    if isempty(dom.mma.Kglob)
        assemble!(dom.mma, pdom1.dofmap, pdom2.dofmap, pdom1.els, pdom2.els, dom.mma.elMats)
        push!(dom.mma.Kglob,SparseArrays.sparse!(dom.mma.I, dom.mma.J, dom.mma.V, dom.ndofs, dom.ndofs, +, dom.mma.klasttouch, dom.mma.csrrowptr, dom.mma.csrcolval, dom.mma.csrnzval, dom.mma.csccolptr, dom.mma.Iptr, dom.mma.Vptr))
        build_index_mapping!(dom)
    else
        assemble!(first(dom.mma.Kglob), dom.mma, pdom1.dofmap, pdom2.dofmap, pdom1.els, pdom2.els, dom.mma.elMats)
    end
    return first(dom.mma.Kglob)
end

function assembleMass!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64}, els::Vector{T}, elMats::Vector{SMatrix{N, N, Float64, NN}}, ndofs) where {N,NN,T<:Tri}
    #k = 1
    nels = length(els)
    @threads for i in 1:nels
        @inbounds el = els[i]
        @inbounds Me  = elMats[i]
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
function assemblePostT!(q::Matrix{Float64}, els::Vector{T}, elMats::Vector{SMatrix{N, M, Float64, NM}}, ndofs) where {N,M,NM,T<:Tri}
    k = 1
    nels = length(els)
    fill!(q,0.0)
    @inbounds for (i,el) in enumerate(els)
        q[el.inds,:] .+= elMats[i]
    end
    return nothing
end