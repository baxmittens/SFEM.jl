using SparseArrays

function assemble!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64}, F::Vector{Float64}, dofmap::Matrix{Int}, els::Vector{Tri3}, elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SVector{N, Float64}}}, ndofs::Int, ndofs_el::Int) where {N,NN}
    k = 1
    nels = length(els)
    nnz_per_el = ndofs_el * ndofs_el
    
    #@threads for i in 1:nels
    for i in 1:nels
        el = els[i]
        Ke  = elMats[i][1]
        @inbounds eldofs = SVector{6,Int}(dofmap[1, el.inds[1]], dofmap[2, el.inds[1]], dofmap[1, el.inds[2]], dofmap[2, el.inds[2]], dofmap[1, el.inds[3]], dofmap[2, el.inds[3]])
        #eldofs = dofmap[SVector{2,Int}(1,2), el.inds]
        k = (i-1)*nnz_per_el + 1        
        @simd for a in 1:ndofs_el
            @inbounds for b in 1:ndofs_el
                I[k] = eldofs[a]
                J[k] = eldofs[b]
                V[k] = Ke[a, b]
                k += 1
            end
        end
    end

    fill!(F,0.0)
    for i in 1:nels
        el = els[i]
        Rint  = elMats[i][2]
        eldofs = dofmap[SVector{2,Int}(1,2), el.inds]
        @inbounds F[eldofs[:]] .-= Rint
    end

    return sparse(I, J, V, ndofs, ndofs)
end

function assembleMass!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64}, dofmap::Matrix{Int}, els::Vector{Tri3}, elMats::Vector{SMatrix{N, N, Float64, NN}}, ndofs::Int, ndofs_el::Int) where {N,NN}
    k = 1
    nels = length(els)
    for (i,el) in enumerate(els)
        Me  = elMats[i]
        @simd for a in 1:ndofs_el
            @inbounds for b in 1:ndofs_el
                I[k] = el.inds[a]
                J[k] = el.inds[b]
                V[k] = Me[a, b]
                k += 1
            end
        end
    end
    return sparse(I, J, V, ndofs, ndofs)
end

function assemblePost!(σ::Matrix{Float64}, εpl::Matrix{Float64}, dofmap::Matrix{Int}, els::Vector{Tri3}, elMats::Vector{Tuple{SMatrix{N, N, Float64, NN}, SMatrix{N, N, Float64, NN}}}, ndofs::Int, ndofs_el::Int) where {N,NN}
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