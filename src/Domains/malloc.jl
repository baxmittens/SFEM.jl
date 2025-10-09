#function thread_ranges(n::Int)
#    nt = Threads.nthreads()
#    ranges = Vector{UnitRange{Int}}(undef, nt)
#    for t in 1:nt
#        start = div((t-1)*n, nt) + 1
#        stop  = div(t*n, nt)
#        ranges[t] = start:stop
#    end
#    return ranges
#end
#


function thread_ranges(n::Int, NDELS::Int)
    nt = Threads.nthreads()
    ranges = Vector{UnitRange{Int}}(undef, nt)

    # Wie viele Elemente pro Thread im Idealfall (als Vielfaches von NDELS)
    base = div(n, nt * NDELS) * NDELS
    remn = n - base * nt  # Rest, der noch übrig bleibt

    start = 1
    for t in 1:nt
        extra = remn > 0 ? min(remn, NDELS) : 0
        stop = min(start + base + extra - 1, n)
        ranges[t] = start:stop
        start = stop + 1
        remn -= extra
    end

    return ranges
end

function split_for_threads_copy(tmp::Vector{T}, NDELS::Int) where {T<:Number}
    n = length(tmp)
    nt = Threads.nthreads()
    chunks = Vector{Vector{T}}(undef, nt)
    ranges = thread_ranges(n,  NDELS)
    #display(ranges)
    for t in 1:nt
    	#display(length(ranges[t]))
        chunks[t] = Vector{T}(undef,length(ranges[t]))
    end
    return chunks
end

struct Malloc{ENNODES,ENNODESSQ}
	U::Vector{Float64}
	ΔU::Vector{Float64}
	F::Vector{Float64}
	I::Vector{Int}
	J::Vector{Int}
	V::Vector{Float64}
	It::Vector{Vector{Int}}
	Jt::Vector{Vector{Int}}
	Vt::Vector{Vector{Float64}}
	thrranges::Vector{UnitRange{Int64}}
	klasttouch::Vector{Int}
	csrrowptr::Vector{Int}
	csrcolval::Vector{Int}
	csrnzval::Vector{Float64}
	csccolptr::Vector{Int}
	Iptr::Vector{Int}
	Vptr::Vector{Float64}
	Im::Vector{Int}
	Jm::Vector{Int}
	Vm::Vector{Float64}
	σ::Matrix{Float64}
	εpl::Matrix{Float64}
	elMats::Vector{Tuple{SMatrix{ENNODES,ENNODES,Float64,ENNODESSQ}, SVector{ENNODES,Float64}}}
	function Malloc(nels,ndofs,ndofs_el,::Type{Val{ennodes}}, nnodes) where {ennodes}
		U = zeros(Float64, ndofs)
		ΔU = zeros(Float64, ndofs)
		F = zeros(Float64, ndofs)
		ndofsq = ndofs_el^2
		nnz_total = nels * ndofsq
		nnz_total_mass = nels * ennodes^2
		I = Vector{Int}(undef, nnz_total)
		J = Vector{Int}(undef, nnz_total)
		V = Vector{Float64}(undef, nnz_total)
		It = split_for_threads_copy(I, ndofsq)
		Jt = split_for_threads_copy(J, ndofsq)
		Vt = split_for_threads_copy(V, ndofsq)

		klasttouch = Vector{Int}(undef, ndofs)
		csrrowptr = Vector{Int}(undef,ndofs+1)
		csrcolval = Vector{Int}(undef,nnz_total)
		csrnzval = Vector{Float64}(undef,nnz_total)
		csccolptr = Vector{Int}(undef, ndofs+1)
		Iptr = Vector{Int}(undef, nnz_total)
		Vptr = Vector{Float64}(undef, nnz_total)
		Im = Vector{Int}(undef, nnz_total_mass)
		Jm = Vector{Int}(undef, nnz_total_mass)
		Vm = Vector{Float64}(undef, nnz_total_mass)
		σ = zeros(Float64, nnodes, 3)
		εpl = zeros(Float64, nnodes, 3)
		elMats = Vector{Tuple{SMatrix{2*ennodes,2*ennodes,Float64,4*ennodes*ennodes}, SVector{2*ennodes,Float64}}}(undef, nels)
		return new{2*ennodes,4*ennodes*ennodes}(U,ΔU,F,I,J,V,It,Jt,Vt,thread_ranges(nnz_total, ndofsq),klasttouch,csrrowptr,csrcolval,csrnzval,csccolptr,Iptr,Vptr,Im,Jm,Vm,σ,εpl,elMats)
	end
end

mutable struct PostDataTS
	U::Vector{Float64}
	σ::Matrix{Float64}
	εpl::Matrix{Float64}
	σ_avg::Vector{Float64}
	PostDataTS(ndofs::Int, nnodes::Int,nels::Int) = new(zeros(Float64, ndofs), zeros(Float64, nnodes, 3), zeros(Float64, nnodes, 3), zeros(Float64, nels))
end

mutable struct PostData
	postdata::Vector{PostDataTS}
	PostData(ndofs, nnodes, nts, nels) = new(PostDataTS[PostDataTS(ndofs, nnodes, nels) for _ in 1:nts])
end