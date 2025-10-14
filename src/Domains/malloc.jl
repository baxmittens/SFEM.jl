
struct ProcessDomainMalloc{ENNODES,ENNODESSQ}
	Im::Vector{Int}
	Jm::Vector{Int}
	Vm::Vector{Float64}
	σ::Matrix{Float64}
	εpl::Matrix{Float64}
	q::Matrix{Float64}
	elMMats::Vector{SMatrix{ENNODES,ENNODES,Float64,ENNODESSQ}}
	function ProcessDomainMalloc(nels, ::Type{Val{ennodes}}, nnodes) where {ennodes}
		nnz_total_mass = nels * ennodes^2
		Im = Vector{Int}(undef, nnz_total_mass)
		Jm = Vector{Int}(undef, nnz_total_mass)
		Vm = Vector{Float64}(undef, nnz_total_mass)
		σ = zeros(Float64, nnodes, 3)
		εpl = zeros(Float64, nnodes, 3)
		q = zeros(Float64, nnodes, 2)
		elMMats = Vector{SMatrix{ennodes,ennodes,Float64,ennodes*ennodes}}(undef, nels)
		return new{ennodes,ennodes*ennodes}(Im,Jm,Vm,σ,εpl,q,elMMats)
	end
end

struct DomainMalloc{ENNODES,ENNODESSQ}
	U::Vector{Float64}
	ΔU::Vector{Float64}
	Uprev::Vector{Float64}
	F::Vector{Float64}
	I::Vector{Int}
	J::Vector{Int}
	V::Vector{Float64}
	klasttouch::Vector{Int}
	csrrowptr::Vector{Int}
	csrcolval::Vector{Int}
	csrnzval::Vector{Float64}
	csccolptr::Vector{Int}
	Iptr::Vector{Int}
	Vptr::Vector{Float64}
	elMats::Vector{Tuple{SMatrix{ENNODES,ENNODES,Float64,ENNODESSQ}, SVector{ENNODES,Float64}}}
	#Kglob::Union{SparseMatrixCSC{Float64, Int64},Nothing}
	Kglob::Vector{SparseMatrixCSC{Float64, Int64}}
	idxmap::Vector{Int}
	luKglob::Vector{SparseArrays.UMFPACK.UmfpackLU{Float64, Int64}}
	function DomainMalloc(nels,ndofs,ndofs_el)
		U = zeros(Float64, ndofs)
		ΔU = zeros(Float64, ndofs)
		Uprev = zeros(Float64, ndofs)
		F = zeros(Float64, ndofs)
		ndofsq = ndofs_el^2
		nnz_total = nels * ndofsq
		I = Vector{Int}(undef, nnz_total)
		J = Vector{Int}(undef, nnz_total)
		V = Vector{Float64}(undef, nnz_total)
		klasttouch = Vector{Int}(undef, ndofs)
		csrrowptr = Vector{Int}(undef,ndofs+1)
		csrcolval = Vector{Int}(undef,nnz_total)
		csrnzval = Vector{Float64}(undef,nnz_total)
		csccolptr = Vector{Int}(undef, ndofs+1)
		Iptr = Vector{Int}(undef, nnz_total)
		Vptr = Vector{Float64}(undef, nnz_total)
		elMats = Vector{Tuple{SMatrix{ndofs_el,ndofs_el,Float64,ndofs_el*ndofs_el}, SVector{ndofs_el,Float64}}}(undef, nels)
		return new{ndofs_el,ndofs_el*ndofs_el}(U,ΔU,Uprev,F,I,J,V,klasttouch,csrrowptr,csrcolval,csrnzval,csccolptr,Iptr,Vptr,elMats,Vector{SparseMatrixCSC{Float64, Int64}}(),Vector{Int}(undef, nnz_total), Vector{SparseArrays.UMFPACK.UmfpackLU{Float64, Int64}}())
	end
end

mutable struct PostDataTS
	pdat::Dict{Symbol, Matrix{Float64}}
	PostDataTS(::Type{LinearElasticity}, nnodes, nels) = new(Dict{Symbol, Matrix{Float64}}(
		:U=>zeros(Float64, nnodes, 2),  
		:σ=>zeros(Float64, nnodes, 3), 
		:εpl=>zeros(Float64, nnodes, 3), 
		:σ_avg=>zeros(Float64, nels, 3)
		))
	PostDataTS(::Type{HeatConduction}, nnodes, nels) = new(Dict{Symbol, Matrix{Float64}}(
		:ΔT=>zeros(Float64, nnodes, 1),
		:q=>zeros(Float64, nnodes, 2)
		))
end

mutable struct PostData
	timesteps::Vector{PostDataTS}
	PostData(::Type{T}, nnodes, nts, nels) where {T<:Process} = new(PostDataTS[PostDataTS(T, nnodes, nels) for _ in 1:nts])
end