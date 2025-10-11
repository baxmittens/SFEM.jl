
struct ProcessDomainMalloc{ENNODES,ENNODESSQ}
	Im::Vector{Int}
	Jm::Vector{Int}
	Vm::Vector{Float64}
	σ::Matrix{Float64}
	εpl::Matrix{Float64}
	elMats::Vector{Tuple{SMatrix{ENNODES,ENNODES,Float64,ENNODESSQ}, SVector{ENNODES,Float64}}}
	function ProcessDomainMalloc(nels,ndofs_el,::Type{Val{ennodes}}, nnodes) where {ennodes}
		ndofsq = ndofs_el^2
		nnz_total_mass = nels * ennodes^2
		Im = Vector{Int}(undef, nnz_total_mass)
		Jm = Vector{Int}(undef, nnz_total_mass)
		Vm = Vector{Float64}(undef, nnz_total_mass)
		σ = zeros(Float64, nnodes, 3)
		εpl = zeros(Float64, nnodes, 3)
		elMats = Vector{Tuple{SMatrix{2*ennodes,2*ennodes,Float64,4*ennodes*ennodes}, SVector{2*ennodes,Float64}}}(undef, nels)
		return new{2*ennodes,4*ennodes*ennodes}(Im,Jm,Vm,σ,εpl,elMats)
	end
end

struct DomainMalloc
	U::Vector{Float64}
	ΔU::Vector{Float64}
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
	function DomainMalloc(nels,ndofs,ndofs_el)
		U = zeros(Float64, ndofs)
		ΔU = zeros(Float64, ndofs)
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
		return new(U,ΔU,F,I,J,V,klasttouch,csrrowptr,csrcolval,csrnzval,csccolptr,Iptr,Vptr)
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
		:temp=>zeros(Float64, nnodes, 1)
		))
end

mutable struct PostData
	timesteps::Vector{PostDataTS}
	PostData(::Type{T}, nnodes, nts, nels) where {T<:Process} = new(PostDataTS[PostDataTS(T, nnodes, nels) for _ in 1:nts])
end