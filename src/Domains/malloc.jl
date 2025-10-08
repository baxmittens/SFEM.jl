struct Malloc{ENNODES,ENNODESSQ}
	U::Vector{Float64}
	ΔU::Vector{Float64}
	F::Vector{Float64}
	I::Vector{Int}
	J::Vector{Int}
	V::Vector{Float64}
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
		nnz_total = nels * ndofs_el^2
		nnz_total_mass = nels * ennodes^2
		I = Vector{Int}(undef, nnz_total)
		J = Vector{Int}(undef, nnz_total)
		V = Vector{Float64}(undef, nnz_total)
		Im = Vector{Int}(undef, nnz_total_mass)
		Jm = Vector{Int}(undef, nnz_total_mass)
		Vm = Vector{Float64}(undef, nnz_total_mass)
		σ = zeros(Float64, nnodes, 3)
		εpl = zeros(Float64, nnodes, 3)
		elMats = Vector{Tuple{SMatrix{2*ennodes,2*ennodes,Float64,4*ennodes*ennodes}, SVector{2*ennodes,Float64}}}(undef, nels)
		return new{2*ennodes,4*ennodes*ennodes}(U,ΔU,F,I,J,V,Im,Jm,Vm,σ,εpl,elMats)
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