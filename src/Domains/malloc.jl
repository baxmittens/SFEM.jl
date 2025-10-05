struct Malloc
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
	function Malloc(nels,ndofs,ndofs_el,ennodes,nnodes)
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
		σ = zeros(Float64, nnodes, ennodes)
		εpl = zeros(Float64, nnodes, ennodes)
		return new(U,ΔU,F,I,J,V,Im,Jm,Vm,σ,εpl)
	end
end

mutable struct PostDataTS
	U::Vector{Float64}
	σ::Matrix{Float64}
	εpl::Matrix{Float64}
	PostDataTS(ndofs::Int, nnodes::Int) = new(zeros(Float64, ndofs), zeros(Float64, nnodes, 3), zeros(Float64, nnodes, 3))
end

mutable struct PostData
	postdata::Vector{PostDataTS}
	PostData(ndofs, nnodes, nts) = new(PostDataTS[PostDataTS(ndofs, nnodes) for _ in 1:nts])
end