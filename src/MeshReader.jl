module MeshReader

using MshReader

struct GmshMesh
	nodes::Matrix{Float64}
	connectivity::Vector{Vector{Int}}
	connectivity_boundary::Vector{Vector{Int}}
	physicalNames::Vector{String}
	elemPhysNums::Vector{Int}
	function GmshMesh(path::String)
		nodesCoordMat, connectivity, physicalNames, elemPhysNums = MshFileReader(path)
		di = findfirst(physicalNames.=="Domain")
		bi = findfirst(physicalNames.=="Boundary")
		return new(nodesCoordMat, connectivity[findall(elemPhysNums.==di)], connectivity[findall(elemPhysNums.==bi)],physicalNames,elemPhysNums)
	end
end

end #module MeshReader