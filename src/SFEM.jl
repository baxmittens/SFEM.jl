module SFEM

	abstract type Process; end
	abstract type LinearElasticity <: Process; end
	abstract type HeatConduction <: Process; end

	include("./Elements.jl")
	include("./IntegrationRules.jl")
	include("./MeshReader.jl")
	include("./Domains.jl")

end #module SFEM