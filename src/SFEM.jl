module SFEM

	abstract type Process; end
	abstract type LinearElasticity <: Process; end
	abstract type HeatConduction <: Process; end

	abstract type MaterialLaw; end
	abstract type NoMaterial <: MaterialLaw; end
	abstract type LinearElastic <: MaterialLaw; end
	abstract type J2Plasticity <: MaterialLaw; end

	include("./Elements.jl")
	include("./IntegrationRules.jl")
	include("./MeshReader.jl")
	include("./Domains.jl")

end #module SFEM