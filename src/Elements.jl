module Elements

using StaticArrays

abstract type GenericElement
end
abstract type ContinuumElement <: GenericElement
end
abstract type ContactElement <: GenericElement
end

struct Tri3 <: ContinuumElement
	nodes::SMatrix{2,3,Float64,6}
	Tri3() = new(SMatrix{2,3,Float64,6}(0.,0.,1.,0.,1.,1.))
end

end #module Elements