function smallDet(M::SMatrix{2,2,Float64,4})
	@inbounds return (M[1,1]*M[2,2] - M[1,2]*M[2,1])
end
function smallDet(M::SMatrix{3,3,Float64,9})
	@inbounds return (M[1,1]*(M[2,2]*M[3,3]-M[2,3]*M[3,2]) - M[1,2]*(M[2,1]*M[3,3]-M[2,3]*M[3,1]) + M[1,3]*(M[2,1]*M[3,2]-M[2,2]*M[3,1]))
end

function Blin0(::Type{Tri{2, 3, NIPs, 6}}, gradN::SMatrix{3,2,Float64,6}) where {NIPs}
    return @SMatrix [
        gradN[1,1]  0.0        gradN[2,1]  0.0        gradN[3,1]  0.0
        0.0         gradN[1,2] 0.0         gradN[2,2] 0.0         gradN[3,2]
        gradN[1,2]  gradN[1,1] gradN[2,2]  gradN[2,1] gradN[3,2]  gradN[3,1]
    ]
end
function Blin0(::Type{Tri{2,6,NIPs,12}}, gradN::SMatrix{6,2,Float64,12}) where {NIPs}
    return @SMatrix [
        gradN[1,1]  0.0        gradN[2,1]  0.0        gradN[3,1]  0.0 			gradN[4,1]  0.0        gradN[5,1]  0.0        gradN[6,1]  0.0	
        0.0         gradN[1,2] 0.0         gradN[2,2] 0.0         gradN[3,2]	0.0         gradN[4,2] 0.0         gradN[5,2] 0.0         gradN[6,2]
        gradN[1,2]  gradN[1,1] gradN[2,2]  gradN[2,1] gradN[3,2]  gradN[3,1]	gradN[4,2]  gradN[4,1] gradN[5,2]  gradN[5,1] gradN[6,2]  gradN[6,1]
    ]
end

function MaterialStiffness(::Type{Val{2}}, E, ν)
	fac = E/((1+ν)*(1-2*ν))
	return fac*SMatrix{3,3,Float64,9}(1-ν,ν,0.,ν,1-ν,0.,0.,0.,(1-2*ν)/2.0)
end