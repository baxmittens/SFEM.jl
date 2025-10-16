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
function NMat(N::SVector{3,Float64})
    return @SMatrix [
        N[1]  0.0   N[2]  0.0   N[3]    0.0
        0.0   N[1]  0.0   N[2]  0.0     N[3]
    ]
end
function NMat(N::SVector{6,Float64})
    return @SMatrix [
        N[1]  0.0   N[2]  0.0   N[3]    0.0     N[4]  0.0   N[5]  0.0   N[6]    0.0 
        0.0   N[1]  0.0   N[2]  0.0     N[3]    0.0   N[4]  0.0   N[5]  0.0     N[6]
    ]
end



function MaterialStiffness(::Type{Val{2}}, matpars::MatPars)
	E,ν = matpars.E, matpars.ν
    fac = E/((1+ν)*(1-2*ν))
	return fac*SMatrix{3,3,Float64,9}(1.0-ν,ν,0.,ν,1-ν,0.,0.,0.,(1.0-2.0*ν)/2.0)
end
function thermal_conductivity(::Type{Val{2}}, matpars::MatPars)
    return SMatrix{2,2,Float64,4}(matpars.k_x,0.,0.,matpars.k_y)
end
function thermal_expansivity(::Type{Val{2}}, matpars::MatPars)
   return SVector{3,Float64}(matpars.α_Tx,matpars.α_Ty,0.0)
end