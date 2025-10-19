function smallDet(M::SMatrix{2,2,Float64,4})
	@inbounds return (M[1,1]*M[2,2] - M[1,2]*M[2,1])
end
function smallDet(M::SMatrix{3,3,Float64,9})
	@inbounds return (M[1,1]*(M[2,2]*M[3,3]-M[2,3]*M[3,2]) - M[1,2]*(M[2,1]*M[3,3]-M[2,3]*M[3,1]) + M[1,3]*(M[2,1]*M[3,2]-M[2,2]*M[3,1]))
end

function NMat(N::SVector{2,Float64})
    return @SMatrix [
        N[1]  0.0   N[2]  0.0
        0.0   N[1]  0.0   N[2]
    ]
end
function NMat(N::SVector{3,Float64})
    return @SMatrix [
        N[1]  0.0   N[2]  0.0   N[3]    0.0
        0.0   N[1]  0.0   N[2]  0.0     N[3]
    ]
end
function NMat(N::SVector{4,Float64})
    return @SMatrix [
        N[1]  0.0   N[2]  0.0   N[3]    0.0   N[4]  0.0
        0.0   N[1]  0.0   N[2]  0.0     N[3]  0.0   N[4]
    ]
end
function NMat(N::SVector{6,Float64})
    return @SMatrix [
        N[1]  0.0   N[2]  0.0   N[3]    0.0     N[4]  0.0   N[5]  0.0   N[6]    0.0 
        0.0   N[1]  0.0   N[2]  0.0     N[3]    0.0   N[4]  0.0   N[5]  0.0     N[6]
    ]
end
function NMat(N::SVector{10,Float64})
    return @SMatrix [
        N[1]  0.0   N[2]  0.0   N[3]    0.0     N[4]  0.0   N[5]  0.0   N[6]    0.0   N[7]    0.0     N[8]  0.0   N[9]  0.0   N[10]    0.0 
        0.0   N[1]  0.0   N[2]  0.0     N[3]    0.0   N[4]  0.0   N[5]  0.0     N[6]  0.0     N[7]    0.0   N[8]  0.0   N[9]  0.0     N[10]
    ]
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
function Blin0(::Type{Tri{2,10,NIPs,20}}, gradN::SMatrix{10,2,Float64,20}) where {NIPs}
    return @SMatrix [
        gradN[1,1]  0.0        gradN[2,1]  0.0        gradN[3,1]  0.0           gradN[4,1]  0.0        gradN[5,1]  0.0        gradN[6,1]  0.0   gradN[7,1]  0.0   gradN[8,1]  0.0        gradN[9,1]  0.0   gradN[10,1]  0.0   
        0.0         gradN[1,2] 0.0         gradN[2,2] 0.0         gradN[3,2]    0.0         gradN[4,2] 0.0         gradN[5,2] 0.0         gradN[6,2]    0.0       gradN[7,2]    0.0    gradN[8,2] 0.0   gradN[9,2] 0.0    gradN[10,2]
        gradN[1,2]  gradN[1,1] gradN[2,2]  gradN[2,1] gradN[3,2]  gradN[3,1]    gradN[4,2]  gradN[4,1] gradN[5,2]  gradN[5,1] gradN[6,2]  gradN[6,1]    gradN[7,2]  gradN[7,1]    gradN[8,2]  gradN[8,1] gradN[9,2]  gradN[9,1] gradN[10,2]  gradN[10,1]
    ]
end

function combine(Kuu::SMatrix{6,6,T,36}, KuT::SMatrix{6,3,T,18}, KTT::SMatrix{3,3,T,9}) where {T}
    return vcat(hcat(Kuu, KuT), hcat(zeros(SMatrix{3,6,Float64,18}), KTT))
end
function combine(Kuu::SMatrix{12,12,T,144}, KuT::SMatrix{12,6,T,72}, KTT::SMatrix{6,6,T,36}) where {T}
    return vcat(hcat(Kuu, KuT), hcat(zeros(SMatrix{6,12,Float64,72}), KTT))
end
function combine(Kuu::SMatrix{12,12,T,144}, KuT::SMatrix{12,3,T,36}, KTT::SMatrix{3,3,T,9}) where {T}
    return vcat(hcat(Kuu, KuT), hcat(zeros(SMatrix{3,12,Float64,36}), KTT))
end
function combine(Kuu::SMatrix{20,20,T,400}, KuT::SMatrix{20,10,T,200}, KTT::SMatrix{10,10,T,100}) where {T}
    return vcat(hcat(Kuu, KuT), hcat(zeros(SMatrix{10,20,Float64,200}), KTT))
end
function combine(Kuu::SMatrix{20,20,T,400}, KuT::SMatrix{20,3,T,60}, KTT::SMatrix{3,3,T,9}) where {T}
    return vcat(hcat(Kuu, KuT), hcat(zeros(SMatrix{3,20,Float64,60}), KTT))
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