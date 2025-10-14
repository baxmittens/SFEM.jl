#function responseTM(εtr::SVector{3,Float64}, ΔTtr::Float64, εpl::SVector{3,Float64})
#	E = 2.1e11
#	ν = 0.3
#	ℂ = MaterialStiffness(Val{2}, E, ν)
#	αT = SVector{3,Float64}(1e-5,1e-5,0.0)
#	return ℂ*(εtr - αT.*ΔTtr), εpl
#end

function responseTM(εtr::SVector{3,Float64}, εpl::SVector{3,Float64}, ΔTtr::Float64)
    # Materialparameter
    E = 2.1e11
    ν = 0.3
    σy = 200.0
    G  = E / (2*(1+ν))

    # 2D Elastizitätsmatrix (plane strain)
    ℂ = MaterialStiffness(Val{2}, E, ν)
    αT = SVector{3,Float64}(1e-5,1e-5,0.0)

    # Trialspannung
    σtr = ℂ * (εtr - εpl - αT.*ΔTtr)

    # Deviatorische Spannung (2D)
    p = (σtr[1] + σtr[2]) / 3.0
    s = σtr .- SVector(p, p, 0.0)

    # Von Mises Spannung
    seq = sqrt(1.5 * (s[1]^2 + s[2]^2 + 2*s[3]^2) / 2)
    #J2 = 0.5 * (s[1]^2 + s[2]^2 + 2*s[3]^2)
    #seq = sqrt(3 * J2)

    f = seq - σy

    if f <= 0
        # elastisch
        σ = σtr
        εpltr = εpl
    else
        # plastisch
        n = s / (sqrt(s[1]^2 + s[2]^2 + 2*s[3]^2))
        #n = s / sqrt(2 * J2)
        Δγ = f / (3.0*G)  # ohne Verfestigung
        s_new = s - 2G * Δγ * n
        σ = s_new .+ SVector(p, p, 0.0)
        εpltr = εpl .+ Δγ * n
    end

    return σ, εpltr
end

function grad(f::Function,σ::Float64)
	h = 1e-8
	gradσ = (f(σ+h)[1]-f(σ)[1])/h
	return gradσ
end

function combine(Kuu::SMatrix{6,6,T,36}, KuT::SMatrix{6,3,T,18}, KTT::SMatrix{3,3,T,9}) where {T}
    return vcat(hcat(Kuu, KuT), hcat(zeros(SMatrix{3,6,Float64,18}), KTT))
end

function ipStiffnessTM(state, 𝐁, 𝐍_temp, grad𝐍_temp, nodalU, nodalT, εpl, detJ, w, Δt)
	
	𝐤 = SMatrix{2,2,Float64,4}(50.0,0.0,0.0,50.0)
	c_p = 450.0
	ϱ = 7000.0
	𝐁tr = transpose(𝐁)
	εtr = 𝐁*nodalU
	ΔTtr = transpose(𝐍_temp)*nodalT

	ℂ1  = grad(x->responseTM(x, εpl, ΔTtr), εtr)
	ℂ2 = grad(x->responseTM(εtr, εpl, x), ΔTtr)

	dVw = detJ*w
	K_uu = 𝐁tr*ℂ1*𝐁*dVw
	K_uT = 𝐁tr*ℂ2*transpose(𝐍_temp)*dVw
	#K_TT = Δt*grad𝐍_temp*𝐤*transpose(grad𝐍_temp)*dVw
	K_TT = grad𝐍_temp*𝐤*transpose(grad𝐍_temp)*dVw
	M = ϱ*c_p*𝐍_temp*transpose(𝐍_temp)*dVw/10000.0
	K = combine(K_uu,K_uT,M+K_TT)

	return K
end
	
function ipRintTM(state, 𝐁, grad𝐍_temp, 𝐍_temp, nodalT, nodalTm1, detJ, w, Δt)
	dVw = detJ*w
	c_p = 450.0
	ϱ = 7000.0
	MΔT = 1/10000.0*ϱ*c_p*𝐍_temp*transpose(𝐍_temp)*(nodalT-nodalTm1)*dVw
	q = grad𝐍_temp*state.qtr*dVw+MΔT
	σ = transpose(𝐁)*state.σtr*dVw
	return vcat(σ,q)
end

@generated function elStiffnessTM(::Type{Val{NIPs}}, ::Type{Val{NNODES1}}, ::Type{Val{NNODES2}}, ::Type{Val{DIM}}, state, 𝐁s, 𝐍s_temp, grad𝐍s_temp, nodalU, nodalT, nodalTm1, εpls, detJs, wips, Δt) where {NIPs,NNODES1,NNODES2,DIM}
	DIMTimesNNODES = (DIM*NNODES1+NNODES2)
	DIMTimesNNODESSQ = DIMTimesNNODES*DIMTimesNNODES
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Rel += ipRintTM(state[$ip], 𝐁s[$ip], grad𝐍s_temp[$ip], 𝐍s_temp[$ip], nodalT, nodalTm1, detJs[$ip], wips[$ip], Δt)
            Kel += ipStiffnessTM(state[$ip], 𝐁s[$ip], 𝐍s_temp[$ip], grad𝐍s_temp[$ip], nodalU, nodalT, εpls[$ip], detJs[$ip], wips[$ip], Δt)
		end)
	end
	return quote
		Rel = zero(SVector{$DIMTimesNNODES, Float64})
        Kel = zero(SMatrix{$DIMTimesNNODES,$DIMTimesNNODES,Float64,$DIMTimesNNODESSQ})
        $body
        return Kel, Rel
	end
end

function elStiffnessTM(el1::Tri{DIM, NNODES1, NIPs, DIMtimesNNodes1}, el2::Tri{DIM, NNODES2, NIPs, DIMtimesNNodes2}, dofmap1, dofmap2, U, Uprev, shapeFuns1, shapeFuns2, actt, Δt) where {DIM, NNODES1, NNODES2, NIPs, DIMtimesNNodes1, DIMtimesNNodes2}
	d𝐍s1 = shapeFuns1.d𝐍s
	d𝐍s2 = shapeFuns2.d𝐍s
	𝐍s2 = shapeFuns2.𝐍s
	wips = shapeFuns1.wips
	elX0 = el1.nodes
	eldofs1 = dofmap1[SVector{DIM,Int}(1:DIM),el1.inds][:]
	eldofs2 = dofmap2[1,el2.inds][:]
	nodalU = U[eldofs1]
	nodalT = U[eldofs2]
	nodalTm1 = Uprev[eldofs2]
	Js1 = ntuple(ip->elX0*d𝐍s1[ip], NIPs)
	detJs1 = ntuple(ip->smallDet(Js1[ip]), NIPs)
	@assert all(detJs1 .> 0) "error: det(JM) < 0"
	invJs = ntuple(ip->inv(Js1[ip]), NIPs)
	grad𝐍s1 = ntuple(ip->d𝐍s1[ip]*invJs[ip], NIPs)
	grad𝐍s2 = ntuple(ip->d𝐍s2[ip]*invJs[ip], NIPs)
	𝐁s = ntuple(ip->Blin0(Tri{DIM, NNODES1, NIPs, DIMtimesNNodes1}, grad𝐍s1[ip]), NIPs)
	if actt == 1
		εpls = ntuple(ip->SVector{3,Float64}(0.,0.,0.), NIPs)
	else
		εpls = ntuple(ip->el1.state.state[ip].εpl[actt-1], NIPs)
	end
	return elStiffnessTM(Val{NIPs}, Val{NNODES1}, Val{NNODES2}, Val{DIM}, el1.state.state, 𝐁s, 𝐍s2, grad𝐍s2, nodalU, nodalT, nodalTm1, εpls, detJs1, wips, Δt)
end

function updateTrialStates!(::Type{LinearElasticity}, ::Type{HeatConduction}, state::IPStateVars2D, 𝐁, grad𝐍_temp, 𝐍_temp, nodalU, nodalT, actt)
	εtr = 𝐁*nodalU
	εpl = actt > 1 ? state.εpl[actt-1] : zeros(SVector{3,Float64})
	ΔTtr = transpose(𝐍_temp)*nodalT
	state.σtr,state.εpltr = responseTM(εtr, εpl, ΔTtr)
	𝐤 = SMatrix{2,2,Float64,4}(50.0,0.0,0.0,50.0)
	state.qtr = 𝐤*transpose(grad𝐍_temp)*nodalT
	return nothing
end

function updateTrialStates!(::Type{LinearElasticity}, ::Type{HeatConduction}, el1::Tri{DIM, NNODES1, NIPs, DIMtimesNNodes1}, el2::Tri{DIM, NNODES2, NIPs, DIMtimesNNodes2}, dofmap1, dofmap2, U, shapeFuns1, shapeFuns2, actt) where {DIM, NNODES1, NNODES2, NIPs, DIMtimesNNodes1, DIMtimesNNodes2}
	d𝐍s1 = shapeFuns1.d𝐍s
	d𝐍s2 = shapeFuns2.d𝐍s
	𝐍s2 = shapeFuns2.𝐍s
	wips = shapeFuns1.wips
	elX0 = el1.nodes
	eldofs1 = dofmap1[SVector{2,Int}(1,2),el1.inds][:]
	eldofs2 = dofmap2[1,el2.inds][:]
	nodalU = U[eldofs1]
	nodalT = U[eldofs2]
	Js = ntuple(ip->elX0*d𝐍s1[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	invJs = ntuple(ip->inv(Js[ip]), NIPs)
	grad𝐍s1 = ntuple(ip->d𝐍s1[ip]*invJs[ip], NIPs)
	grad𝐍s2 = ntuple(ip->d𝐍s2[ip]*invJs[ip], NIPs)
	𝐁s = ntuple(ip->Blin0(Tri{DIM, NNODES1, NIPs, DIMtimesNNodes1}, grad𝐍s1[ip]), NIPs)
	foreach((ipstate,𝐁, grad𝐍temp, 𝐍_temp)->updateTrialStates!(LinearElasticity, HeatConduction, ipstate, 𝐁, grad𝐍temp, 𝐍_temp, nodalU, nodalT, actt), el1.state.state, 𝐁s, grad𝐍s2, 𝐍s2)
	return nothing
end