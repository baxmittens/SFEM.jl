function grad(f::Function,σ::Float64)
	h = 1e-8
	gradσ = (f(σ+h)[1]-f(σ)[1])/h
	return gradσ
end

function ipStiffnessTM(state, matpars, 𝐁, 𝐍_temp, grad𝐍_temp, nodalU, nodalT, εpl, detJ, w, Δt)
	𝐁tr = transpose(𝐁)
	εtr = 𝐁*nodalU
	ΔTtr = transpose(𝐍_temp)*nodalT
	ℂ2 = grad(x->response(matpars,εtr, εpl, x), ΔTtr)
	dVw = detJ*w
	K_uu = ipStiffness(state, matpars, 𝐁, nodalU, εpl, detJ, w, ΔTtr)
	K_uT = 𝐁tr*ℂ2*transpose(𝐍_temp)*dVw
	MK_TT = ipStiffnessT(state, matpars, grad𝐍_temp, 𝐍_temp, nodalT, detJ, w, Δt)
	K = combine(K_uu,K_uT,MK_TT)
	return K
end
	
function ipRintTM(state, matpars, 𝐁, grad𝐍_temp, 𝐍, 𝐍_temp, nodalU, nodalT, nodalTm1, εpl, detJ, w, Δt, X0, actt)
	ΔTr = transpose(𝐍_temp)*nodalT
	σ = ipRint(state, matpars, 𝐁, 𝐍, nodalU, εpl, detJ, w, X0, actt, ΔTr)
	q = ipRintT(state, matpars, grad𝐍_temp, 𝐍_temp, nodalT, nodalTm1, detJ, w, Δt, X0, actt)
	return vcat(σ,q)
end

@generated function elStiffnessTM(::Type{Val{NIPs}}, ::Type{Val{NNODES1}}, ::Type{Val{NNODES2}}, ::Type{Val{DIM}}, state, matpars, 𝐁s, 𝐍s, 𝐍s_temp, grad𝐍s_temp, nodalU, nodalT, nodalTm1, εpls, detJs, wips, Δt, X0s, actt) where {NIPs,NNODES1,NNODES2,DIM}
	DIMTimesNNODES = (DIM*NNODES1+NNODES2)
	DIMTimesNNODESSQ = DIMTimesNNODES*DIMTimesNNODES
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Rel += ipRintTM(state[$ip], matpars, 𝐁s[$ip], grad𝐍s_temp[$ip], 𝐍s[$ip], 𝐍s_temp[$ip], nodalU, nodalT, nodalTm1, εpls[$ip], detJs[$ip], wips[$ip], Δt, X0s[$ip], actt)
            Kel += ipStiffnessTM(state[$ip], matpars, 𝐁s[$ip], 𝐍s_temp[$ip], grad𝐍s_temp[$ip], nodalU, nodalT, εpls[$ip], detJs[$ip], wips[$ip], Δt)
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
	𝐁s, 𝐍s, nodalU, εpls, detJs, wips, X0s = elStiffnessVals(el1, dofmap1, U, shapeFuns1, actt)
	grad𝐍sT, 𝐍sT, nodalT, nodalTm1, _, _, _ = elStiffnessTVals(el2, dofmap2, U, Uprev, shapeFuns2, actt)
	return elStiffnessTM(Val{NIPs}, Val{NNODES1}, Val{NNODES2}, Val{DIM}, el1.state.state, el1.matpars, 𝐁s, 𝐍s, 𝐍sT, grad𝐍sT, nodalU, nodalT, nodalTm1, εpls, detJs, wips, Δt, X0s, actt)
end

function updateTrialStates!(::Type{LinearElasticity}, ::Type{HeatConduction}, state::IPStateVars2D, matpars, 𝐁, grad𝐍_temp, 𝐍_temp, nodalU, nodalT, actt)
	εtr = 𝐁*nodalU
	εpl = actt > 1 ? state.εpl[actt-1] : zeros(SVector{3,Float64})
	ΔTtr = transpose(𝐍_temp)*nodalT
	state.σtr,state.εpltr = response(matpars, εtr, εpl, ΔTtr)
	𝐤 = thermal_conductivity(Val{2}, matpars)
	state.qtr = 𝐤*transpose(grad𝐍_temp)*nodalT
	return nothing
end

function updateTrialStates!(::Type{LinearElasticity}, ::Type{HeatConduction}, el1::Tri{DIM, NNODES1, NIPs, DIMtimesNNodes1}, el2::Tri{DIM, NNODES2, NIPs, DIMtimesNNodes2}, dofmap1, dofmap2, U, shapeFuns1, shapeFuns2, actt) where {DIM, NNODES1, NNODES2, NIPs, DIMtimesNNodes1, DIMtimesNNodes2}
	𝐁s, _, nodalU, _, _, _ = elStiffnessVals(el1, dofmap1, U, shapeFuns1, actt)
	grad𝐍s, 𝐍s, nodalT, _, _, _ = elStiffnessTVals(el2, dofmap2, U, U, shapeFuns2, actt)
	foreach((ipstate,𝐁, grad𝐍temp, 𝐍_temp)->updateTrialStates!(LinearElasticity, HeatConduction, ipstate, el1.matpars, 𝐁, grad𝐍temp, 𝐍_temp, nodalU, nodalT, actt), el1.state.state, 𝐁s, grad𝐍s, 𝐍s)
	return nothing
end