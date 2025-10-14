
function grad(f::Function,σ::Float64)
	h = 1e-8
	gradσ = (f(σ+h)[1]-f(σ)[1])/h
	return gradσ
end

function combine(Kuu::SMatrix{6,6,T,36}, KuT::SMatrix{6,3,T,18}, KTT::SMatrix{3,3,T,9}) where {T}
    return vcat(hcat(Kuu, KuT), hcat(zeros(SMatrix{3,6,Float64,18}), KTT))
end
#𝐁, 𝐍_temp, grad𝐍_temp, εpl, detJ, w, state = 𝐁s[1], 𝐍sT[1], grad𝐍sT[1], εpls[1], detJs[1], wips[1], dom.processes[1].els[1].state.state[1]
function ipStiffnessTM(state, 𝐁, 𝐍_temp, grad𝐍_temp, nodalU, nodalT, εpl, detJ, w, Δt)
	dVw = detJ*w
	εtr = 𝐁*nodalU
	ΔTtr = transpose(𝐍_temp)*nodalT
	ℂ2 = grad(x->response(εtr, εpl, x), ΔTtr)
	K_uu = ipStiffness(state, 𝐁, nodalU, εpl, detJ, w, Δt)
	K_uT = transpose(𝐁)*ℂ2*transpose(𝐍_temp)*dVw
	K_TT = ipStiffnessT(state, grad𝐍_temp, 𝐍_temp, nodalT, detJ, w, Δt)
	K = combine(K_uu,K_uT,K_TT)
	return K
end
	
function ipRintTM(state, 𝐁, grad𝐍_temp, detJ, w, Δt)
	σ = ipRint(state, 𝐁, detJ, w)
	q = ipRintT(state, grad𝐍_temp, detJ, w, Δt)
	return vcat(σ,q)
end

@generated function elStiffnessTM(::Type{Val{NIPs}}, ::Type{Val{NNODES1}}, ::Type{Val{NNODES2}}, ::Type{Val{DIM}}, state, 𝐁s, 𝐍s_temp, grad𝐍s_temp, nodalU, nodalT, εpls, detJs, wips, Δt) where {NIPs,NNODES1,NNODES2,DIM}
	DIMTimesNNODES = (DIM*NNODES1+NNODES2)
	DIMTimesNNODESSQ = DIMTimesNNODES*DIMTimesNNODES
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Rel += ipRintTM(state[$ip], 𝐁s[$ip], grad𝐍s_temp[$ip], detJs[$ip], wips[$ip], Δt)
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

#Δt = 1.0
#el1, dofmap1, U, ΔU, shapeFuns1, actt = dom.processes[1].els[1], dom.processes[1].dofmap, dom.mma.U, dom.mma.ΔU, dom.processes[1].shapeFuns, dom.actt;
#el2, dofmap2, shapeFuns2 = dom.processes[2].els[1], dom.processes[2].dofmap, dom.processes[2].shapeFuns;
#import SFEM.Elements: _elStiffnessT, _elStiffness, response, ipStiffness, ipStiffnessT, ipRint, ipRintT
#𝐁s, nodalU, εpls, detJs, wips = _elStiffness(el1, dofmap1, U, ΔU, shapeFuns1, actt)
#grad𝐍sT, 𝐍sT, nodalT, _, _ = _elStiffnessT(el2, dofmap2, U, ΔU, shapeFuns2, actt, Δt)

function elStiffnessTM(el1::Tri{DIM, NNODES1, NIPs, DIMtimesNNodes1}, el2::Tri{DIM, NNODES2, NIPs, DIMtimesNNodes2}, dofmap1, dofmap2, U, shapeFuns1, shapeFuns2, actt, Δt) where {DIM, NNODES1, NNODES2, NIPs, DIMtimesNNodes1, DIMtimesNNodes2}
	𝐁s, _, nodalU, εpls, detJs, wips = _elStiffness(el1, dofmap1, U, shapeFuns1, actt)
	grad𝐍sT, 𝐍sT, nodalT, _, _ = _elStiffnessT(el2, dofmap2, U, shapeFuns2)
	return elStiffnessTM(Val{NIPs}, Val{NNODES1}, Val{NNODES2}, Val{DIM}, el1.state.state, 𝐁s, 𝐍sT, grad𝐍sT, nodalU, nodalT, εpls, detJs, wips, Δt)
end

function updateTrialStates!(::Type{LinearElasticity}, ::Type{HeatConduction}, state::IPStateVars2D, 𝐁, grad𝐍_temp, 𝐍_temp, nodalU, nodalT, nodalTm1, actt)
	updateTrialStates!(HeatConduction, state, grad𝐍_temp, 𝐍_temp, nodalT, nodalTm1, actt)
	updateTrialStates!(LinearElasticity, state, 𝐁, nodalU, actt)
	return nothing
end

function updateTrialStates!(::Type{LinearElasticity}, ::Type{HeatConduction}, el1::Tri{DIM, NNODES1, NIPs, DIMtimesNNodes1}, el2::Tri{DIM, NNODES2, NIPs, DIMtimesNNodes2}, dofmap1, dofmap2, U, Uprev, shapeFuns1, shapeFuns2, actt) where {DIM, NNODES1, NNODES2, NIPs, DIMtimesNNodes1, DIMtimesNNodes2}
	eldofs2 = dofmap2[1,el2.inds][:]
	nodalTm1 = Uprev[eldofs2]
	𝐁s, _, nodalU, _, detJs, wips = _elStiffness(el1, dofmap1, U, shapeFuns1, actt)
	grad𝐍sT, 𝐍sT, nodalT, _, _ = _elStiffnessT(el2, dofmap2, U, shapeFuns2)
	foreach((ipstate,𝐁, grad𝐍temp, 𝐍_temp)->updateTrialStates!(LinearElasticity, HeatConduction, ipstate, 𝐁, grad𝐍temp, 𝐍_temp, nodalU, nodalT, nodalTm1, actt), el1.state.state, 𝐁s, grad𝐍sT, 𝐍sT)
	return nothing
end