
function ipStiffnessT(state, matpars, grad𝐍_temp, 𝐍_temp, nodalT, detJ, w, Δt)
	𝐤 = thermal_conductivity(Val{2}, matpars)
	c_p,ϱ = matpars.c_p,matpars.ϱ
	dVw = detJ*w
	K_TT = grad𝐍_temp*𝐤*transpose(grad𝐍_temp)*dVw
	M = 1/Δt*ϱ*c_p*𝐍_temp*transpose(𝐍_temp)*dVw
	return M+K_TT
end
	
function ipRintT(state, matpars, grad𝐍_temp, 𝐍_temp, nodalT, nodalTm1, detJ, w, Δt)
	𝐤 = thermal_conductivity(Val{2}, matpars)
	c_p,ϱ = matpars.c_p,matpars.ϱ
	dVw = detJ*w
	MΔT = 1.0/Δt*ϱ*c_p*𝐍_temp*transpose(𝐍_temp)*(nodalT-nodalTm1)*dVw
	q = grad𝐍_temp*𝐤*transpose(grad𝐍_temp)*nodalT*dVw
	return MΔT+q
end

@generated function elStiffnessT(::Type{Val{NIPs}}, ::Type{Val{NNODES}}, ::Type{Val{DIM}}, state, matpars, grad𝐍s_temp, 𝐍s, nodalT, nodalTm1, detJs, wips, Δt) where {NIPs,NNODES,DIM}
	DIMTimesNNODES = NNODES
	DIMTimesNNODESSQ = DIMTimesNNODES*DIMTimesNNODES
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Rel += ipRintT(state[$ip], matpars, grad𝐍s_temp[$ip], 𝐍s[$ip], nodalT, nodalTm1, detJs[$ip], wips[$ip], Δt)
            Kel += ipStiffnessT(state[$ip], matpars, grad𝐍s_temp[$ip], 𝐍s[$ip], nodalT, detJs[$ip], wips[$ip], Δt)
		end)
	end
	return quote
		Rel = zero(SVector{$DIMTimesNNODES, Float64})
        Kel = zero(SMatrix{$DIMTimesNNODES,$DIMTimesNNODES,Float64,$DIMTimesNNODESSQ})
        $body
        return Kel, Rel
	end
end

function elStiffnessTVals(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes, Mat}, dofmap, U, Uprev, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes, Mat}
	d𝐍s = shapeFuns.d𝐍s
	𝐍s = shapeFuns.𝐍s
	wips = shapeFuns.wips
	elX0 = el.nodes
	eldofs = dofmap[1,el.inds][:]
	nodalT = U[eldofs]
	nodalTm1 = Uprev[eldofs]
	Js = ntuple(ip->elX0*d𝐍s[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(JM) < 0"
	invJs = ntuple(ip->inv(Js[ip]), NIPs)
	grad𝐍s = ntuple(ip->d𝐍s[ip]*invJs[ip], NIPs)
	return grad𝐍s, 𝐍s, nodalT, nodalTm1, detJs, wips
end

function elStiffnessT(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes, Mat}, matpars, dofmap, U, Uprev, shapeFuns, actt, Δt) where {DIM, NNODES, NIPs, DIMtimesNNodes, Mat}
	grad𝐍s, 𝐍s, nodalT, nodalTm1, detJs, wips = elStiffnessTVals(el, dofmap, U, Uprev, shapeFuns, actt)
	return elStiffnessT(Val{NIPs}, Val{NNODES}, Val{DIM}, el.state.state, matpars, grad𝐍s, 𝐍s, nodalT, nodalTm1, detJs, wips)
end

@generated function elPostT(::Type{Val{NIPs}}, ::Type{Val{NNODES}}, state, 𝐍s, detJs, wips, actt) where {NIPs, NNODES}
	#println(typeof(wips)," ",wips)
	NNODES2 = NNODES*2
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			qe += elPost(𝐍s[$ip], state[$ip].q[actt], detJs[$ip], wips[$ip])
		end)
	end
	return quote
        qe = zero(SMatrix{NNODES,2,Float64,$NNODES2})
        $body
        return qe
	end
end

function elPostT(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes, Mat}, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes, Mat}
	𝐍s = shapeFuns.𝐍s
	d𝐍s = shapeFuns.d𝐍s
	wips = shapeFuns.wips
	elX0 = el.nodes
	Js = ntuple(ip->elX0*d𝐍s[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	return elPostT(Val{NIPs}, Val{NNODES}, el.state.state, 𝐍s, detJs, wips, actt)
end

function updateTrialStates!(::Type{HeatConduction}, state::IPStateVars2D, matpars, grad𝐍_temp, nodalT)
	𝐤 = thermal_conductivity(Val{2}, matpars)
	state.qtr = 𝐤*transpose(grad𝐍_temp)*nodalT
	return nothing
end

function updateTrialStates!(::Type{HeatConduction}, el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, dofmap, U, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	grad𝐍s, _, nodalT, _, _, _ = elStiffnessTVals(el, dofmap, U, U, shapeFuns, actt)
	foreach((ipstate,grad𝐍)->updateTrialStates!(HeatConduction, ipstate, el.matpars,  grad𝐍, nodalT), el.state.state, grad𝐍s)
	return nothing
end