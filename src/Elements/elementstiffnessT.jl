
function ipStiffnessT(state, matpars, grad𝐍_temp, 𝐍_temp, nodalT, detJ, w, Δt)
	𝐤 = thermal_conductivity(Val{2}, matpars)
	c_p,ϱ = matpars.c_p,matpars.ϱ
	dVw = detJ*w
	K_TT = grad𝐍_temp*𝐤*transpose(grad𝐍_temp)*dVw
	M = 1/Δt*ϱ*c_p*𝐍_temp*transpose(𝐍_temp)*dVw
	return M+K_TT
end
	
function ipRintT(state, matpars, grad𝐍_temp, 𝐍_temp, nodalT, nodalTm1, detJ, w, Δt, X0, actt)
	𝐤 = thermal_conductivity(Val{2}, matpars)
	c_p,ϱ = matpars.c_p,matpars.ϱ
	dVw = detJ*w
	MΔT = 1.0/Δt*ϱ*c_p*𝐍_temp*transpose(𝐍_temp)*(nodalT-nodalTm1)*dVw
	q = grad𝐍_temp*𝐤*transpose(grad𝐍_temp)*nodalT*dVw
	qbar = 𝐍_temp*bodyforceT(X0, matpars, actt)*dVw
	return MΔT+q-qbar
end

@generated function elStiffnessT(::Type{Val{NIPs}}, ::Type{Val{NNODES}}, ::Type{Val{DIM}}, state, matpars, grad𝐍s_temp, 𝐍s, nodalT, nodalTm1, detJs, wips, Δt, X0s, actt) where {NIPs,NNODES,DIM}
	DIMTimesNNODES = NNODES
	DIMTimesNNODESSQ = DIMTimesNNODES*DIMTimesNNODES
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Rel += ipRintT(state[$ip], matpars, grad𝐍s_temp[$ip], 𝐍s[$ip], nodalT, nodalTm1, detJs[$ip], wips[$ip], Δt, X0s[$ip], actt)
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

function elStiffnessTVals(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, d𝐍s::NTuple{NIPs, SMatrix{NNODES,DIM,Float64,DIMtimesNNodes}}, 𝐍s::NTuple{NIPs, SVector{NNODES,Float64}}, elX0::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}, dofmap, U, Uprev, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	X0s = elX0s(elX0, 𝐍s)
	Js = Jacobis(elX0, d𝐍s)
	eldofs = dofmap[1,el.inds][:]
	nodalT = U[eldofs]
	nodalTm1 = Uprev[eldofs]
	detJs = DetJs(Js)
	@assert all(detJs .> 0) "error: det(JM) < 0"
	invJs = elInvJs(Js)
	grad𝐍s = Grad𝐍s(d𝐍s, invJs)
	return grad𝐍s, 𝐍s, nodalT, nodalTm1, detJs, X0s
end

function elStiffnessT(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, matpars, dofmap, U, Uprev, shapeFuns, actt, Δt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	grad𝐍s, 𝐍s, nodalT, nodalTm1, detJs, X0s = elStiffnessTVals(el, shapeFuns.d𝐍s, shapeFuns.𝐍s, el.nodes, dofmap, U, Uprev, actt)
	return elStiffnessT(Val{NIPs}, Val{NNODES}, Val{DIM}, el.state.state, matpars, grad𝐍s, 𝐍s, nodalT, nodalTm1, detJs, shapeFuns.wips, Δt, X0s, actt)
end

function elFT(fun::Function, 𝐍, X0, detJ, w::Float64, actt)
	dVw = detJ*w
	return 𝐍*fun(X0, actt)*dVw
end

@generated function elFT(::Type{Val{NIPs}}, ::Type{Val{NNODES}}, fun, 𝐍s, X0s, detJs, wips, actt) where {NIPs, NNODES}
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			qe += elFT(fun, 𝐍s[$ip], X0s[$ip], detJs[$ip], wips[$ip], actt)
		end)
	end
	return quote
        qe = zero(SVector{NNODES,Float64})
        $body
        return qe
	end
end

function elFT(fun::Function, el::Line{DIM, NNODES, NIPs, DIMtimesNNodes}, d𝐍s::NTuple{NIPs, SMatrix{NNODES,DIM2,Float64,DIM2timesNNodes}}, 𝐍s::NTuple{NIPs, SVector{NNODES,Float64}}, elX0::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}, wips::SVector{NIPs,Float64}, actt) where {DIM, DIM2, NNODES, NIPs, DIMtimesNNodes, DIM2timesNNodes}
	X0s = elX0s(elX0, 𝐍s)
	Js = Jacobis(elX0, d𝐍s)
	detJs = ntuple(ip->norm(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	return elFT(Val{NIPs}, Val{NNODES}, fun, 𝐍s, X0s, detJs, wips, actt)
end

function elFT(fun::Function, el::Line{DIM, NNODES, NIPs, DIMtimesNNodes}, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	return elFT(fun, el, shapeFuns.d𝐍s, shapeFuns.𝐍s, el.nodes, shapeFuns.wips, actt)
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

function elPostT(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, d𝐍s::NTuple{NIPs, SMatrix{NNODES,DIM,Float64,DIMtimesNNodes}}, 𝐍s::NTuple{NIPs, SVector{NNODES,Float64}}, elX0::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}, wips::SVector{NIPs,Float64}, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	Js = Jacobis(elX0, d𝐍s)
	detJs = DetJs(Js)
	@assert all(detJs .> 0) "error: det(J) < 0"
	return elPostT(Val{NIPs}, Val{NNODES}, el.state.state, 𝐍s, detJs, wips, actt)
end

function elPostT(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	return elPostT(el, shapeFuns.d𝐍s, shapeFuns.𝐍s, el.nodes, shapeFuns.wips, actt)
end

function updateTrialStates!(::Type{HeatConduction}, state::IPStateVars2D, matpars, grad𝐍_temp, nodalT)
	𝐤 = thermal_conductivity(Val{2}, matpars)
	state.qtr = 𝐤*transpose(grad𝐍_temp)*nodalT
	return nothing
end

function updateTrialStates!(::Type{HeatConduction}, el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, dofmap, U, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	grad𝐍s, _, nodalT, _, _, _ = elStiffnessTVals(el, shapeFuns.d𝐍s, shapeFuns.𝐍s, el.nodes, dofmap, U, U, actt)
	foreach((ipstate,grad𝐍)->updateTrialStates!(HeatConduction, ipstate, el.matpars,  grad𝐍, nodalT), el.state.state, grad𝐍s)
	return nothing
end