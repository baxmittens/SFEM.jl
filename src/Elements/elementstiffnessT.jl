
function ipStiffnessT(state, grad𝐍_temp, nodalT, detJ, w)
	𝐤 = SMatrix{2,2,Float64,4}(50.,0.,0.,50.)
	dVw = detJ*w
	K_TT = grad𝐍_temp*𝐤*transpose(grad𝐍_temp)*dVw
	return K_TT
end
	
function ipRintT(state, grad𝐍_temp, nodalT, detJ, w) # needs update
	dVw = detJ*w
	return grad𝐍_temp*state.qtr*dVw
end

@generated function elStiffnessT(::Type{Val{NIPs}}, ::Type{Val{NNODES}}, ::Type{Val{DIM}}, state, grad𝐍s_temp, nodalT, detJs, wips) where {NIPs,NNODES,DIM}
	DIMTimesNNODES = NNODES
	DIMTimesNNODESSQ = DIMTimesNNODES*DIMTimesNNODES
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Rel += ipRintT(state[$ip], grad𝐍s_temp[$ip], nodalT, detJs[$ip], wips[$ip])
            Kel += ipStiffnessT(state[$ip], grad𝐍s_temp[$ip], nodalT, detJs[$ip], wips[$ip])
		end)
	end
	return quote
		Rel = zero(SVector{$DIMTimesNNODES, Float64})
        Kel = zero(SMatrix{$DIMTimesNNODES,$DIMTimesNNODES,Float64,$DIMTimesNNODESSQ})
        $body
        return Kel, Rel
	end
end

function elStiffnessT(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, dofmap, U, ΔU, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	d𝐍s = shapeFuns.d𝐍s
	wips = shapeFuns.wips
	elT0 = el.nodes
	eldofs = dofmap[1,el.inds][:]
	nodalT = U[eldofs]
	Js = ntuple(ip->elT0*d𝐍s[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(JM) < 0"
	invJs = ntuple(ip->inv(Js[ip]), NIPs)
	grad𝐍s = ntuple(ip->d𝐍s[ip]*invJs[ip], NIPs)
	return elStiffnessT(Val{NIPs}, Val{NNODES}, Val{DIM}, el.state.state, grad𝐍s, nodalT, detJs, wips)
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

function elPostT(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	𝐍s = shapeFuns.𝐍s
	d𝐍s = shapeFuns.d𝐍s
	wips = shapeFuns.wips
	elX0 = el.nodes
	Js = ntuple(ip->elX0*d𝐍s[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	return elPostT(Val{NIPs}, Val{NNODES}, el.state.state, 𝐍s, detJs, wips, actt)
end

function updateTrialStates!(::Type{HeatConduction}, state::IPStateVars2D, grad𝐍_temp, nodalT, actt)
	𝐤 = SMatrix{2,2,Float64,4}(50.0,0.0,0.0,50.0)
	state.qtr = 𝐤*transpose(grad𝐍_temp)*nodalT
	return nothing
end

function updateTrialStates!(::Type{HeatConduction}, el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, dofmap, U, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	d𝐍s = shapeFuns.d𝐍s
	elX0 = el.nodes
	eldofs = dofmap[1,el.inds][:]
	nodalT = U[eldofs]
	Js = ntuple(ip->elX0*d𝐍s[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	invJs = ntuple(ip->inv(Js[ip]), NIPs)
	grad𝐍s = ntuple(ip->d𝐍s[ip]*invJs[ip], NIPs)
	foreach((ipstate,grad𝐍)->updateTrialStates!(HeatConduction, ipstate, grad𝐍, nodalT, actt), el.state.state, grad𝐍s)
	return nothing
end