
#function response(matpars, εtr::SVector{3,Float64}, εpl::SVector{3,Float64}, ΔT=0.0)
#    ℂ = MaterialStiffness(Val{2}, matpars)
#    αT = thermal_expansivity(Val{2}, matpars)
#	return ℂ * (εtr - αT.*ΔT), εpl
#end

using LinearAlgebra, StaticArrays

function response(matpars, εtr::SVector{3,Float64}, εpl::SVector{3,Float64}, ΔT=0.0)
    E,ν,σy = matpars.E, matpars.ν, matpars.σy
    G  = E / (2*(1+ν))
    ℂ = MaterialStiffness(Val{2}, matpars)
    αT = thermal_expansivity(Val{2}, matpars)
    σtr = ℂ * (εtr - εpl - αT.*ΔT)
    p = (σtr[1] + σtr[2]) / 3.0
    s = σtr .- SVector(p, p, 0.0)
    seq = sqrt(1.5 * (s[1]^2 + s[2]^2 + 2*s[3]^2) / 2)
    f = seq - σy
    if f <= 0
        σ = σtr
        εpltr = εpl
    else
        n = s / (sqrt(s[1]^2 + s[2]^2 + 2*s[3]^2))
        Δγ = f / (3.0*G)
        s_new = s - 2G * Δγ * n
        σ = s_new .+ SVector(p, p, 0.0)
        εpltr = εpl .+ Δγ * n
    end

    return σ, εpltr
end


@generated function grad(f::Function,σ::SVector{3,Float64})
	exprs = Vector{Expr}()
	for j = 1:3
		push!(exprs, :((f(σ+αs[:,$j])[1]-fσ)./h))
	end
	return quote
		h = 10.0^-8
		αs = SMatrix{3,3,Float64,9}(LinearAlgebra.I)*h
		fσ = f(σ)[1]
		σs = $(Expr(:tuple, exprs...))
		return SMatrix{3,3,Float64,9}(σs[1]...,σs[2]...,σs[3]...)
	end
end

function ipStiffness(state, matpars, 𝐁, nodalU, εpl, detJ, w, ΔT=0.0)
	𝐁tr = transpose(𝐁)
	εtr = 𝐁*nodalU
	ℂnum = grad(x->response(matpars, x, εpl, ΔT), εtr)
	dVw = detJ*w
	return 𝐁tr*ℂnum*𝐁*dVw
end

function ipRint(state, matpars, 𝐁, 𝐍, nodalU, εpl, detJ, w, X0, actt, ΔT=0.0)
	dVw = detJ*w
	εtr = 𝐁*nodalU
	σtr = response(matpars, εtr, εpl, ΔT)[1]
	b = transpose(NMat(𝐍))*bodyforceM(X0, matpars, actt)*matpars.ϱ
	return transpose(𝐁)*σtr*dVw - b*dVw
end

@generated function elStiffness(::Type{Val{NIPs}}, ::Type{Val{NNODES}}, ::Type{Val{DIM}}, state, matpars, 𝐁s, 𝐍s, nodalU, εpls, detJs, wips, X0s, actt) where {NIPs,NNODES,DIM}
	DIMTimesNNODES = DIM*NNODES
	DIMTimesNNODESSQ = DIMTimesNNODES*DIMTimesNNODES
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Rel += ipRint(state[$ip], matpars, 𝐁s[$ip], 𝐍s[$ip], nodalU, εpls[$ip], detJs[$ip], wips[$ip], X0s[$ip], actt)
            Kel += ipStiffness(state[$ip], matpars, 𝐁s[$ip], nodalU, εpls[$ip], detJs[$ip], wips[$ip])
		end)
	end
	return quote
		Rel = zero(SVector{$DIMTimesNNODES, Float64})
        Kel = zero(SMatrix{$DIMTimesNNODES,$DIMTimesNNODES,Float64,$DIMTimesNNODESSQ})
        $body
        return Kel, Rel
	end
end

function elStiffnessVals(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, d𝐍s::NTuple{NIPs, SMatrix{NNODES,DIM,Float64,DIMtimesNNodes}}, 𝐍s::NTuple{NIPs, SVector{NNODES,Float64}}, elX0::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}, dofmap, U, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	eldofs = dofmap[SVector{DIM,Int}(1:DIM),el.inds][:]
	nodalU = U[eldofs]
	#X0s = ntuple(ip->elX0*𝐍s[ip], NIPs)
	#Js = ntuple(ip->elX0*d𝐍s[ip], NIPs)
	#detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	#@assert all(detJs .> 0) "error: det(J) < 0"
	#invJs = ntuple(ip->inv(Js[ip]), NIPs)
	#grad𝐍s = ntuple(ip->d𝐍s[ip]*invJs[ip], NIPs)
	#𝐁s = ntuple(ip->Blin0(Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, grad𝐍s[ip]), NIPs)
	X0s = elX0s(elX0, 𝐍s)
	Js = Jacobis(elX0, d𝐍s)
	detJs = DetJs(Js)
	@assert all(detJs .> 0) "error: det(J) < 0"
	invJs = elInvJs(Js)
	grad𝐍s = Grad𝐍s(d𝐍s, invJs)
	𝐁s = el𝐁s(Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, grad𝐍s)
	if actt == 1
		εpls = ntuple(ip->SVector{3,Float64}(0.,0.,0.), NIPs)
	else
		εpls = ntuple(ip->el.state.state[ip].εpl[actt-1], NIPs)
	end
	return 𝐁s, 𝐍s, nodalU, εpls, detJs, X0s
end

function elStiffness(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, matpars, dofmap, U, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	𝐁s, 𝐍s, nodalU, εpls, detJs, X0s = elStiffnessVals(el, shapeFuns.d𝐍s, shapeFuns.𝐍s, el.nodes, dofmap, U, actt)
	return elStiffness(Val{NIPs}, Val{NNODES}, Val{DIM}, el.state.state, el.matpars, 𝐁s, 𝐍s, nodalU, εpls, detJs, shapeFuns.wips, X0s, actt)
end

function elFM(fun::Function, 𝐍, X0, detJ, w::Float64, actt)
	dVw = detJ*w
	return transpose(NMat(𝐍))*fun(X0, actt)*dVw
end

@generated function elFM(::Type{Val{NIPs}}, ::Type{Val{NNODES}}, ::Type{Val{DIM}}, fun, 𝐍s, X0s, detJs, wips, actt) where {NIPs, NNODES, DIM}
	DIMTimesNNODES = DIM*NNODES
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			qe += elFM(fun, 𝐍s[$ip], X0s[$ip], detJs[$ip], wips[$ip], actt)
		end)
	end
	return quote
        qe = zero(SVector{$DIMTimesNNODES,Float64})
        $body
        return qe
	end
end

function elFM(fun::Function, el::Line{DIM, NNODES, NIPs, DIMtimesNNodes}, d𝐍s::NTuple{NIPs, SMatrix{NNODES,DIM2,Float64,DIM2timesNNodes}}, 𝐍s::NTuple{NIPs, SVector{NNODES,Float64}}, elX0::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}, wips::SVector{NIPs,Float64}, actt) where {DIM, DIM2, NNODES, NIPs, DIMtimesNNodes, DIM2timesNNodes}
	X0s = elX0s(elX0, 𝐍s)
	Js = Jacobis(elX0, d𝐍s)
	detJs = ntuple(ip->norm(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	return elFM(Val{NIPs}, Val{NNODES}, Val{DIM}, fun, 𝐍s, X0s, detJs, wips, actt)
end

function elFM(fun::Function, el::Line{DIM, NNODES, NIPs, DIMtimesNNodes}, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	return elFM(fun, el, shapeFuns.d𝐍s, shapeFuns.𝐍s, el.nodes, shapeFuns.wips, actt)
end

function ipMass(𝐍, detJ, w)
	dVw = detJ*w
	return 𝐍*transpose(𝐍)*dVw
end

@generated function elMass(::Type{Val{NIPs}},::Type{Val{NNODES}}, 𝐍s, detJs, wips) where {NIPs,NNODES}
	NNODESSQ = NNODES*NNODES
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Me += ipMass(𝐍s[$ip], detJs[$ip], wips[$ip])
		end)
	end
	return quote
        Me = zero(SMatrix{NNODES,NNODES,Float64,$NNODESSQ})
        $body
        return Me
	end
end

function elMass(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, d𝐍s::NTuple{NIPs, SMatrix{NNODES,DIM,Float64,DIMtimesNNodes}}, 𝐍s::NTuple{NIPs, SVector{NNODES,Float64}}, elX0::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}, wips::SVector{NIPs,Float64}) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	Js = Jacobis(elX0, d𝐍s)
	detJs = DetJs(Js)
	@assert all(detJs .> 0) "error: det(J) < 0"
	return elMass(Val{NIPs}, Val{NNODES}, 𝐍s, detJs, wips)
end

function elMass(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, shapeFuns) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	𝐍s = shapeFuns.𝐍s
	d𝐍s = shapeFuns.d𝐍s
	wips = shapeFuns.wips
	elX0 = el.nodes
	return elMass(el, d𝐍s, 𝐍s, elX0, wips)
end

function elPost(𝐍, vals, detJ, w::Float64)
	dVw = detJ*w
	return 𝐍*transpose(vals)*dVw
end

@generated function elPost(::Type{Val{NIPs}}, ::Type{Val{NNODES}}, state, 𝐍s, detJs, wips, actt) where {NIPs, NNODES}
	NNODES3 = NNODES*3
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			σe += elPost(𝐍s[$ip], state[$ip].σ[actt], detJs[$ip], wips[$ip])
			εple += elPost(𝐍s[$ip], state[$ip].εpl[actt], detJs[$ip], wips[$ip])
		end)
	end
	return quote
        σe = zero(SMatrix{NNODES,3,Float64,$NNODES3})
        εple = zero(SMatrix{NNODES,3,Float64,$NNODES3})
        $body
        return σe,εple
	end
end

function elPost(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, d𝐍s::NTuple{NIPs, SMatrix{NNODES,DIM,Float64,DIMtimesNNodes}}, 𝐍s::NTuple{NIPs, SVector{NNODES,Float64}}, elX0::SMatrix{DIM,NNODES,Float64,DIMtimesNNodes}, wips::SVector{NIPs,Float64}, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	Js = Jacobis(elX0, d𝐍s)
	detJs =  DetJs(Js)
	@assert all(detJs .> 0) "error: det(J) < 0"
	return elPost(Val{NIPs}, Val{NNODES}, el.state.state, 𝐍s, detJs, wips, actt)
end

function elPost(el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	return elPost(el, shapeFuns.d𝐍s, shapeFuns.𝐍s, el.nodes, shapeFuns.wips, actt)
end

function updateTrialStates!(::Type{LinearElasticity}, state::IPStateVars2D, matpars, 𝐁, nodalU, actt)
	εtr = 𝐁*nodalU
	εpl = actt>1 ? state.εpl[actt-1] : zeros(SVector{3,Float64})
	state.σtr,state.εpltr = response(matpars, εtr, εpl)
	return nothing
end

function updateTrialStates!(::Type{LinearElasticity}, el::Tri{DIM, NNODES, NIPs, DIMtimesNNodes}, dofmap, U, shapeFuns, actt) where {DIM, NNODES, NIPs, DIMtimesNNodes}
	𝐁s, _, nodalU, _, _, _ = elStiffnessVals(el, shapeFuns.d𝐍s, shapeFuns.𝐍s, el.nodes, dofmap, U, actt)
	foreach((ipstate,𝐁)->updateTrialStates!(LinearElasticity, ipstate, el.matpars, 𝐁, nodalU, actt), el.state.state, 𝐁s)
	return nothing
end