
function smallDet(M::SMatrix{2,2,Float64,4})
	@inbounds return (M[1,1]*M[2,2] - M[1,2]*M[2,1])
end
function smallDet(M::SMatrix{3,3,Float64,9})
	@inbounds return (M[1,1]*(M[2,2]*M[3,3]-M[2,3]*M[3,2]) - M[1,2]*(M[2,1]*M[3,3]-M[2,3]*M[3,1]) + M[1,3]*(M[2,1]*M[3,2]-M[2,2]*M[3,1]))
end

function Blin0(::Type{Tri3}, gradN::SMatrix{3,2,Float64,6})
	return SMatrix{3,6,Float64,18}(
		gradN[1,1],0.0,gradN[1,2],
		0.0,gradN[1,2],gradN[1,1],
		gradN[2,1],0.0,gradN[2,2],
		0.0,gradN[2,2],gradN[2,1],
		gradN[3,1],0.0,gradN[3,2],
		0.0,gradN[3,2],gradN[3,1])
end

function MaterialStiffness(::Type{Val{2}}, E, ν)
	fac = E/((1+ν)*(1-2*ν))
	return fac*SMatrix{3,3,Float64,9}(1-ν,ν,0.,ν,1-ν,0.,0.,0.,(1-2*ν)/2.0)
end

#function response(εtr, εpl)
#	E = 1e6
#	ν = 0.25
#	ℂ = MaterialStiffness(Val{2}, E, ν)
#	return ℂ*εtr, εpl
#end

using LinearAlgebra, StaticArrays

function response(εtr::SVector{3,Float64}, εpl::SVector{3,Float64})
    # Materialparameter
    E  = 1e6
    ν  = 0.25
    σy = 200.0
    G  = E / (2*(1+ν))

    # 2D Elastizitätsmatrix (plane strain)
    ℂ = MaterialStiffness(Val{2}, E, ν)

    # Trialspannung
    σtr = ℂ * (εtr - εpl)

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
        n = s / (sqrt(s[1]^2 + s[2]^2 + 2*s[3]^2) + eps())
        #n = s / sqrt(2 * J2)
        Δγ = f / (3.0*G)  # ohne Verfestigung
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
		h = 10.0^-9
		αs = SMatrix{3,3,Float64,9}(LinearAlgebra.I)*h
		fσ = f(σ)[1]
		σs = $(Expr(:tuple, exprs...))
		return SMatrix{3,3,Float64,9}(σs[1]...,σs[2]...,σs[3]...)
	end
end

function ipStiffness(state, 𝐁, nodalU, εpl, detJ, w)
	𝐁tr = transpose(𝐁)
	εtr = 𝐁*nodalU
	ℂnum = grad(x->response(x, εpl), εtr)
	dVw = detJ*w
	return 𝐁tr*ℂnum*𝐁*dVw
end

function ipRint(state, 𝐁, nodalU, εpl, σ, detJ, w)
	E = 1e6
	ν = 0.25
	𝐁tr = transpose(𝐁)
	ℂ = MaterialStiffness(Val{2}, E, ν)
	ε = 𝐁*nodalU
	σ = ℂ * (ε-εpl)
	#display(hcat(σ,state.σtr))
	dVw = detJ*w
	return 𝐁tr*state.σtr*dVw
end

@generated function elStiffness(::Type{Val{NIPs}}, state, 𝐁s, nodalU, εpls, σs, detJs, wips) where {NIPs}
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Rel += ipRint(state[$ip], 𝐁s[$ip], nodalU, εpls[$ip], σs[$ip], detJs[$ip], wips[$ip])
            Kel += ipStiffness(state[$ip], 𝐁s[$ip], nodalU, εpls[$ip], detJs[$ip], wips[$ip])
		end)
	end
	return quote
		Rel = zero(SVector{6, Float64})
        Kel = zero(SMatrix{6,6,Float64,36})
        $body
        return Kel, Rel
	end
end

function elStiffness(el::Tri3{NIPs}, dofmap, U, ΔU, shapeFuns, actt) where {NIPs}
	d𝐍s = shapeFuns.d𝐍s
	wips = shapeFuns.wips
	elX0 = el.nodes
	eldofs = dofmap[SVector{2,Int}(1,2),el.inds][:]
	nodalU = U[eldofs]
	Js = ntuple(ip->elX0*d𝐍s[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	invJs = ntuple(ip->inv(Js[ip]), NIPs)
	grad𝐍s = ntuple(ip->d𝐍s[ip]*invJs[ip], NIPs)
	𝐁s = ntuple(ip->Blin0(Tri3, grad𝐍s[ip]), NIPs)
	if actt == 1
		εpls = ntuple(ip->SVector{3,Float64}(0.,0.,0.), NIPs)
		σs = ntuple(ip->SVector{3,Float64}(0.,0.,0.), NIPs)
	else
		εpls = ntuple(ip->el.state.state[ip].εpl[actt-1], NIPs)
		σs = ntuple(ip->el.state.state[ip].σtr, NIPs)
	end
	return elStiffness(Val{NIPs}, el.state.state, 𝐁s, nodalU, εpls, σs, detJs, wips)
end

function ipMass(𝐍, detJ, w)
	dVw = detJ*w
	return 𝐍*transpose(𝐍)*dVw
end

@generated function elMass(::Type{Val{NIPs}}, 𝐍s, detJs, wips) where {NIPs}
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			Me += ipMass(𝐍s[$ip], detJs[$ip], wips[$ip])
		end)
	end
	return quote
        Me = zero(SMatrix{3,3,Float64,9})
        $body
        return Me
	end
end

function elMass(el::Tri3{NIPs}, dofmap, shapeFuns) where {NIPs}
	𝐍s = shapeFuns.𝐍s
	d𝐍s = shapeFuns.d𝐍s
	wips = shapeFuns.wips
	elX0 = el.nodes
	eldofs = dofmap[SVector{2,Int}(1,2),el.inds][:]
	Js = ntuple(ip->elX0*d𝐍s[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	return elMass(Val{NIPs}, 𝐍s, detJs, wips)
end

function elPost(𝐍, vals, detJ, w)
	dVw = detJ*w
	return 𝐍*transpose(vals)*dVw
end

@generated function elPost(::Type{Val{NIPs}}, state, 𝐍s, detJs, wips, actt) where {NIPs}
	body = Expr(:block)
	for ip in 1:NIPs
		push!(body.args, quote
			σe += elPost(𝐍s[$ip], state[$ip].σ[actt], detJs[$ip], wips[$ip])
			εple += elPost(𝐍s[$ip], state[$ip].εpl[actt], detJs[$ip], wips[$ip])
		end)
	end
	return quote
        σe = zero(SMatrix{3,3,Float64,9})
        εple = zero(SMatrix{3,3,Float64,9})
        $body
        return σe,εple
	end
end

function elPost(el::Tri3{NIPs}, dofmap, shapeFuns, actt) where {NIPs}
	𝐍s = shapeFuns.𝐍s
	d𝐍s = shapeFuns.d𝐍s
	wips = shapeFuns.wips
	elX0 = el.nodes
	eldofs = dofmap[SVector{2,Int}(1,2),el.inds][:]
	Js = ntuple(ip->elX0*d𝐍s[ip], NIPs)
	detJs = ntuple(ip->smallDet(Js[ip]), NIPs)
	@assert all(detJs .> 0) "error: det(J) < 0"
	return elPost(Val{NIPs}, el.state.state, 𝐍s, detJs, wips, actt)
end