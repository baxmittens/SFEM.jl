@generated function globalToLocalGuess(xPt::SVector{N,Float64}, nodalX::SMatrix{N,M,Float64,NM})  where {N,M,NM}
    indsmat = SVector{3,Int}(1:3) 
    inds = SVector{N,Int}(2:N+1)
    return quote
        A = vcat(ones(SMatrix{1,3,Float64}), nodalX[:, $indsmat])
        x = vcat(ones(SVector{1,Float64}), xPt)
        b = inv(A)*x
        return b[$inds]
    end
end

function Tri_checkInHullConstraint(ec::SVector{2,Float64})
    s = ec[1]+ec[2] - 1.0 > 1e-10
    sm = ec[1] < -1e-10 || ec[2] < -1e-10
    if sm || s
        return false
    else
        return true
    end
end

function globalToLocal(xPt::SVector{N,Float64}, nodalX::SMatrix{N,M,Float64,NM}, shapeFuns, dshapeFuns) where {N,M,NM}
    ec = globalToLocalGuess(xPt, nodalX)
    iter = 0
    maxIter = 20
    nrm = Inf
    tol = 1e-9
    while nrm > tol
        iter += 1
        if iter > maxIter
            break
        end
        Ne = SVector{M,Float64}(ntuple(i->evaluate(shapeFuns[i], ec), M))
        dNe = SMatrix{M,N,Float64,NM}(flatten_tuple(ntuple(dir->ntuple(i->evaluate(dshapeFuns[i][dir], ec), M), N)))
        A = nodalX*dNe
        b = (nodalX*Ne - xPt)
        dec = -inv(A)*b
        ec += dec
        nrm = norm(dec)
    end
    if iter > maxIter
        success = false
    else
        success = Tri_checkInHullConstraint(ec)
    end
    return (success, ec)
end
