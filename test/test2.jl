include("../src/SFEM.jl")

using .SFEM
using .SFEM.Elements: Tri3Ref, Tri6Ref, evaluateShapeFuns, dim, nnodes, flatten_tuple, shape_functions, monomials, Tri3, Tri6
using .SFEM.IntegrationRules: gaussSimplex
using .SFEM.MeshReader: GmshMesh

using StaticArrays
import DensePolynomials
import DensePolynomials: DensePoly, coordinates, evaluate, fast_binomial, diff
using LinearAlgebra
using SparseArrays

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

function ipStiffness(elX0, d𝐍, nodalU, w)
	J = elX0*d𝐍
	detJ = smallDet(J)
	@assert detJ > 0 "error: det(J) < 0"
	invJ = inv(J)
	grad𝐍 = d𝐍 * invJ
	𝐁 = Blin0(Tri3, grad𝐍)
	E = 1e6
	ν = 0.25
	ℂ = MaterialStiffness(Val{2}, E, ν)
	dVw = detJ*w
	return transpose(𝐁)*ℂ*𝐁*dVw
end

function elStiffness(el, dofmap, U, ΔU)
	elX0 = el.nodes
	eldofs = dofmap[SVector{2,Int}(1,2),el.inds]
	nodalU = U[eldofs].+ΔU[eldofs]
	return reduce(+, map((d𝐍,w)->ipStiffness(elX0, d𝐍, nodalU, w), d𝐍s, wips))
end

function assemble!(I, J, V, dofmap, els, elMats, ndofs)
    k = 1
    for (i,el) in enumerate(els)
        Ke     = elMats[i]
        eldofs = dofmap[SVector{2,Int}(1,2),el.inds]
        for a in 1:ndofs_el
            for b in 1:ndofs_el
                I[k] = eldofs[a]
                J[k] = eldofs[b]
                V[k] = Ke[a, b]
                k += 1
            end
        end
    end
    return sparse(I, J, V, ndofs, ndofs)
end

function simpleDirichletBCExample!(U, mesh, Urval)
	inds_xc_0 = findall(x->isapprox(x[1],0.0,atol=1e-9), eachrow(mesh.nodes))
	inds_yc_0 = findall(x->isapprox(x[2],0.0,atol=1e-9), eachrow(mesh.nodes))
	inds_xc_1 = findall(x->isapprox(x[1],1.0,atol=1e-9), eachrow(mesh.nodes))
	uc_y_0 = dofmap[2,inds_yc_0]
	uc_x_0 = dofmap[1,inds_xc_0]
	uc_x_1 = dofmap[1,inds_xc_1]
	cmap = vcat(uc_y_0, uc_x_0, uc_x_1)
	ucmap = setdiff(1:ndofs, cmap)
	U[uc_x_1] .= Urval
	return ucmap,cmap
end

function malloc(nels,ndofs,ndofs_el,nts)
	Us = zeros(Float64, ndofs, nts)
	ΔUs = zeros(Float64, ndofs, nts)
	Fs = zeros(Float64, ndofs, nts)
	nnz_total = nels * ndofs_el^2
	I = Vector{Int}(undef, nnz_total)
	J = Vector{Int}(undef, nnz_total)
	V = Vector{Float64}(undef, nnz_total)
	return I,J,V,Us,ΔUs,Fs
end

function solve!(I,J,V,U,ΔU,F,els,ndofs,t,Δt,mesh)
	elMats = SMatrix{6, 6, Float64, 36}[elStiffness(el, dofmap, U, ΔU) for el in els];
	Kglob = assemble!(I, J, V, dofmap, els, elMats, ndofs)
	ucmap,cmap = simpleDirichletBCExample!(U, mesh, t)
	ΔU[ucmap] = Kglob[ucmap, ucmap] \ ( F[ucmap] - Kglob[ucmap, cmap] * U[cmap])
	F[cmap] =  Kglob[cmap, ucmap] * (U[ucmap].+ΔU[ucmap]) + Kglob[cmap, cmap] * U[cmap]
	return nothing
end

function newtonraphsonsolve!(I,J,V,U,ΔU,F,els,ndofs,t,Δt,mesh)
	normdu = Inf
	solve!(I,J,V,U,ΔU,F,els,ndofs,t,Δt,mesh)
end

function tsolve!(I,J,V,Us,ΔUs,Fs,els,ndofs,ts,mesh)
	U,ΔU,F = Us[:,1],ΔUs[:,1],Fs[:,1]
	newtonraphsonsolve!(I,J,V,U,ΔU,F,els,ndofs,ts[1],0.0,mesh)
	Us[:,1] .+= ΔU
	for (tim1,t) in enumerate(ts[2:end])
		Us[:,ti] .= Us[:,tim1]
		ti = tim1+1
		Δt = ts[ti]-ts[tim1]
		U,ΔU,F = U[:,ti],ΔUs[:,ti],F[:,ti]
		newtonraphsonsolve!(I,J,V,U,ΔU,F,els,ndofs,ts[ti],Δt,mesh)
		Us[:,ti] .+= ΔU
	end
end

meshfilepath = "../models/2d/patchtest.msh"
mesh = GmshMesh(meshfilepath)
els = Tri3[Tri3(SMatrix{2,3,Float64,6}(mesh.nodes[elinds,1:2]'), SVector{3,Int}(elinds)) for elinds in mesh.connectivity]
nels = length(els)

tri3 = Tri3Ref()
𝐍s,d𝐍s,wips = evaluateShapeFuns(tri3, gaussSimplex, 3)

ndofs = size(mesh.nodes,1)*dim(tri3)
ndofs_el = length(els[1].inds)*dim(els[1])
dofmap = reshape(1:ndofs,2,:)

ts = [1.0]
I,J,V,Us,ΔUs,Fs = malloc(nels,ndofs,ndofs_el,length(ts))

tsolve!(I,J,V,Us,ΔUs,Fs,els,ndofs,ts,mesh)

using GLMakie

conn = mesh.connectivity
# Originale Knotenkoordinaten
X = mesh.nodes[:,1]
Y = mesh.nodes[:,2]

# Verschiebungen an den Knoten
Ux = Us[dofmap[1,:],end]
Uy = Us[dofmap[2,:],end]

# Deformierte Koordinaten
Xd = X .+ Ux
Yd = Y .+ Uy

# Verschiebungsbetrag an jedem Knoten (für Farbgebung)
#Umag = sqrt.(Ux.^2 .+ Uy.^2)

# --- Koordinaten zu Nx2-Matrix zusammenfassen ---
points = Point2f.(Xd, Yd)  # Vektor von Point2f
# oder alternativ: hcat(Xd, Yd)'

# --- Mesh plotten ---
fig = Figure(size=(900,900))
ax = Axis(fig[1,1], aspect=DataAspect())

mesh!(ax, points, hcat(conn...)', color=Ux, colormap=:viridis)

for c in conn
    lines!(ax, Xd[c], Yd[c], color=:gray, linewidth=1)
end

#for (i, (x, y)) in enumerate(zip(X, Y))
#    text!(ax, string(i), position=(x, y), align=(:center, :center), color=:black, fontsize=12)
#end

Colorbar(fig[1,2], colormap=:viridis, label="|U|")

fig
