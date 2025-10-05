using GLMakie

function plot(dom)
	mesh = dom.mesh
	conn = mesh.connectivity
	# Originale Knotenkoordinaten
	X = mesh.nodes[:,1]
	Y = mesh.nodes[:,2]
	
	# Verschiebungen an den Knoten
	U = dom.mma.U
	Ux = U[dom.dofmap[1,:]]
	Uy = U[dom.dofmap[2,:]]
	
	# Deformierte Koordinaten
	Xd = X .+ Ux
	Yd = Y .+ Uy
	
	# Verschiebungsbetrag an jedem Knoten (für Farbgebung)
	#Umag = sqrt.(Ux.^2 .+ Uy.^2)
	
	# --- Koordinaten zu Nx2-Matrix zusammenfassen ---
	points = Point2f.(Xd, Yd)  # Vektor von Point2f
	# oder alternativ: hcat(Xd, Yd)'
	
	# --- Mesh plotten ---
	fig = Figure(size=(1000,500))
	ax = Axis(fig[1,1], aspect=DataAspect())
	
	mesh!(ax, points, hcat(conn...)', color=Ux)#, colormap=:viridis)
	
	#for c in conn
	#    lines!(ax, Xd[c], Yd[c], color=:gray, linewidth=1)
	#end
	
	#for (i, (x, y)) in enumerate(zip(X, Y))
	#    text!(ax, string(i), position=(x, y), align=(:center, :center), color=:black, fontsize=12)
	#end
	
	Colorbar(fig[1,2], label="Ux", tellheight=false)
	
	display(fig)
end
