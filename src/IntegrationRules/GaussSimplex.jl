using StaticArrays

function gaussSimplex(dim::Int, nip::Int)
	if dim == 2
	#  integration points for two dimensions
		if nip == 1 # polynomial degree 1
			m1 = 0.333333333333333
			r  = SVector{1,Float64}(m1)
			s  = SVector{1,Float64}(m1)
			w  = SVector{1,Float64}(0.5)
			return ((r,s), w)
		elseif nip == 3 # polynomial degree 2
			w1 = 0.166666666666667
			w2 = 0.666666666666667
			r = SVector{3,Float64}(w1, w2, w1)
			s = SVector{3,Float64}(w1, w1, w2)
			w = SVector{3,Float64}(w1, w1, w1)
			return ((r,s), w)
		elseif nip == 4 # polynomial degree 3
			w1 = 0.260416666666667
			w2 = -0.281250000000000
			r1 = 0.200000000000000
			r2 = 0.600000000000000
			r3 = 0.333333333333333
			r = SVector{4,Float64}(r1, r2, r1, r3)
			s = SVector{4,Float64}(r1, r1, r2, r3)
			w = SVector{4,Float64}(w1, w1, w1, w2)
			return ((r,s), w)
		elseif nip == 6 # polynomial degree 4
			a = 0.445948490915965
			b = 0.091576213509771
			w1 = 0.111690794839005
			w2 = 0.054975871827661
			r = SVector{6,Float64}(a, 1.0-2.0*a, a, b, 1.0-2.0*b, b)
			s = SVector{6,Float64}(a, a, 1.0-2.0*a, b, b, 1.0-2.0*b)
			w = SVector{6,Float64}(w1, w1, w1, w2, w2, w2)
			return ((r,s), w)
		else
			error("gaussSimplex: dim $dim, nip $nip combination not supported")
		end
		return nothing
	end
end