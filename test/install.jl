filedir = @__DIR__
import Pkg
Pkg.add("StaticArrays")
Pkg.add(url="https://github.com/baxmittens/DensePolynomials.jl.git")
Pkg.add("GLMakie")
Pkg.add("NearestNeighbors")
Pkg.add("GeometryBasics")
Pkg.develop(path=joinpath(filedir,".."))