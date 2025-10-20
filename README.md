# SFEM.jl
Statically typed and sized finite element implementation

Implemented is a thermo-mechanically coupled elastoplastic model. The time discretization is backwards Euler and the solution scheme is a standard Newton-Raphson method for triangles up to order 3.

The objective was to write a fairly simple standard finite element discretization with emphasis on computing speed which is realized by statically typing all matrix sizes and types. The src-folder contains just under 2000 lines of code.

```
github.com/AlDanial/cloc v 2.06  T=0.02 s (815.8 files/s, 116283.3 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Julia                           15            165             36           1937
-------------------------------------------------------------------------------
SUM:                            15            165             36           1937
-------------------------------------------------------------------------------
``` 

## Installation

In the terminal
```
git clone https://github.com/baxmittens/SFEM.jl.git
cd SFEM.jl/test
julia
```

In Julia
```
include("install.jl") # takes some minutes
include("beam_medium_p3p1.jl")
```
