include("../src/SFEM.jl")

using .SFEM
using .SFEM.Elements: Tri3

tri3 = Tri3()
display(tri3.nodes)