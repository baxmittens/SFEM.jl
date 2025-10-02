using Combinatorics

"""
    
    *** not working code created by chatgpt. needs fixing ***

    grundmann_moeller(d, s)

Erzeuge Integrationsknoten und Gewichte der Grundmann–Möller-Regel
für das Standard-Simplex in Dimension `d` mit Parameter `s`.

- `d`: Dimension des Simplex (z. B. 2 = Dreieck, 3 = Tetraeder)
- `s`: Genauigkeitsparameter (exakt für Polynome bis Grad 2s+1)

Gibt eine Liste von `(x, w)` zurück, wobei `x` der Punkt (als Vektor)
und `w` das Gewicht ist.
"""
function grundmann_moeller(d::Int, s::Int)
    points = []
    volume = 1 / factorial(d)  # Volumen des Standard-Simplex
    for r in 0:s
        # Gewicht für diesen Level
        c_r = ((-1)^r * binomial(d+s, s-r)) / binomial(d+s, s)

        # Alle α mit Summe = r
        for α in MultisetPartitions(fill(1, r), d+1)
            counts = countmap(α)
            vec = [get(counts, i, 0) for i in 1:d]  # nur d Koordinaten, letzte = Rest
            x = [vi / (d+s) for vi in vec]
            x_last = (d+s - sum(vec)) / (d+s)
            x_full = vcat(x, x_last)  # baryzentrische Koordinaten
            push!(points, (x_full[1:end-1], c_r * volume))
        end
    end
    return points
end

# Beispiel: Dreieck (d=2), s=1 -> exakt bis Grad 3
pts = grundmann_moeller(2, 1)
for (p, w) in pts
    println("Punkt = $p, Gewicht = $w")
end
