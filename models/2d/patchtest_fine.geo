// Rechteck-Geometrie mit Netzverfeinerung

// Parameter für die Netzdichte (kleiner = feiner)
lc = 0.01;

// Vier Eckpunkte (Ursprung + 1x1 Quadrat)
Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {1.0, 0.0, 0.0, lc};
Point(3) = {1.0, 1.0, 0.0, lc};
Point(4) = {0.0, 1.0, 0.0, lc};

// Linien des Rechtecks
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Linien-Schleife und Fläche
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Physikalische Gruppen (optional, nützlich für FE-Software)
Physical Surface("Domain") = {1};
Physical Line("Boundary") = {1,2,3,4};