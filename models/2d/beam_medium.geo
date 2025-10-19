// Rechteck-Geometrie mit Netzverfeinerung
//Mesh.MshFileVersion = 2.2;
Mesh.SaveAll = 0;
// Parameter für die Netzdichte (kleiner = feiner)
lc = 0.05;

// Vier Eckpunkte (Ursprung + 1x1 Quadrat)
Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {10.0, 0.0, 0.0, lc};
Point(3) = {10.0, 1.0, 0.0, lc};
Point(4) = {0.0, 1.0, 0.0, lc};

// Linien des Rechtecks
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Linien-Schleife und Fläche
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Physical Surface("domain") = {1};