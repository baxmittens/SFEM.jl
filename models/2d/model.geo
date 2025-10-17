//------------------------------------------------------------
// 2D Geometry: Rectangle (50x200) with two concentric circles
//------------------------------------------------------------
SetFactory("OpenCASCADE");

// === PARAMETERS ===
Lx = 50;
Ly = 200;
r1 = 2;    // inner radius (canister)
r2 = 3;    // outer radius (backfill)
lc_inner = 0.2;
lc_outer = 8;

//------------------------------------------------------------
// === GEOMETRY ===

Point(1) = {-25.0, -100.0, 0.0, lc_outer};
Point(2) = {25.0, -100.0, 0.0, lc_outer};
Point(3) = {25.0, 100.0, 0.0, lc_outer};
Point(4) = {-25.0, 100.0, 0.0, lc_outer};

// Linien des Rechtecks
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Linien-Schleife und Fläche
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Outer rectangle (base domain)
//Rectangle(1) = {-Lx/2, -Ly/2, 0, Lx, Ly};

// Inner and outer disks
Disk(2) = {0, 0, 0, r2, r2}; // backfill + canister
Disk(3) = {0, 0, 0, r1, r1}; // canister only

// Boolean differences
backfill[]   = BooleanDifference{ Surface{2}; Delete; }{ Surface{3}; Delete; };
outerdomain[] = BooleanDifference{ Surface{1}; Delete; }{ Surface{backfill[0]}; };

//------------------------------------------------------------
// === PHYSICAL GROUPS ===
Physical Surface("Canister", 0) = {3};
Physical Surface("Backfill", 1) = {backfill[0]};
Physical Surface("Domain",   2) = {outerdomain[0]};



// Verfeinerung an den Kreisrändern:
Characteristic Length{ PointsOf{ Surface{3}; } } = lc_outer;
Characteristic Length{ PointsOf{ Surface{backfill[0]}; } } = lc_inner;

Mesh 2;
