// Cylinder with spiral cutout
//
// Author: S.B. Lunowa

SetFactory("OpenCASCADE");

// Parameters
h = 0.03;    // mesh size
R = 1;       // outer radius
r = 0.675;   // small radius
rr = 0.25;   // coil radius
L = 4;       // length
nturns = 5;  // number of turns
nt = 200;    // number of points

// Create cylinder: base center (0,0,0), axis along x with height 0.75*L and radius R
Cylinder(1) = {0, 0, 0, 0.75*L, 0, 0, R};

// Create points along the spiral with radius r
For i In {0:nt}
  t = i * 2*Pi*nturns / nt;
  x = r * Cos(t);
  y = r * Sin(t);
  z = i * L / nt - 0.125*L;
  Point(100+i) = {z, x, y};
EndFor
Spline(10) = {100:(100+nt)}; // Join points into a spline
Wire(10) = {10};
// We extrude a disk of radius rr along the wire (open spline):
Disk(10) = {-0.125*L,r,0, rr};
Rotate{ {0, 1, 0}, {-0.125*L,r,0}, L/(2*Pi*r*nturns) } { Surface{10}; }
Extrude { Surface{10}; } Using Wire {10}
Delete{ Surface{10}; }
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

// Create points along the spiral with radius 0.25
For i In {0:nt}
  t = i * 2*Pi*nturns / nt + Pi;
  x = rr * Cos(t);
  y = rr * Sin(t);
  z = i * L / nt - 0.125*L;
  Point(500+i) = {z, x, y};
EndFor
Spline(20) = {500:(500+nt)}; // Join points into a spline
Wire(20) = {20};
// We extrude a disk of radius rr along the wire (open spline):
Disk(20) = {-0.125*L,-rr,0, rr};
Rotate{ {0, 1, 0}, {-0.125*L,-rr,0}, -L/(2*Pi*rr*nturns) } { Surface{20}; }
Extrude { Surface{20}; } Using Wire {20}
Delete{ Surface{20}; }

BooleanDifference(5) = { Volume{3}; Delete; }{ Volume{4}; Delete; };

Physical Volume("mesh") = {5};
Physical Surface("in",  1) = {3};
Physical Surface("out", 2) = {2};
Physical Surface("wall",3) = {1,4,5};

// Activate the calculation of mesh element sizes based on curvature (target of 20 elements per 2*Pi radians):
Mesh.MeshSizeFromCurvature = 25;
// Constraint the min and max element sizes to stay within reasonnable values:
Mesh.MeshSizeMin = 0.25*h;
Mesh.MeshSizeMax = h;
