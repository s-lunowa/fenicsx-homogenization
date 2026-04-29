// Cylinder with spiral cutout
//
// Author: S.B. Lunowa

SetFactory("OpenCASCADE");

// Parameters
h = 0.035;   // mesh size
N_eps = 4;   // number of unit cells
r = 0.25;    // unit cell radius of cylinder
eps = 1/N_eps;

// Create cube
Box(1000) = {0,0,0, 1,1,1};

// Create cylinders with radius r along direction (1,1,1) and cut them from the cube
For i In {0:2*N_eps}
    For j In {0:2*N_eps}
        Cylinder(1+i*(2*N_eps+1)+j) = {(i-N_eps-1)*eps, (j-N_eps-1)*eps, -eps, 1+2*eps, 1+2*eps, 1+2*eps, r*eps};
        BooleanDifference(1001+i*(2*N_eps+1)+j) = { Volume{1000+i*(2*N_eps+1)+j}; Delete; }{ Volume{1+i*(2*N_eps+1)+j}; Delete; };
    EndFor
EndFor

Physical Volume("mesh", 1) = {1001+4*N_eps*(N_eps+1)};
Physical Surface("in", 1) = {1};
If (N_eps == 1)
    Physical Surface("out", 2) = {11};
    Physical Surface("wall", 3) = {2:10,12:15};
ElseIf (N_eps == 2)
    Physical Surface("out", 2) = {23};
    Physical Surface("wall", 3) = {2:22,24:30};
ElseIf (N_eps == 3)
    Physical Surface("out", 2) = {33};
    Physical Surface("wall", 3) = {2:32,34:51};
ElseIf (N_eps == 4)
    Physical Surface("out", 2) = {45};
    Physical Surface("wall", 3) = {2:44,46:78};
ElseIf (N_eps == 5)
    Physical Surface("out", 2) = {59};
    Physical Surface("wall", 3) = {2:58,60:111};
EndIf

// Activate the calculation of mesh element sizes based on curvature (target of 20 elements per 2*Pi radians):
Mesh.MeshSizeFromCurvature = 30;
// Constraint the min and max element sizes to stay within reasonnable values:
Mesh.MeshSizeMin = 0.5*h;
Mesh.MeshSizeMax = h;
