// Alternative 3D circular tube geometry
//
// Author: S.B. Lunowa

SetFactory("OpenCASCADE");

// Parameters
r = 1;    // radius
L = 3.0;  // length
h = 0.08; // target mesh size

// Create cylinder: base center (0,0,0), axis along x, radius r, height L

// create symmetric circle in the y-z plane at x=0 and extrude along x
Point(1) = {0,  r,  0, h};
Point(2) = {0,  0,  r, h};
Point(3) = {0, -r,  0, h};
Point(4) = {0,  0, -r, h};
Point(5) = {0,  0,  0, h}; // center for circle arcs
Circle(1) = {1, 5, 2};
Circle(2) = {2, 5, 3};
Circle(3) = {3, 5, 4};
Circle(4) = {4, 5, 1};
Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

// Extrude the planar disk along x to form the tube. Layers controls the number
// of element layers along x; Recombine gives structured hexa-like elements.
Extrude {L, 0, 0} {
  Surface{1};
  Layers{Ceil(L/h)};
}

// Tag physical surfaces
// We can use BooleanFragments to extract faces and label them
// but OpenCASCADE automatically labels sub-entities we can refer to

// Define the fluid volume
Physical Volume("fluid", 1) = {1};

// Extract boundary surfaces
// The cylinder command automatically creates 3 surface entities:
//  - lateral wall
//  - circular outlet (x=L)
//  - circular inlet (x=0)
Physical Surface("inlet", 1)  = {1};   // bottom disk
Physical Surface("outlet", 2) = {6};   // top disk
Physical Surface("wall", 3)   = {2:5}; // curved surface

Mesh.MeshSizeMin = 0.5*h;
Mesh.MeshSizeMax = h;
Mesh.ElementOrder = 2;      // quadratic (curved) mesh
// Mesh.HighOrderOptimize = 2; // smooth curved geometry
