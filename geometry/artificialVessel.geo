// 3D artificial vessel with coil
//
// Author: S.B. Lunowa

h = 0.5;

// add vessel and reparametrize
Merge "artificial_vessel/vessel.stl";

DefineConstant[
  // Angle between two triangles above which an edge is considered as sharp
  angle = {40, Min 20, Max 120, Step 1,
    Name "Parameters/Angle for surface detection"},
  // For complex geometries, patches can be too complex, too elongated or too
  // large to be parametrized; setting the following option will force the
  // creation of patches that are amenable to reparametrization:
  forceParametrizablePatches = {0, Choices{0,1},
    Name "Parameters/Create surfaces guaranteed to be parametrizable"},
  // For open surfaces include the boundary edges in the classification process:
  includeBoundary = 1,
  // Force curves to be split on given angle:
  curveAngle = 180
];
ClassifySurfaces{angle * Pi/180, includeBoundary, forceParametrizablePatches,
                 curveAngle * Pi / 180};

// Create a geometry for all the discrete curves and surfaces in the mesh, by computing a parametrization for each one
CreateGeometry;

Surface Loop(1) = Surface{:};

// add coil
Merge "artificial_vessel/coilsMerged_remeshed.stl";
Surface Loop(2)={5};

// remove coil from vessel
Volume(1) = {1,2};

Physical Volume("mesh") = {1};
Physical Surface("in",  1) = {4};
Physical Surface("out", 2) = {3};
Physical Surface("wall",3) = {2,5};

Field[1] = Extend;
Field[1].SurfacesList = {5};
Field[1].SizeMax = h;
Background Field = 1;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.AngleToleranceFacetOverlap = 0.01;
Mesh.MeshSizeMax = h;
// Mesh.MeshSizeMin = 0.25 * h;