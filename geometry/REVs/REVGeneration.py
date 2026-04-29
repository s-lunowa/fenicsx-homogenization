import bpy
import numpy as np
# Script to reconstruct REVs for the real case.
#
# This script must be run using Blender’s Python environment (bpy).
# It will NOT work with a standard Python interpreter.
#
# To run it from the terminal:
# blender --background --python REVGeneration.py
#
# Requirements:
# - Blender must be installed
# - Run the command from this directory (or adjust file paths accordingly)

# Remove everything
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import the OBJ mesh
filepath = "../real_case/coils.obj"
bpy.ops.wm.obj_import(
    filepath=filepath,
    forward_axis='Y',
    up_axis='Z',
)

base_obj = bpy.context.selected_objects[0]
print("Imported object:", base_obj.name)

# Output directory
outdir = "./"

# Positions to process
positions = -1*np.array([
    [6.008,   6.0037, 3.4961],
    [8,       3.8,    3],
    [7,       6.5,    4],
    [5.67,    6.82,   2.6],
    [7.4735,  6.262,  3.9615],
    [8.3517,  5.2054, 6.60],
    [5.68,    7.27,   2.61],
    [6.144,   7.72,   3.4894],
    [9.1149,  6.9253, 3.5424],
    [6.9426,  7.15,   3.6673],
    [9.2,     4.07,   3.45],
    [7.5,     7.11,   1.98],
    [7.2,     5.0,    2.8],
    [7.89,    3.97,   3.98]
])

# Store volumes
volumes = []

# -------------------------------
# Main loop for all coordinates
# -------------------------------

for i, pos in enumerate(positions, start=1):
    # Duplicate the imported mesh (so each run is independent)
    bpy.ops.object.select_all(action='DESELECT')
    base_obj.select_set(True)
    bpy.ops.object.duplicate()
    mesh_copy = bpy.context.selected_objects[0]

    # Create cube
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    cube = bpy.context.active_object
    cube.name = f"REV_cube_{i}"

    # Move cube
    cube.location = pos

    # Boolean intersection
    bool_mod = cube.modifiers.new(name="BooleanIntersect", type='BOOLEAN')
    bool_mod.operation = 'INTERSECT'
    bool_mod.object = mesh_copy

    # Apply modifier
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)
    

    # Export result
    outfile = f"{outdir}/REV_{i}.obj"
    bpy.ops.object.select_all(action='DESELECT')
    cube.select_set(True)   # export only cube
    bpy.ops.wm.obj_export(
        filepath=outfile,
        export_selected_objects=True,
        forward_axis='Y',
        up_axis='Z'
    )

    # Cleanup objects before next iteration
    bpy.ops.object.delete()      # delete cube result
    mesh_copy.select_set(True)
    bpy.ops.object.delete()      # delete duplicate mesh

    print(f"Exported REV_{i}.obj")

print("All exports complete!")
