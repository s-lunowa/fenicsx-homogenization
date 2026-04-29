import numpy as np
import vtk
import h5py

def write_grid_hdf5_pointcloud(get_flattened_grid, grid_shape, fileOutName: str, mask=None, **fields):
    """
    Write cell-centered data and coordinates to HDF5.
    Points and all fields are stored as datasets.

    Parameters
    ----------
    fileOutName : str
        Output filename, '.h5' appended if missing.
    mask : array-like of bool or int, optional
        Boolean mask of length nCells selecting which points to include.
    **fields : dict
        Scalar or vector fields with shapes:
            - (nx, ny, nz)
            - (nCells,)
            - (nCells, 3)
    """

    nx, ny, nz = grid_shape
    n_cells = nx * ny * nz

    # ---- Validate and reshape points ----
    points = np.asarray(get_flattened_grid)
    if points.shape[0] != n_cells:
        raise ValueError(f"Expected {n_cells} cell centers but got {points.shape[0]}")

    # ---- Apply mask ----
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size != n_cells:
            raise ValueError(f"Mask has {mask.size} elements, expected {n_cells}")

        points = points[mask]
        print(f"Mask applied: writing {len(points)} / {n_cells} points")
    else:
        print(f"No mask provided: writing all {n_cells} points")

    nPts = len(points)

    # ---- Prepare filename ----
    if not (fileOutName.endswith(".h5") or fileOutName.endswith(".hdf5")):
        fileOutName += ".h5"

    # ---- Write HDF5 ----
    with h5py.File(fileOutName, "w") as h5:

        # Store metadata
        h5.create_dataset("grid_shape", data=np.array(grid_shape, dtype=int))
        if mask is not None:
            h5.create_dataset("mask", data=mask)

        # Store points
        h5.create_dataset("points", data=points.astype(np.float64))

        # Create a group for fields
        field_group = h5.create_group("fields")

        # ---- Add all fields ----
        for key, arr in fields.items():
            data = np.asarray(arr)

            # Reshape 3D arrays into 1D
            if data.ndim == 3 and data.shape == (nx, ny, nz):
                data = data.flatten(order="F")

            # Vector field: shape (n, 3)
            elif data.ndim == 2 and data.shape[1] == 3:
                pass

            # Scalar 1D array
            elif data.ndim == 1 and data.size == n_cells:
                pass

            else:
                raise ValueError(f"Unsupported data shape for '{key}': {data.shape}")

            # Apply mask
            if mask is not None:
                data = data[mask]

            field_group.create_dataset(key, data=data.astype(np.float64))

            print(f"Stored field '{key}' → shape {data.shape}")

    print(f"Wrote HDF5 point cloud: {fileOutName}")
    return fileOutName


def write_grid_vtk_pointcloud(get_flattened_grid, grid_shape, fileOutName: str, mask=None, **fields):
    """
    Write cell-centered data as a VTK point cloud (vtkPolyData).

    Each cell center is written as a single point with attached scalar/vector
    attributes. Optionally, a mask can select which points to include.

    Parameters
    ----------
    fileOutName : str
        Output file name ('.vtk' appended if missing).
    mask : array-like of bool or int, optional
        Boolean or integer mask of length nx*ny*nz selecting which points to write.
        If None, all points are included.
    **fields : dict[str, array-like]
        Scalar or vector fields defined per cell, e.g. porosity, masks, etc.
        Each array can be:
        - 3D array of shape (nx, ny, nz)
        - 1D array of length nx*ny*nz
        - 2D array of shape (nx*ny*nz, 3) for vectors
    """

    nx, ny, nz = grid_shape
    n_cells = nx * ny * nz

    points = np.asarray(get_flattened_grid)
    if points.shape[0] != n_cells:
        raise ValueError(f"Expected {n_cells} cell centers but got {points.shape[0]}")

    # --- Apply mask if provided ---
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size != n_cells:
            raise ValueError(f"Mask has {mask.size} elements but expected {n_cells}")
        points = points[mask]
        print(f"Mask applied: writing {points.shape[0]} / {n_cells} points")
    else:
        print(f"No mask provided: writing all {n_cells} points")

    nPts = points.shape[0]
    print(f"Creating VTK point cloud with {nPts} points ({nx}×{ny}×{nz})")

    # --- Convert numpy points to vtkPoints ---
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToDouble()
    for p in points:
        vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

    # --- Create PolyData container ---
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Represent each point as a vertex so they appear in ParaView
    vertices = vtk.vtkCellArray()
    for i in range(nPts):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    polydata.SetVerts(vertices)

    # ===========================================================
    # --- Helper: attach scalar/vector data to points -----------
    # ===========================================================
    def add_point_data(poly, name: str, data):
        data = np.asarray(data)
        if data.ndim == 3 and data.shape == (nx, ny, nz):
            data = data.flatten(order="F")
        elif data.ndim == 2 and data.shape[1] == 3:
            pass  # vector data
        elif data.ndim != 1:
            raise ValueError(f"Unsupported data shape for '{name}': {data.shape}")

        # Apply mask if given
        if mask is not None:
            data = data[mask]

        arr = vtk.vtkDoubleArray()
        arr.SetName(name)

        if data.ndim == 1:
            arr.SetNumberOfComponents(1)
            for val in data:
                arr.InsertNextValue(float(val))
            poly.GetPointData().AddArray(arr)
        elif data.ndim == 2 and data.shape[1] == 3:
            arr.SetNumberOfComponents(3)
            for gx, gy, gz in data:
                arr.InsertNextTuple3(float(gx), float(gy), float(gz))
            poly.GetPointData().AddArray(arr)

    # --- Add all user-supplied scalar/vector fields ---
    for key, arr in fields.items():
        print(f"Adding field '{key}' as point data.")
        add_point_data(polydata, key, arr)

    # --- Write file ---
    if not fileOutName.endswith(".vtk"):
        fileOutName += ".vtk"

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fileOutName)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Write()

    print(f"Wrote VTK point cloud ({nPts} points) with {len(fields)} fields to {fileOutName}")
    return polydata


def write_grid_vtk_cellcentered(grid, grid_shape, sample_width, fileOutName: str, **fields):
    """
    Write a structured VTK grid where data (e.g., scalar fields, masks, etc.)
    are defined at **cell centers** rather than corners.

    Parameters
    ----------
    fileOutName : str
        Output file name ('.vtk' appended if missing).
    **fields : dict of str -> array-like
        Additional scalar or vector fields (cell-centered) to attach to the VTK file.
        Each value can be:
        - 3D array of shape (nx, ny, nz)
        - 1D array of length nx*ny*nz
        - 2D array of shape (nx*ny*nz, 3) for vectors
    """
    nx, ny, nz = grid_shape
    dx = sample_width

    # Extract coordinate axes (cell-centered)
    xq = grid[0][:, 0, 0]
    yq = grid[1][0, :, 0]
    zq = grid[2][0, 0, :]

    # --- Shift and extend coordinates to get cell corners ---
    x_nodes = np.concatenate(([xq[0] - dx / 2.0], xq + dx / 2.0))
    y_nodes = np.concatenate(([yq[0] - dx / 2.0], yq + dx / 2.0))
    z_nodes = np.concatenate(([zq[0] - dx / 2.0], zq + dx / 2.0))

    nxp, nyp, nzp = len(x_nodes), len(y_nodes), len(z_nodes)
    print(f"Creating structured VTK grid: {nx}×{ny}×{nz} cells ({nxp}×{nyp}×{nzp} corner points)")

    # --- Create coordinate mesh for corners ---
    X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")

    # --- Create VTK points (corners) ---
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToDouble()
    for k in range(nzp):
        for j in range(nyp):
            for i in range(nxp):
                vtk_points.InsertNextPoint(float(X[i, j, k]),
                                        float(Y[i, j, k]),
                                        float(Z[i, j, k]))

    # --- Create structured grid (corner-based) ---
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nxp, nyp, nzp)
    grid.SetPoints(vtk_points)

    # ===========================================================
    # --- Helper: attach scalar/vector data to CELLS ------------
    # ===========================================================
    def add_cell_data(grid, name: str, data):
        data = np.asarray(data)
        n_cells = nx * ny * nz
        data_array = vtk.vtkDoubleArray()
        data_array.SetName(name)

        # --- Check shape ---
        if data.ndim == 3 and data.shape == (nx, ny, nz):
            data = data.flatten(order="F")

        if data.ndim == 1:
            if data.size != n_cells:
                raise ValueError(f"'{name}' has {data.size} elements but expected {n_cells}")
            data_array.SetNumberOfComponents(1)
            for val in data:
                data_array.InsertNextValue(float(val))
            grid.GetCellData().AddArray(data_array)

        elif data.ndim == 2 and data.shape[1] == 3:
            if data.shape[0] != n_cells:
                raise ValueError(f"'{name}' has {data.shape[0]} vectors but expected {n_cells}")
            data_array.SetNumberOfComponents(3)
            for gx, gy, gz in data:
                data_array.InsertNextTuple3(float(gx), float(gy), float(gz))
            grid.GetCellData().AddArray(data_array)

        else:
            raise ValueError(f"Unsupported data shape for '{name}': {data.shape}")

    # --- Add any user-supplied scalar/vector fields ---
    for key, arr in fields.items():
        print(f"Adding field '{key}' as cell data.")
        add_cell_data(grid, key, arr)

    # --- Write file ---
    if not fileOutName.endswith(".vtk"):
        fileOutName += ".vtk"

    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(fileOutName)
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()
    writer.Write()

    print(f"Wrote structured grid ({nx}×{ny}×{nz} cells) with {len(fields)} extra fields to {fileOutName}")
    return grid