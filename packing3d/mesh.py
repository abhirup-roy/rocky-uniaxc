import numpy as np

class Mesh:
    def __init__(self, system, divisions, **params):
        """
        Initialize a mesh object.

        Args:
            divisions (dict): Number of divisions for each dimension.
            system (str): Coordinate system, either 'cartesian' or 'cylindrical'.
            params: Additional parameters:
                - For 'cartesian': x_min, x_max, y_min, y_max, z_min, z_max, cylindrical_mesh_shape.
                - For 'cylindrical': cylinder_radius, cylinder_base_level, cylinder_height, constant_volume.
        """
        self.system = system.lower()
        self.divisions = divisions
        self.params = params

        # Validate input arguments
        self._validate_inputs()

        # Precompute total number of cells
        self.total_cells = self._compute_total_cells()

        # Preallocate storage for cell indices and boundaries
        self.cell_indices = np.empty((self.total_cells, 3), dtype=int)
        self.cell_boundaries = np.empty((self.total_cells, 6))  # 6 boundaries per cell

        # Generate the mesh
        self.generate()

    def _validate_inputs(self):
        """
        Validate input parameters based on the specified system.
        """
        if self.system not in ["cartesian", "cylindrical"]:
            raise ValueError("Invalid system. Choose 'cartesian' or 'cylindrical'.")

        if self.system == "cartesian":
            required_params = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
            if not required_params.issubset(self.params):
                raise ValueError(f"Cartesian system requires parameters: {required_params}")
            if not {"x", "y", "z"}.issubset(self.divisions):
                raise ValueError("Cartesian system requires divisions for 'x', 'y', and 'z'.")
            self._validate_cartesian_params()

        elif self.system == "cylindrical":
            required_params = {"cylinder_radius", "cylinder_base_level", "cylinder_height"}
            if not required_params.issubset(self.params):
                print(self.params)
                raise ValueError(f"Cylindrical system requires parameters: {required_params}")
            if not {"r", "theta", "z"}.issubset(self.divisions):
                raise ValueError("Cylindrical system requires divisions for 'r', 'theta', and 'z'.")
            self._validate_cylindrical_params()

    def _validate_cartesian_params(self):
        """
        Additional validation for Cartesian mesh parameters.
        """
        if self.params["x_min"] >= self.params["x_max"]:
            raise ValueError("'x_min' must be less than 'x_max'.")
        if self.params["y_min"] >= self.params["y_max"]:
            raise ValueError("'y_min' must be less than 'y_max'.")
        if self.params["z_min"] >= self.params["z_max"]:
            raise ValueError("'z_min' must be less than 'z_max'.")
        if any(d <= 0 for d in self.divisions.values()):
            raise ValueError("Divisions for 'x', 'y', and 'z' must be positive integers.")

    def _validate_cylindrical_params(self):
        """
        Additional validation for Cylindrical mesh parameters.
        """
        if self.params["cylinder_radius"] <= 0:
            raise ValueError("'cylinder_radius' must be a positive number.")
        if self.params["cylinder_height"] <= 0:
            raise ValueError("'cylinder_height' must be a positive number.")
        if any(d <= 0 for d in self.divisions.values()):
            raise ValueError("Divisions for 'r', 'theta', and 'z' must be positive integers.")

    def _compute_total_cells(self):
        """
        Compute the total number of cells based on the system and divisions.
        """
        if self.system == "cartesian":
            return self.divisions["x"] * self.divisions["y"] * self.divisions["z"]
        elif self.system == "cylindrical":
            r_div, z_div = self.divisions["r"], self.divisions["z"]
            theta_div = self.divisions.get("theta", 3)
            constant_volume = self.params.get("constant_volume", True)
            if constant_volume:
                return (r_div**2) * z_div  # Include inner cells
            else:
                num_slice_cells = 1 + np.sum(
                    [int(np.round(theta_div*(2*n - 1)/3))
                     for n in range(2, r_div + 1)]
                                            )

                return (num_slice_cells) * z_div

    def generate(self):
        """
        Generate the mesh based on the system and divisions.
        """
        if self.system == "cartesian":
            self._generate_cartesian_mesh()
        elif self.system == "cylindrical":
            self._generate_simple_cylindrical_mesh()

    def _generate_cartesian_mesh(self):
        """
        Generate a Cartesian mesh using numpy arrays.
        """
        x_div, y_div, z_div = self.divisions["x"], self.divisions["y"], self.divisions["z"]
        cylindrical_mesh_shape = self.params.get("cylindrical_mesh_shape", False)

        if cylindrical_mesh_shape:
            radius = self.params["cylinder_radius"]
            base_level = self.params["cylinder_base_level"]
            height = self.params["cylinder_height"]
            x_bounds = np.linspace(-radius, radius, x_div + 1)
            y_bounds = np.linspace(-radius, radius, y_div + 1)
            z_bounds = np.linspace(base_level, base_level + height, z_div + 1)
        else:
            x_bounds = np.linspace(self.params["x_min"], self.params["x_max"], x_div + 1)
            y_bounds = np.linspace(self.params["y_min"], self.params["y_max"], y_div + 1)
            z_bounds = np.linspace(self.params["z_min"], self.params["z_max"], z_div + 1)

        index = 0
        for i in range(x_div):
            for j in range(y_div):
                for k in range(z_div):
                    x_min, x_max = x_bounds[i], x_bounds[i + 1]
                    y_min, y_max = y_bounds[j], y_bounds[j + 1]
                    z_min, z_max = z_bounds[k], z_bounds[k + 1]

                    if cylindrical_mesh_shape:
                        # Check if the cell is inside the cylinder
                        corners = [
                            (x_min, y_min), (x_min, y_max),
                            (x_max, y_min), (x_max, y_max),
                        ]
                        distances = [np.sqrt(x**2 + y**2) for x, y in corners]
                        if not all(d >= radius for d in distances):
                            self.cell_indices[index] = (i, j, k)
                            self.cell_boundaries[index] = [
                                x_min, x_max,
                                y_min, y_max,
                                z_min, z_max,
                            ]
                            index += 1
                    else:
                        self.cell_indices[index] = (i, j, k)
                        self.cell_boundaries[index] = [
                            x_min, x_max,
                            y_min, y_max,
                            z_min, z_max,
                        ]
                        index += 1

        # Trim unused elements if cylindrical mesh shape reduces cells
        self.cell_indices = self.cell_indices[:index]
        self.cell_boundaries = self.cell_boundaries[:index]

    def _generate_simple_cylindrical_mesh(self):
        """
        Generate a simple uniform cylindrical mesh with constant cell volume.
        The region is divided into:
        - z_divisions (vertical)
        - theta_divisions (angular, uniformly)
        - r_divisions (radial) computed such that each cell has constant volume.
        
        The radial boundaries are computed as:
        r_bounds[i] = cylinder_radius * sqrt(i / r_divisions) for i = 0,...,r_divisions.
        
        Each cell's boundaries are stored in the format:
        [r_min, r_max, theta_min, theta_max, z_min, z_max].
        """
        # Extract parameters
        cylinder_radius = self.params["cylinder_radius"]
        base_level = self.params["cylinder_base_level"]
        height = self.params["cylinder_height"]
        r_div = self.divisions["r"]
        theta_div = self.divisions["theta"]
        z_div = self.divisions["z"]

        # Compute radial boundaries so that each cell has constant volume.
        # Volume ∝ (r_max^2 - r_min^2); using square-root spacing yields constant differences.
        r_bounds = np.array([cylinder_radius * np.sqrt(i / r_div) for i in range(r_div + 1)])
        # Uniform angular boundaries
        theta_bounds = np.linspace(0, 2 * np.pi, theta_div + 1)
        # Uniform vertical boundaries
        z_bounds = np.linspace(base_level, base_level + height, z_div + 1)

        # Total number of cells
        total_cells = r_div * theta_div * z_div
        self.total_cells = total_cells

        # Preallocate storage for cell indices and boundaries
        self.cell_indices = np.empty((total_cells, 3), dtype=int)
        self.cell_boundaries = np.empty((total_cells, 6))

        index = 0
        # Loop over vertical layers, then radial and angular divisions
        for k in range(z_div):
            for i in range(r_div):
                for j in range(theta_div):
                    r_min = r_bounds[i]
                    r_max = r_bounds[i + 1]
                    theta_min = theta_bounds[j]
                    theta_max = theta_bounds[j + 1]
                    z_min = z_bounds[k]
                    z_max = z_bounds[k + 1]

                    # Store the cell's indices and boundaries
                    self.cell_indices[index] = (i, j, k)
                    self.cell_boundaries[index] = [r_min, r_max, theta_min, theta_max, z_min, z_max]
                    index += 1



    def _generate_cylindrical_mesh(self):
        """
        Generate a cylindrical mesh with division indices, including a special
        inner cell at each z-layer, using preallocated numpy arrays.
        """
        radius = self.params["cylinder_radius"]
        base_level = self.params["cylinder_base_level"]
        height = self.params["cylinder_height"]
        r_div = self.divisions["r"]
        z_div = self.divisions["z"]
        theta_div = self.divisions.get("theta", 3)
        constant_volume = self.params.get("constant_volume", True)

        if constant_volume:
            if theta_div != 3:
                raise ValueError("Theta divisions must equal 3 for constant volume cells.")
            radius_inner = radius / r_div
        else:
            radius_inner = radius / r_div

        # Precompute radial and vertical boundaries
        radial_bounds = np.linspace(radius_inner, radius, r_div)
        z_bounds = np.linspace(base_level, base_level + height, z_div + 1)

        # Target volume for cells in the first radial layer
        radius_factor = radial_bounds[1]**2 - radius_inner**2
        cell_height = height / z_div
        target_volume = np.pi * radius_factor * cell_height / theta_div

        # Populate mesh
        index = 0
        for k in range(len(z_bounds) - 1):
            z_min, z_max = z_bounds[k], z_bounds[k + 1]

            # Add the special inner cell for this z-layer
            self.cell_indices[index] = (0, 0, k)
            self.cell_boundaries[index] = [
                -radius_inner, radius_inner,  # Radial boundaries
                0, 2 * np.pi,  # Angular boundaries
                z_min, z_max,  # Vertical boundaries
            ]
            index += 1

            # Loop through radial layers outside the special cell
            for i in range(len(radial_bounds) - 1):
                r_min, r_max = radial_bounds[i], radial_bounds[i + 1]
                layer_volume = np.pi * (r_max**2 - r_min**2) * (z_max - z_min)

                # Calculate angular divisions for this layer to maintain constant volume
                if theta_div == 1:
                    layer_theta_divisions = 1
                else:
                    layer_theta_divisions = max(1, int(round(layer_volume / target_volume)))

                # Angular boundaries for this layer
                theta_bounds = np.linspace(0, 2 * np.pi, layer_theta_divisions + 1)

                for j in range(len(theta_bounds) - 1):
                    theta_min, theta_max = theta_bounds[j], theta_bounds[j + 1]

                    # Append regular cell boundaries and indices
                    self.cell_indices[index] = (i + 1, j, k)
                    self.cell_boundaries[index] = [
                        r_min, r_max,
                        theta_min, theta_max,
                        z_min, z_max,
                    ]
                    index += 1

        # Trim unused elements if necessary
        self.cell_indices = self.cell_indices[:index]
        self.cell_boundaries = self.cell_boundaries[:index]
    
    def get_mesh_boundaries(self):
        """
        Return the boundaries of all cells in the mesh as a numpy array.

        Returns:
            np.ndarray: An array of shape (num_cells, 6), where each row
                        contains the boundaries of a cell in the format:
                        [r_min, r_max, theta_min, theta_max, z_min, z_max].
        """
        return self.cell_boundaries

    def get_cell_boundaries(self, index):
        """Retrieve the boundaries for a specific cell."""
        if index < 0 or index >= self.total_cells:
            raise IndexError("Cell index out of bounds.")
        return self.cell_boundaries[index]
