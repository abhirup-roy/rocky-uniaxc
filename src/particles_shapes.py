import os


class Shape:
    def __init__(
        self,
        shape_type: str,
        radius: float | dict[float, float],
        vert_ar: float | None = None,
        horiz_ar: float | None = None,
        smoothness: float | None = None,
        n_corners: int | None = None,
        sq_degree: float | None = None,
        size_type: str = "equivalent_diameter",
    ) -> None:
        """
        Base shape class for Rocky DEM particles.
        Parameters:
            radius (float | dict[float, float] | None): Radius of the particle or a dictionary with PSD.
            vert_ar (float | None): Vertical aspect ratio.
            horiz_ar (float | None): Horizontal aspect ratio.
            smoothness (float | None): Smoothness of the particle surface.
            n_corners (int | None): Number of corners for polygonal shapes.
            sq_degree (float | None): Superquadric degree for square-like shapes.
        """
        self.radius = radius
        self.vert_ar = vert_ar
        self.horiz_ar = horiz_ar
        self.smoothness = smoothness
        self.n_corners = n_corners
        self.sq_degree = sq_degree

        if shape_type not in [
            "sphere",
            "polyhedron",
            "sphero_cylinder",
            "sphero_polygon",
            "sphero_polyhedron",
            "briquete",
            "faceted_cylinder",
            "assembly",
            "straight_fiber",
            "custom_fiber",
            "custom_shell",
            "custom_polyhedron",
        ]:
            raise ValueError(f"Invalid shape type: {shape_type}. Check for typos.")
        else:
            self.shape_type = shape_type

        if size_type not in ["sieve", "equivalent_diameter", "original_size_scale"]:
            raise ValueError(f"Invalid size type: {size_type}. Check for typos.")
        else:
            self.size_type = size_type

    def particle2rocky(self, particle, material, rolling_friction: float = 0.0) -> None:
        """Set the particle size distribution for the shape.
        Parameters:
            radius (float | dict[float, float]): Radius of the particle or a dictionary with
                particle sizes and their proportions.
            material (str): Material of the particle.
            rolling_friction (str): Rolling friction type. Default is 'none'."""

        if isinstance(self.radius, float) or isinstance(self.radius, int):
            size_distr_lst = particle.GetSizeDistributionList()
            size_distr_lst.Clear()

            psd = size_distr_lst.New()
            psd.SetSize(float(self.radius), "m")
            psd.SetCumulativePercentage(100)

        # If it is a dictionary, create a particle size distribution
        # with multiple sizes
        elif isinstance(self.radius, dict):
            # Check if the values are valid
            if sum(self.radius.values()) == 1:
                radius = {k: v * 100 for k, v in self.radius.items()}
            elif sum(self.radius.values()) == 100:
                pass
            else:
                raise ValueError(
                    "The size dict values must sum to 1 or 100."
                    "Please provide a valid dictionary."
                )
            # Create a new particle and size distribution list
            size_distr_lst = particle.GetSizeDistributionList()
            size_distr_lst.Clear()

            # Create a new PSD for each particle size
            init_pct = 100
            sorted_dict = dict(sorted(self.radius.items(), reverse=True))
            for i, (size, proportion) in enumerate(sorted_dict):
                # Create a new PSD for each particle size
                # and set the size and cumulative percentage
                # Use exec to create a variable with the name of the PSD
                # This is not recommended, but it works for this case ;)
                exec(f"psd{i} = size_distr_lst.New()")
                exec(f'psd{i}.SetSize(size, "m")')
                exec(f"psd{i}.SetCumulativePercentage(init_pct)")
                init_pct -= proportion

        # Set particle material
        particle.SetMaterial(material)
        if rolling_friction != "none":
            particle.SetRollingResistance(rolling_friction)
        else:
            raise TypeError("Radius must be a float, int or a dictionary.")

        particle.SetShape(self.shape_type)
        particle.SetSizeType(self.size_type)

        if self.vert_ar:
            particle.SetVerticalAspectRatio(self.vert_ar)
        if self.horiz_ar:
            particle.SetHorizontalAspectRatio(self.horiz_ar)
        if self.smoothness:
            particle.SetSmoothness(self.smoothness)
        if self.n_corners:
            particle.SetNumberOfCorners(self.n_corners)
        if self.sq_degree:
            particle.SetSuperquadricDegree(self.sq_degree)


class Sphere(Shape):
    def __init__(self, radius: float | dict[float, float]) -> None:
        super().__init__(shape_type="sphere", radius=radius)
        self.radius = radius


class Polyhedron(Shape):
    def __init__(
        self,
        radius: float | dict[float, float],
        vert_ar: float,
        horiz_ar: float,
        n_corners: int,
        superquadric_degree: float,
    ) -> None:
        self.radius = radius
        self.vert_ar = vert_ar
        self.horiz_ar = horiz_ar
        self.n_corners = n_corners
        self.sq_degree = superquadric_degree
        super().__init__(
            shape_type="polyhedron",
            radius=radius,
            vert_ar=vert_ar,
            horiz_ar=horiz_ar,
            n_corners=n_corners,
            sq_degree=superquadric_degree,
        )


class SpheroCylinder(Shape):
    """Sphero-cylinder shape for Rocky DEM particles."""

    def __init__(self, radius: float | dict[float, float], vert_ar: float) -> None:
        super().__init__(shape_type="sphero_cylinder", radius=radius, vert_ar=vert_ar)


class CustomPolyhedron(Shape):
    def __init__(self, stl_path: str, radius: float | dict[float, float]) -> None:
        stl_path = os.path.abspath(stl_path)
        if not os.path.exists(stl_path):
            raise FileNotFoundError(f"STL file not found: {stl_path}")
        else:
            self.stl_path = stl_path
        self.radius = radius

    def particle2rocky(self, particle, material, rolling_friction: float = 0.0):
        particle.ImportFromSTL(stl_filename=self.stl_path, scale=1.0)
        if material:
            particle.SetMaterial(material)
        else:
            raise ValueError("Material must be provided for custom polyhedron shapes.")

        particle.SetMaterial(material)
        if rolling_friction != "none":
            particle.SetRollingResistance(rolling_friction)
        else:
            raise TypeError("Radius must be a float, int or a dictionary.")

        particle.SetShape(self.shape_type)
        particle.SetSizeType(self.size_type)
