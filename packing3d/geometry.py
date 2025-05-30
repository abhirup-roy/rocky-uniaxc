# This module contains all the necessary geometrical functions required by
# the packing density calculator

import numpy as np

def simpson(x, y):
    """
    Numerically integrates a function using Simpson's 1/3 rule for evenly
    spaced points.

    If the number of points is odd, the entire range is integrated using
    Simpson's rule. If the number of points is even, Simpson's rule is applied
    to the first `n-1` points, and the trapezoidal rule is applied to the last
    interval.

    Parameters
    ----------
    x : array_like
        1D array of x-coordinates at which the function is sampled. The
        x-coordinates must be evenly spaced within a tolerance of `1e-10`.
    y : array_like
        1D array of y-coordinates (function values) corresponding to the
        x-coordinates.

    Returns
    -------
    float
        The estimated integral of the function.

    Raises
    ------
    ValueError
        If `x` and `y` have different lengths.
        If `x` is not evenly spaced within a tolerance of `1e-10`.

    Example
    --------
    Integrate a cubic function between 1 and 5:
    
    >>> import numpy as np
    >>> def func(x):
    ...     return x**3 - 6*x**2 + 11*x - 6
    >>> x = np.linspace(1, 5, num=1000)
    >>> y = func(x)
    >>> result = simpsons(x, y)
    16.0000  # Approximation of the integral
    """

    # Check if x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # Check for evenly spaced points
    epsilon = 1e-10
    spacings = np.diff(x)
    if not np.allclose(spacings, spacings[0], atol=epsilon):
        raise ValueError("""x-coordinates are not evenly spaced within
                         tolerance of epsilon = 1e-10.""")

    num_points = len(x)

    if num_points % 2 == 0:
        print(f"Warning: Even number of points detected. Applying trapezoidal \
rule for the last interval.")
        # Apply Simpson's rule to the first n-1 points
        h = x[1] - x[0]
        n = num_points - 2  # Exclude the last point
        result = (h / 3) * (
            y[0] + y[n] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2])
                           )
        # Apply trapezoidal rule for the last interval
        result += 0.5 * (x[-1] - x[-2]) * (y[-1] + y[-2])
        return result

    # Standard Simpson's rule
    h = x[1] - x[0]
    n = num_points - 1
    result = (h / 3) * (
        y[0] + y[-1] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2])
                       )
    return result


def double_circle_intersection(r, a, b) -> float:
    """
    Calculate the area of intersection between a circle and two perpendicular
    chords.

    Args:
        r (float): Radius of the circle.
        a (float): Distance of the first chord's center from the circle center.
        b (float): Distance of the second chord's center from the circle center

    Returns:
        float: The area of the intersected region.

    Raises:
        ValueError: If the geometry does not permit an intersection.

    :Example:
    >>> double_circle_intersection(2, 0, 0)
    3.141592653589793
    """
    # Evaluating cases
    if a**2 + b**2 < r**2:
        # a and b are contained within the circle, double intersection exists

        # Calculate the sector area (the area of the circle wedge)
        sector_area = 0.5 * r**2 * \
            (3*np.pi/2 + np.arcsin(a / r) + np.arcsin(b / r))

        # Calculate areas of the two remaining triangles
        triangle_area = 0.5 * a * \
            (np.sqrt(r**2 - a**2) - b) + 0.5 * b * (np.sqrt(r**2 - b**2) - a)

        # Circle area minus sector area and triangles
        intersection_area = np.pi*r**2 - sector_area - triangle_area

        return intersection_area
    else:
        # a and b outside or on circle
        # Special cases, a double intersection does not exist
        if (a >= 0 and b >= 0) or a >= r or b >= r:
            return 0  # No circle
        elif a <= -r and b <= -r:
            return np.pi*r**2  # Full circle
        # Segment with centre-chord distance = b
        elif (a < 0 and 0 < b < r) or (a <= -r and -r < b <= 0):
            return r**2 * (np.arccos(b/r) - (b/r) * np.sqrt(1 - (b/r)**2))
        # Segment with centre-chord distance = a
        elif (b < 0 and 0 < a < r) or (b <= -r and -r < a <= 0):
            return r**2 * (np.arccos(a/r) - (a/r) * np.sqrt(1 - (a/r)**2))
        elif -r < a < 0 and -r < b < 0:
            # Circle missing two minor segments, with centre-chord distances
            # -a and -b
            return np.pi*r**2 - (
                r**2 * (np.arccos(-b/r) + (b/r) * np.sqrt(1 - (b/r)**2))) - (
                r**2 * (np.arccos(-a/r) + (a/r) * np.sqrt(1 - (a/r)**2)))


def single_cap_intersection(R, a) -> float:
    """
    Function that evaluates the (analytical) volume of a spherical cap. The
    sphere has radius R, and the distance from the sphere's centre to the
    boundary is a.

    :Example:
    >>> single_cap_intersection(1, 0.2)
    1.47445415208481
    """
    return np.pi * (R - a)**2 * (2 * R + a) / 3


def double_cap_intersection(R, a, b) -> float:
    """
    Function that evaluates the volume of a double spherical cap intersection.
    The sphere has radius R, and the distances from the centre to the
    boundaries are a and b

    The cross sectional area function can safely just be integrated from -R
    to R. However, this may be wasteful as this can include regions where the
    cross sectional area is zero. The integration limits are set as small as
    possible, such that they just encapsulate the cap volume.

    :Example:
    >>> double_cap_intersection(1, 0.2, 0.3)
    0.3974826065772735
    """

    if a**2 + b**2 <= R**2:
        # a and b are contained within sphere, double cap intersection exists
        if a < 0 and b < 0:
            c_lim_upper = R
        elif a < 0:
            c_lim_upper = np.sqrt(R**2 - b**2)
        elif b < 0:
            c_lim_upper = np.sqrt(R**2 - a**2)
        else:
            c_lim_upper = np.sqrt(R**2 - a**2 - b**2)
    else:
        # Short-circuiting for cases which have analytical solutions
        # (perfect accuracy and reduces computational load)
        if a > 0 and b > 0:
            # No intersection
            return 0
        elif a < 0 and b > 0:
            # Single cap intersection, with centre-chord distance = b
            return np.pi * (R - b)**2 * (3 * R - (R - b)) / 3
        elif b < 0 and a > 0:
            # Single cap intersection, with centre-chord distance = a
            return np.pi * (R - a)**2 * (3 * R - (R - a)) / 3
        else:
            # Sphere missing two caps, with centre-chord distances -a and -b
            return 4/3 * np.pi * R**3 - (
                np.pi * (R + a)**2 * (3 * R - (R + a)) / 3) - (
                np.pi * (R + b)**2 * (3 * R - (R + b)) / 3)

    # The double cap intersection is symmetrical, so c_lim_lower is set to 0
    # and the volume doubled
    c_lim_lower = 0
    return 2*triple_cap_integrator(R, a, b, c_lim_lower, c_lim_upper,
                                   num_simpson_sample_points=3)


def triple_cap_intersection(R, a, b, c) -> float:
    """
    Function that evaluates the volume of a triple cap intersection. The sphere
    has radius R, and the distance from the sphere's centre to the boundaries
    are a, b and c.

    The cross sectional area function must now be carefully integrated to
    include the intersection with the boundary defined by c. The upper
    integration limit is set as low as possible, such that it still entirely
    encapsulates the cap volume. The lower integration limit is set as c,
    unless the cap is symmetrical (c <= -c_lim_upper) or there is no
    intersection (c >= c_lim_upper).

    :Example:
    >>> triple_cap_intersection(1, 0.3, 0.1, 0.2)
    0.16451538109365088
    """

    if a**2 + b**2 <= R**2:
        # a and b are contained within sphere
        # This means a triple cap intersection can exist (depending on c)
        if a < 0 and b < 0:
            c_lim_upper = R
        elif a < 0:
            c_lim_upper = np.sqrt(R**2 - b**2)
        elif b < 0:
            c_lim_upper = np.sqrt(R**2 - a**2)
        else:
            c_lim_upper = np.sqrt(R**2 - a**2 - b**2)
    else:
        # Short-circuiting for cases which have analytical solutions
        # (perfect accuracy and reduces computational load)
        if a > 0 and b > 0:
            # No intersection
            return 0
        elif a < 0 and b > 0:
            if c <= -np.sqrt(R**2 - b**2):
                # Single cap intersection, with centre-chord distance = b
                return np.pi * (R - b)**2 * (3 * R - (R - b)) / 3
            elif c >= np.sqrt(R**2 - b**2):
                # No intersection
                return 0
            else:
                c_lim_upper = np.sqrt(R**2 - b**2)
        elif b < 0 and a > 0:
            if c <= -np.sqrt(R**2 - a**2):
                # Single cap intersection, with centre-chord distance = a
                return np.pi * (R - a)**2 * (3 * R - (R - a)) / 3
            elif c >= np.sqrt(R**2 - a**2):
                # No intersection
                return 0
            else:
                c_lim_upper = np.sqrt(R**2 - a**2)
        elif c > 0 and a < -np.sqrt(R**2 - c**2) and b < -np.sqrt(R**2 - c**2):
            # Single cap intersection, with centre-chord distance = c
            return np.pi * (R - c)**2 * (3 * R - (R - c)) / 3
        elif b < 0 and a < 0:
            if c <= -max(np.sqrt(R**2 - a**2), np.sqrt(R**2 - b**2)):
                # Sphere missing three single caps, with centre-chord distances
                # -a, -b, and -c
                return 4/3 * np.pi * R**3 - (
                    np.pi * (R + a)**2 * (3 * R - (R + a)) / 3) - (
                    np.pi * (R + b)**2 * (3 * R - (R + b)) / 3) - (
                    np.pi * (R + c)**2 * (3 * R - (R + c)) / 3)
            else:
                c_lim_upper = R
        else:
            c_lim_upper = R

    if c >= c_lim_upper:
        # No intersection
        return 0
    elif c <= -c_lim_upper:
        # Symmetrical -> double cap intersection
        c_lim_lower = -c_lim_upper
    else:
        # c intersects the double cap intersection
        # -> integrate between c and c_lim_upper
        c_lim_lower = c

    return triple_cap_integrator(R, a, b, c_lim_lower, c_lim_upper)


def triple_cap_integrator(R, a, b,
                          c_lim_lower, c_lim_upper,
                          num_simpson_sample_points=7) -> float:
    """
    Function for integrating the differential volume of slices of a double
    spherical cap intersection. R is the radius of the sphere, a and b are the
    distances of two planes from the sphere's centre. c_lim_lower and
    c_lim_upper are the integration limits in the third dimension.
    """

    # 6 sample points, can be improved if you're being very precise
    c_values = np.linspace(c_lim_lower, c_lim_upper,
                           num=num_simpson_sample_points)
    radius_values = np.sqrt(R**2 - c_values**2)
    cross_sectional_area_values = np.array(
        [double_circle_intersection(r, a, b) for r in radius_values])

    # Integrate cross sectional slice throughout the volume
    volume = simpson(y=cross_sectional_area_values, x=c_values)
    return volume


def circle_rectangle_intersection(x_min, x_max, y_min, y_max, circle_radius):
    """
    Compute the area of intersection between a circle and a rectangle.

    Args:
        x_min (float): Minimum x-coordinate of the rectangle.
        x_max (float): Maximum x-coordinate of the rectangle.
        y_min (float): Minimum y-coordinate of the rectangle.
        y_max (float): Maximum y-coordinate of the rectangle.
        circle_radius (float): Radius of the circle, assumed to be centered at
                               the origin (0, 0).

    Returns:
        float: The area of the intersection between the circle and the
               rectangle.
    """

    # Compute distances of all corners from the cylinder's centre (x=0, y=0)
    corners = [
        (x_min, y_min), (x_min, y_max),
        (x_max, y_min), (x_max, y_max),
    ]
    distances = [np.sqrt(x**2 + y**2) for x, y in corners]

    num_outside = sum(d >= circle_radius for d in distances)

    if abs(x_min) <= abs(x_max):
        x_closest, x_furthest = (x_min, x_max)
        x_inv = False
    else:
        x_closest, x_furthest = (x_max, x_min)
        x_inv = True

    if abs(y_min) <= abs(y_max):
        y_closest, y_furthest = (y_min, y_max)
        y_inv = False
    else:
        y_closest, y_furthest = (y_max, y_min)
        y_inv = True

    # Match-case based on the number of corners outside the cylinder
    match num_outside:
        case 0:
            # Fully inside the cylinder
            return (x_max - x_min) * (y_max - y_min)

        case 4:
            # Fully outside the cylinder

            critical_boundary = sorted([abs(x_min), abs(x_max),
                                        abs(y_min), abs(y_max)])[2]

            straddling_axis = x_min*x_max < 0 or y_min*y_max < 0

            if critical_boundary < circle_radius and straddling_axis:
                overlap_factor = critical_boundary/circle_radius
                overlap_area = circle_radius**2 * (np.arccos(overlap_factor) -
                                                   (overlap_factor) *
                                                   np.sqrt(1 - (
                                                       overlap_factor)**2))
                return overlap_area
            else:
                return 0

        case 3:
            # Exactly three corners are outside
            a = -x_closest if x_inv else x_closest
            b = -y_closest if y_inv else y_closest
            intersection_area = double_circle_intersection(circle_radius, a, b)

            # Volume is the area of intersection extruded along the z-direction
            return intersection_area

        case 2:
            # Exactly two corners are outside
            a = -x_closest if x_inv else x_closest
            b = -y_closest if y_inv else y_closest
            intersection_area = double_circle_intersection(circle_radius, a, b)

            # Additional correction for partial overlap
            a_extra1 = max(-x_closest if x_inv else x_closest,
                           -y_closest if y_inv else y_closest)
            b_extra1 = min(-y_furthest if y_inv else y_furthest,
                           -x_furthest if x_inv else x_furthest)

            extra_intersection_area = double_circle_intersection(circle_radius,
                                                                 a_extra1,
                                                                 b_extra1)

            furthest_boundary = max(abs(x_min), abs(x_max),
                                    abs(y_min), abs(y_max))

            straddling_axis = x_min*x_max < 0 or y_min*y_max < 0

            if furthest_boundary < circle_radius and straddling_axis:
                overlap_factor = furthest_boundary/circle_radius
                overlap_area = circle_radius**2 * (np.arccos(overlap_factor) -
                                                   (overlap_factor) *
                                                   np.sqrt(1 - (
                                                       overlap_factor)**2))
            else:
                overlap_area = 0

            # Volume is the area of intersection extruded along the z-direction
            return (intersection_area - extra_intersection_area - overlap_area)

        case 1:
            # Exactly one corner is outside
            a = -x_closest if x_inv else x_closest
            b = -y_closest if y_inv else y_closest
            intersection_area = double_circle_intersection(circle_radius, a, b)

            # Additional correction for partial overlap
            a_extra1 = max(-x_closest if x_inv else x_closest,
                           -y_closest if y_inv else y_closest)
            b_extra1 = min(-y_furthest if y_inv else y_furthest,
                           -x_furthest if x_inv else x_furthest)

            a_extra2 = min(-x_closest if x_inv else x_closest,
                           -y_closest if y_inv else y_closest)
            b_extra2 = max(-y_furthest if y_inv else y_furthest,
                           -x_furthest if x_inv else x_furthest)
            # a_extra2 = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
            # b_extra2 = min(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
            extra_intersection_area = (
                double_circle_intersection(circle_radius, a_extra1, b_extra1) +
                double_circle_intersection(circle_radius, a_extra2, b_extra2)
            )

            # Volume is the area of intersection extruded along the z-direction
            return (intersection_area - extra_intersection_area)

        case _:
            raise ValueError(f"""Unexpected number of corners outside the
                             cylinder: {num_outside}""")


def circle_circle_intersection(R_C, R_c, r):
    """
    Compute the intersection area between two circles in 2D.

    Args:
        R_C (float): Radius of the first circle.
        R_c (float): Radius of the second circle.
        r (float): Distance between the centers of the circles.

    Returns:
        float: The area of intersection between the two circles.
    """
    if r == 0:
        return np.pi * R_C**2
    r_i = r/2 + 0.5 * (R_C**2 - R_c**2) / r
    r_ic = r - r_i
    epsilon = 1e-10
    if abs(r_i - R_C) < R_C * epsilon:
        r_i = R_C
    if abs(r_i + R_C) < R_C * epsilon:
        r_i = -R_C
    if abs(r_ic - R_c) < R_c * epsilon:
        r_ic = R_c
    if abs(r_ic + R_c) < R_c * epsilon:
        r_ic = -R_c

    I1 = R_c**2 * (np.arccos(r_ic/R_c) - (r_ic/R_c)*np.sqrt(1-(r_ic/R_c)**2))
    I2 = R_C**2 * (np.arccos(r_i/R_C) - (r_i/R_C)*np.sqrt(1-(r_i/R_C)**2))
    return I1 + I2


def circle_annulus_intersection(r_p, r, r_overlap, min_boundary=False):
    """
    Compute the intersection area between a circle and an annular region.

    Args:
        r_p (float): Radius of the particle's cross-section.
        r (float): Radius of the annulus.
        r_overlap (float): Overlap distance in the radial direction.
        min_boundary (bool): True if the overlap is with the minimum radius
                             boundary; False otherwise.

    Returns:
        float: Intersection area between the circle and the annulus.
    """

    # Handle special cases
    if r_overlap >= r_p:
        # No intersection
        return 0
    elif r_overlap <= -r_p:
        # Full intersection
        return np.pi * r_p**2
    elif min_boundary:
        return np.pi * r_p**2 - circle_circle_intersection(
                                                        r + r_overlap, r_p, r)
    elif not min_boundary:
        return circle_circle_intersection(r - r_overlap, r_p, r)
    else:
        print("Invalid case encountered in circle_annulus_intersection")
        return 0


def sphere_cylinder_integrator(R_p, r, r_overlap, z_lim_lower, z_lim_upper,
                               min_boundary=False,
                               num_simpson_sample_points=7):
    """
    Numerically integrate the volume of a sphere intersecting with a
    cylindrical boundary.

    Args:
        R_p (float): Radius of the sphere.
        r (float): Radius of the cylinder.
        r_overlap (float): Overlap distance in the radial direction.
        z_lim_lower (float): Lower limit of integration in the z-direction.
        z_lim_upper (float): Upper limit of integration in the z-direction.
        min_boundary (bool): Whether the overlap is with the minimum radial
                             boundary.
        num_simpson_sample_points (int): Number of points for Simpson's rule
                                         integration.

    Returns:
        float: The volume of the sphere-cylinder intersection.
    """
    # 6 sample points, can be improved if you're being very precise
    z_values = np.linspace(z_lim_lower, z_lim_upper,
                           num=num_simpson_sample_points)
    r_p_values = np.sqrt(R_p**2 - z_values**2)
    cross_sectional_area_values = np.array(
        [circle_annulus_intersection(r_p, r, r_overlap, min_boundary)
         for r_p in r_p_values])

    # Integrate cross sectional slice throughout the volume
    volume = simpson(y=cross_sectional_area_values, x=z_values)
    return volume


def sphere_cylinder_intersection(R_p, r, r_overlap, min_boundary):
    """
    Compute the volume of intersection between a sphere and a cylindrical
    boundary.

    Args:
        R_p (float): Radius of the sphere.
        r (float): Radius of the cylinder.
        r_overlap (float): Overlap in the radial direction.
        min_boundary (bool): True if the overlap is with the minimum boundary.

    Returns:
        float: Volume of intersection.
    """
    # Need to determine z_lim_lower and z_lim_upper
    # Handle special cases
    if r_overlap >= R_p:
        # No intersection
        return 0
    elif r_overlap <= -R_p:
        # Full sphere
        return (4/3) * np.pi * R_p**3
    elif r_overlap > 0:
        z_lim_upper = np.sqrt(R_p**2 - r_overlap**2)
    elif r_overlap <= 0:
        z_lim_upper = R_p
    else:
        print("Invalid case encountered in circle_annulus_intersection")
        return 0
    # Symmetrical about z=0
    z_lim_lower = 0
    volume = 2*sphere_cylinder_integrator(R_p, r, r_overlap, z_lim_lower,
                                          z_lim_upper, min_boundary,
                                          num_simpson_sample_points=7)
    return volume


def sphere_cylinder_plane_intersection(R_p, r, r_overlap, z_overlap,
                                       min_boundary):
    """
    Calculate the volume of intersection between a sphere, a cylinder, and a
    plane.

    Args:
        R_p (float): Radius of the particle (sphere).
        r (float): Radius of the cylinder.
        r_overlap (float): Overlap in the radial direction.
        z_overlap (float): Overlap in the vertical (z) direction.
        min_boundary (bool): True if the overlap is with the minimum boundary;
                             False otherwise.

    Returns:
        float: Volume of intersection between the sphere, cylinder, and plane.
    """

    # Need to determine z_lim_lower and z_lim_upper
    # Handle special cases
    if r_overlap >= R_p:
        # No intersection
        return 0
    elif r_overlap <= -R_p:
        # Spherical cap with centre-plane distance=z_overlap
        print("Single cap intersection calculated when impossible")
        return single_cap_intersection(R_p, z_overlap)
    elif r_overlap > 0:
        z_lim_upper = np.sqrt(R_p**2 - r_overlap**2)
    elif r_overlap <= 0:
        z_lim_upper = R_p
    else:
        print("Invalid case encountered in circle_annulus_intersection")
        return 0

    if z_overlap > z_lim_upper:
        # Cannot be an intersection
        return 0
    elif z_overlap < -z_lim_upper:
        # Symmetrical
        z_lim_lower = -z_lim_upper
    else:
        # z_overlap intersects the sphere-cylinder intersection
        # integrate between z_overlap and z_lim_upper
        z_lim_lower = z_overlap

    volume = sphere_cylinder_integrator(R_p, r, r_overlap, z_lim_lower,
                                        z_lim_upper, min_boundary,
                                        num_simpson_sample_points=7)
    return volume


def compute_cell_volume(boundaries, system="cartesian", cylinder_radius=None):
    """
    Calculate the volume of a cuboidal or cylindrical cell, adjusting for
    overlap with a cylindrical boundary when specified for cuboids.

    Args:
        boundaries (dict): Dictionary defining the cell's x, y, z boundaries.
        cell_type (int): Type of cell to calculate volume for. 1: Cartesian,
        2: Cylindrical.
        cylinder_radius (float, optional): Radius of the cylinder for overlap
        calculations.

    Returns:
        float: The volume of the cell.
    """

    if system == "cartesian":


        # Cartesian cell type specified
        x_min, x_max, y_min, y_max, z_min, z_max = boundaries

        if cylinder_radius is None:
            # Regular Cartesian volume if no cylinder is defined
            return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

        return circle_rectangle_intersection(x_min, x_max, y_min, y_max,
                                             cylinder_radius) * (z_max - z_min)

    elif system == "cylindrical":
        # Cylindrical cell type specified
        r_min, r_max, theta_min, theta_max, z_min, z_max = boundaries

        delta_theta = angular_difference(theta_min, theta_max)

        if r_min < 0:  # Full cylindrical cell
            return np.pi * r_max**2 * (z_max - z_min)
        elif theta_min == 0 and theta_max == 2*np.pi:
            return np.pi * (r_max**2 - r_min**2) * (z_max - z_min)
        elif delta_theta > 2*np.pi:
            return np.pi * (r_max**2 - r_min**2) * (z_max - z_min)
        else:
            outer = 0.5 * r_max**2 * (delta_theta) * (z_max - z_min)
            inner = 0.5 * r_min**2 * (delta_theta) * (z_max - z_min)
            return outer - inner
    else:
        raise ValueError("Invalid coordinate system specified")


def angular_difference(theta1, theta2):
    """
    Calculate the shortest angular difference between two angles,
    wrapped to the range [0, 2π], treating 0 and 2π as a full-circle difference.

    Args:
        theta1 (float or np.ndarray): First angle(s) in radians.
        theta2 (float or np.ndarray): Second angle(s) in radians.

    Returns:
        float or np.ndarray: The angular difference, wrapped to [0, 2π],
        with special handling for 0 and 2π.
    """
    delta_theta = (theta2 - theta1) % (2 * np.pi)
    
    # Special case: if delta_theta is 0 but theta1 != theta2, treat as 2π
    full_circle_mask = (theta1 != theta2) & (delta_theta == 0)
    if np.any(full_circle_mask):
        delta_theta = np.where(full_circle_mask, 2 * np.pi, delta_theta)
    
    return delta_theta


def theta_within_range(theta_data, theta_min, theta_max,
                       factor, mode="inside"):
    """
    Check whether angles are within or outside a periodic range.

    Args:
        theta_data (np.ndarray): Array of angles in radians.
        theta_min (float): Minimum bound of the range.
        theta_max (float): Maximum bound of the range.
        factor (float): Adjustment factor for angular overlap.
        mode (str): "inside" to check within range, "outside" for outside
                    ange.

    Returns:
        np.ndarray: Boolean array indicating whether each angle is within
                    or outside the range, depending on the mode.
    """
    # Adjust bounds for particle radii
    adjusted_min = (theta_min + factor) % (2 * np.pi)
    adjusted_max = (theta_max - factor) % (2 * np.pi)

    # Create masks for standard and wrapped ranges
    standard_mask = (adjusted_min <= adjusted_max)
    wrapped_mask = ~standard_mask  # Complement of standard_mask

    if mode == "inside":
        # Inside the range
        standard_range = (theta_data >= adjusted_min) & (
            theta_data <= adjusted_max)
        wrapped_range = (theta_data >= adjusted_min) | (
            theta_data <= adjusted_max)
    elif mode == "outside":
        # Outside the range
        standard_range = (theta_data < adjusted_min) | (
            theta_data > adjusted_max)
        wrapped_range = (theta_data < adjusted_min) & (
            theta_data > adjusted_max)
    else:
        raise ValueError("Mode must be 'inside' or 'outside'")

    # Combine results based on range type
    return (standard_mask & standard_range) | (wrapped_mask & wrapped_range)


def calculate_angular_overlap_factor(r_data, radii):
    """
    Calculate the angular factor for cylindrical overlap calculations.

    Args:
        r_data (np.ndarray): Radial distances of particles from the origin.
        radii (np.ndarray): Radii of the particles.

    Returns:
        np.ndarray: Angular factors for each particle, based on the ratio of
                    particle radius to radial distance.
    """
    # Prevent division by zero for particles near r = 0
    safe_r_data = np.maximum(r_data, radii)

    # Calculate angular factor using arcsin
    factor = np.arcsin(np.clip(radii / safe_r_data, -1, 1))

    return factor


def convert_to_cylindrical(x_data, y_data, centre=(0, 0)):
    # Shift the coordinate system by the specified centre
    x_shifted = x_data - centre[0]
    y_shifted = y_data - centre[1]
    
    # Compute the radial distances and angular positions relative to the new centre
    r_data = np.sqrt(x_shifted**2 + y_shifted**2)
    theta_data = np.arctan2(y_shifted, x_shifted)
    theta_data = (theta_data + 2 * np.pi) % (2 * np.pi)
    
    return r_data, theta_data

