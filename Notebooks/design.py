# design.py

def calculate_area_of_steel(moment, pile_cap_shear_depth):
    """
    Calculate the area of steel based on the given moment and pile cap shear depth.

    Parameters:
        moment (float): The moment value.
        pile_cap_shear_depth (float): The pile cap shear depth.

    Returns:
        float: The area of steel computed as moment / (4 * pile_cap_shear_depth).
    """
    return moment / (4 * pile_cap_shear_depth * 12)


def calculate_punching_shear_capacity(
    column_size: tuple[float, float],
    concrete_strength_psi: float,
    pile_cap_shear_depth: float,
    shear_perimeter: float,
    lambda_factor: float,
    column_location: str
) -> float:
    """
    Compute the factored punching‑shear capacity (kips).

    Parameters:
        column_size: (width, height) of column (ft)
        concrete_strength_psi: concrete compressive strength (psi)
        pile_cap_shear_depth: effective shear depth (ft)
        shear_perimeter: shear perimeter length (ft)
        lambda_factor: modification factor for lightweight concrete
        column_location: "interior", "edge", or "corner"

    Returns:
        phi * Vc (kips)
    """
    width, height = column_size
    beta = min(width, height) / max(width, height)

    loc = column_location.lower()
    if loc == "interior":
        alpha = 40
    elif loc == "edge":
        alpha = 30
    else:
        alpha = 20

    phi = 0.75
    sqrt_fc = concrete_strength_psi**0.5

    term1 = (2 + 4 / beta) * lambda_factor * sqrt_fc / 1000 * shear_perimeter * pile_cap_shear_depth
    term2 = (2 + alpha * pile_cap_shear_depth / shear_perimeter) * lambda_factor * sqrt_fc / 1000 * shear_perimeter * pile_cap_shear_depth
    term3 = 4 * lambda_factor * sqrt_fc / 1000 * shear_perimeter * pile_cap_shear_depth

    Vnc = min(term1, term2, term3)
    return phi * Vnc
