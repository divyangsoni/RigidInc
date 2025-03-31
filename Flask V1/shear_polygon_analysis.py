#!/usr/bin/env python
# coding: utf-8

"""
shear_polygon_analysis.py

This module provides a function to analyze shear reactions relative to a shear polygon
around a column within a pile cap structure.
"""

import numpy as np
from shapely.geometry import Polygon, box

def analyze_shear_polygon_reactions(
    polygon_vertices,
    pile_layout,
    pile_size,
    column_centroid,
    column_size,
    shear_depth,
    pile_reactions
):
    """
    Analyze reactions outside the shear polygon around a column, and categorize piles.

    Parameters:
    - polygon_vertices: np.ndarray of pile cap polygon [[x1, y1], [x2, y2], ...]
    - pile_layout: np.ndarray of pile center coordinates [[x1, y1], ...]
    - pile_size: float, size of each square pile (ft)
    - column_centroid: tuple (x, y) for column center
    - column_size: tuple (width, height) of column (ft)
    - shear_depth: float, shear depth to expand the critical polygon (ft)
    - pile_reactions: np.ndarray of reactions at each pile (kip)

    Returns:
    - total_reaction_outside: float, sum of reactions outside the shear polygon (kip)
    - shear_polygon_coords: list of coordinates of the shear polygon (list)
    - shear_polygon_perimeter: float, perimeter of the shear polygon (ft)
    - inside_piles: list of pile indices fully inside the shear polygon
    - outside_piles: list of pile indices fully outside the shear polygon
    - intersecting_piles: list of pile indices partially inside/outside the shear polygon
    """

    # Create pile cap polygon
    pile_cap_polygon = Polygon(polygon_vertices)

    if not pile_cap_polygon.is_valid:
        raise ValueError("Invalid pile cap polygon provided.")

    # Shear polygon dimensions (centered on column centroid)
    col_x, col_y = column_centroid
    col_width, col_height = column_size

    # Expand by shear depth / 2 on all sides
    shear_width = col_width + shear_depth
    shear_height = col_height + shear_depth

    # Coordinates of shear polygon BEFORE trimming
    shear_min_x = col_x - shear_width / 2
    shear_max_x = col_x + shear_width / 2
    shear_min_y = col_y - shear_height / 2
    shear_max_y = col_y + shear_height / 2

    # Create shear polygon box
    shear_polygon = box(shear_min_x, shear_min_y, shear_max_x, shear_max_y)

    # Trim to pile cap polygon (if it extends outside)
    shear_polygon_trimmed = shear_polygon.intersection(pile_cap_polygon)

    # Compute perimeter length of shear polygon
    shear_polygon_perimeter = shear_polygon_trimmed.length

    # Prepare totals
    total_reaction_outside = 0.0

    # Prepare lists for pile classifications
    inside_piles = []
    outside_piles = []
    intersecting_piles = []

    # Loop through each pile and check its relation to shear polygon
    half_size = pile_size / 2
    pile_area = pile_size ** 2

    for idx, (px, py) in enumerate(pile_layout):
        reaction = pile_reactions[idx]

        # Create pile box
        pile_box = box(px - half_size, py - half_size, px + half_size, py + half_size)

        # Fully inside shear polygon => no contribution to reaction outside
        if shear_polygon_trimmed.contains(pile_box):
            inside_piles.append(idx + 1)

        # Fully outside shear polygon => full reaction counts
        elif not shear_polygon_trimmed.intersects(pile_box):
            outside_piles.append(idx + 1)
            total_reaction_outside += reaction

        # Intersecting pile => partial reaction contribution
        else:
            intersecting_piles.append(idx + 1)
            
            # Get portion of pile box outside the shear polygon
            outside_piece = pile_box.difference(shear_polygon_trimmed)
            outside_area = outside_piece.area

            # Calculate reaction proportional to the area outside
            reaction_contribution = reaction * (outside_area / pile_area)
            total_reaction_outside += reaction_contribution

    # Get coordinates of trimmed shear polygon for output
    shear_polygon_coords = list(shear_polygon_trimmed.exterior.coords)

    # Return all results
    return (
    float(total_reaction_outside),  # Convert to float!
    shear_polygon_coords,
    shear_polygon_perimeter,
    inside_piles,
    outside_piles,
    intersecting_piles
)
