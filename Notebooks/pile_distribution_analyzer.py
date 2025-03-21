#!/usr/bin/env python
# coding: utf-8

"""
pile_distribution_analyzer.py

This module provides functions to:
1. Analyze pile areas and reactions in relation to a line cutting through the pile cap.
2. Calculate moments above/below or left/right of a section line.

Author: [Your Name]
Date: [Today's Date]
"""

import numpy as np
from shapely.geometry import Polygon, LineString, box
from shapely.ops import split

# ======================================================
# Function 1: Analyze pile areas and reactions by section line
# ======================================================

def analyze_pile_distribution_with_reactions(
    polygon_vertices: np.ndarray,
    line_type: str,
    line_value: tuple,
    pile_layout: np.ndarray,
    pile_size: float,
    pile_reactions: np.ndarray
) -> tuple:
    """
    Analyze pile areas and reactions in relation to a line cutting through the pile cap.

    Parameters:
    - polygon_vertices: numpy array of polygon vertices [[x1, y1], [x2, y2], ...]
    - line_type: 'regular' (y=mx+c) or 'vertical' (x=constant)
    - line_value: tuple (m, c) for regular or (x,) for vertical line
    - pile_layout: numpy array of pile center coordinates [[x1, y1], [x2, y2], ...]
    - pile_size: size of square piles (ft)
    - pile_reactions: array/list of pile reactions corresponding to pile_layout (kip)

    Returns:
    - total_area_above: float, area above/right of the line
    - total_area_below: float, area below/left of the line
    - piles_above: list of pile numbers fully above/right of the line
    - piles_below: list of pile numbers fully below/left of the line
    - intersected_area_above: float, area of intersected piles above/right of line
    - intersected_area_below: float, area of intersected piles below/left of line
    - intersected_piles: list of pile numbers intersected by the line
    - total_reaction_above: float, total reaction above/right of the line
    - total_reaction_below: float, total reaction below/left of the line
    - intersected_pile_geoms: dict with intersected pile pieces and their reactions
    """

    pile_cap_polygon = Polygon(polygon_vertices)

    if not pile_cap_polygon.is_valid:
        raise ValueError("Invalid pile cap polygon.")

    min_x, min_y, max_x, max_y = pile_cap_polygon.bounds
    extend = max(max_x - min_x, max_y - min_y) * 10

    if line_type == 'regular':
        m, c = line_value
        x1, x2 = min_x - extend, max_x + extend
        y1, y2 = m * x1 + c, m * x2 + c
        cutting_line = LineString([(x1, y1), (x2, y2)])

    elif line_type == 'vertical':
        x = line_value[0]
        y1, y2 = min_y - extend, max_y + extend
        cutting_line = LineString([(x, y1), (x, y2)])

    else:
        raise ValueError("line_type must be 'regular' or 'vertical'")

    # Initialize data containers
    piles_above, piles_below, intersected_piles = [], [], []
    area_above, area_below = 0.0, 0.0
    reaction_above, reaction_below = 0.0, 0.0
    intersected_area_above, intersected_area_below = 0.0, 0.0
    intersected_reaction_above, intersected_reaction_below = 0.0, 0.0
    intersected_pile_geoms = {}

    half_size = pile_size / 2
    pile_area = pile_size ** 2

    for idx, (px, py) in enumerate(pile_layout):
        pile_box = box(px - half_size, py - half_size, px + half_size, py + half_size)
        reaction = pile_reactions[idx]

        if pile_box.crosses(cutting_line):
            intersected_piles.append(idx + 1)

            try:
                split_pile = split(pile_box, cutting_line)
                intersected_pile_geoms[idx + 1] = []

                for piece in split_pile.geoms:
                    centroid = piece.centroid
                    cx, cy = centroid.x, centroid.y
                    piece_area = piece.area
                    piece_reaction = reaction * (piece_area / pile_area)

                    intersected_pile_geoms[idx + 1].append({
                        'centroid': (cx, cy),
                        'area': piece_area,
                        'reaction': piece_reaction
                    })

                    if line_type == 'regular':
                        y_on_line = m * cx + c
                        if cy > y_on_line:
                            intersected_area_above += piece_area
                            intersected_reaction_above += piece_reaction
                        else:
                            intersected_area_below += piece_area
                            intersected_reaction_below += piece_reaction

                    elif line_type == 'vertical':
                        if cx > line_value[0]:
                            intersected_area_above += piece_area
                            intersected_reaction_above += piece_reaction
                        else:
                            intersected_area_below += piece_area
                            intersected_reaction_below += piece_reaction

            except Exception as e:
                print(f"Warning: Could not split pile {idx + 1}. Error: {e}")

        else:
            if line_type == 'regular':
                y_on_line = m * px + c
                if py > y_on_line:
                    piles_above.append(idx + 1)
                    area_above += pile_area
                    reaction_above += reaction
                else:
                    piles_below.append(idx + 1)
                    area_below += pile_area
                    reaction_below += reaction

            elif line_type == 'vertical':
                if px > line_value[0]:
                    piles_above.append(idx + 1)
                    area_above += pile_area
                    reaction_above += reaction
                else:
                    piles_below.append(idx + 1)
                    area_below += pile_area
                    reaction_below += reaction

    total_area_above = area_above + intersected_area_above
    total_area_below = area_below + intersected_area_below
    total_reaction_above = reaction_above + intersected_reaction_above
    total_reaction_below = reaction_below + intersected_reaction_below

    return (
        total_area_above,
        total_area_below,
        piles_above,
        piles_below,
        intersected_area_above,
        intersected_area_below,
        intersected_piles,
        float(total_reaction_above),
        float(total_reaction_below),
        intersected_pile_geoms
    )


# ======================================================
# Function 2: Calculate moments above/below or left/right of section line
# ======================================================

def calculate_moments_about_section_line(
    line_type: str,
    line_value: tuple,
    pile_layout: np.ndarray,
    pile_reactions: np.ndarray,
    piles_above: list,
    piles_below: list,
    intersected_pile_geoms: dict
) -> tuple:
    """
    Calculate moments about a section line from pile reactions.

    Returns:
    - moment_above: float, moment of reactions above/right of the section line
    - moment_below: float, moment of reactions below/left of the section line
    """
    moment_above = 0.0
    moment_below = 0.0

    if line_type == 'regular':
        m, c = line_value
        def perp_distance(x0, y0):
            return abs(m * x0 - y0 + c) / np.sqrt(m**2 + 1)

    elif line_type == 'vertical':
        x_line = line_value[0]
        def perp_distance(x0, y0):
            return abs(x0 - x_line)

    else:
        raise ValueError("line_type must be 'regular' or 'vertical'")

    # Fully above piles
    for idx in piles_above:
        px, py = pile_layout[idx - 1]
        reaction = pile_reactions[idx - 1]
        dist = perp_distance(px, py)
        moment_above += reaction * dist

    # Fully below piles
    for idx in piles_below:
        px, py = pile_layout[idx - 1]
        reaction = pile_reactions[idx - 1]
        dist = perp_distance(px, py)
        moment_below += reaction * dist

    # Intersected piles
    for pile_id, pieces in intersected_pile_geoms.items():
        for piece in pieces:
            cx, cy = piece['centroid']
            piece_reaction = piece['reaction']
            dist = perp_distance(cx, cy)

            if line_type == 'regular':
                y_on_line = m * cx + c
                if cy > y_on_line:
                    moment_above += piece_reaction * dist
                else:
                    moment_below += piece_reaction * dist

            elif line_type == 'vertical':
                if cx > line_value[0]:
                    moment_above += piece_reaction * dist
                else:
                    moment_below += piece_reaction * dist

    return float(moment_above), float(moment_below)

__all__ = ["analyze_pile_distribution_with_reactions", "calculate_moments_about_section_line"]


