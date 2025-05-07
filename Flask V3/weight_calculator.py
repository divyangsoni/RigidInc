#!/usr/bin/env python
# coding: utf-8

"""
weight_calculator.py

This module provides a function to calculate the weight of concrete and soil over a specified area
of a pile cap section.
"""


def calculate_concrete_soil_weight(area, pile_cap_thickness, soil_depth_above, concrete_density, soil_density):
    """
    Calculates the weight of concrete and soil over a given area of pile cap.

    Parameters:
    - area (float): Area of the pile cap section (sq.ft.)
    - pile_cap_thickness (float): Thickness of the pile cap (ft)
    - soil_depth_above (float): Depth of soil above the pile cap (ft)
    - concrete_density (float): Density of concrete (pcf)
    - soil_density (float): Density of soil (pcf)

    Returns:
    - concrete_weight (float): Weight of concrete over the area (kips)
    - soil_weight (float): Weight of soil over the area (kips)
    """

    # Input validation
    if area < 0:
        raise ValueError("Area cannot be negative.")
    if pile_cap_thickness < 0 or soil_depth_above < 0:
        raise ValueError("Thickness and soil depth cannot be negative.")
    if concrete_density < 0 or soil_density < 0:
        raise ValueError("Material densities must be positive.")

    # Calculate volumes (in cubic feet)
    concrete_volume = area * pile_cap_thickness
    soil_volume = area * soil_depth_above

    # Calculate weights (convert pcf to kips by dividing by 1000)
    concrete_weight = (concrete_volume * concrete_density) / 1000  # kips
    soil_weight = (soil_volume * soil_density) / 1000  # kips

    return concrete_weight, soil_weight
