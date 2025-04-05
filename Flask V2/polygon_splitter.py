#!/usr/bin/env python
# coding: utf-8

"""
polygon_splitter.py

This module provides a function to split a polygon into two areas based on a line,
and calculate the first area moment (area * centroid distance to line).
Useful for structural engineering applications like pile cap one-way shear and moment analysis.
"""

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import split


def split_polygon_by_line(polygon_vertices, line_type, line_value, column_centroid=None):
    from shapely.geometry import Polygon, LineString
    from shapely.ops import split
    import numpy as np

    polygon = Polygon(polygon_vertices)

    if not polygon.is_valid:
        raise ValueError("Invalid polygon.")

    min_x, min_y, max_x, max_y = polygon.bounds
    extend = max(max_x - min_x, max_y - min_y) * 10

    if line_type == 'regular':
        m, c = line_value
        x1 = min_x - extend
        x2 = max_x + extend
        y1 = m * x1 + c
        y2 = m * x2 + c
        line = LineString([(x1, y1), (x2, y2)])

    elif line_type == 'vertical':
        x = line_value[0]
        y1 = min_y - extend
        y2 = max_y + extend
        line = LineString([(x, y1), (x, y2)])

    else:
        raise ValueError("Invalid line type")

    try:
        split_polygons = split(polygon, line)
    except Exception as e:
        print("Could not split polygon:", e)
        return 0.0, 0.0, 0.0, 0.0

    if len(split_polygons.geoms) < 2:
        print("Polygon not split!")
        area = polygon.area
        return area, 0.0, 0.0, 0.0

    areas = []
    moments = []

    for poly in split_polygons.geoms:
        centroid = poly.centroid
        x, y = centroid.x, centroid.y
        poly_area = poly.area

        if line_type == 'regular':
            y_on_line = m * x + c
            distance = abs(y - y_on_line)
            if y > y_on_line:
                areas.append(('above', poly_area))
                moments.append(('above', poly_area * distance))
            else:
                areas.append(('below', poly_area))
                moments.append(('below', poly_area * distance))

        elif line_type == 'vertical':
            x_on_line = line_value[0]
            distance = abs(x - x_on_line)
            if x > x_on_line:
                areas.append(('right', poly_area))
                moments.append(('right', poly_area * distance))
            else:
                areas.append(('left', poly_area))
                moments.append(('left', poly_area * distance))

    if line_type == 'regular':
        area_above = sum(a for s, a in areas if s == 'above')
        area_below = sum(a for s, a in areas if s == 'below')
        moment_above = sum(m for s, m in moments if s == 'above')
        moment_below = sum(m for s, m in moments if s == 'below')
        return area_above, area_below, moment_above, moment_below

    elif line_type == 'vertical':
        area_right = sum(a for s, a in areas if s == 'right')
        area_left = sum(a for s, a in areas if s == 'left')
        moment_right = sum(m for s, m in moments if s == 'right')
        moment_left = sum(m for s, m in moments if s == 'left')
        return area_right, area_left, moment_right, moment_left
    
def polygon_intersection(polygon_vertices, line_type, line_value):
    """
    Calculates the length of the line segment formed by the intersection of the
    extended splitting line (defined by line_type and line_value) with the polygon.

    Parameters:
    - polygon_vertices: numpy array or list of [x, y] vertices defining the polygon.
    - line_type: 'regular' (interpreted as y = m*x + c) or 'vertical' (x = constant)
    - line_value: tuple (m, c) for 'regular' or (x,) for 'vertical'

    Returns:
    - length: float, the total length of the intersection line segment within the polygon.
    """
    # Create the polygon
    polygon = Polygon(polygon_vertices)
    if not polygon.is_valid:
        raise ValueError("Invalid polygon.")

    min_x, min_y, max_x, max_y = polygon.bounds
    extend = max(max_x - min_x, max_y - min_y) * 10

    # Define the extended line based on line_type.
    if line_type == 'regular':
        m, c = line_value
        x1 = min_x - extend
        x2 = max_x + extend
        y1 = m * x1 + c
        y2 = m * x2 + c
        line = LineString([(x1, y1), (x2, y2)])
    elif line_type == 'vertical':
        x = line_value[0]
        y1 = min_y - extend
        y2 = max_y + extend
        line = LineString([(x, y1), (x, y2)])
    else:
        raise ValueError("Invalid line type")

    # Compute the intersection of the line with the polygon.
    intersection = polygon.intersection(line)
    
    # Depending on the geometry type, calculate the total length.
    total_length = 0.0
    if intersection.is_empty:
        total_length = 0.0
    elif intersection.geom_type == 'LineString':
        total_length = intersection.length
    elif intersection.geom_type == 'MultiLineString':
        for seg in intersection.geoms:
            total_length += seg.length
    else:
        # If it's a GeometryCollection or other type, try to extract LineStrings.
        try:
            for geom in intersection:
                if geom.geom_type == 'LineString':
                    total_length += geom.length
        except TypeError:
            pass

    return total_length
