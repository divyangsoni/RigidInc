from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split
from matplotlib.patches import Polygon as MplPolygon

def split_polygon_by_line(polygon_vertices, line_type, line_value, column_centroid=None):
    """
    Splits a polygon into two areas based on a line.

    Args:
    - polygon_vertices (np.ndarray): Vertices of the polygon [[x, y], ...].
    - line_type (str): 'regular' or 'vertical'.
    - line_value (tuple): (m, c) if regular or (x,) if vertical.
    - column_centroid (tuple): Not used in this version, but kept for future extension.

    Returns:
    - Tuple of two areas: (area_above_or_right, area_below_or_left)
    """
    # Create the main polygon
    polygon = Polygon(polygon_vertices)
    
    if not polygon.is_valid:
        raise ValueError("Input polygon is invalid! Check vertices order and if it's closed.")

    # Get bounds for extending the line beyond the polygon
    min_x, min_y, max_x, max_y = polygon.bounds
    extend = max(max_x - min_x, max_y - min_y) * 10  # Extend line far beyond bounds

    # Create the line geometry
    if line_type == 'regular':
        m, c = line_value
        # Pick x values far beyond polygon
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
        raise ValueError("line_type must be 'regular' or 'vertical'")

    # Split the polygon
    try:
        split_polygons = split(polygon, line)
    except Exception as e:
        print("Polygon could not be split:", e)
        return 0.0, 0.0

    # If not split, return the area as one side and zero on the other
    if len(split_polygons.geoms) < 2:
        print("Polygon was not split by the line.")
        return polygon.area, 0.0

    # Decide which side is "above/right" vs "below/left"
    areas = []
    for poly in split_polygons.geoms:
        centroid = poly.centroid
        x, y = centroid.x, centroid.y

        if line_type == 'regular':
            y_on_line = m * x + c
            if y > y_on_line:
                areas.append(('above', poly.area))
            else:
                areas.append(('below', poly.area))

        elif line_type == 'vertical':
            if x > line_value[0]:
                areas.append(('right', poly.area))
            else:
                areas.append(('left', poly.area))

    # Sum areas
    if line_type == 'regular':
        area_above = sum(a for s, a in areas if s == 'above')
        area_below = sum(a for s, a in areas if s == 'below')
        return area_above, area_below

    elif line_type == 'vertical':
        area_right = sum(a for s, a in areas if s == 'right')
        area_left = sum(a for s, a in areas if s == 'left')
        return area_right, area_left