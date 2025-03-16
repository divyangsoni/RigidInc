import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString, box
from shapely.ops import split

# Input Parameters
column_size = (5.0, 10.0)  # Column dimensions (x, y) in ft
column_eccentricity = (0.0, 0)  # Eccentricity of column from center (x, y) in ft
column_centroid = (8, 11)  # Center of the column in ft
pile_cap_thickness = 5.5  # Pile cap thickness in ft
pile_embedment = 1.0  # Depth of pile embedment in pile cap (ft)
soil_depth_above = 2.0  # Soil depth above pile cap (ft)
soil_density = 120  # Soil density in pcf
concrete_density = 150  # Concrete density in pcf
pile_cap_shear_depth = 3.72  # Effective shear depth (ft)

# Pile Properties
pile_shape = "square"
pile_size = 2.0
max_pile_compression = 120  # kips
max_pile_tension = 150      # kips

# Column Loads (kips and kip-ft)
Fx, Fy, Fz = 39, 0.0, -4175
Mx, My = -8689, 1479

# Pile Layout
pile_layout = np.array([
    [2, 2], [8, 2], [14, 2],
    [2, 8], [8, 8], [14, 8],
    [2, 14], [8, 14], [14, 14],
    [2, 20], [8, 20], [14, 20]
])

# Pile Cap Shape
pile_cap_vertices = np.array([
    [0, 0], [16, 0], [16, 22], [0, 22]
])

# --- CHECK PILES INSIDE CAP ---
min_x, min_y = np.min(pile_cap_vertices, axis=0)
max_x, max_y = np.max(pile_cap_vertices, axis=0)

for i, (px, py) in enumerate(pile_layout):
    if not (min_x <= px <= max_x and min_y <= py <= max_y):
        raise ValueError(f"Pile {i+1} at ({px}, {py}) is outside the pile cap!")

# --- WEIGHTS ---
pile_cap_volume = 22 * 16 * pile_cap_thickness
pile_cap_weight = (pile_cap_volume * concrete_density) / 1000

soil_volume = 22 * 16 * soil_depth_above
soil_weight = (soil_volume * soil_density) / 1000

pile_embedment_weight = (pile_embedment * pile_size**2 * len(pile_layout) * concrete_density) / 1000

total_weight = pile_cap_weight + soil_weight - pile_embedment_weight

# --- CENTROID ---
pile_centroid = np.mean(pile_layout, axis=0)

# --- LOADS WITH ECCENTRICITY ---
Mx_adj = Mx + Fz * column_eccentricity[1]
My_adj = My - Fz * column_eccentricity[0]

# --- EQUILIBRIUM EQUATIONS ---
num_piles = len(pile_layout)
A = np.zeros((3, num_piles))

b = np.array([-Fz + total_weight, -Mx_adj, My_adj])

for i, (px, py) in enumerate(pile_layout):
    A[0, i] = 1
    A[1, i] = py - pile_centroid[1]
    A[2, i] = px - pile_centroid[0]

# --- SOLVE FOR REACTIONS ---
reactions, _, _, _ = np.linalg.lstsq(A, b.reshape(-1, 1), rcond=None)
reactions = reactions.flatten()

# --- CHECK CAPACITY ---
for i, R in enumerate(reactions):
    if R > max_pile_compression:
        print(f"WARNING: Pile {i+1} exceeds max compression ({R:.2f} > {max_pile_compression} kips)")
    if R < -max_pile_tension:
        print(f"WARNING: Pile {i+1} exceeds max tension ({abs(R):.2f} > {max_pile_tension} kips)")

# --- DISPLAY RESULTS ---
results = pd.DataFrame({
    "Pile": np.arange(1, num_piles + 1),
    "X (ft)": pile_layout[:, 0],
    "Y (ft)": pile_layout[:, 1],
    "Reaction (kips)": reactions
})

print("\nPile Reactions:\n", results)

# ==============================
# FUNCTION: SPLIT POLYGON BY LINE
# ==============================

def split_polygon_by_line(polygon_vertices, line_type, line_value, column_centroid=None):
    polygon = Polygon(polygon_vertices)

    if not polygon.is_valid:
        raise ValueError("Input polygon is invalid!")

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
        raise ValueError("line_type must be 'regular' or 'vertical'")

    try:
        split_polygons = split(polygon, line)
    except Exception as e:
        print("Polygon could not be split:", e)
        return 0.0, 0.0

    if len(split_polygons.geoms) < 2:
        print("Polygon was not split by the line.")
        return polygon.area, 0.0

    areas = []
    for poly in split_polygons.geoms:
        centroid = poly.centroid
        x, y = centroid.x, centroid.y

        if line_type == 'regular':
            y_on_line = m * x + c
            areas.append(('above' if y > y_on_line else 'below', poly.area))
        elif line_type == 'vertical':
            areas.append(('right' if x > line_value[0] else 'left', poly.area))

    if line_type == 'regular':
        area_above = sum(a for s, a in areas if s == 'above')
        area_below = sum(a for s, a in areas if s == 'below')
        return area_above, area_below
    else:
        area_right = sum(a for s, a in areas if s == 'right')
        area_left = sum(a for s, a in areas if s == 'left')
        return area_right, area_left

# ==============================
# FUNCTION: ANALYZE PILE DISTRIBUTION WITH REACTIONS
# ==============================

def analyze_pile_distribution_with_reactions(polygon_vertices, line_type, line_value, pile_layout, pile_size, pile_reactions):
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
        cutting_line = LineString([(x, min_y - extend), (x, max_y + extend)])
    else:
        raise ValueError("line_type must be 'regular' or 'vertical'")

    piles_above, piles_below, intersected_piles = [], [], []
    area_above, area_below = 0.0, 0.0
    reaction_above, reaction_below = 0.0, 0.0
    intersected_area_above, intersected_area_below = 0.0, 0.0
    intersected_reaction_above, intersected_reaction_below = 0.0, 0.0

    half_size = pile_size / 2
    pile_area = pile_size ** 2

    for idx, (px, py) in enumerate(pile_layout):
        pile_box = box(px - half_size, py - half_size, px + half_size, py + half_size)
        reaction = reactions[idx]

        if pile_box.crosses(cutting_line):
            intersected_piles.append(idx + 1)
            try:
                split_pile = split(pile_box, cutting_line)
                for piece in split_pile.geoms:
                    centroid = piece.centroid
                    cx, cy = centroid.x, centroid.y
                    piece_area = piece.area
                    piece_reaction = reaction * (piece_area / pile_area)

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
        total_area_above, total_area_below,
        piles_above, piles_below,
        intersected_area_above, intersected_area_below,
        intersected_piles,
        total_reaction_above, total_reaction_below
    )

# ==============================
# EXAMPLE USE:
# ==============================

line_type = 'vertical'
line_value = (10.5,)

(
    area_above, area_below,
    piles_above, piles_below,
    intersected_area_above, intersected_area_below,
    intersected_piles,
    total_reaction_above, total_reaction_below
) = analyze_pile_distribution_with_reactions(
    pile_cap_vertices,
    line_type,
    line_value,
    pile_layout,
    pile_size,
    reactions
)

print(f"Total area right of line: {area_above:.2f} sq.ft.")
print(f"Total area left of line: {area_below:.2f} sq.ft.\n")
print(f"Piles right of line: {piles_above}")
print(f"Piles left of line: {piles_below}\n")
print(f"Intersected piles: {intersected_piles}")
print(f"Intersected areas right: {intersected_area_above:.2f} sq.ft.")
print(f"Intersected areas left: {intersected_area_below:.2f} sq.ft.\n")
print(f"Total reaction right of line: {total_reaction_above:.2f} kips")
print(f"Total reaction left of line: {total_reaction_below:.2f} kips")
