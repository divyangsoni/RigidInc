# %%
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from pile_distribution_analyzer import (analyze_pile_distribution_with_reactions, calculate_moments_about_section_line)
from weight_calculator import calculate_concrete_soil_weight
from shear_polygon_analysis import analyze_shear_polygon_reactions
from polygon_splitter import split_polygon_by_line

# %%
# Input Parameters
column_size = (2.0, 5.0)  # Column dimensions (x, y) in ft
column_eccentricity = (0.0, 0)  # Eccentricity of column from center (x, y) in ft
column_centroid = (7.5, 7.5)  # center of the column in ft
pile_cap_thickness = 3.167  # Pile cap thickness in ft
pile_embedment = 1.0  # Depth of pile embedment in pile cap (ft)
soil_depth_above = 2.0  # Soil depth above pile cap (ft)
soil_density = 0  # Soil density in pcf
concrete_density = 0  # Concrete density in pcf
pile_cap_shear_depth = 2.792  # Effective shear depth (ft)

# %%
# Pile Properties
pile_shape = "square"  # Either "square" or "circular"
pile_size = 0.833  # Pile size (ft)
max_pile_compression = 120  # Max pile resistance in compression (kips)
max_pile_tension = 120  # Max pile resistance in tension (kips)

# %%
# Column Loads (kips and kip-ft)
Fx, Fy, Fz = 0, 0.0, -960  # Forces in kips
Mx, My = 0, 0  # Moments in kip-ft

# %%
# Pile Layout (x, y coordinates in ft) - measured from bottom-left corner
pile_layout = np.array([
    [3, 4], [7.5, 4], [12, 4],
    [5.25, 7.5], [9.75, 7.5],
    [3, 11], [7.5, 11], [12, 11]
])  # (ft)

# %%
# Pile Cap Shape (Vertices of pile cap)
pile_cap_vertices = np.array([
    [0, 0], [15, 0], [15, 15], [0, 15]
])  # Square pile cap

# %%
# Verify all piles are within the pile cap boundary
min_x, min_y = np.min(pile_cap_vertices, axis=0)
max_x, max_y = np.max(pile_cap_vertices, axis=0)

for i, (px, py) in enumerate(pile_layout):
    if not (min_x <= px <= max_x and min_y <= py <= max_y):
        raise ValueError(f"Pile {i+1} at ({px}, {py}) is outside the pile cap!")

# %%
# Compute Self-weight of Pile Cap and Soil
pile_cap_volume = Polygon(pile_cap_vertices).area * pile_cap_thickness  # ft³
pile_cap_weight = (pile_cap_volume * concrete_density) / 1000  # kips

soil_volume = Polygon(pile_cap_vertices).area * soil_depth_above  # ft³
soil_weight = (soil_volume * soil_density) / 1000  # kips

# Deduct pile embedment weight
pile_embedment_weight = (pile_embedment * pile_size**2 * len(pile_layout) * concrete_density) / 1000  # kips
total_weight = pile_cap_weight + soil_weight - pile_embedment_weight

# %%
# Compute centroid of pile locations
pile_centroid = np.mean(pile_layout, axis=0)

# %%
# Adjust loads for eccentricity
Mx_adj = Mx + Fz * column_eccentricity[1]  # Moment about x
My_adj = My - Fz * column_eccentricity[0]  # Moment about y

# %%
# Formulating Equilibrium Equations
num_piles = len(pile_layout)
A = np.zeros((3, num_piles))  # 3 equations (ΣFz=0, ΣMx=0, ΣMy=0)

# Right-hand side of the system of equations (Fix My sign)
b = np.array([-Fz + total_weight, -Mx_adj, My_adj])  # <-- FIX HERE

for i, (px, py) in enumerate(pile_layout):
    A[0, i] = 1  # ΣFz = 0
    A[1, i] = py - pile_centroid[1]  # ΣMx = 0 (Moment arm about x-axis)
    A[2, i] = px - pile_centroid[0]  # ΣMy = 0 (Moment arm about y-axis)

# Solve for reactions using least squares (in case of near-singular matrices)
reactions, _, _, _ = np.linalg.lstsq(A, b.reshape(-1, 1), rcond=None)

# %%
# Ensure piles are within capacity
for i, R in enumerate(reactions.flatten()):
    if R > max_pile_compression:
        print(f"WARNING: Pile {i+1} exceeds max compression ({R:.2f} > {max_pile_compression} kips)")
    if R < -max_pile_tension:
        print(f"WARNING: Pile {i+1} exceeds max tension ({abs(R):.2f} > {max_pile_tension} kips)")

# %%
# Display Results
results = pd.DataFrame({
    "Pile": np.arange(1, num_piles + 1),
    "X (ft)": pile_layout[:, 0],
    "Y (ft)": pile_layout[:, 1],
    "Reaction (kips)": reactions.flatten()
})

print("\nPile Reactions:\n", results)

# %%
# Example 1: Regular line (y = 0x + 12.792)
line_type = 'regular'
line_value = (0, 12.792)

# Function returns: area_above, area_below, first_area_moment_above, first_area_moment_below
area_above, area_below, area_moment_above, area_moment_below = split_polygon_by_line(
    pile_cap_vertices, line_type, line_value
)

print(f"Areas above the line: {area_above:.2f} sq.ft.")
print(f"Areas below the line: {area_below:.2f} sq.ft.")
print(f"First area moment above the line: {area_moment_above:.2f} sq.ft.-ft")
print(f"First area moment below the line: {area_moment_below:.2f} sq.ft.-ft\n")


# Example 2: Vertical line (x = 8.5)
line_type = 'vertical'
line_value = (8.5,)

area_right, area_left, area_moment_right, area_moment_left = split_polygon_by_line(
    pile_cap_vertices, line_type, line_value
)

print(f"Areas to the right of the line: {area_right:.2f} sq.ft.")
print(f"Areas to the left of the line: {area_left:.2f} sq.ft.")
print(f"First area moment to the right of the line: {area_moment_right:.2f} sq.ft.-ft")
print(f"First area moment to the left of the line: {area_moment_left:.2f} sq.ft.-ft\n")


# Example 3: Another Vertical Line (x = 8.0)
line_type = 'vertical'
line_value = (8.0,)

area_right, area_left, area_moment_right, area_moment_left = split_polygon_by_line(
    pile_cap_vertices, line_type, line_value
)

print(f"Areas to the right of the line (x = 8.0): {area_right:.2f} sq.ft.")
print(f"Areas to the left of the line (x = 8.0): {area_left:.2f} sq.ft.")
print(f"First area moment to the right of the line: {area_moment_right:.2f} sq.ft.-ft")
print(f"First area moment to the left of the line: {area_moment_left:.2f} sq.ft.-ft")


# %%
# -----------------------------------
# Inputs (make sure these variables are defined)
# -----------------------------------
line_type = 'vertical'
line_value = (8.5,)  # x = 11.292 (for vertical line)


# -----------------------------------
# Analyze pile distribution with reactions
# -----------------------------------
(
    area_above, area_below,
    piles_above, piles_below,
    intersected_area_above, intersected_area_below,
    intersected_piles,
    total_reaction_above, total_reaction_below,
    intersected_pile_geoms
) = analyze_pile_distribution_with_reactions(
    polygon_vertices=pile_cap_vertices,
    line_type=line_type,
    line_value=line_value,
    pile_layout=pile_layout,
    pile_size=pile_size,
    pile_reactions=reactions
)

# -----------------------------------
# Calculate moments about section line
# -----------------------------------
moment_above, moment_below = calculate_moments_about_section_line(
    line_type=line_type,
    line_value=line_value,
    pile_layout=pile_layout,
    pile_reactions=reactions,
    piles_above=piles_above,
    piles_below=piles_below,
    intersected_pile_geoms=intersected_pile_geoms
)

# -----------------------------------
# Print results
# -----------------------------------
print(f"\n=== Areas ===")
print(f"Total area right of line: {area_above:.2f} sq.ft.")
print(f"Total area left of line: {area_below:.2f} sq.ft.")

print(f"\n=== Piles ===")
print(f"Piles right of line: {piles_above}")
print(f"Piles left of line: {piles_below}")

print(f"\n=== Intersected Piles ===")
print(f"Intersected piles: {intersected_piles}")
print(f"Intersected areas right of line: {intersected_area_above:.2f} sq.ft.")
print(f"Intersected areas left of line: {intersected_area_below:.2f} sq.ft.")

print(f"\n=== Reactions ===")
print(f"Total reaction right of line: {total_reaction_above:.2f} kips")
print(f"Total reaction left of line: {total_reaction_below:.2f} kips")

print(f"\n=== Moments ===")
print(f"Moment right of line: {moment_above:.2f} kip-ft")
print(f"Moment left of line: {moment_below:.2f} kip-ft")



# %%
# Example arguments
area = 100.0  # sq.ft.
pile_cap_thickness = 5.5  # ft
soil_depth_above = 2.0    # ft
concrete_density = 150    # pcf
soil_density = 120        # pcf

# Run the function
concrete_wt, soil_wt = calculate_concrete_soil_weight(
    area,
    pile_cap_thickness,
    soil_depth_above,
    concrete_density,
    soil_density
)

print(f"Concrete weight over {area} sq.ft.: {concrete_wt:.3f} kips")
print(f"Soil weight over {area} sq.ft.: {soil_wt:.3f} kips")


# %%

# 1. Analyze pile distribution by line
results = analyze_pile_distribution_with_reactions(
    pile_cap_vertices,
    line_type='vertical',
    line_value=(10.5,),
    pile_layout=pile_layout,
    pile_size=2.0,
    pile_reactions=reactions
)

(
    total_area_above,
    total_area_below,
    piles_above,
    piles_below,
    intersected_area_above,
    intersected_area_below,
    intersected_piles,
    total_reaction_above,
    total_reaction_below,
    intersected_pile_geoms
) = results

# 2. Calculate moments
moment_above, moment_below = calculate_moments_about_section_line(
    line_type='vertical',
    line_value=(10.5,),
    pile_layout=pile_layout,
    pile_reactions=reactions,
    piles_above=piles_above,
    piles_below=piles_below,
    intersected_pile_geoms=intersected_pile_geoms
)

print(f"Moment right of line: {moment_above:.2f} kip-ft")
print(f"Moment left of line: {moment_below:.2f} kip-ft")


# %%
# Example usage (make sure your variables are properly defined in the main program!)
result = analyze_shear_polygon_reactions(
    polygon_vertices=pile_cap_vertices,
    pile_layout=pile_layout,
    pile_size=pile_size,
    column_centroid=column_centroid,
    column_size=column_size,
    shear_depth=pile_cap_shear_depth,
    pile_reactions=reactions
)

(
    total_reaction_outside,
    shear_polygon_coords,
    shear_polygon_perimeter,
    inside_piles,
    outside_piles,
    intersecting_piles
) = result

print(f"Total reaction outside shear polygon: {total_reaction_outside:.2f} kip")
print(f"Shear polygon perimeter length: {shear_polygon_perimeter:.2f} ft")
print(f"Shear polygon coordinates: {shear_polygon_coords}\n")

print(f"Piles fully inside shear polygon: {inside_piles}")
print(f"Piles fully outside shear polygon: {outside_piles}")
print(f"Piles intersecting shear polygon: {intersecting_piles}")



