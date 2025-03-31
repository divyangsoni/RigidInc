# visualization.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from shapely.geometry import Polygon


def plot_foundation_analysis(
    pile_cap_vertices,
    pile_layout,
    pile_size,
    pile_reactions, 
    column_size,
    column_centroid,
    pile_cap_thickness,
    pile_cap_shear_depth,
    one_way_shear_1_line_type,
    one_way_shear_1_line_value,
    one_way_shear_2_line_type,
    one_way_shear_2_line_value,
    one_way_moment_1_line_type,
    one_way_moment_1_line_value,
    one_way_moment_2_line_type,
    one_way_moment_2_line_value,
    one_way_shear_section_1,
    one_way_shear_section_2,
    one_way_moment_section_4,
    one_way_moment_section_3,
    area_of_steel_section_1,
    area_of_steel_section_2,
    shear_polygon_coords,
    total_reaction_outside,
    shear_capacity_1,
    shear_capacity_2,
    punching_shear_capacity: float,
    utilization_ratio: float
):
    """
    Plot the foundation analysis results.
    """

    # Set the font to Arial (or any other available font)
    plt.rcParams['font.family'] = 'Georgia'
    
    # First, determine the extents of the pile cap to adapt the figure size:
    min_x, min_y = np.min(pile_cap_vertices, axis=0)
    max_x, max_y = np.max(pile_cap_vertices, axis=0)
    width_ft = max_x - min_x
    height_ft = max_y - min_y

    # Set a scale factor (inches per foot) for the figure size
    scale = 1.9  # Adjust this value as needed
    fig, ax = plt.subplots(figsize=(width_ft * scale, height_ft * scale))

    # -------------------------------
    # 1. Plot the Pile Cap
    # -------------------------------
    pile_cap_poly = MplPolygon(pile_cap_vertices, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(pile_cap_poly)

    # -------------------------------
    # 2. Plot Each Pile + Reaction
    # -------------------------------
    half_pile = pile_size / 2.0
    for i, (px, py) in enumerate(pile_layout):
        lower_left = (px - half_pile, py - half_pile)
        pile_rect = Rectangle(lower_left, pile_size, pile_size,
                            edgecolor='blue', facecolor='lightblue', linewidth=1)
        ax.add_patch(pile_rect)
        # Pile number at center
        ax.text(px, py, str(i+1), color='blue', fontsize=10, ha='center', va='center')
        # Reaction value just below
        ax.text(px, py - (half_pile + 0.1),
            f"{pile_reactions[i].item():.3f} kips",
            color='black', fontsize=8, ha='center', va='top')



    # -------------------------------
    # 3. Plot the Column
    # -------------------------------
    column_width, column_height = column_size
    column_cx, column_cy = column_centroid
    col_lower_left = (column_cx - column_width / 2, column_cy - column_height / 2)
    column_rect = Rectangle(col_lower_left, column_width, column_height,
                            edgecolor='magenta', facecolor='none', linewidth=2)
    ax.add_patch(column_rect)
    ax.text(column_cx, column_cy, 'Column', color='magenta', fontsize=10, ha='center', va='center')



    # -------------------------------
    # 4. Plot the One-Way Shear Section Lines with Labeled Values
    # -------------------------------
    # Section 1: One-way shear section (regular line)
    if one_way_shear_1_line_type == 'regular':
        m, c = one_way_shear_1_line_value
        x_vals = np.linspace(min_x - 5, max_x + 5, 100)
        y_vals = m * x_vals + c
        ax.plot(x_vals, y_vals, color='green', linestyle='--', linewidth=2,
                label=f"One-way shear @ section 1: {one_way_shear_section_1:.3f} kips")

    # Section 2: One-way shear section (vertical line)
    if one_way_shear_2_line_type == 'vertical':
        x_val = one_way_shear_2_line_value[0]
        y_vals = np.linspace(min_y - 5, max_y + 5, 100)
        x_vals = np.full_like(y_vals, x_val)
        ax.plot(x_vals, y_vals, color='cyan', linestyle='--', linewidth=2,
                label=f"One-way shear @ section 2: {one_way_shear_section_2:.3f} kips")

    # -------------------------------
    # 5. Plot the One-Way Moment Section Lines with Labeled Values
    # -------------------------------
    # Section 1: One-way moment section (regular line)
    if one_way_moment_1_line_type == 'regular':
        m, c = one_way_moment_1_line_value
        x_vals = np.linspace(min_x - 5, max_x + 5, 100)
        y_vals = m * x_vals + c
        ax.plot(x_vals, y_vals, color='orange', linestyle=':', linewidth=2,
                label=f"One-way moment @ section 1: {one_way_moment_section_4:.3f} kip-ft "
                f"(A = {area_of_steel_section_1:.3f} sq. in.)")

    # Section 2: One-way moment section (vertical line)
    if one_way_moment_2_line_type == 'vertical':
        x_val = one_way_moment_2_line_value[0]
        y_vals = np.linspace(min_y - 5, max_y + 5, 100)
        x_vals = np.full_like(y_vals, x_val)
        ax.plot(x_vals, y_vals, color='red', linestyle=':', linewidth=2,
                label=f"One-way moment @ section 2: {one_way_moment_section_3:.3f} kip-ft " 
                f"(A = {area_of_steel_section_2:.3f} sq. in.)")

    # -------------------------------
    # 6. Plot the Shear Polygon (Punching Shear Polygon) with Label Including total_reaction_outside
    # -------------------------------
    if shear_polygon_coords:
        punching_shear_poly = MplPolygon(shear_polygon_coords, edgecolor='blue', facecolor='none',
                                         linewidth=2, linestyle='-.',
                                         label=f"Punching shear: {total_reaction_outside:.3f} kips")
        ax.add_patch(punching_shear_poly)

    # -------------------------------
    # 7. Annotate the Pile Cap Thickness
    # -------------------------------
    # Position the annotation just outside the bottom-right corner of the pile cap.
    offset_x = 0.25  # adjust as needed
    offset_y = -0.5  # adjust as needed
    ax.text(min_x + offset_x, min_y + offset_y, f"Thickness = {12*pile_cap_thickness:.3f} in.",
            fontsize=10, color='black', ha='left', va='top')
    
    # -------------------------------
    # 7.5. Annotate the Column Dimensions
    # -------------------------------
    # Position the annotation just outside the bottom-right corner of the pile cap.
    offset_x = 0.25  # adjust as needed
    offset_y = -0.25  # adjust as needed
    ax.text(max_x + offset_x, min_y + offset_y, 
        f"Column Width = {column_width*12:.3f} in.\nColumn Height = {column_height*12:.3f} in.",
        fontsize=10, color='black', ha='right', va='top')
    
    # -------------------------------
    # 8. Annotate Shear Capacity & Utilization
    # -------------------------------

    offset_x = 0.25
    offset_y = -2.5
    utilization_shear_1 = one_way_shear_section_1 / shear_capacity_1 * 100
    utilization_shear_2 = one_way_shear_section_2 / shear_capacity_2 * 100
    text = (
        f"One-way shear capacity (Section 1): {shear_capacity_1:.3f} kips, "
        f"Utilization: {utilization_shear_1:.1f}%\n"
        f"One-way shear capacity (Section 2): {shear_capacity_2:.3f} kips, "
        f"Utilization: {utilization_shear_2:.1f}%\n"
        f"Punching shear capacity: {punching_shear_capacity:.3f} kips, "
        f"Utilization: {utilization_ratio:.1f}%"
    )
    ax.text(min_x + offset_x, min_y + offset_y, text,
            fontsize=10, color='black', ha='left', va='top')

    # -------------------------------
    # 9. Annotate the Pile Cap Shear Thickness
    # -------------------------------
    # Position the annotation just outside the bottom-right corner of the pile cap.
    offset_x = 0.25  # adjust as needed
    offset_y = -1.0  # adjust as needed
    ax.text(min_x + offset_x, min_y + offset_y, f"Shear Depth = {12*pile_cap_shear_depth:.3f} in.",
            fontsize=10, color='black', ha='left', va='top')
    

    # -------------------------------
    # 10. Annotate Shear Polygon Perimeter
    # -------------------------------
    if shear_polygon_coords:
        shear_poly = Polygon(shear_polygon_coords)
        shear_perimeter = shear_poly.length
        # Adjust offsets relative to the thickness annotation:
        offset_x2 = 0.25    # Same as thickness offset_x
        offset_y2 = -1.5    # Slightly lower than thickness annotation (adjust as needed)
        ax.text(min_x + offset_x2, min_y + offset_y2,
                f"Punching Shear Perimeter: {shear_perimeter:.3f} ft",
                fontsize=10, color='black', ha='left', va='top')
        



    # -------------------------------
    # Final Plot Settings
    # -------------------------------
    ax.set_xlim(min_x - 5, max_x + 5)
    ax.set_ylim(min_y - 5, max_y + 5)
    ax.set_aspect('equal')
    ax.set_title("Analysis Results for Foundation Supported by Rigid Inclusions")
    plt.xlabel("X (ft)")
    plt.ylabel("Y (ft)")
    plt.grid(True)
    plt.legend()
    plt.show()
