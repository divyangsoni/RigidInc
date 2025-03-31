import pandas as pd
from shapely.geometry import Polygon, LineString, Point # Keep shapely imports if used by your modules
# from shapely.ops import split # Keep if used by your modules
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting in Flask
import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon # Keep if used by visualization.py
import numpy as np
import io
import base64
from flask import Flask, render_template, request, flash # Added logging
import logging

# --- Import your custom modules ---
# Ensure these files are in the same directory or accessible via Python path
from pile_distribution_analyzer import (analyze_pile_distribution_with_reactions, calculate_moments_about_section_line)
from weight_calculator import calculate_concrete_soil_weight
from shear_polygon_analysis import analyze_shear_polygon_reactions
from polygon_splitter import split_polygon_by_line, polygon_intersection # Make sure polygon_intersection is imported if used directly
from visualization import plot_foundation_analysis
from design import calculate_area_of_steel, calculate_punching_shear_capacity, concrete_shear_capacity_simple

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_123!' # Needed for flash messages

# --- Default Values (Keep these accessible) ---
# Convert numpy arrays in defaults to string representations suitable for textarea
def format_array_for_textarea(arr):
    return '\n'.join([f"{row[0]}, {row[1]}" for row in arr])

defaults = {
    "column_size": ("38.66, 1"),
    "column_eccentricity": ("0, 0"),
    "column_centroid": ("25, 9"),
    "pile_cap_thickness": 3 + 4/12,
    "pile_embedment": 1.0,
    "soil_depth_above": 2.0,
    "soil_density": 0.0, # Make sure defaults are floats where needed
    "concrete_density": 0.0, # Make sure defaults are floats where needed
    "concrete_strength_psi": 3000,
    "column_location": "interior",
    "lambda_factor": 1.0,
    "pile_shape": "square",
    "pile_size": 0.8863,
    "max_pile_compression": 120.0,
    "max_pile_tension": 120.0,
    "Fx": 0.0,
    "Fy": 0.0,
    "Fz": -30 * 120 * 1.29,
    "Mx": 0.0,
    "My": 0.0,
    "num_piles": 30, # Default number (will be overwritten by parsed layout)
    "pile_layout_str": format_array_for_textarea(np.array([
        [3.25, 4.5], [8.083, 4.5], [12.9167, 4.5], [17.75, 4.5], [22.583, 4.5], [27.4167, 4.5], [32.25, 4.5], [37.083, 4.5], [41.9167, 4.5], [46.75, 4.5],
        [3.25, 9], [8.083, 9], [12.9167, 9], [17.75, 9], [22.583, 9], [27.4167, 9], [32.25, 9], [37.083, 9], [41.9167, 9], [46.75, 9],
        [3.25, 13.5], [8.083, 13.5], [12.9167, 13.5], [17.75, 13.5], [22.583, 13.5], [27.4167, 13.5], [32.25, 13.5], [37.083, 13.5], [41.9167, 13.5], [46.75, 13.5],
    ])),
     "pile_cap_shape_type": "rectangle", # Keep the actual type default
     "pile_cap_vertices_str": format_array_for_textarea(np.array([
         [0+1, 0+2.5], [50-1, 0+2.5], [50-1, 18-2.5], [0+1, 18-2.5]
     ])),
     "rect_bottom_left": ("1, 2.5"),
     "rect_top_right": ("49, 15.5"),
     # Store original numpy arrays too for internal use if parsing fails
     "_pile_layout_default_np": np.array([
         [3.25, 4.5], [8.083, 4.5], [12.9167, 4.5], [17.75, 4.5], [22.583, 4.5], [27.4167, 4.5], [32.25, 4.5], [37.083, 4.5], [41.9167, 4.5], [46.75, 4.5],
         [3.25, 9], [8.083, 9], [12.9167, 9], [17.75, 9], [22.583, 9], [27.4167, 9], [32.25, 9], [37.083, 9], [41.9167, 9], [46.75, 9],
         [3.25, 13.5], [8.083, 13.5], [12.9167, 13.5], [17.75, 13.5], [22.583, 13.5], [27.4167, 13.5], [32.25, 13.5], [37.083, 13.5], [41.9167, 13.5], [46.75, 13.5],
     ]),
    "_pile_cap_vertices_default_np": np.array([
         [0+1, 0+2.5], [50-1, 0+2.5], [50-1, 18-2.5], [0+1, 18-2.5]
    ]),
    # --- NEW DEFAULTS ---
    "assumed_cover_in": 3.0,
    "assumed_bar_dia_in": 10.0 # Represents #10 bar -> 10/8 inch diameter
}

# --- Helper function for parsing coordinate pairs ---
def parse_coord_pair(input_str, default_tuple):
    if not input_str:
        return default_tuple
    try:
        parts = [float(v.strip()) for v in input_str.split(',')]
        if len(parts) == 2:
            return tuple(parts)
        else:
            # Raise error to be caught below
            raise ValueError("Input string must contain exactly two comma-separated numbers.")
    except ValueError:
        # Let the main error handler catch this
        raise ValueError(f"Invalid coordinate pair format: '{input_str}'. Expected 'x, y'.")

# --- Helper function for parsing multi-line coordinates (like pile layout) ---
def parse_multi_line_coords(input_str, default_np_array):
    if not input_str:
        return default_np_array
    lines = input_str.strip().split('\n')
    coords_list = []
    try:
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue # Skip empty lines
            parts = [float(v.strip()) for v in line.split(',')]
            if len(parts) == 2:
                coords_list.append(parts)
            else:
                raise ValueError(f"Line {i+1} ('{line}') does not contain exactly two comma-separated numbers.")
        if not coords_list: # Handle case where input was just whitespace
             return default_np_array
        return np.array(coords_list)
    except ValueError as e:
        # Let the main error handler catch this
        raise ValueError(f"Invalid multi-line coordinate format: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Display form with default values
        return render_template('index.html', defaults=defaults, results=None, form_data=defaults) # Pass defaults as form_data initially

    # --- POST Request: Process form and run calculation ---
    form_data = request.form.to_dict() # Get submitted data
    results_data = {}
    plot_url = None
    pile_reactions_html = None
    warnings = [] # To collect warnings like pile capacity exceeded
    calculation_log = [] # To store print-like outputs for debugging/comparison

    try:
        # --- Get and Validate Inputs (using defaults if empty) ---
        calculation_log.append("--- Parsing Inputs ---")
        column_size = parse_coord_pair(form_data.get('column_size'), defaults['column_size'])
        column_eccentricity = parse_coord_pair(form_data.get('column_eccentricity'), defaults['column_eccentricity'])
        column_centroid = parse_coord_pair(form_data.get('column_centroid'), defaults['column_centroid'])

        pile_cap_thickness = float(form_data.get('pile_cap_thickness') or defaults['pile_cap_thickness'])
        pile_embedment = float(form_data.get('pile_embedment') or defaults['pile_embedment'])
        soil_depth_above = float(form_data.get('soil_depth_above') or defaults['soil_depth_above'])
        soil_density = float(form_data.get('soil_density') or defaults['soil_density'])
        concrete_density = float(form_data.get('concrete_density') or defaults['concrete_density'])
        concrete_strength_psi = int(form_data.get('concrete_strength_psi') or defaults['concrete_strength_psi'])
        column_location = form_data.get('column_location') or defaults['column_location']
        if column_location not in ["interior", "edge", "corner"]: column_location = defaults['column_location'] # Basic validation
        lambda_factor = float(form_data.get('lambda_factor') or defaults['lambda_factor'])

        # --- NEW INPUTS ---
        assumed_cover_in = float(form_data.get('assumed_cover_in') or defaults['assumed_cover_in'])
        assumed_bar_dia_in = float(form_data.get('assumed_bar_dia_in') or defaults['assumed_bar_dia_in'])
        # --- END NEW INPUTS ---

        pile_shape = form_data.get('pile_shape') or defaults['pile_shape']
        if pile_shape not in ["square", "circular"]: pile_shape = defaults['pile_shape'] # Basic validation
        pile_size = float(form_data.get('pile_size') or defaults['pile_size'])
        max_pile_compression = float(form_data.get('max_pile_compression') or defaults['max_pile_compression'])
        max_pile_tension = float(form_data.get('max_pile_tension') or defaults['max_pile_tension'])

        Fx = float(form_data.get('Fx') or defaults['Fx'])
        Fy = float(form_data.get('Fy') or defaults['Fy'])
        Fz = float(form_data.get('Fz') or defaults['Fz'])
        Mx = float(form_data.get('Mx') or defaults['Mx'])
        My = float(form_data.get('My') or defaults['My'])

        # Parse Pile Layout
        pile_layout_str = form_data.get('pile_layout_str', '').strip()
        if not pile_layout_str:
             pile_layout = defaults['_pile_layout_default_np']
             form_data['pile_layout_str'] = defaults['pile_layout_str'] # Update form_data for re-display
             calculation_log.append("Using default pile layout.")
        else:
             pile_layout = parse_multi_line_coords(pile_layout_str, defaults['_pile_layout_default_np'])
             calculation_log.append(f"Parsed {len(pile_layout)} piles from input.")
        num_piles = len(pile_layout) # Set num_piles based on actual layout used

        # Parse Pile Cap Shape and Vertices
        pile_cap_shape_type = form_data.get('pile_cap_shape_type') or defaults['pile_cap_shape_type']
        pile_cap_vertices = None
        calculation_log.append(f"Pile cap shape type selected: {pile_cap_shape_type}")

        if pile_cap_shape_type == 'rectangle':
            rect_bl_str = form_data.get('rect_bottom_left', f"{defaults['rect_bottom_left'][0]},{defaults['rect_bottom_left'][1]}")
            rect_tr_str = form_data.get('rect_top_right', f"{defaults['rect_top_right'][0]},{defaults['rect_top_right'][1]}")
            form_data['rect_bottom_left'] = rect_bl_str # Ensure form_data has the value used
            form_data['rect_top_right'] = rect_tr_str

            bl_coords = parse_coord_pair(rect_bl_str, defaults['rect_bottom_left'])
            tr_coords = parse_coord_pair(rect_tr_str, defaults['rect_top_right'])

            bl_x, bl_y = bl_coords
            tr_x, tr_y = tr_coords
            if tr_x > bl_x and tr_y > bl_y:
                 pile_cap_vertices = np.array([
                     [bl_x, bl_y], [tr_x, bl_y], [tr_x, tr_y], [bl_x, tr_y]
                 ])
                 calculation_log.append("Defined rectangular pile cap vertices from input.")
            else:
                 flash("Error: Rectangle top-right coordinates must be greater than bottom-left. Using default pile cap shape.", "danger")
                 pile_cap_vertices = defaults['_pile_cap_vertices_default_np']
                 form_data['pile_cap_vertices_str'] = defaults['pile_cap_vertices_str'] # Reset textarea
                 calculation_log.append("Using default pile cap vertices due to rectangle input error.")

        elif pile_cap_shape_type == 'polygon':
             vertices_str = form_data.get('pile_cap_vertices_str', '').strip()
             if not vertices_str:
                 pile_cap_vertices = defaults['_pile_cap_vertices_default_np']
                 form_data['pile_cap_vertices_str'] = defaults['pile_cap_vertices_str'] # Update form_data
                 calculation_log.append("Using default pile cap vertices (polygon input empty).")
             else:
                 pile_cap_vertices = parse_multi_line_coords(vertices_str, defaults['_pile_cap_vertices_default_np'])
                 calculation_log.append(f"Parsed {len(pile_cap_vertices)} polygon vertices from input.")
             if len(pile_cap_vertices) < 3:
                 flash("Error: Polygon must have at least 3 vertices. Using default pile cap shape.", "danger")
                 pile_cap_vertices = defaults['_pile_cap_vertices_default_np']
                 form_data['pile_cap_vertices_str'] = defaults['pile_cap_vertices_str'] # Reset textarea
                 calculation_log.append("Using default pile cap vertices due to polygon input error (<3 vertices).")
        else: # Fallback if shape type is weird
            flash("Error: Invalid pile cap shape type selected. Using default pile cap shape.", "danger")
            pile_cap_vertices = defaults['_pile_cap_vertices_default_np']
            form_data['pile_cap_vertices_str'] = defaults['pile_cap_vertices_str'] # Update form_data
            calculation_log.append("Using default pile cap vertices due to invalid shape type.")

        calculation_log.append("--- Inputs Parsed ---")

        # --- Start of Calculation Code (Literal copy blocks) ---

        ### START ORIGINAL BLOCK ###
        # ---- Final Calculations based on Inputs ----
        # Calculate effective shear depth AFTER pile_cap_thickness is determined
        # Assuming standard cover and bar sizes as in the original calculation
        # You might want to make cover and bar sizes inputs as well for more flexibility
        # --- MODIFIED TO USE INPUTS ---
        # Note: Original script used hardcoded 3.0 and 10.0 here.
        # This calculation now uses the input variables `assumed_cover_in` and `assumed_bar_dia_in`
        pile_cap_shear_depth = pile_cap_thickness - (assumed_cover_in / 12.0) - (assumed_bar_dia_in / 8.0 / 12.0) - (assumed_bar_dia_in / 8.0 / 2.0 / 12.0) # ft
        # --- END MODIFICATION ---

        # print("\n--- Input Summary ---") # Handled by Input Echo tab
        # calculation_log.append(f"Column Size (x,y): {column_size} ft")
        # ... etc ... No need to log all inputs again here
        calculation_log.append(f"Calculated Effective Shear Depth (d): {pile_cap_shear_depth:.3f} ft")
        # calculation_log.append(f"Pile Cap Vertices Used:\n{pile_cap_vertices}")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Section 1: Regular line (y = mx + c)
        offset_1 = column_centroid[1] + column_size[1] / 2 + pile_cap_shear_depth
        one_way_shear_1_line_type = 'regular'
        one_way_shear_1_line_value = (0, offset_1)

        # Section 2: Vertical line (x = c)
        offset_2 = column_centroid[0] + column_size[0] / 2 + pile_cap_shear_depth
        one_way_shear_2_line_type = 'vertical'
        one_way_shear_2_line_value = (offset_2,)
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Section 1: Regular line (y = mx + c)
        offset_3 = column_centroid[1] + column_size[1] / 2
        one_way_moment_1_line_type = 'regular'
        one_way_moment_1_line_value = (0, offset_3)

        # Section 2: Vertical line (x = c)
        offset_4 = column_centroid[0] + column_size[0] / 2
        one_way_moment_2_line_type = 'vertical'
        one_way_moment_2_line_value = (offset_4,)

        # Verify all piles are within the pile cap boundary
        try:
             # Use Shapely polygon for robust check (as implemented previously)
             pile_cap_poly = Polygon(pile_cap_vertices)
        except Exception as e:
             raise ValueError(f"Invalid pile cap vertices definition for geometry checks: {e}")

        for i, (px, py) in enumerate(pile_layout):
            # Use a small buffer tolerance for floating point checks with contains()
             if not (pile_cap_poly.buffer(1e-9).contains(Point(px, py))):
                  raise ValueError(f"Pile {i+1} at ({px:.3f}, {py:.3f}) is outside the defined pile cap boundary!")
        calculation_log.append("Pile location check passed.")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Compute Self-weight of Pile Cap and Soil
        pile_cap_poly_area = Polygon(pile_cap_vertices).area # Calculate area once
        pile_cap_volume = pile_cap_poly_area * pile_cap_thickness  # ft³
        pile_cap_weight = (pile_cap_volume * concrete_density) / 1000 if concrete_density else 0 # kips

        soil_volume = pile_cap_poly_area * soil_depth_above  # ft³
        soil_weight = (soil_volume * soil_density) / 1000 if soil_density else 0 # kips

        # Deduct pile embedment weight
        # Adapting slightly for pile_shape input, using approximation for circular area
        if pile_shape == 'square':
            pile_single_area = pile_size**2
        elif pile_shape == 'circular':
            pile_single_area = np.pi * (pile_size / 2)**2
        else: # Default to square if shape is unknown (shouldn't happen with validation)
            pile_single_area = pile_size**2
            warnings.append(f"Unknown pile shape '{pile_shape}', assuming square for weight deduction.")

        pile_embedment_weight = (pile_embedment * pile_single_area * num_piles * concrete_density) / 1000 if concrete_density and num_piles > 0 else 0 # kips
        total_weight = pile_cap_weight + soil_weight - pile_embedment_weight
        calculation_log.append(f"Calculated Total Self-Weight (incl. soil, less piles): {total_weight:.2f} kips")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Compute centroid of pile locations
        pile_centroid = np.mean(pile_layout, axis=0) if num_piles > 0 else np.array([pile_cap_poly.centroid.x, pile_cap_poly.centroid.y]) # Use cap centroid if no piles
        calculation_log.append(f"Pile Centroid: ({pile_centroid[0]:.3f}, {pile_centroid[1]:.3f})")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Adjust loads for eccentricity
        Mx_adj = Mx + Fz * column_eccentricity[1]  # Moment about x
        My_adj = My - Fz * column_eccentricity[0]  # Moment about y
        calculation_log.append(f"Adjusted Moments: Mx_adj={Mx_adj:.2f} kip-ft, My_adj={My_adj:.2f} kip-ft")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Formulating Equilibrium Equations
        # Note: num_piles was already determined from the actual pile_layout
        A = np.zeros((3, num_piles))  # 3 equations (ΣFz=0, ΣMx=0, ΣMy=0)

        # Right-hand side of the system of equations (Fix My sign - as per original script)
        b = np.array([-Fz + total_weight, -Mx_adj, My_adj]) # <-- Original script had this sign convention

        reactions = np.zeros((num_piles, 1)) # Initialize reactions array

        if num_piles > 0:
            for i, (px, py) in enumerate(pile_layout):
                A[0, i] = 1                    # ΣFz = 0
                A[1, i] = py - pile_centroid[1]  # ΣMx = 0 (Moment arm about x-axis)
                A[2, i] = px - pile_centroid[0]  # ΣMy = 0 (Moment arm about y-axis)

            # Solve for reactions using least squares (in case of near-singular matrices)
            try:
                reactions, _, _, _ = np.linalg.lstsq(A, b.reshape(-1, 1), rcond=None)
                calculation_log.append("Solved for pile reactions using least squares.")
            except np.linalg.LinAlgError as e:
                raise ValueError(f"Could not solve for pile reactions (LinAlgError): {e}. Check pile layout and loads.")
        else:
             calculation_log.append("No piles defined, skipping reaction calculation.")
             # Check if vertical load is balanced if no piles
             if abs(-Fz + total_weight) > 1e-6: # Tolerance for floating point
                 warnings.append("WARNING: No piles defined, but vertical load Fz is not balanced by self-weight.")

        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Ensure piles are within capacity
        # (This block just prints warnings in the original, now collects them)
        for i, R in enumerate(reactions.flatten()):
             if R > max_pile_compression:
                 warnings.append(f"WARNING: Pile {i+1} reaction exceeds max compression ({R:.2f} > {max_pile_compression:.2f} kips)")
                 calculation_log.append(f"WARNING: Pile {i+1} compression check FAILED.")
             if R < -max_pile_tension:
                 warnings.append(f"WARNING: Pile {i+1} reaction exceeds max tension ({abs(R):.2f} > {max_pile_tension:.2f} kips)")
                 calculation_log.append(f"WARNING: Pile {i+1} tension check FAILED.")
        calculation_log.append("Pile capacity checks complete.")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Display Results (Prepared as DataFrame for HTML table)
        results_df = pd.DataFrame({
            "Pile": np.arange(1, num_piles + 1),
            "X (ft)": pile_layout[:, 0] if num_piles > 0 else [],
            "Y (ft)": pile_layout[:, 1] if num_piles > 0 else [],
            "Reaction (kips)": reactions.flatten() if num_piles > 0 else []
        })
        # Convert DataFrame to HTML for display - moved near end of try block
        # calculation_log.append("\nPile Reactions:\n"+ results_df.to_string(index=False)) # Log string version
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Section 1: Function returns: area_above, area_below, first_area_moment_above, first_area_moment_below
        area_above, area_below, _, _ = split_polygon_by_line( # Ignored moment outputs here
            pile_cap_vertices, one_way_shear_1_line_type, one_way_shear_1_line_value
        )
        area_above = area_above or 0 # Handle None return if line doesn't split
        area_below = area_below or 0 # Handle None return

        calculation_log.append(f"Areas above the shear line 1: {area_above:.2f} sq.ft.")
        calculation_log.append(f"Areas below the shear line 1: {area_below:.2f} sq.ft.")

        area_B = min(area_above, area_below)
        calculation_log.append(f"Area B (Shear Section 1 - Min Area): {area_B:.2f} sq.ft.")

        # Renamed variables to avoid clash (_s2 for shear line 2)
        area_right_s2, area_left_s2, _, _ = split_polygon_by_line( # Ignored moment outputs here
            pile_cap_vertices, one_way_shear_2_line_type, one_way_shear_2_line_value
        )
        area_right_s2 = area_right_s2 or 0 # Handle None
        area_left_s2 = area_left_s2 or 0 # Handle None

        calculation_log.append(f"\nAreas to the right of the shear line 2: {area_right_s2:.2f} sq.ft.")
        calculation_log.append(f"Areas to the left of the shear line 2: {area_left_s2:.2f} sq.ft.")

        area_A = min(area_right_s2, area_left_s2)
        calculation_log.append(f"Area A (Shear Section 2 - Min Area): {area_A:.2f} sq.ft.")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Calculate the intersection length for the regular line.
        intersection_length_reg = polygon_intersection(pile_cap_vertices, one_way_shear_1_line_type, one_way_shear_1_line_value) or 0
        calculation_log.append(f"Intersection length for shear line 1 (regular/horizontal): {intersection_length_reg:.3f} ft")

        # Calculate the intersection length for the vertical line.
        intersection_length_ver = polygon_intersection(pile_cap_vertices, one_way_shear_2_line_type, one_way_shear_2_line_value) or 0
        calculation_log.append(f"Intersection length for shear line 2 (vertical): {intersection_length_ver:.3f} ft")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Section 1 (Area B)
        concrete_wt_B, soil_wt_B = calculate_concrete_soil_weight(
            area_B, pile_cap_thickness, soil_depth_above, concrete_density, soil_density
        )
        calculation_log.append(f"Concrete weight over Area B ({area_B:.3f} sq.ft.): {concrete_wt_B:.3f} kips")
        calculation_log.append(f"Soil weight over Area B ({area_B:.3f} sq.ft.): {soil_wt_B:.3f} kips")

        # Section 2 (Area A)
        concrete_wt_A, soil_wt_A = calculate_concrete_soil_weight(
            area_A, pile_cap_thickness, soil_depth_above, concrete_density, soil_density
        )
        calculation_log.append(f"\nConcrete weight over Area A ({area_A:.3f} sq.ft.): {concrete_wt_A:.3f} kips")
        calculation_log.append(f"Soil weight over Area A ({area_A:.3f} sq.ft.): {soil_wt_A:.3f} kips")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # -----------------------------------
        # Section 1: Analyze pile distribution with reactions (Shear Line 1)
        # -----------------------------------
        # Renamed outputs slightly to avoid clashes (e.g., _s1 for Shear Line 1)
        (
            area_above_pile_s1, area_below_pile_s1, # Pile areas
            piles_above_s1, piles_below_s1,
            intersected_area_above_s1, intersected_area_below_s1,
            intersected_piles_s1,
            total_reaction_above_1, total_reaction_below_1, # Keep original reaction var names
            intersected_pile_geoms_s1 # Keep original geometry var name suffix
        ) = analyze_pile_distribution_with_reactions(
            polygon_vertices=pile_cap_vertices,
            line_type=one_way_shear_1_line_type,
            line_value=one_way_shear_1_line_value,
            pile_layout=pile_layout,
            pile_size=pile_size,
            pile_reactions=reactions
        )

        # --- Log Outputs ---
        calculation_log.append(f"\n=== Areas (Shear Section 1) ===")
        calculation_log.append(f"Total pile area above line 1: {area_above_pile_s1:.2f} sq.ft.")
        calculation_log.append(f"Total pile area below line 1: {area_below_pile_s1:.2f} sq.ft.")
        calculation_log.append(f"\n=== Piles (Shear Section 1) ===")
        calculation_log.append(f"Piles above line 1: {piles_above_s1}")
        calculation_log.append(f"Piles below line 1: {piles_below_s1}")
        calculation_log.append(f"\n=== Intersected Piles (Shear Section 1) ===")
        calculation_log.append(f"Intersected piles: {intersected_piles_s1}")
        calculation_log.append(f"Intersected areas above line 1: {intersected_area_above_s1:.2f} sq.ft.")
        calculation_log.append(f"Intersected areas below line 1: {intersected_area_below_s1:.2f} sq.ft.")
        calculation_log.append(f"\n=== Reactions (Shear Section 1) ===")
        calculation_log.append(f"Total reaction above line 1: {total_reaction_above_1:.2f} kips")
        calculation_log.append(f"Total reaction below line 1: {total_reaction_below_1:.2f} kips")
        # --- End Log ---

        # -----------------------------------
        # Section 2: Analyze pile distribution with reactions (Shear Line 2)
        # -----------------------------------
         # Renamed outputs slightly to avoid clashes (e.g., _s2 for Shear Line 2)
        (
            area_right_pile_s2, area_left_pile_s2, # Pile areas ('above' is 'right' for vertical)
            piles_right_s2, piles_left_s2,         # ('above' is 'right')
            intersected_area_right_s2, intersected_area_left_s2, # ('above' is 'right')
            intersected_piles_s2,
            total_reaction_above_2, total_reaction_below_2, # Keep original reaction var names ('above' is 'right')
            intersected_pile_geoms_s2 # Keep original geometry var name suffix
        ) = analyze_pile_distribution_with_reactions(
            polygon_vertices=pile_cap_vertices,
            line_type=one_way_shear_2_line_type,
            line_value=one_way_shear_2_line_value,
            pile_layout=pile_layout,
            pile_size=pile_size,
            pile_reactions=reactions
        )

        # --- Log Outputs ---
        calculation_log.append(f"\n=== Areas (Shear Section 2) ===")
        calculation_log.append(f"Total pile area to the right of line 2: {area_right_pile_s2:.2f} sq.ft.")
        calculation_log.append(f"Total pile area to the left of line 2: {area_left_pile_s2:.2f} sq.ft.")
        calculation_log.append(f"\n=== Piles (Shear Section 2) ===")
        calculation_log.append(f"Piles to the right of line 2: {piles_right_s2}")
        calculation_log.append(f"Piles to the left of line 2: {piles_left_s2}")
        calculation_log.append(f"\n=== Intersected Piles (Shear Section 2) ===")
        calculation_log.append(f"Intersected piles: {intersected_piles_s2}")
        calculation_log.append(f"Intersected areas to the right of line 2: {intersected_area_right_s2:.2f} sq.ft.")
        calculation_log.append(f"Intersected areas to the left of line 2: {intersected_area_left_s2:.2f} sq.ft.")
        calculation_log.append(f"\n=== Reactions (Shear Section 2) ===")
        calculation_log.append(f"Total reaction to the right of line 2 (Reaction Above 2): {total_reaction_above_2:.2f} kips")
        calculation_log.append(f"Total reaction to the left of line 2 (Reaction Below 2): {total_reaction_below_2:.2f} kips")
        # --- End Log ---
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # --- REVERTED TO ORIGINAL CALCULATION ---
        # The original script calculated net shear based on the "above" reaction directly
        one_way_shear_section_1 = total_reaction_above_1 - concrete_wt_B - soil_wt_B
        one_way_shear_section_2 = total_reaction_above_2 - concrete_wt_A - soil_wt_A
        # --- END REVERSION ---

        calculation_log.append(f"\n=== One-Way Shear (kips) ===")
        calculation_log.append(f"Net one-way shear at Section 1 (using Reaction Above 1): {one_way_shear_section_1:.2f} kips")
        calculation_log.append(f"Net one-way shear at Section 2 (using Reaction Above 2): {one_way_shear_section_2:.2f} kips")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        shear_capacity_1 = concrete_shear_capacity_simple(lambda_factor, concrete_strength_psi,
                                                        intersection_length_reg, pile_cap_shear_depth)

        # Original script had this check/fallback logic
        if shear_capacity_1 == 0 and intersection_length_reg == 0: # Add check for length=0
            intersection_length_mom1 = polygon_intersection(pile_cap_vertices, one_way_moment_1_line_type, one_way_moment_1_line_value) or 0
            calculation_log.append(f"Shear line 1 intersection zero, checking moment line 1 intersection: {intersection_length_mom1:.3f} ft")
            shear_capacity_1 = concrete_shear_capacity_simple(lambda_factor, concrete_strength_psi,
                                                            intersection_length_mom1, pile_cap_shear_depth)

        calculation_log.append(f"Design shear capacity φVc (kips) along Section 1: {shear_capacity_1:.3f}")


        shear_capacity_2 = concrete_shear_capacity_simple(lambda_factor, concrete_strength_psi,
                                                        intersection_length_ver, pile_cap_shear_depth)

        # Original script had this check/fallback logic
        if shear_capacity_2 == 0 and intersection_length_ver == 0: # Add check for length=0
            intersection_length_mom2 = polygon_intersection(pile_cap_vertices, one_way_moment_2_line_type, one_way_moment_2_line_value) or 0
            calculation_log.append(f"Shear line 2 intersection zero, checking moment line 2 intersection: {intersection_length_mom2:.3f} ft")
            shear_capacity_2 = concrete_shear_capacity_simple(lambda_factor, concrete_strength_psi,
                                                             intersection_length_mom2, pile_cap_shear_depth) # MISTAKE IN ORIGINAL SCRIPT? Used reg length here. Correcting to use intersection_length_mom2. IF USER WANTS EXACT BUGGY BEHAVIOR, CHANGE BACK to intersection_length_reg. Assuming user wants correct logic here.

        calculation_log.append(f"Design shear capacity φVc (kips) along Section 2: {shear_capacity_2:.3f}")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Section 1 (Moment Line 1): Function returns: area_above, area_below, first_area_moment_above, first_area_moment_below
        # Renaming vars _m1 for Moment Line 1
        area_above_m1, area_below_m1, area_moment_above_m1, area_moment_below_m1 = split_polygon_by_line(
            pile_cap_vertices, one_way_moment_1_line_type, one_way_moment_1_line_value
        )
        area_above_m1 = area_above_m1 or 0
        area_below_m1 = area_below_m1 or 0
        area_moment_above_m1 = area_moment_above_m1 or 0
        area_moment_below_m1 = area_moment_below_m1 or 0

        calculation_log.append(f"\n=== Moment Line 1 Split ===")
        calculation_log.append(f"Areas above the moment line 1: {area_above_m1:.2f} sq.ft.")
        calculation_log.append(f"Areas below the moment line 1: {area_below_m1:.2f} sq.ft.")
        calculation_log.append(f"First area moment above the line 1: {area_moment_above_m1:.2f} ft³")
        calculation_log.append(f"First area moment below the line 1: {area_moment_below_m1:.2f} ft³\n")

        # These variables correspond to Section 4 in the original output naming
        area_D = min(area_above_m1, area_below_m1)
        first_moment_area_D = min(area_moment_above_m1, area_moment_below_m1)
        calculation_log.append(f"Area D (Moment Section 1 - Min Area): {area_D:.2f} sq.ft.")
        calculation_log.append(f"First moment of area D (Moment Section 1 - Min Moment Area): {first_moment_area_D:.2f} ft³")

        # Section 2 (Moment Line 2): Function returns: area_above(right), area_below(left), first_area_moment_above(right), first_area_moment_below(left)
        # Renaming vars _m2 for Moment Line 2
        area_right_m2, area_left_m2, area_moment_right_m2, area_moment_left_m2 = split_polygon_by_line(
            pile_cap_vertices, one_way_moment_2_line_type, one_way_moment_2_line_value
        )
        area_right_m2 = area_right_m2 or 0
        area_left_m2 = area_left_m2 or 0
        area_moment_right_m2 = area_moment_right_m2 or 0
        area_moment_left_m2 = area_moment_left_m2 or 0


        calculation_log.append(f"\n=== Moment Line 2 Split ===")
        calculation_log.append(f"Areas right of the moment line 2: {area_right_m2:.2f} sq.ft.")
        calculation_log.append(f"Areas left the moment line 2: {area_left_m2:.2f} sq.ft.")
        calculation_log.append(f"First area moment right of the line 2: {area_moment_right_m2:.2f} ft³")
        calculation_log.append(f"First area moment left of the line 2: {area_moment_left_m2:.2f} ft³\n")

        # These variables correspond to Section 3 in the original output naming
        area_C = min(area_right_m2, area_left_m2)
        first_moment_area_C = min(area_moment_right_m2, area_moment_left_m2)
        calculation_log.append(f"Area C (Moment Section 2 - Min Area): {area_C:.2f} sq.ft.")
        calculation_log.append(f"First moment of area C (Moment Section 2 - Min Moment Area): {first_moment_area_C:.2f} ft³")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Section 1 (Moment D corresponds to Moment Line 1)
        concrete_moment_D, soil_moment_D = calculate_concrete_soil_weight(
             first_moment_area_D, pile_cap_thickness, soil_depth_above, concrete_density, soil_density
        )
        calculation_log.append(f"\n=== Concrete and Soil Moments (Section D / Line 1) ===")
        calculation_log.append(f"Concrete moment (over first moment area D {first_moment_area_D:.3f} ft³): {concrete_moment_D:.3f} kip-ft")
        calculation_log.append(f"Soil moment (over first moment area D {first_moment_area_D:.3f} ft³): {soil_moment_D:.3f} kip-ft")


        # Section 2 (Moment C corresponds to Moment Line 2)
        concrete_moment_C, soil_moment_C = calculate_concrete_soil_weight(
            first_moment_area_C, pile_cap_thickness, soil_depth_above, concrete_density, soil_density
        )
        calculation_log.append(f"\n=== Concrete and Soil Moments (Section C / Line 2) ===")
        calculation_log.append(f"Concrete moment (over first moment area C {first_moment_area_C:.3f} ft³): {concrete_moment_C:.3f} kip-ft")
        calculation_log.append(f"Soil moment (over first moment area C {first_moment_area_C:.3f} ft³): {soil_moment_C:.3f} kip-ft")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # -----------------------------------
        # Section 1: Analyze pile distribution with reactions (Moment Line 1)
        # -----------------------------------
        # Renaming output vars _m1 for Moment Line 1
        (
            area_above_pile_m1, area_below_pile_m1, # Pile areas
            piles_above_m1, piles_below_m1,
            intersected_area_above_m1, intersected_area_below_m1,
            intersected_piles_m1,
            total_reaction_above_4, total_reaction_below_4, # KEEP ORIGINAL REACTION VAR NAMES (used 4 in original script here)
            intersected_pile_geoms_4 # KEEP ORIGINAL GEOM VAR NAME (used 4 in original script here)
        ) = analyze_pile_distribution_with_reactions(
            polygon_vertices=pile_cap_vertices,
            line_type=one_way_moment_1_line_type, # Moment Line 1
            line_value=one_way_moment_1_line_value,
            pile_layout=pile_layout,
            pile_size=pile_size,
            pile_reactions=reactions
        )

        # -----------------------------------
        # Calculate moments about section line (Moment Line 1)
        # -----------------------------------
        # Use the original variable names from the corresponding `calculate_moments...` call
        moment_above_4, moment_below_4 = calculate_moments_about_section_line(
            line_type=one_way_moment_1_line_type,
            line_value=one_way_moment_1_line_value,
            pile_layout=pile_layout,
            pile_reactions=reactions,
            piles_above=piles_above_m1, # Use consistent pile lists
            piles_below=piles_below_m1, # Use consistent pile lists
            intersected_pile_geoms=intersected_pile_geoms_4 # Use the geom output matching reaction names
        )

        # --- Log Outputs ---
        calculation_log.append(f"\n=== Pile Analysis (Moment Line 1 / Section 4) ===")
        # calculation_log.append(f"Total pile area above line 1: {area_above_pile_m1:.2f} sq.ft.") # Redundant logging?
        # calculation_log.append(f"Total pile area below line 1: {area_below_pile_m1:.2f} sq.ft.")
        calculation_log.append(f"Piles above line 1: {piles_above_m1}")
        calculation_log.append(f"Piles below line 1: {piles_below_m1}")
        # calculation_log.append(f"Intersected piles: {intersected_piles_m1}") # Redundant logging?
        # calculation_log.append(f"Intersected areas above line 1: {intersected_area_above_m1:.2f} sq.ft.")
        # calculation_log.append(f"Intersected areas below line 1: {intersected_area_below_m1:.2f} sq.ft.")
        calculation_log.append(f"Total reaction above line 1 (Reaction Above 4): {total_reaction_above_4:.2f} kips")
        calculation_log.append(f"Total reaction below line 1 (Reaction Below 4): {total_reaction_below_4:.2f} kips")
        calculation_log.append(f"Moment above line (Moment Above 4): {moment_above_4:.2f} kip-ft")
        calculation_log.append(f"Moment below line (Moment Below 4): {moment_below_4:.2f} kip-ft")
        # --- End Log ---


        # -----------------------------------
        # Section 2: Analyze pile distribution with reactions (Moment Line 2)
        # -----------------------------------
        # Renaming output vars _m2 for Moment Line 2
        (
            area_right_pile_m2, area_left_pile_m2, # Pile areas ('above' = right)
            piles_right_m2, piles_left_m2, # ('above' = right)
            intersected_area_right_m2, intersected_area_left_m2, # ('above' = right)
            intersected_piles_m2,
            total_reaction_above_3, total_reaction_below_3, # KEEP ORIGINAL REACTION VAR NAMES (used 3 in original script here)
            intersected_pile_geoms_3 # KEEP ORIGINAL GEOM VAR NAME (used 3 in original script here)
        ) = analyze_pile_distribution_with_reactions(
            polygon_vertices=pile_cap_vertices,
            line_type=one_way_moment_2_line_type, # Moment Line 2
            line_value=one_way_moment_2_line_value,
            pile_layout=pile_layout,
            pile_size=pile_size,
            pile_reactions=reactions
        )

        # -----------------------------------
        # Calculate moments about section line (Moment Line 2)
        # -----------------------------------
        # Use the original variable names from the corresponding `calculate_moments...` call
        moment_above_3, moment_below_3 = calculate_moments_about_section_line( # Note: Original script named these 'above'/'below' even for vertical line
            line_type=one_way_moment_2_line_type,
            line_value=one_way_moment_2_line_value,
            pile_layout=pile_layout,
            pile_reactions=reactions,
            piles_above=piles_right_m2, # Use consistent pile lists ('above' means right here)
            piles_below=piles_left_m2,  # Use consistent pile lists ('below' means left here)
            intersected_pile_geoms=intersected_pile_geoms_3 # Use the geom output matching reaction names
        )

        # --- Log Outputs ---
        calculation_log.append(f"\n=== Pile Analysis (Moment Line 2 / Section 3) ===")
        # calculation_log.append(f"Total pile area right of line 2: {area_right_pile_m2:.2f} sq.ft.") # Redundant?
        # calculation_log.append(f"Total pile area left of line 2: {area_left_pile_m2:.2f} sq.ft.")
        calculation_log.append(f"Piles right of line 2: {piles_right_m2}")
        calculation_log.append(f"Piles left of line 2: {piles_left_m2}")
        # calculation_log.append(f"Intersected piles: {intersected_piles_m2}") # Redundant?
        # calculation_log.append(f"Intersected areas right of line 2: {intersected_area_right_m2:.2f} sq.ft.")
        # calculation_log.append(f"Intersected areas left of line 2: {intersected_area_left_m2:.2f} sq.ft.")
        calculation_log.append(f"Total reaction right of line 2 (Reaction Above 3): {total_reaction_above_3:.2f} kips")
        calculation_log.append(f"Total reaction left of line 2 (Reaction Below 3): {total_reaction_below_3:.2f} kips")
        calculation_log.append(f"Moment right of line (Moment Above 3): {moment_above_3:.2f} kip-ft") # Note original naming convention 'above'
        calculation_log.append(f"Moment left of line (Moment Below 3): {moment_below_3:.2f} kip-ft") # Note original naming convention 'below'
        # --- End Log ---
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # --- REVERTED TO ORIGINAL CALCULATION ---
        # The original script calculated net moment based on the "above" moment directly
        one_way_moment_section_3 = moment_above_3 - concrete_moment_C - soil_moment_C
        one_way_moment_section_4 = moment_above_4 - concrete_moment_D - soil_moment_D
        # --- END REVERSION ---

        calculation_log.append(f"\n=== One-Way Moment (kip-ft) ===")
        # Note: Original print statements had Section 1 = moment_4, Section 2 = moment_3
        calculation_log.append(f"Net one-way moment at Section 1 (using Moment Above 4): {one_way_moment_section_4:.2f} kip-ft")
        calculation_log.append(f"Net one-way moment at Section 2 (using Moment Above 3): {one_way_moment_section_3:.2f} kip-ft")


        area_of_steel_section_1 = calculate_area_of_steel(one_way_moment_section_4, pile_cap_shear_depth)
        area_of_steel_section_2 = calculate_area_of_steel(one_way_moment_section_3, pile_cap_shear_depth)
        calculation_log.append(f"Area of steel for Section 1 (from Moment 4): {area_of_steel_section_1:.3f} sq. in.")
        calculation_log.append(f"Area of steel for Section 2 (from Moment 3): {area_of_steel_section_2:.3f} sq. in.")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        # Total reaction outside punching shear polygon
        result = analyze_shear_polygon_reactions( # Keep original var name 'result'
            polygon_vertices=pile_cap_vertices,
            pile_layout=pile_layout,
            pile_size=pile_size,
            column_centroid=column_centroid,
            column_size=column_size,
            shear_depth=pile_cap_shear_depth,
            pile_reactions=reactions
        )

        # Unpack using original variable names
        (
            total_reaction_outside,
            shear_polygon_coords,
            shear_polygon_perimeter,
            inside_piles,
            outside_piles,
            intersecting_piles
        ) = result

        calculation_log.append(f"\n=== Punching Shear Analysis ===")
        calculation_log.append(f"Total reaction outside shear polygon: {total_reaction_outside:.2f} kip")
        calculation_log.append(f"Shear polygon perimeter length b0: {shear_polygon_perimeter:.2f} ft")
        # calculation_log.append(f"Shear polygon coordinates: {shear_polygon_coords}\n") # Can be long
        calculation_log.append(f"Piles fully inside shear polygon: {inside_piles}")
        calculation_log.append(f"Piles fully outside shear polygon: {outside_piles}")
        calculation_log.append(f"Piles intersecting shear polygon: {intersecting_piles}")
        ### END ORIGINAL BLOCK ###


        ### START ORIGINAL BLOCK ###
        Vc = calculate_punching_shear_capacity(
            column_size,
            concrete_strength_psi ,
            pile_cap_shear_depth*12, # d in inches
            shear_polygon_perimeter*12, # b0 in inches
            lambda_factor,
            column_location,
        )
        calculation_log.append(f"Punching shear capacity φVc = {Vc:.3f} kips")

        # Avoid division by zero if Vc is somehow zero
        utilization_ratio = (total_reaction_outside / Vc * 100) if Vc > 0 else float('inf')
        calculation_log.append(f"Punching shear utilization ratio = {utilization_ratio:.2f} %")
        ### END ORIGINAL BLOCK ###

        # Final Summary Print Block (Mimicked by results_data for display)
        calculation_log.append("\n--- Final Results Summary ---")
        calculation_log.append(f"One-Way Shear Section 1: Vu={one_way_shear_section_1:.2f} kips | φVc={shear_capacity_1:.2f} kips")
        calculation_log.append(f"One-Way Shear Section 2: Vu={one_way_shear_section_2:.2f} kips | φVc={shear_capacity_2:.2f} kips")
        calculation_log.append(f"One-Way Moment Section 1 (from M4): Mu={one_way_moment_section_4:.2f} kip-ft | As={area_of_steel_section_1:.3f} in²")
        calculation_log.append(f"One-Way Moment Section 2 (from M3): Mu={one_way_moment_section_3:.2f} kip-ft | As={area_of_steel_section_2:.3f} in²")
        calculation_log.append(f"Two-Way Shear: Vu={total_reaction_outside:.2f} kip | φVc={Vc:.3f} kips | Util={utilization_ratio:.2f} %")


        # --- Prepare Results for Display ---
        results_data = {
            'input_summary': { # Store processed inputs for summary display
                 'column_size': column_size, 'column_eccentricity': column_eccentricity, 'column_centroid': column_centroid,
                 'pile_cap_thickness': pile_cap_thickness, 'pile_cap_shear_depth': pile_cap_shear_depth,
                 'pile_embedment': pile_embedment, 'soil_depth_above': soil_depth_above, 'soil_density': soil_density,
                 'concrete_density': concrete_density, 'concrete_strength_psi': concrete_strength_psi,
                 'assumed_cover_in': assumed_cover_in, # New input
                 'assumed_bar_dia_in': assumed_bar_dia_in, # New input
                 'column_location': column_location, 'lambda_factor': lambda_factor, 'pile_shape': pile_shape,
                 'pile_size': pile_size, 'max_pile_compression': max_pile_compression, 'max_pile_tension': max_pile_tension,
                 'Fx': Fx, 'Fy': Fy, 'Fz': Fz, 'Mx': Mx, 'My': My, 'num_piles': num_piles,
                 'pile_cap_vertices': pile_cap_vertices.tolist() # Convert numpy array for easier display
            },
            'one_way_shear': {
                'section_1_net': one_way_shear_section_1, 'section_2_net': one_way_shear_section_2,
                'section_1_capacity': shear_capacity_1, 'section_2_capacity': shear_capacity_2,
                'section_1_util': (one_way_shear_section_1 / shear_capacity_1 * 100) if shear_capacity_1 > 0 else float('inf'),
                'section_2_util': (one_way_shear_section_2 / shear_capacity_2 * 100) if shear_capacity_2 > 0 else float('inf')
            },
            'one_way_moment': {
                 # Use the original script's naming convention for output sections
                'section_1_net': one_way_moment_section_4, # Section 1 in output uses moment_4
                'section_2_net': one_way_moment_section_3, # Section 2 in output uses moment_3
                'section_1_steel': area_of_steel_section_1, # Steel for moment_4
                'section_2_steel': area_of_steel_section_2  # Steel for moment_3
            },
            'punching_shear': {
                'reaction_outside': total_reaction_outside, 'capacity': Vc, 'utilization': utilization_ratio,
                'perimeter': shear_polygon_perimeter, #'coords': shear_polygon_coords # Coords can be long, maybe omit from summary
                'inside_piles': inside_piles, 'outside_piles': outside_piles, 'intersecting_piles': intersecting_piles
            }
        }

        # Convert Pile Reactions DataFrame to HTML
        pile_reactions_html = results_df.to_html(classes='table table-striped table-sm', index=False, float_format='{:.2f}'.format) if num_piles > 0 else "<p>No piles defined.</p>"

        # --- Generate Plot ---
        # Ensure the plot function uses the variables consistent with the original final summary
        fig, ax = plot_foundation_analysis(
             pile_cap_vertices, pile_layout, pile_size, reactions, column_size, column_centroid,
             pile_cap_thickness, pile_cap_shear_depth,
             one_way_shear_1_line_type, one_way_shear_1_line_value, one_way_shear_2_line_type, one_way_shear_2_line_value,
             one_way_moment_1_line_type, one_way_moment_1_line_value, one_way_moment_2_line_type, one_way_moment_2_line_value,
             one_way_shear_section_1, one_way_shear_section_2, # Vu values
             one_way_moment_section_4, one_way_moment_section_3, # Mu values (Note order for consistency)
             area_of_steel_section_1, area_of_steel_section_2, # As values (Note order for consistency)
             shear_polygon_coords, total_reaction_outside, # Punching Vu
             shear_capacity_1, shear_capacity_2, # 1-way Vc values
             Vc, utilization_ratio # Punching Vc and Util
        )

        # Save plot to memory
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        calculation_log.append("Plot generated successfully.")
        # --- End of Calculation Code ---

    except ValueError as e:
        flash(f"Input Error: {e}", "danger")
        app.logger.warning(f"Input Validation Error: {e}") # Log validation errors
        calculation_log.append(f"ERROR: Input Error - {e}")
        # Return form with submitted data and error
        return render_template('index.html', defaults=defaults, results=None, form_data=form_data, warnings=warnings, calculation_log=calculation_log) # Pass log even on error
    except Exception as e:
        flash(f"An unexpected error occurred during calculation: {e}", "danger")
        app.logger.error(f"Calculation Error: {e}", exc_info=True) # Log the full error stack trace for debugging
        calculation_log.append(f"FATAL ERROR during calculation: {e}")
        # Return form with submitted data and error
        return render_template('index.html', defaults=defaults, results=None, form_data=form_data, warnings=warnings, calculation_log=calculation_log) # Pass log even on error

    # --- Render template with results ---
    calculation_log.append("--- Calculation Complete ---")
    return render_template('index.html',
                           defaults=defaults,
                           results=results_data,
                           pile_reactions_html=pile_reactions_html,
                           plot_url=plot_url,
                           form_data=form_data, # Pass submitted data back to repopulate form
                           warnings=warnings, # Pass warnings
                           calculation_log=calculation_log) # Pass calculation log

if __name__ == '__main__':
    # Basic logging configuration
    logging.basicConfig(level=logging.INFO) # Log INFO level messages and above
    # To log DEBUG level (more verbose), change to logging.DEBUG
    # logging.basicConfig(level=logging.DEBUG)

    app.run(host="0.0.0.0", port=5000, debug=True)  # debug=True is helpful during development, REMOVE for production