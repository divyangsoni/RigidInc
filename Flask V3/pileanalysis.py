import pandas as pd
from shapely.geometry import Polygon, LineString, Point # Keep needed imports
# from shapely.ops import split # Keep if needed
import numpy as np
import traceback # For detailed error logging
import matplotlib.pyplot as plt # Needed for plt.show() in example

# --- Import your custom modules (Ensure they are accessible) ---
try:
    from pile_distribution_analyzer import (analyze_pile_distribution_with_reactions, calculate_moments_about_section_line)
    from weight_calculator import calculate_concrete_soil_weight
    from shear_polygon_analysis import analyze_shear_polygon_reactions
    from polygon_splitter import split_polygon_by_line, polygon_intersection
    from design import calculate_area_of_steel, calculate_punching_shear_capacity, concrete_shear_capacity_simple
    from visualization import plot_foundation_analysis # <<< ADDED IMPORT
except ImportError as ie:
    print(f"Error importing custom modules: {ie}")
    # In a real app, proper error handling/logging is needed here
    # Make sure visualization.py is in the same directory or Python path
    raise # Re-raise for Flask to catch if running in that context

# --- Default Values (Internal to the module) ---
# Default numpy arrays (used if parsing fails or input is empty)
default_pile_layout_np = np.array([
    [3.25, 4.5], [8.083, 4.5], [12.9167, 4.5], [17.75, 4.5], [22.583, 4.5], [27.4167, 4.5], [32.25, 4.5], [37.083, 4.5], [41.9167, 4.5], [46.75, 4.5],
    [3.25, 9], [8.083, 9], [12.9167, 9], [17.75, 9], [22.583, 9], [27.4167, 9], [32.25, 9], [37.083, 9], [41.9167, 9], [46.75, 9],
    [3.25, 13.5], [8.083, 13.5], [12.9167, 13.5], [17.75, 13.5], [22.583, 13.5], [27.4167, 13.5], [32.25, 13.5], [37.083, 13.5], [41.9167, 13.5], [46.75, 13.5],
])
default_pile_cap_vertices_np = np.array([
    [0+1, 0+2.5], [50-1, 0+2.5], [50-1, 18-2.5], [0+1, 18-2.5]
])

# Defaults dictionary (used internally for parsing)
internal_defaults = {
    "column_size": "38.66, 1",
    "column_eccentricity": "0,0",
    "column_centroid": "25,9",
    "pile_cap_thickness": str(int((3 + 4/12)*1000)/1000),
    "pile_embedment": "1.0",
    "soil_depth_above": "2.0",
    "soil_density": "0.0",
    "concrete_density": "0.0",
    "concrete_strength_psi": "3000",
    "column_location": "interior",
    "lambda_factor": "1.0",
    "pile_shape": "square",
    "pile_size": "0.8863",
    "max_pile_compression": "120.0",
    "max_pile_tension": "120.0",
    "Fx": "0.0",
    "Fy": "0.0",
    "Fz": str(-30 * 120 * 1.29),
    "Mx": "0.0",
    "My": "0.0",
    "pile_layout_str": '\n'.join([f"{r[0]}, {r[1]}" for r in default_pile_layout_np]), # Default string format
    "pile_cap_shape_type": "rectangle",
    "pile_cap_vertices_str": '\n'.join([f"{r[0]}, {r[1]}" for r in default_pile_cap_vertices_np]), # Default string format
    "rect_bottom_left": "1, 2.5",
    "rect_top_right": "49,15.5",
    "assumed_cover_in": "3.0",
    "assumed_bar_dia_in": "10.0"
}

# --- Internal Parsing Helper Functions ---
def _parse_coord_pair(input_str, default_tuple_str):
    """Parses 'x, y' string, returns tuple(float, float). Raises ValueError on error."""
    val = input_str.strip() if input_str else None
    if not val:
        default_val = default_tuple_str
    else:
        default_val = val # Use input if provided

    try:
        # Handle case where default_val might already be a tuple or list
        if isinstance(default_val, (tuple, list)) and len(default_val) == 2:
            parts_to_parse = default_val
        elif isinstance(default_val, str):
            parts_to_parse = default_val.split(',')
        else:
            raise ValueError("Invalid format for coordinate pair source.")

        parts = [float(v.strip()) for v in parts_to_parse]
        if len(parts) == 2:
            return tuple(parts)
        else:
            raise ValueError("Input string must contain exactly two comma-separated numbers.")
    except ValueError:
        raise ValueError(f"Invalid coordinate pair format: '{default_val}'. Expected 'x, y'.")
    except Exception as e:
        raise ValueError(f"Unexpected error parsing coordinate pair '{default_val}': {e}")

def _parse_multi_line_coords(input_str, default_np_array):
    """Parses multi-line 'x, y' string, returns numpy array. Raises ValueError on error."""
    val = input_str.strip() if input_str else None
    if not val:
        return default_np_array # Use the default numpy array if input is empty

    lines = val.split('\n')
    coords_list = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue # Skip empty lines
        try:
            parts = [float(v.strip()) for v in line.split(',')]
            if len(parts) == 2:
                coords_list.append(parts)
            else:
                raise ValueError(f"Line {i+1} ('{line}') does not contain exactly two comma-separated numbers.")
        except ValueError as e:
            raise ValueError(f"Invalid format on line {i+1}: '{line}'. Expected 'x, y'. Original error: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error parsing line {i+1} ('{line}'): {e}")

    if not coords_list: # Handle case where input was just whitespace or empty lines
        return default_np_array

    return np.array(coords_list)

# --- Main Analysis Function (Modified Signature) ---
def perform_pile_cap_analysis(form_data: dict):
    """
    Performs the pile cap analysis based on raw form data dictionary.
    Handles parsing, defaulting, calculations, and visualization internally.

    Args:
        form_data (dict): A dictionary containing the raw string inputs,
                          typically from request.form.to_dict().

    Returns:
        dict: A dictionary containing the calculated results, intermediate values,
              and potentially matplotlib figure and axes objects ('figure', 'axes').
              Returns {'status': 'error', 'message': str} on failure.
              'figure' and 'axes' will be None if visualization fails.
    """
    results = {}
    warnings = []
    calculation_steps = [] # To store intermediate print-like messages if desired
    parsed_inputs = {} # Store parsed values for potential echo/debugging

    # Initialize fig and ax to None
    fig = None
    ax = None
    results['figure'] = fig
    results['axes'] = ax

    try:
        calculation_steps.append("--- Parsing Inputs ---")

        # --- Parse and Validate Inputs (using internal defaults) ---
        parsed_inputs['column_size'] = column_size = _parse_coord_pair(form_data.get('column_size'), internal_defaults['column_size'])
        parsed_inputs['column_eccentricity'] = column_eccentricity = _parse_coord_pair(form_data.get('column_eccentricity'), internal_defaults['column_eccentricity'])
        parsed_inputs['column_centroid'] = column_centroid = _parse_coord_pair(form_data.get('column_centroid'), internal_defaults['column_centroid'])

        parsed_inputs['pile_cap_thickness'] = pile_cap_thickness = float(form_data.get('pile_cap_thickness') or internal_defaults['pile_cap_thickness'])
        parsed_inputs['pile_embedment'] = pile_embedment = float(form_data.get('pile_embedment') or internal_defaults['pile_embedment'])
        parsed_inputs['soil_depth_above'] = soil_depth_above = float(form_data.get('soil_depth_above') or internal_defaults['soil_depth_above'])
        parsed_inputs['soil_density'] = soil_density = float(form_data.get('soil_density') or internal_defaults['soil_density'])
        parsed_inputs['concrete_density'] = concrete_density = float(form_data.get('concrete_density') or internal_defaults['concrete_density'])
        parsed_inputs['concrete_strength_psi'] = concrete_strength_psi = int(form_data.get('concrete_strength_psi') or internal_defaults['concrete_strength_psi'])
        parsed_inputs['lambda_factor'] = lambda_factor = float(form_data.get('lambda_factor') or internal_defaults['lambda_factor'])
        parsed_inputs['pile_size'] = pile_size = float(form_data.get('pile_size') or internal_defaults['pile_size'])
        parsed_inputs['max_pile_compression'] = max_pile_compression = float(form_data.get('max_pile_compression') or internal_defaults['max_pile_compression'])
        parsed_inputs['max_pile_tension'] = max_pile_tension = float(form_data.get('max_pile_tension') or internal_defaults['max_pile_tension'])
        parsed_inputs['Fx'] = Fx = float(form_data.get('Fx') or internal_defaults['Fx'])
        parsed_inputs['Fy'] = Fy = float(form_data.get('Fy') or internal_defaults['Fy'])
        parsed_inputs['Fz'] = Fz = float(form_data.get('Fz') or internal_defaults['Fz'])
        parsed_inputs['Mx'] = Mx = float(form_data.get('Mx') or internal_defaults['Mx'])
        parsed_inputs['My'] = My = float(form_data.get('My') or internal_defaults['My'])
        parsed_inputs['assumed_cover_in'] = assumed_cover_in = float(form_data.get('assumed_cover_in') or internal_defaults['assumed_cover_in'])
        parsed_inputs['assumed_bar_dia_in'] = assumed_bar_dia_in = float(form_data.get('assumed_bar_dia_in') or internal_defaults['assumed_bar_dia_in'])

        column_location = form_data.get('column_location') or internal_defaults['column_location']
        if column_location not in ["interior", "edge", "corner"]:
            warnings.append(f"Invalid column location '{column_location}', using default '{internal_defaults['column_location']}'.")
            column_location = internal_defaults['column_location']
        parsed_inputs['column_location'] = column_location

        pile_shape = form_data.get('pile_shape') or internal_defaults['pile_shape']
        if pile_shape not in ["square", "circular"]:
            warnings.append(f"Invalid pile shape '{pile_shape}', using default '{internal_defaults['pile_shape']}'.")
            pile_shape = internal_defaults['pile_shape']
        parsed_inputs['pile_shape'] = pile_shape

        pile_layout_str = form_data.get('pile_layout_str', '').strip()
        pile_layout = _parse_multi_line_coords(pile_layout_str, default_pile_layout_np)
        if pile_layout is default_pile_layout_np and pile_layout_str:
              warnings.append("Failed to parse pile layout, using default layout.")
        elif not pile_layout_str:
              calculation_steps.append("Empty pile layout input, using default layout.")
        parsed_inputs['num_piles'] = len(pile_layout)

        pile_cap_shape_type = form_data.get('pile_cap_shape_type') or internal_defaults['pile_cap_shape_type']
        pile_cap_vertices = None
        calculation_steps.append(f"Pile cap shape type selected: {pile_cap_shape_type}")

        if pile_cap_shape_type == 'rectangle':
            rect_bl_str = form_data.get('rect_bottom_left') # Use get directly
            rect_tr_str = form_data.get('rect_top_right')
            bl_coords = _parse_coord_pair(rect_bl_str, internal_defaults['rect_bottom_left'])
            tr_coords = _parse_coord_pair(rect_tr_str, internal_defaults['rect_top_right'])
            bl_x, bl_y = bl_coords
            tr_x, tr_y = tr_coords
            if tr_x > bl_x and tr_y > bl_y:
                pile_cap_vertices = np.array([[bl_x, bl_y], [tr_x, bl_y], [tr_x, tr_y], [bl_x, tr_y]])
                calculation_steps.append("Defined rectangular pile cap vertices.")
            else:
                raise ValueError("Rectangle top-right coordinates must be greater than bottom-left.")
        elif pile_cap_shape_type == 'polygon':
            vertices_str = form_data.get('pile_cap_vertices_str', '').strip()
            pile_cap_vertices = _parse_multi_line_coords(vertices_str, default_pile_cap_vertices_np)
            if pile_cap_vertices is default_pile_cap_vertices_np and vertices_str:
                 warnings.append("Failed to parse polygon vertices, using default shape.")
            elif not vertices_str:
                 calculation_steps.append("Empty polygon vertices input, using default shape.")
            if len(pile_cap_vertices) < 3:
                raise ValueError("Polygon must have at least 3 vertices.")
            calculation_steps.append(f"Parsed {len(pile_cap_vertices)} polygon vertices.")
        else:
            raise ValueError("Invalid pile cap shape type selected.")

        parsed_inputs['pile_cap_vertices'] = pile_cap_vertices.tolist() # Store as list

        calculation_steps.append("--- Input Parsing Complete ---")
        results['parsed_inputs'] = parsed_inputs # Store parsed inputs for potential echo

        # --- Start Core Calculation Blocks (Identical to previous version) ---
        num_piles = len(pile_layout)
        results['num_piles'] = num_piles

        # <<< BLOCK 1: Final Calculations (Shear Depth) >>>
        pile_cap_shear_depth = pile_cap_thickness - (assumed_cover_in / 12.0) - (assumed_bar_dia_in / 8.0 / 12.0) - (assumed_bar_dia_in / 8.0 / 2.0 / 12.0) # ft
        results['pile_cap_shear_depth_ft'] = pile_cap_shear_depth
        calculation_steps.append(f"Calculated Effective Shear Depth (d): {pile_cap_shear_depth:.3f} ft")
        calculation_steps.append("Block 1: Shear Depth Calculation Complete.")

        # <<< BLOCK 2: Define Shear Sections >>>
        offset_1 = column_centroid[1] + column_size[1] / 2 + pile_cap_shear_depth
        one_way_shear_1_line_type = 'regular'
        one_way_shear_1_line_value = (0, offset_1)
        offset_2 = column_centroid[0] + column_size[0] / 2 + pile_cap_shear_depth
        one_way_shear_2_line_type = 'vertical'
        one_way_shear_2_line_value = (offset_2,)
        calculation_steps.append("Block 2: Shear Section Definition Complete.")

        # <<< BLOCK 3: Define Moment Sections & Validate Piles >>>
        offset_3 = column_centroid[1] + column_size[1] / 2
        one_way_moment_1_line_type = 'regular'
        one_way_moment_1_line_value = (0, offset_3)
        offset_4 = column_centroid[0] + column_size[0] / 2
        one_way_moment_2_line_type = 'vertical'
        one_way_moment_2_line_value = (offset_4,)
        try:
            pile_cap_poly = Polygon(pile_cap_vertices)
            if not pile_cap_poly.is_valid:
                 raise ValueError(f"Invalid pile cap polygon geometry (Self-intersection?). Vertices: {pile_cap_vertices.tolist()}")
        except Exception as e:
            raise ValueError(f"Invalid pile cap vertices definition for geometry checks: {e}")
        for i, (px, py) in enumerate(pile_layout):
            if not pile_cap_poly.buffer(1e-9).contains(Point(px, py)):
                 if not pile_cap_poly.boundary.buffer(1e-9).contains(Point(px, py)):
                      raise ValueError(f"Pile {i+1} at ({px:.3f}, {py:.3f}) is outside the defined pile cap boundary!")
                 else:
                      calculation_steps.append(f"Note: Pile {i+1} at ({px:.3f}, {py:.3f}) lies on the pile cap boundary.")
        calculation_steps.append("Pile location check passed.")
        calculation_steps.append("Block 3: Moment Section Definition & Pile Validation Complete.")

        # <<< BLOCK 4: Compute Self-weight >>>
        pile_cap_poly_area = pile_cap_poly.area
        pile_cap_volume = pile_cap_poly_area * pile_cap_thickness
        pile_cap_weight = (pile_cap_volume * concrete_density) / 1000 if concrete_density else 0
        soil_volume = pile_cap_poly_area * soil_depth_above
        soil_weight = (soil_volume * soil_density) / 1000 if soil_density else 0
        if pile_shape == 'square': pile_single_area = pile_size**2
        elif pile_shape == 'circular': pile_single_area = np.pi * (pile_size / 2)**2
        else: pile_single_area = pile_size**2
        pile_embedment_weight = (pile_embedment * pile_single_area * num_piles * concrete_density) / 1000 if concrete_density and num_piles > 0 else 0
        total_weight = pile_cap_weight + soil_weight - pile_embedment_weight
        results['total_self_weight_kips'] = total_weight
        calculation_steps.append(f"Calculated Total Self-Weight (incl. soil, less piles): {total_weight:.2f} kips")
        calculation_steps.append("Block 4: Self-Weight Calculation Complete.")

        # <<< BLOCK 5: Compute Pile Centroid >>>
        pile_centroid = np.mean(pile_layout, axis=0) if num_piles > 0 else np.array([pile_cap_poly.centroid.x, pile_cap_poly.centroid.y])
        results['pile_centroid'] = pile_centroid.tolist()
        calculation_steps.append(f"Pile Centroid: ({pile_centroid[0]:.3f}, {pile_centroid[1]:.3f})")
        calculation_steps.append("Block 5: Pile Centroid Calculation Complete.")

        # <<< BLOCK 6: Adjust Loads for Eccentricity >>>
        Mx_adj = Mx + Fz * column_eccentricity[1]
        My_adj = My - Fz * column_eccentricity[0]
        results['adjusted_Mx_kip_ft'] = Mx_adj
        results['adjusted_My_kip_ft'] = My_adj
        calculation_steps.append(f"Adjusted Moments: Mx_adj={Mx_adj:.2f} kip-ft, My_adj={My_adj:.2f} kip-ft")
        calculation_steps.append("Block 6: Load Adjustment Complete.")

        # <<< BLOCK 7: Solve for Pile Reactions >>>
        A = np.zeros((3, num_piles)); b = np.array([-Fz + total_weight, -Mx_adj, My_adj])
        reactions = np.zeros((num_piles, 1))
        if num_piles > 0:
            if num_piles < 3: warnings.append(f"WARNING: Only {num_piles} pile(s) defined. System may be unstable.")
            if num_piles >= 2:
                if np.allclose(pile_layout[:, 0], pile_layout[0, 0]): warnings.append("WARNING: Piles vertically collinear.")
                if np.allclose(pile_layout[:, 1], pile_layout[0, 1]): warnings.append("WARNING: Piles horizontally collinear.")
            for i, (px, py) in enumerate(pile_layout):
                A[0, i] = 1; A[1, i] = py - pile_centroid[1]; A[2, i] = px - pile_centroid[0]
            try:
                reactions, residuals, rank, s = np.linalg.lstsq(A, b.reshape(-1, 1), rcond=None)
                calculation_steps.append(f"Solved for pile reactions using least squares (Rank: {rank}).")
                if rank < 3 and num_piles >= 3: warnings.append(f"WARNING: Pile reaction system may be underdetermined (Rank {rank} < 3).")
            except np.linalg.LinAlgError as e: raise ValueError(f"Could not solve for pile reactions (LinAlgError): {e}.")
        else:
            calculation_steps.append("No piles defined, skipping reaction calculation.")
            if abs(-Fz + total_weight) > 1e-6: warnings.append("WARNING: No piles, vertical load Fz not balanced by self-weight.")
        results['pile_reactions_kips'] = reactions.flatten().tolist() # Store as list
        # KEEP 'reactions' as numpy array for plotting function
        calculation_steps.append("Block 7: Reaction Calculation Complete.")

        # <<< BLOCK 8: Check Pile Capacities >>>
        for i, R in enumerate(reactions.flatten()):
            if R > max_pile_compression: warnings.append(f"WARNING: Pile {i+1} reaction exceeds max compression ({R:.2f} > {max_pile_compression:.2f} kips)")
            if R < -max_pile_tension: warnings.append(f"WARNING: Pile {i+1} reaction exceeds max tension ({abs(R):.2f} > {max_pile_tension:.2f} kips)")
        calculation_steps.append("Block 8: Pile Capacity Check Complete.")

        # <<< BLOCK 9: Display Results (DataFrame for logging/optional return) >>>
        results_df = pd.DataFrame({
            "Pile": np.arange(1, num_piles + 1),
            "X (ft)": pile_layout[:, 0] if num_piles > 0 else [],
            "Y (ft)": pile_layout[:, 1] if num_piles > 0 else [],
            "Reaction (kips)": reactions.flatten() if num_piles > 0 else []
        })
        results['pile_reactions_dataframe_str'] = results_df.to_string(index=False, float_format='{:.2f}'.format)
        calculation_steps.append("\nPile Reactions Table Generated.")
        calculation_steps.append("Block 9: Reaction Table Generation Complete.")

        # <<< BLOCK 10: Split Polygon for Shear Areas >>>
        area_above, area_below, _, _ = split_polygon_by_line(pile_cap_vertices, one_way_shear_1_line_type, one_way_shear_1_line_value)
        area_B = min(area_above or 0, area_below or 0)
        results['shear_area_B_sqft'] = area_B
        calculation_steps.append(f"Area B (Shear Section 1 - Min Area): {area_B:.2f} sq.ft.")
        area_right_s2, area_left_s2, _, _ = split_polygon_by_line(pile_cap_vertices, one_way_shear_2_line_type, one_way_shear_2_line_value)
        area_A = min(area_right_s2 or 0, area_left_s2 or 0)
        results['shear_area_A_sqft'] = area_A
        calculation_steps.append(f"Area A (Shear Section 2 - Min Area): {area_A:.2f} sq.ft.")
        calculation_steps.append("Block 10: Shear Area Splitting Complete.")

        # <<< BLOCK 11: Calculate Shear Intersection Lengths >>>
        intersection_length_reg = polygon_intersection(pile_cap_vertices, one_way_shear_1_line_type, one_way_shear_1_line_value) or 0
        results['shear_intersection_len_1_ft'] = intersection_length_reg
        calculation_steps.append(f"Intersection length for shear line 1 (regular/horizontal): {intersection_length_reg:.3f} ft")
        intersection_length_ver = polygon_intersection(pile_cap_vertices, one_way_shear_2_line_type, one_way_shear_2_line_value) or 0
        results['shear_intersection_len_2_ft'] = intersection_length_ver
        calculation_steps.append(f"Intersection length for shear line 2 (vertical): {intersection_length_ver:.3f} ft")
        calculation_steps.append("Block 11: Shear Intersection Length Calculation Complete.")

        # <<< BLOCK 12: Calculate Concrete/Soil Weights for Shear Areas >>>
        concrete_wt_B, soil_wt_B = calculate_concrete_soil_weight(area_B, pile_cap_thickness, soil_depth_above, concrete_density, soil_density)
        results['concrete_weight_shear_B_kips'] = concrete_wt_B; results['soil_weight_shear_B_kips'] = soil_wt_B
        calculation_steps.append(f"Concrete/Soil weight over Area B: {concrete_wt_B:.3f} / {soil_wt_B:.3f} kips")
        concrete_wt_A, soil_wt_A = calculate_concrete_soil_weight(area_A, pile_cap_thickness, soil_depth_above, concrete_density, soil_density)
        results['concrete_weight_shear_A_kips'] = concrete_wt_A; results['soil_weight_shear_A_kips'] = soil_wt_A
        calculation_steps.append(f"Concrete/Soil weight over Area A: {concrete_wt_A:.3f} / {soil_wt_A:.3f} kips")
        calculation_steps.append("Block 12: Shear Area Weight Calculation Complete.")

        # <<< BLOCK 13: Analyze Pile Distribution for Shear >>>
        (_, _, piles_above_s1, piles_below_s1, _, _, intersected_piles_s1, total_reaction_above_1, _, _) = analyze_pile_distribution_with_reactions(
            polygon_vertices=pile_cap_vertices, line_type=one_way_shear_1_line_type, line_value=one_way_shear_1_line_value,
            pile_layout=pile_layout, pile_size=pile_size, pile_reactions=reactions)
        results['total_reaction_shear_1_above_kips'] = total_reaction_above_1; results['piles_shear_1_above'] = piles_above_s1
        results['piles_shear_1_below'] = piles_below_s1; results['piles_shear_1_intersected'] = intersected_piles_s1
        calculation_steps.append(f"Total reaction above shear line 1: {total_reaction_above_1:.2f} kips")
        (_, _, piles_right_s2, piles_left_s2, _, _, intersected_piles_s2, total_reaction_above_2, _, _) = analyze_pile_distribution_with_reactions(
            polygon_vertices=pile_cap_vertices, line_type=one_way_shear_2_line_type, line_value=one_way_shear_2_line_value,
            pile_layout=pile_layout, pile_size=pile_size, pile_reactions=reactions)
        results['total_reaction_shear_2_above_kips'] = total_reaction_above_2; results['piles_shear_2_right'] = piles_right_s2
        results['piles_shear_2_left'] = piles_left_s2; results['piles_shear_2_intersected'] = intersected_piles_s2
        calculation_steps.append(f"Total reaction right of shear line 2 ('above'): {total_reaction_above_2:.2f} kips")
        calculation_steps.append("Block 13: Pile Distribution Analysis for Shear Complete.")

        # <<< BLOCK 14: Calculate Net One-Way Shear >>>
        one_way_shear_section_1 = total_reaction_above_1 - concrete_wt_B - soil_wt_B
        one_way_shear_section_2 = total_reaction_above_2 - concrete_wt_A - soil_wt_A
        results['net_one_way_shear_1_kips'] = one_way_shear_section_1; results['net_one_way_shear_2_kips'] = one_way_shear_section_2
        calculation_steps.append(f"Net one-way shear at Section 1: {one_way_shear_section_1:.2f} kips")
        calculation_steps.append(f"Net one-way shear at Section 2: {one_way_shear_section_2:.2f} kips")
        calculation_steps.append("Block 14: Net One-Way Shear Calculation Complete.")

        # <<< BLOCK 15: Calculate Concrete Shear Capacity >>>
        shear_capacity_1 = concrete_shear_capacity_simple(lambda_factor, concrete_strength_psi, intersection_length_reg, pile_cap_shear_depth)
        if shear_capacity_1 == 0 and intersection_length_reg == 0:
            intersection_length_mom1 = polygon_intersection(pile_cap_vertices, one_way_moment_1_line_type, one_way_moment_1_line_value) or 0
            calculation_steps.append(f"Shear line 1 intersection zero, checking moment line 1 intersection: {intersection_length_mom1:.3f} ft")
            shear_capacity_1 = concrete_shear_capacity_simple(lambda_factor, concrete_strength_psi, intersection_length_mom1, pile_cap_shear_depth)
        results['shear_capacity_1_kips'] = shear_capacity_1
        calculation_steps.append(f"Design shear capacity φVc (kips) along Section 1: {shear_capacity_1:.3f}")
        shear_capacity_2 = concrete_shear_capacity_simple(lambda_factor, concrete_strength_psi, intersection_length_ver, pile_cap_shear_depth)
        if shear_capacity_2 == 0 and intersection_length_ver == 0:
            intersection_length_mom2 = polygon_intersection(pile_cap_vertices, one_way_moment_2_line_type, one_way_moment_2_line_value) or 0
            calculation_steps.append(f"Shear line 2 intersection zero, checking moment line 2 intersection: {intersection_length_mom2:.3f} ft")
            shear_capacity_2 = concrete_shear_capacity_simple(lambda_factor, concrete_strength_psi, intersection_length_mom2, pile_cap_shear_depth)
        results['shear_capacity_2_kips'] = shear_capacity_2
        calculation_steps.append(f"Design shear capacity φVc (kips) along Section 2: {shear_capacity_2:.3f}")
        calculation_steps.append("Block 15: Concrete Shear Capacity Calculation Complete.")

        # <<< BLOCK 16: Split Polygon for Moment Areas >>>
        _, _, area_moment_above_m1, area_moment_below_m1 = split_polygon_by_line(pile_cap_vertices, one_way_moment_1_line_type, one_way_moment_1_line_value)
        first_moment_area_D = min(area_moment_above_m1 or 0, area_moment_below_m1 or 0)
        results['moment_first_moment_area_D_ft3'] = first_moment_area_D
        calculation_steps.append(f"First moment of area D (Moment Section 1): {first_moment_area_D:.2f} ft³")
        _, _, area_moment_right_m2, area_moment_left_m2 = split_polygon_by_line(pile_cap_vertices, one_way_moment_2_line_type, one_way_moment_2_line_value)
        first_moment_area_C = min(area_moment_right_m2 or 0, area_moment_left_m2 or 0)
        results['moment_first_moment_area_C_ft3'] = first_moment_area_C
        calculation_steps.append(f"First moment of area C (Moment Section 2): {first_moment_area_C:.2f} ft³")
        calculation_steps.append("Block 16: Moment Area Splitting Complete.")

        # <<< BLOCK 17: Calculate Concrete/Soil Moments >>>
        concrete_moment_D, soil_moment_D = calculate_concrete_soil_weight(first_moment_area_D, pile_cap_thickness, soil_depth_above, concrete_density, soil_density)
        results['concrete_moment_D_kip_ft'] = concrete_moment_D; results['soil_moment_D_kip_ft'] = soil_moment_D
        calculation_steps.append(f"Concrete/Soil moment (Section D): {concrete_moment_D:.3f} / {soil_moment_D:.3f} kip-ft")
        concrete_moment_C, soil_moment_C = calculate_concrete_soil_weight(first_moment_area_C, pile_cap_thickness, soil_depth_above, concrete_density, soil_density)
        results['concrete_moment_C_kip_ft'] = concrete_moment_C; results['soil_moment_C_kip_ft'] = soil_moment_C
        calculation_steps.append(f"Concrete/Soil moment (Section C): {concrete_moment_C:.3f} / {soil_moment_C:.3f} kip-ft")
        calculation_steps.append("Block 17: Concrete/Soil Moment Calculation Complete.")

        # <<< BLOCK 18: Analyze Pile Distribution & Moments >>>
        (_, _, piles_above_m1, piles_below_m1, _, _, intersected_piles_m1, _, _, intersected_pile_geoms_4) = analyze_pile_distribution_with_reactions(
            polygon_vertices=pile_cap_vertices, line_type=one_way_moment_1_line_type, line_value=one_way_moment_1_line_value,
            pile_layout=pile_layout, pile_size=pile_size, pile_reactions=reactions)
        results['piles_moment_1_above'] = piles_above_m1; results['piles_moment_1_below'] = piles_below_m1; results['piles_moment_1_intersected'] = intersected_piles_m1
        moment_above_4, moment_below_4 = calculate_moments_about_section_line(
            line_type=one_way_moment_1_line_type, line_value=one_way_moment_1_line_value, pile_layout=pile_layout, pile_reactions=reactions,
            piles_above=piles_above_m1, piles_below=piles_below_m1, intersected_pile_geoms=intersected_pile_geoms_4)
        results['pile_moment_section_4_above_kip_ft'] = moment_above_4
        calculation_steps.append(f"Pile moment above moment line 1 (Section 4): {moment_above_4:.2f} kip-ft")
        (_, _, piles_right_m2, piles_left_m2, _, _, intersected_piles_m2, _, _, intersected_pile_geoms_3) = analyze_pile_distribution_with_reactions(
            polygon_vertices=pile_cap_vertices, line_type=one_way_moment_2_line_type, line_value=one_way_moment_2_line_value,
            pile_layout=pile_layout, pile_size=pile_size, pile_reactions=reactions)
        results['piles_moment_2_right'] = piles_right_m2; results['piles_moment_2_left'] = piles_left_m2; results['piles_moment_2_intersected'] = intersected_piles_m2
        moment_above_3, moment_below_3 = calculate_moments_about_section_line(
            line_type=one_way_moment_2_line_type, line_value=one_way_moment_2_line_value, pile_layout=pile_layout, pile_reactions=reactions,
            piles_above=piles_right_m2, piles_below=piles_left_m2, intersected_pile_geoms=intersected_pile_geoms_3)
        results['pile_moment_section_3_above_kip_ft'] = moment_above_3
        calculation_steps.append(f"Pile moment right of moment line 2 ('above', Section 3): {moment_above_3:.2f} kip-ft")
        calculation_steps.append("Block 18: Pile Distribution & Moment Analysis Complete.")

        # <<< BLOCK 19: Calculate Net One-Way Moment & Steel Area >>>
        one_way_moment_section_3 = moment_above_3 - concrete_moment_C - soil_moment_C
        one_way_moment_section_4 = moment_above_4 - concrete_moment_D - soil_moment_D
        results['net_one_way_moment_section_1_kip_ft'] = one_way_moment_section_4; results['net_one_way_moment_section_2_kip_ft'] = one_way_moment_section_3
        calculation_steps.append(f"Net one-way moment at Section 1 (from M4): {one_way_moment_section_4:.2f} kip-ft")
        calculation_steps.append(f"Net one-way moment at Section 2 (from M3): {one_way_moment_section_3:.2f} kip-ft")
        intersection_len_mom1 = polygon_intersection(pile_cap_vertices, one_way_moment_1_line_type, one_way_moment_1_line_value) or 0
        intersection_len_mom2 = polygon_intersection(pile_cap_vertices, one_way_moment_2_line_type, one_way_moment_2_line_value) or 0
        results['moment_intersection_len_1_ft'] = intersection_len_mom1; results['moment_intersection_len_2_ft'] = intersection_len_mom2
        area_of_steel_section_1 = calculate_area_of_steel(one_way_moment_section_4, pile_cap_shear_depth)
        area_of_steel_section_2 = calculate_area_of_steel(one_way_moment_section_3, pile_cap_shear_depth)
        results['area_of_steel_section_1_sqin'] = area_of_steel_section_1; results['area_of_steel_section_2_sqin'] = area_of_steel_section_2
        calculation_steps.append(f"Area of steel for Section 1 (from Moment 4): {area_of_steel_section_1:.3f} sq. in.")
        calculation_steps.append(f"Area of steel for Section 2 (from Moment 3): {area_of_steel_section_2:.3f} sq. in.")
        calculation_steps.append("Block 19: Net Moment & Steel Area Calculation Complete.")

        # <<< BLOCK 20: Analyze Punching Shear Polygon >>>
        # Capture the shear_polygon_coords directly for plotting
        (total_reaction_outside, shear_polygon_coords_plot, shear_polygon_perimeter, inside_piles, outside_piles, intersecting_piles) = analyze_shear_polygon_reactions(
            polygon_vertices=pile_cap_vertices, pile_layout=pile_layout, pile_size=pile_size, column_centroid=column_centroid,
            column_size=column_size, shear_depth=pile_cap_shear_depth, pile_reactions=reactions)
        results['punching_shear_reaction_outside_kips'] = total_reaction_outside
        results['punching_shear_polygon_coords'] = [list(coord) for coord in shear_polygon_coords_plot] if shear_polygon_coords_plot else []
        results['punching_shear_perimeter_ft'] = shear_polygon_perimeter if shear_polygon_perimeter else 0.0
        results['punching_shear_inside_piles'] = inside_piles; results['punching_shear_outside_piles'] = outside_piles; results['punching_shear_intersecting_piles'] = intersecting_piles
        calculation_steps.append(f"Punching Shear Analysis: Reaction Outside={total_reaction_outside:.2f} kip, Perimeter={results['punching_shear_perimeter_ft']:.2f} ft")
        calculation_steps.append("Block 20: Punching Shear Polygon Analysis Complete.")

        # <<< BLOCK 21: Calculate Punching Shear Capacity & Utilization >>>
        Vc = calculate_punching_shear_capacity(
            column_size, concrete_strength_psi , pile_cap_shear_depth*12, results['punching_shear_perimeter_ft']*12, lambda_factor, column_location)
        results['punching_shear_capacity_Vc_kips'] = Vc
        calculation_steps.append(f"Punching shear capacity φVc = {Vc:.3f} kips")
        utilization_ratio = (total_reaction_outside / Vc * 100) if Vc > 0 else float('inf')
        results['punching_shear_utilization_percent'] = utilization_ratio
        calculation_steps.append(f"Punching shear utilization ratio = {utilization_ratio:.2f} %")
        calculation_steps.append("Block 21: Punching Shear Capacity Calculation Complete.")

        # --- Store section line definitions for potential use in plotting ---
        results['section_lines'] = {
            'shear_1': {'type': one_way_shear_1_line_type, 'value': one_way_shear_1_line_value},
            'shear_2': {'type': one_way_shear_2_line_type, 'value': one_way_shear_2_line_value},
            'moment_1': {'type': one_way_moment_1_line_type, 'value': one_way_moment_1_line_value},
            'moment_2': {'type': one_way_moment_2_line_type, 'value': one_way_moment_2_line_value},
        }
        # --- End Core Calculation Blocks ---

        # <<< Call Visualization Function >>>
        calculation_steps.append("--- Generating Visualization ---")
        try:
            fig, ax = plot_foundation_analysis(
                pile_cap_vertices=pile_cap_vertices,
                pile_layout=pile_layout,
                pile_size=pile_size,
                pile_reactions=reactions, # Pass the numpy array directly
                column_size=column_size,
                column_centroid=column_centroid,
                pile_cap_thickness=pile_cap_thickness,
                pile_cap_shear_depth=pile_cap_shear_depth,
                one_way_shear_1_line_type=one_way_shear_1_line_type,
                one_way_shear_1_line_value=one_way_shear_1_line_value,
                one_way_shear_2_line_type=one_way_shear_2_line_type,
                one_way_shear_2_line_value=one_way_shear_2_line_value,
                one_way_moment_1_line_type=one_way_moment_1_line_type,
                one_way_moment_1_line_value=one_way_moment_1_line_value,
                one_way_moment_2_line_type=one_way_moment_2_line_type,
                one_way_moment_2_line_value=one_way_moment_2_line_value,
                one_way_shear_section_1=one_way_shear_section_1,
                one_way_shear_section_2=one_way_shear_section_2,
                one_way_moment_section_4=one_way_moment_section_4, # Mapped to section 1 in plot
                one_way_moment_section_3=one_way_moment_section_3, # Mapped to section 2 in plot
                area_of_steel_section_1=area_of_steel_section_1,
                area_of_steel_section_2=area_of_steel_section_2,
                shear_polygon_coords=shear_polygon_coords_plot, # Use the captured variable
                total_reaction_outside=total_reaction_outside,
                shear_capacity_1=shear_capacity_1,
                shear_capacity_2=shear_capacity_2,
                punching_shear_capacity=Vc,
                utilization_ratio=utilization_ratio
            )
            results['figure'] = fig # Add fig to results
            results['axes'] = ax   # Add ax to results
            calculation_steps.append("Visualization generated successfully.")
        except Exception as plot_e:
            # Log the error but don't crash the whole analysis
            error_details_plot = traceback.format_exc()
            calculation_steps.append(f"ERROR generating plot: {plot_e}\n{error_details_plot}")
            warnings.append(f"Visualization failed: {plot_e}")
            results['figure'] = None # Indicate failure
            results['axes'] = None


        calculation_steps.append("--- Analysis Complete ---")
        results['status'] = 'success'
        results['warnings'] = warnings
        results['log'] = calculation_steps
        # 'figure' and 'axes' are already in results dict
        return results

    except ValueError as ve:
        calculation_steps.append(f"ERROR (ValueError): {ve}")
        # Include fig/ax (which will be None) in error return
        return {'status': 'error', 'message': f"ValueError: {ve}", 'log': calculation_steps, 'warnings': warnings, 'parsed_inputs': parsed_inputs, 'figure': fig, 'axes': ax}
    except ImportError as ie:
        calculation_steps.append(f"ERROR (ImportError): {ie}")
        # Include fig/ax (which will be None) in error return
        return {'status': 'error', 'message': f"ImportError: {ie}. Check module paths.", 'log': calculation_steps, 'warnings': warnings, 'parsed_inputs': parsed_inputs, 'figure': fig, 'axes': ax}
    except Exception as e:
        error_details = traceback.format_exc()
        calculation_steps.append(f"FATAL ERROR: {e}\n{error_details}")
        # Include fig/ax (which will be None) in error return
        return {'status': 'error', 'message': f"Unexpected Error: {e}", 'log': calculation_steps, 'warnings': warnings, 'parsed_inputs': parsed_inputs, 'figure': fig, 'axes': ax}

# --- Example Usage (Modified to show the plot) ---
if __name__ == '__main__':
    # Example form data dictionary (mimicking request.form.to_dict())
    example_form_data = {
        "column_size": "38.66, 1", "column_eccentricity": "0, 0", "column_centroid": "25, 9",
        "pile_cap_thickness": str(3 + 4/12), "pile_embedment": "1.0", "soil_depth_above": "2.0",
        "soil_density": "0.0", "concrete_density": "0.0", "concrete_strength_psi": "3000",
        "column_location": "interior", "lambda_factor": "1.0", "pile_shape": "square",
        "pile_size": "0.8863", "max_pile_compression": "120.0", "max_pile_tension": "120.0",
        "Fx": "0.0", "Fy": "0.0", "Fz": str(-30 * 120 * 1.29), "Mx": "0.0", "My": "0.0",
        "pile_layout_str": internal_defaults["pile_layout_str"], # Use default string
        "pile_cap_shape_type": "rectangle",
        "rect_bottom_left": "1, 2.5", # Example rectangle input
        "rect_top_right": "49, 15.5",
        "pile_cap_vertices_str": "", # Empty for rectangle case
        "assumed_cover_in": "3.0",
        "assumed_bar_dia_in": "10.0"
    }

    print("--- Running Analysis with Example Form Data ---")
    # Call the analysis function with the form data dictionary
    analysis_results = perform_pile_cap_analysis(example_form_data)

    # Process the results
    if analysis_results.get('status') == 'success':
        print("\n--- Analysis Successful ---")
        # Access results using keys from the returned dictionary
        print(f"\nInput Summary (Parsed):")
        # Example: print(f"  Column Size (Parsed): {analysis_results['parsed_inputs']['column_size']}")
        print(f"  Pile Cap Thickness: {analysis_results['parsed_inputs']['pile_cap_thickness']:.3f} ft")
        print(f"  Effective Shear Depth (d): {analysis_results.get('pile_cap_shear_depth_ft', 'N/A'):.3f} ft")
        print(f"  Number of Piles: {analysis_results.get('num_piles', 'N/A')}")

        print(f"\nPile Reactions:")
        print(analysis_results.get('pile_reactions_dataframe_str', 'N/A'))

        print(f"\nOne-Way Shear:")
        shear_1_cap = analysis_results.get('shear_capacity_1_kips', 1E-9)
        shear_2_cap = analysis_results.get('shear_capacity_2_kips', 1E-9)
        shear_1_util = (analysis_results.get('net_one_way_shear_1_kips', 0) / shear_1_cap * 100) if shear_1_cap != 0 else float('inf')
        shear_2_util = (analysis_results.get('net_one_way_shear_2_kips', 0) / shear_2_cap * 100) if shear_2_cap != 0 else float('inf')
        print(f"  Section 1: Vu={analysis_results.get('net_one_way_shear_1_kips', 'N/A'):.2f} kips | φVc={shear_1_cap:.2f} kips | Util={shear_1_util:.1f}%")
        print(f"  Section 2: Vu={analysis_results.get('net_one_way_shear_2_kips', 'N/A'):.2f} kips | φVc={shear_2_cap:.2f} kips | Util={shear_2_util:.1f}%")

        print(f"\nOne-Way Moment:")
        print(f"  Section 1 (Horiz): Mu={analysis_results.get('net_one_way_moment_section_1_kip_ft', 'N/A'):.2f} kip-ft | As={analysis_results.get('area_of_steel_section_1_sqin', 'N/A'):.3f} sq.in")
        print(f"  Section 2 (Vert) : Mu={analysis_results.get('net_one_way_moment_section_2_kip_ft', 'N/A'):.2f} kip-ft | As={analysis_results.get('area_of_steel_section_2_sqin', 'N/A'):.3f} sq.in")

        print(f"\nPunching (Two-Way) Shear:")
        print(f"  Vu = {analysis_results.get('punching_shear_reaction_outside_kips', 'N/A'):.2f} kips")
        print(f"  φVc = {analysis_results.get('punching_shear_capacity_Vc_kips', 'N/A'):.2f} kips")
        print(f"  Utilization = {analysis_results.get('punching_shear_utilization_percent', 'N/A'):.1f} %")
        print(f"  Perimeter (b0) = {analysis_results.get('punching_shear_perimeter_ft', 'N/A'):.2f} ft")

        if analysis_results.get('warnings'):
            print("\n--- Warnings ---")
            for warning in analysis_results['warnings']:
                print(f"- {warning}")

        # <<< Check for and display the plot >>>
        print("\n--- Visualization ---")
        if analysis_results.get('figure') and analysis_results.get('axes'):
            print("Plot generated. Displaying...")
            fig = analysis_results['figure']
            # fig.savefig("foundation_analysis_plot.png") # Optional: save the figure
            plt.show() # Display the plot interactively
        else:
            print("Plot generation failed or was skipped.")

    else:
        print("\n--- Analysis Failed ---")
        print(f"Error: {analysis_results.get('message', 'Unknown error')}")
        print("\n--- Calculation Log (contains error details) ---")
        for step in analysis_results.get('log', []):
            print(step)