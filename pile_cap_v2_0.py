import numpy as np
import pandas as pd

# Input Parameters
column_size = (5.0, 10.0)  # Column dimensions (x, y) in ft
column_eccentricity = (0.0, 0)  # Eccentricity of column from center (x, y) in ft
pile_cap_thickness = 5.5  # Pile cap thickness in ft
pile_embedment = 1.0  # Depth of pile embedment in pile cap (ft)
soil_depth_above = 2.0  # Soil depth above pile cap (ft)
soil_density = 120  # Soil density in pcf
concrete_density = 150  # Concrete density in pcf
pile_cap_shear_depth = 3.72  # Effective shear depth (ft)

# Pile Properties
pile_shape = "square"  # Either "square" or "circular"
pile_size = 2.0  # Pile size (ft)
max_pile_compression = 120  # Max pile resistance in compression (kips)
max_pile_tension = 150  # Max pile resistance in tension (kips)

# Column Loads (kips and kip-ft)
Fx, Fy, Fz = 0, 0.0, -1440  # Forces in kips
Mx, My = 0, 0  # Moments in kip-ft

# Pile Layout (x, y coordinates in ft) - measured from bottom-left corner
pile_layout = np.array([
    [2, 2], [8, 2], [14, 2],
    [2, 8], [8, 8], [14, 8],
    [2,14], [8,14], [14,14],
    [2,20], [8,20], [14,20]
])  # (ft)

# Pile Cap Shape (Vertices of pile cap)
pile_cap_vertices = np.array([
    [0, 0], [16, 0], [16, 22], [0, 22]
])  # Square pile cap

# Verify all piles are within the pile cap boundary
min_x, min_y = np.min(pile_cap_vertices, axis=0)
max_x, max_y = np.max(pile_cap_vertices, axis=0)

for i, (px, py) in enumerate(pile_layout):
    if not (min_x <= px <= max_x and min_y <= py <= max_y):
        raise ValueError(f"Pile {i+1} at ({px}, {py}) is outside the pile cap!")

# Compute Self-weight of Pile Cap and Soil
pile_cap_volume = 10 * 10 * pile_cap_thickness  # ft³
pile_cap_weight = (pile_cap_volume * concrete_density) / 1000  # kips

soil_volume = 10 * 10 * soil_depth_above  # ft³
soil_weight = (soil_volume * soil_density) / 1000  # kips

# Deduct pile embedment weight
pile_embedment_weight = (pile_embedment * pile_size**2 * len(pile_layout) * concrete_density) / 1000  # kips
total_weight = pile_cap_weight + soil_weight - pile_embedment_weight

# Compute centroid of pile locations
pile_centroid = np.mean(pile_layout, axis=0)

# Adjust loads for eccentricity
Mx_adj = Mx + Fz * column_eccentricity[1]  # Moment about x
My_adj = My - Fz * column_eccentricity[0]  # Moment about y

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

# Ensure piles are within capacity
for i, R in enumerate(reactions.flatten()):
    if R > max_pile_compression:
        print(f"WARNING: Pile {i+1} exceeds max compression ({R:.2f} > {max_pile_compression} kips)")
    if R < -max_pile_tension:
        print(f"WARNING: Pile {i+1} exceeds max tension ({abs(R):.2f} > {max_pile_tension} kips)")

# Display Results
results = pd.DataFrame({
    "Pile": np.arange(1, num_piles + 1),
    "X (ft)": pile_layout[:, 0],
    "Y (ft)": pile_layout[:, 1],
    "Reaction (kips)": reactions.flatten()
})

print("\nPile Reactions:\n", results)
