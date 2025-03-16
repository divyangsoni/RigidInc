import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given Data
L = 10  # Pile cap length (ft)
B = 10  # Pile cap width (ft)
H = 4   # Pile cap depth (ft)
fc = 4000  # Concrete strength (psi)
fy = 60000  # Steel yield strength (psi)
cover = 3/12  # Cover in ft
d = H - cover - 1/12  # Effective depth
P = 450  # Column axial load (kips)
M_col = 45  # Column moment (kip-ft)
pile_capacity = 120  # Max pile capacity (kips)
pile_spacing = 6  # Pile spacing (ft)
pile_diameter = 2  # Pile diameter (ft)
soil_bearing_capacity = 2000 / 144  # psf converted to psi
phi_shear = 0.75
phi_flexure = 0.9

def pile_reactions(P, M_col, pile_spacing):
    """Calculate pile reactions considering axial load and moment."""
    e = M_col / P  # Eccentricity
    R1 = (P / 4) + (M_col / (2 * pile_spacing))
    R2 = (P / 4) - (M_col / (2 * pile_spacing))
    return [R1, R2, R2, R1]  # Reactions at piles in sequence

def shear_stress(x, y):
    """Shear stress function decreasing radially from the column."""
    r = np.sqrt(x**2 + y**2) + 1  # Avoid division by zero
    return P / (4 * np.pi * r**2)  # Shear stress reduces with distance

# 3D Shear Stress Visualization
x_vals = np.linspace(-L/2, L/2, 30)
y_vals = np.linspace(-B/2, B/2, 30)
X, Y = np.meshgrid(x_vals, y_vals)
Z = shear_stress(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)

# Pile Positions
pile_positions = [(-pile_spacing/2, -pile_spacing/2), (pile_spacing/2, -pile_spacing/2),
                  (-pile_spacing/2, pile_spacing/2), (pile_spacing/2, pile_spacing/2)]
for px, py in pile_positions:
    ax.scatter(px, py, 0, color='blue', s=150, label="Pile Locations")

# Diagonal Shear Planes
diag_x = [-pile_spacing/2, pile_spacing/2]
diag_y = [pile_spacing/2, -pile_spacing/2]
ax.plot(diag_x, diag_y, [0, 0], 'g--', label="Diagonal Shear Planes")
ax.set_xlabel("Pile Cap Width (ft)")
ax.set_ylabel("Pile Cap Length (ft)")
ax.set_zlabel("Shear Stress (psi)")
ax.set_title("3D Shear Stress Distribution & Failure Planes")
ax.legend()
plt.show()

# Reinforcement Design
Mu = P * (pile_spacing / 2) + M_col  # Moment at critical section (kip-ft)
As_flexure = (Mu * 12) / (0.9 * fy * d)  # Required steel area (sq. in)
As_min = 0.0018 * L * H * 144  # Minimum steel requirement
As_required = max(As_flexure, As_min)

# Shear Check
Vu = P / 2  # Shear force at critical section
phi_Vc = phi_shear * 2 * np.sqrt(fc) * L * d  # Concrete shear capacity
stirrup_spacing = max(3, min((phi_Vc / Vu) * d, 12)) if Vu > phi_Vc else "Not Required"

# Soil Bearing Check
bearing_pressure = P / (L * B)  # psi
bearing_check = "Pass" if bearing_pressure <= soil_bearing_capacity else "Fail"

# Print Design Results
print("Pile Cap Analysis & Design Results")
print(f"Pile Reactions: {pile_reactions(P, M_col, pile_spacing)} kips per pile")
print(f"Moment at Critical Section: {Mu:.2f} kip-ft")
print(f"Required Flexural Steel Area: {As_required:.2f} sq.in")
print(f"Shear Capacity of Concrete: {phi_Vc:.2f} kips")
print(f"Stirrup Spacing Required: {stirrup_spacing} inches")
print(f"Soil Bearing Check: {bearing_check}")
