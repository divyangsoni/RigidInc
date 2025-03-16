# OpenSees TCL script for pile cap analysis with Drucker-Prager material
# Authors: K.Petek, P.Mackenzie-Helnwein, P.Arduino, U.Washington

wipe
model BasicBuilder -ndm 3 -ndf 3

# Define Material (Drucker-Prager)
nDMaterial DruckerPrager 1 30000 0.3 1000 35 15 0.1 0.2

# Define Nodes
node 1  0.0  0.0  0.0
node 2  1.0  0.0  0.0
node 3  1.0  1.0  0.0
node 4  0.0  1.0  0.0
node 5  0.0  0.0  1.0
node 6  1.0  0.0  1.0
node 7  1.0  1.0  1.0
node 8  0.0  1.0  1.0

# Define Elements (8-node Brick Elements)
element stdBrick 1  1 2 3 4 5 6 7 8 1

# Boundary Conditions (Fixing Base)
fix 1 1 1 1
fix 2 1 1 1
fix 3 1 1 1
fix 4 1 1 1

# Define Load Pattern
pattern Plain 1 Linear {
    load 7 0.0 0.0 -100.0  ;# Load applied at node 7
}

# Analysis Settings
constraints Transformation
numberer RCM
system Mumps
test NormDispIncr 1.0e-3 500 0
algorithm NewtonLineSearch
integrator LoadControl 5.0e-8
analysis Static

# Run Analysis
set ok [analyze 100]

if {$ok != 0} {
    puts "⚠️ Analysis failed, retrying with smaller step size..."
    integrator LoadControl 1.0e-8
    analyze 500
}

puts "🎉 Analysis complete!"
