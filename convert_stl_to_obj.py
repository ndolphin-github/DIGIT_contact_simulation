"""
Convert STL to OBJ format for obj2mjcf processing
"""
import trimesh

# Load STL
mesh = trimesh.load("assets/ur5e/assets/Objects/Hole_hexa.STL")

# Export as OBJ
mesh.export("assets/ur5e/assets/Objects/Hole_hexa.obj")

print("✓ Converted")


#After running this conversion code run this line:
# Option 1: Smaller threshold (더 세밀한 분해)
# obj2mjcf --obj-dir assets\ur5e\assets\Objects --obj-filter "Hole_hexa" --save-mjcf --decompose --compile-model --overwrite --coacd-args.threshold 0.01 --coacd-args.max-convex-hull 30

# Option 2: Current (threshold 0.1, max 10개)
# obj2mjcf --obj-dir assets\ur5e\assets\Objects --obj-filter "Hole_hexa" --save-mjcf --decompose --compile-model --overwrite --coacd-args.threshold 0.1 --coacd-args.max-convex-hull 10