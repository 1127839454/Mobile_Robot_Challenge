import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Manually click four corners to get image coordinates
clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)}: ({x}, {y})")

img = cv2.imread("imagevideo.jpg")  # Replace with your own image
cv2.imshow("Click 4 Corners (TL, TR, BR, BL)", img)
cv2.setMouseCallback("Click 4 Corners (TL, TR, BR, BL)", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Real-world coordinates of the four clicked corners (unit: cm)
world_coords = np.array([
    [-163, 316],      
    [163, 316],      
    [8, 16],
    [-8,16],        
], dtype=np.float32)

# Convert image coordinates to NumPy array
image_coords = np.array(clicked_points, dtype=np.float32)

# -------------------------------
# 3. Compute the Homography matrix
H, _ = cv2.findHomography(image_coords, world_coords)
np.savez("homography_data.npz", H=H)
print("Homography saved to homography_data.npz")

# -------------------------------
# 4. Test mapping multiple pixel points to the ground plane
pixel_test_list = [
    (309, 226),
    (247, 226),
    (108, 250),
    (73, 250)
]

pixel_array = np.array([[pt] for pt in pixel_test_list], dtype=np.float32)
world_mapped = cv2.perspectiveTransform(pixel_array, H)

# -------------------------------
# 5. Visualize the mapped world coordinates
fig, ax = plt.subplots()

# Draw the mapped area polygon
polygon = np.vstack([world_coords, world_coords[0]])
ax.plot(polygon[:, 0], polygon[:, 1], 'b-', label='Board Area')
ax.scatter(world_coords[:, 0], world_coords[:, 1], color='blue')

# Draw mapped test points
for i, pt in enumerate(world_mapped):
    x, y = pt[0]
    ax.plot(x, y, 'rx', markersize=10)
    ax.text(x + 0.2, y, f"{i+1}: ({x:.1f}, {y:.1f})", color='red')

ax.set_title("Pixel Points → Ground Coordinates Mapping (Unit: cm)")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
