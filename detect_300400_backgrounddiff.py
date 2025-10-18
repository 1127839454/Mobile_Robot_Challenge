import cv2
import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
from car import Car
import time
# Load homography matrix (unit: meters)
homography_file = 'homography_data.npz'
import math
def calc_perpendicular_instruction(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2

    if abs(y2 - y1) < 1e-3:
        print("[INFO] Original line is nearly horizontal.")
        return {
            "intersect": None,
            "angle_deg": 90,
            "direction": "none",
            "distance": 0
        }

    if abs(x2 - x1) < 1e-3:
        print("[INFO] Original line is nearly vertical.")
        return {
            "intersect": (xm, 0),
            "angle_deg": 0,
            "direction": "left" if xm < 0 else "right",
            "distance": abs(xm)
        }

    k = (y2 - y1) / (x2 - x1)
    k_perp = -1 / k
    x_intersect = xm - ym / k_perp
    angle = math.degrees(math.atan(abs(k_perp)))
    direction = "left" if x_intersect < 0 else "right"

    return {
        "intersect": (x_intersect, 0),
        "angle_deg": angle,
        "direction": direction,
        "distance": abs(x_intersect)
    }

def move_to_gap_by_points(car, p1, p2):
    info = calc_perpendicular_instruction(p1, p2)

    if info["intersect"] is None:
        print("[ERROR] Mid-perpendicular is vertical; no x-axis intersection.")
        return

    direction = info["direction"]
    distance = info["distance"]
    angle = info["angle_deg"]

    if direction == "right":
        print("[ACTION] Turn left 90")
        car.motor.setMotorModel(-1500, 1500)
        time.sleep(0.4)
    elif direction == "left":
        print("[ACTION] Turn right 90")
        car.motor.setMotorModel(1500, -1500)
        time.sleep(0.4)

    print(f"[ACTION] Move forward {distance:.1f} cm")
    distance = distance / 10 + 3
    duration = distance / 15.0
    car.motor.setMotorModel(1500, 1500)
    time.sleep(duration)
    car.motor.setMotorModel(0, 0)

    turn_angle = 180 - angle
    print(f"[ACTION] Turn to face gap, angle {turn_angle:.1f}")
    # right_turn_duration = turn_angle / 90.0 * 0.38
    right_turn_duration = (turn_angle / 90.0) * 0.37
    if turn_angle > 0:
        if direction == "right":
            car.motor.setMotorModel(1500, -1500)
        elif direction == "left":
            car.motor.setMotorModel(-1500, 1500)
        time.sleep(right_turn_duration)
        car.motor.setMotorModel(0, 0)



def pixel_to_world(H, pt):
    x, y = pt
    vec = np.array([x, y, 1.0])
    world = H @ vec
    world /= world[2]
    return world[:2]

def capture_background(cam, num_frames=30):
    frames = []
    for _ in range(num_frames):
        fb = cam.get_frame()
        arr = np.frombuffer(fb, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    bg = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return bg

def align_frame_to_background(bg, frame, max_features=2000, keep_percent=0.15):
    bg_gray    = cv2.cvtColor(bg,    cv2.COLOR_BGR2GRAY)
    fr_gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kps1, des1 = orb.detectAndCompute(bg_gray,    None)
    kps2, des2 = orb.detectAndCompute(fr_gray, None)
    if des1 is None or des2 is None or len(kps1)<4 or len(kps2)<4:
        return frame

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    keep = int(len(matches) * keep_percent)
    matches = matches[:max(keep, 4)]

    pts_bg    = np.float32([ kps1[m.queryIdx ].pt for m in matches ]).reshape(-1,1,2)
    pts_frame = np.float32([ kps2[m.trainIdx ].pt for m in matches ]).reshape(-1,1,2)

    H_align, mask = cv2.findHomography(pts_frame, pts_bg, cv2.RANSAC, 5.0)
    if H_align is None:
        return frame

    h, w = bg.shape[:2]
    frame_aligned = cv2.warpPerspective(frame, H_align, (w, h),
                                        flags=cv2.INTER_LINEAR)

    return frame_aligned


if __name__ == "__main__":
    H = np.load(homography_file)['H']

    # Robot is now defined as world origin, consistent with calibration
    robot_world = np.array([0.0, 0.0])

    cam = Camera()
    cam.start_stream()

    time.sleep(1.0)

    bg_frames = []
    for _ in range(20):
        fb = cam.get_frame()
        arr = np.frombuffer(fb, dtype=np.uint8)
        frm = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frm is not None:
            bg_frames.append(frm)
    bg = np.median(np.stack(bg_frames, axis=0), axis=0).astype(np.uint8)
    print(">> Background captured.")

    input(">> press to continue...")

    ground_points = []
    gap_center = None
    gap_y = None
    gap_world = None

    try:
        while len(ground_points) < 4:
            frame_bytes = cam.get_frame()
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("[WARN] Frame decode failed, skipping.")
                continue

            height, width = frame.shape[:2]


            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # edges = cv2.Canny(blur, 30, 100)
            # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            aligned = align_frame_to_background(bg, frame)

            diff = cv2.absdiff(bg, frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            cv2.imwrite('dbg_background.png', bg)          
            cv2.imwrite('dbg_frame.png', frame)             
            cv2.imwrite('dbg_diff_gray.png', gray)        
            cv2.imwrite('dbg_mask_raw.png', mask)                        

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            ground_points.clear()
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y , w, h = cv2.boundingRect(cnt)
                print(x, y, w, h )

                if area < 100:
                    continue
                if (y + h) > 0.5 * height:
                    tl = (x,y )
                    tr = (x + w, y)
                    bl = (x, y + h)
                    br = (x + w,  y  + h)

                    # Reordered for logical bottom-center reference: [bl, br, tr, tl]
                    corners_px = [tl, tr, br, bl]
                    corners_world = []
                    for pt in corners_px:
                         world_pt = pixel_to_world(H, pt)
                         x_world, y_world = world_pt[0], world_pt[1]
                         corners_world.append((x_world, y_world + 50))

                   # corners_world = [pixel_to_world(H, pt) for pt in corners_px]

                    tl_world = corners_world[0]  # Bottom-left
                    tr_world = corners_world[1]  # Bottom-right
                    br_world = corners_world[2]  # Top-right
                    bl_world = corners_world[3]  # Top-left


                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    for pt in corners_px:
                        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

                    print("[INFO] Detected object corners:")
                    for i, px in enumerate(corners_px):
                        print(f"   Pixel Corner {i+1}: {px}")

                    print("[INFO] Transformed to world coordinates:")
                    for i, wp in enumerate(corners_world):
                        print(f"   World Corner {i+1}: ({wp[0]:.2f}, {wp[1]:.2f})")

                    ground_points.append(br_world)
                    ground_points.append(bl_world)
                    cv2.imshow("Ground Object Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C). Exiting...")

    finally:
        cam.stop_stream()
        cam.close()
        cv2.destroyAllWindows()

        # ----- Visualize simplified robot map -----
        # ----- Visualize 2D robot map in cm -----
        # ----- Visualize multiple obstacles as horizontal lines -----
        if len(ground_points) >= 4 and len(ground_points) % 2 == 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(-60,60)
            ax.set_ylim(0, 220)
            ax.set_aspect('equal')
            ax.grid(True)

            # Plot Robot
            ax.plot(0, 0, 'ro', markersize=12, label='Robot')
            ax.text(5, 5, 'Robot', color='red')

            # Draw obstacle lines for every pair
            obstacle_count = len(ground_points) // 2
            for i in range(obstacle_count):
                p1 = ground_points[2*i]
                p2 = ground_points[2*i+1]

                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=4, label='Obstacle' if i == 0 else "")
                ax.text(p1[0], p1[1] + 10, f"Obj{i+1}", color='black')

            # Draw gap between first two objects
            obj1_center = np.mean([ground_points[0], ground_points[1]], axis=0)
            obj2_center = np.mean([ground_points[2], ground_points[3]], axis=0)
            gap_center = (obj1_center + obj2_center) / 2

            # Gap line
            ax.plot(gap_center[0], gap_center[1], 'bx', markersize=10, label='Gap Center')
            ax.text(gap_center[0] + 5, gap_center[1], 'Gap', color='blue')
            ax.plot([0, gap_center[0]], [0, gap_center[1]], 'b--', label='Path to Gap')

            # Scale bar
            ax.plot([50, 100], [-30, -30], 'k-', linewidth=2)
            ax.text(75, -45, '50 cm', ha='center')

            ax.set_title("Robot Navigation Map (Unit: cm)")
            ax.legend()
            plt.tight_layout()
            plt.savefig("challenge_view.png")
            plt.show()
            car = Car()
            p1 = ground_points[1]
            p2 = ground_points[2]
            move_to_gap_by_points(car,p1, p2)
            time.sleep(2)
            car.motor.setMotorModel(1500, 1500)
            time.sleep(2)
            car.motor.setMotorModel(1500, 1500)
            time.sleep(3)
            

      
            time.sleep(3)



